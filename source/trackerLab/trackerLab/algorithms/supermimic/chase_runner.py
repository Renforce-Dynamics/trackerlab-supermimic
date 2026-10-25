"""Downstream task training with the SuperMimic imagined-gradient update."""

from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time


def train(args, app_launcher, run_start):
    import torch
    from torch import nn
    import trackerLab.algorithms.supermimic.networks as networks
    from trackerLab.algorithms.supermimic.networks import Critic, Policy
    from trackerLab.algorithms.supermimic.chase_networks import TaskPolicy, TaskWorldModel
    from trackerLab.algorithms.supermimic.chase_buffer import ChaseReplayBuffer
    from trackerLab.tracker_env.supermimic.env import make_sim, launch_simulation
    from trackerLab.tracker_env.supermimic.chase_dodge import ChaseDodgeSim
    from trackerLab.tracker_env.mdp.supermimic import RewardCfg, unflatten_state, normalize_xi
    from trackerLab.tracker_env.mdp.supermimic_chase_dodge import ChaseDodgeMDP, TERMS, advance_scenario
    from trackerLab.kinematics.supermimic.robot import RobotKinematics
    from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
    from trackerTask.supermimic.robots import get_robot
    from trackerTask.supermimic.go2_chase_dodge_cfg import ChaseDodgeCfg, make_task_cfg

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    device = args.device
    spec = get_robot("go2")
    cfg = make_task_cfg()
    cfg.num_envs, cfg.env_spacing = args.num_envs, args.env_spacing
    cfg.friction_range, cfg.reset_noise = args.friction_range, args.reset_noise
    task = ChaseDodgeCfg()
    value_bound = task.max_step_reward / (1 - args.gamma)
    kin = RobotKinematics(device, spec)
    motion = MotionLib.from_config(args.motion_config, kin, device)
    mdp = ChaseDodgeMDP(kin, motion, torch.tensor(spec.default_q, device=device),
                       cfg=RewardCfg(**cfg.reward_overrides), obs_mode=args.obs_mode, task_cfg=task)
    goal_dim = spec.goal_frame_dim * args.goal_frames
    pi = TaskPolicy(mdp.obs_dim, goal_dim, hist=args.hist, num_joints=12,
                    action_limit=args.action_limit).to(device)
    wm = TaskWorldModel(num_joints=12).to(device)
    critic = Critic(spec.state_dim + spec.context_dim + task.task_feature_dim + goal_dim).to(device)
    teacher = Policy(mdp.obs_dim, goal_dim, hist=args.hist, num_joints=12,
                     action_limit=args.action_limit).to(device)
    source_path = args.resume or args.warm_start
    if source_path is None:
        raise ValueError("Chase/dodge requires --warm_start from a Go2 gait or --resume from this task")
    source = torch.load(source_path, map_location=device, weights_only=False)
    if args.resume:
        if source["args"]["task"] != "go2_chase_dodge":
            raise ValueError("--resume requires a chase/dodge checkpoint")
        pi.load_state_dict(source["pi"])
        wm.load_state_dict(source["wm"])
        critic.load_state_dict(source["critic"])
        teacher.load_state_dict(source["teacher"])
    else:
        if source["args"]["task"] != "go2":
            raise ValueError("--warm_start requires a Go2 gait checkpoint")
        for field in ("hist", "goal_frames", "obs_mode", "action_limit"):
            if source["args"].get(field) != getattr(args, field):
                raise ValueError(f"Warm-start {field} must match the gait")
        pi.load_gait(source["pi"])
        wm.load_gait(source["wm"])
        teacher.load_state_dict(source["pi"])
        # Retain moments while allowing the task's dynamics distribution to adapt.
        wm.in_norm.count.fill_(100000)
        wm.out_norm.count.fill_(100000)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    start, sim_steps = 0, 0
    if args.resume:
        for name, opt in (("opt_pi", opt_pi), ("opt_wm", opt_wm), ("opt_v", opt_v)):
            opt.load_state_dict(source[name])
        start, sim_steps = source["iteration"], source["sim_steps"]
    if start >= args.iterations:
        raise ValueError("Target iteration must exceed the checkpoint")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    def save_json(name, value):
        (log_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    save_json("args.json", {"arguments": vars(args), "task": asdict(task)})
    save_json("reward_config.json", {"terms": TERMS, "max_step_reward": task.max_step_reward,
                                    "value_bound": task.max_step_reward / (1 - args.gamma),
                                    "source": "Local downstream task design; not the unreleased video author's objective"})
    with open(source_path, "rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    initialization = {"source_checkpoint": source_path, "source_sha256": source_sha,
                      "source_iteration": source["iteration"], "source_task": source["args"]["task"],
                      "actor_loaded": True, "world_model_loaded": True, "critic_reset": not bool(args.resume)}
    save_json("transfer.json", initialization)
    save_json("motion_library.json", {"names": motion.names, "provenance": motion.provenance,
                                      "role": "reset state and auxiliary gait conditioning"})
    del source
    buffer = ChaseReplayBuffer(args.buffer_steps, args.num_envs, device)
    error_floor = torch.tensor([.003]*3 + [.005]*3 + [.02]*3 + [.05]*3 + [.003]*12 + [.05]*12, device=device)

    def next_frames(f):
        clip = motion.clip_id[f]
        return torch.where(f + 1 >= motion.clip_end[clip], motion.clip_start[clip], f + 1)

    def critic_input(S, f, xi, c):
        return torch.cat((mdp.task_critic_features(S, xi, c), motion.goal(f, args.goal_frames)), -1)

    def value(x):
        raw = critic(x)
        return raw.clamp(max=value_bound) if args.bounded_critic else raw

    def wm_error(window):
        S = unflatten_state(window["s"][:, 0])
        estimates = []
        for j in range(args.wm_window):
            S = wm.step(S, window["a"][:, j], normalize_xi(window["xi"][:, j]), window["scenario"][:, j])
            estimates.append(S)
        stacked = tuple(torch.stack([s[k] for s in estimates], 1) for k in range(6))
        error = (wm.state_error(stacked, unflatten_state(window["s1"])) /
                 torch.maximum(wm.out_norm.std(), error_floor)).square().mean(-1)
        return (error * window["mask"]).sum() / window["mask"].sum().clamp_min(1)

    def imag_step(S, hist, f, xi, c):
        a = pi(hist, motion.goal(f, args.goal_frames), mdp.task_features(S, c))
        S1 = wm.step(S, a, normalize_xi(xi), c)
        f1, c1 = next_frames(f), advance_scenario(c)
        obs = mdp.add_noise(mdp.obs(S1, a, xi, f1, c1))
        return S1, a, torch.cat((hist[:, 1:], obs[:, None]), 1), f1, c1

    def reward_head(S, f, a, previous, c):
        return mdp.task_reward(S, f, a, previous, c), mdp.task_soft_termination(S, c)

    wm_error_fn = torch.compile(wm_error) if args.compile else wm_error
    imag_fn = torch.compile(imag_step) if args.compile else imag_step
    reward_fn = torch.compile(reward_head) if args.compile else reward_head
    skipped = 0

    def policy_update(iteration):
        nonlocal skipped
        s, h = buffer.sample_starts(args.pi_batch, args.hist)
        S = unflatten_state(s["s"])
        f, xi, c, previous = s["f"], s["xi"], s["scenario"], s["a_prev"]
        with torch.no_grad():
            hist = mdp.add_noise(mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"], h["scenario"]))
            teacher_action = teacher(hist, motion.goal(f, args.goal_frames))
        for net in (wm, critic):
            for p in net.parameters():
                p.requires_grad_(False)
        X = [critic_input(S, f, normalize_xi(xi), c).detach()]
        rewards, deaths, actions = [], [], []
        for _ in range(args.horizon):
            S, a, hist, f, c = imag_fn(S, hist, f, xi, c)
            r, d = reward_fn(S, f, a, previous, c)
            rewards.append(r)
            deaths.append(d)
            actions.append(a)
            X.append(critic_input(S, f, normalize_xi(xi), c).detach())
            previous = a
        r, d = torch.stack(rewards), torch.stack(deaths)
        alive = torch.cumprod(1 - d, 0)
        before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        discount = args.gamma ** torch.arange(args.horizon, device=device)[:, None]
        objective = (discount * before * r).sum(0)
        objective = objective + args.gamma**args.horizon * alive[-1] * value(critic_input(S, f, normalize_xi(xi), c))
        anchor_weight = 0.1 * max(0, 1 - iteration / 2500)
        anchor = (actions[0] - teacher_action).square().mean()
        loss = -objective.mean() + anchor_weight * anchor
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite task objective")
        opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(pi.parameters(), 1.)
        nn.utils.clip_grad_value_(pi.parameters(), .1)
        if torch.isfinite(grad):
            opt_pi.step()
        else:
            skipped += 1
        for net in (wm, critic):
            for p in net.parameters():
                p.requires_grad_(True)
        X = torch.stack(X)
        critic.norm.update(X)
        with torch.no_grad():
            V = value(X)
            raw_values = critic(X)
            R = V[-1]
            targets = []
            for t in range(args.horizon - 1, -1, -1):
                R = r[t].detach() + args.gamma * (1 - d[t].detach()) * ((1 - args.lam) * V[t+1] + args.lam * R)
                targets.append(R)
            Y = torch.stack(targets[::-1]).flatten()
        xf = X[:-1].flatten(0, 1)
        for _ in range(args.critic_epochs):
            for idx in torch.randperm(len(Y), device=device).tensor_split(args.critic_minibatches):
                vloss = (critic(xf[idx]) - Y[idx]).square().mean()
                opt_v.zero_grad(set_to_none=True)
                vloss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
                opt_v.step()
        return [float(loss.detach()), float(vloss.detach()), float(r.detach().mean()/.1),
                float(grad), float(alive[-1].detach().mean()), float((raw_values > value_bound).float().mean())]

    sim_cfg = make_sim(device, args, cfg)
    with launch_simulation(sim_cfg, args, app_launcher=app_launcher):
        env = ChaseDodgeSim(sim_cfg, args.num_envs, motion, mdp, cfg, gamma=args.gamma)
        history = torch.zeros(args.num_envs, args.hist, mdp.obs_dim, device=device)
        stats = []

        @torch.no_grad()
        def collect(n):
            nonlocal history, sim_steps
            for _ in range(n):
                S = env.state()
                o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f, env.scenario))
                history = torch.where((env.ep_step == 0)[:, None, None], o[:, None].expand_as(history),
                                      torch.cat((history[:, 1:], o[:, None]), 1))
                features = mdp.task_features(S, env.scenario)
                pi.task_norm.update(features)
                a = pi(history, motion.goal(env.f, args.goal_frames), features)
                a = (a + args.action_noise * torch.randn_like(a)).clamp(-4, 4)
                out = env.step(a)
                buffer.add(out)
                S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
                wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
                wm.out_norm.update(wm.targets(S0, S1))
                wm.bar_norm.update(wm.task_inputs(S0, out["scenario"]))
                sim_steps += args.num_envs
                done = out["done"]
                denominator = done.sum().clamp_min(1)
                task_done = done & (out["episode_mode"] > 0)
                stats.append(torch.stack((
                    out["r"].mean()/.1, (out["terminated"].float().sum()/denominator),
                    out["collision"].float().mean(), (out["ep_ret_norm"]*done).sum()/denominator,
                    (out["episode_success"] & task_done).sum()/task_done.sum().clamp_min(1),
                    *[out["reward_terms"][k].mean()/.1 for k in TERMS],
                )))

        print(f"Transfer gait into chase/dodge: {args.num_envs} envs, {args.iterations} task iterations", flush=True)
        collect(max(args.warmup_steps, args.wm_window))
        begin, last_log = time.monotonic(), time.monotonic()
        header = ["iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_rew",
                  "pi_gradnorm", "imag_survival", "value_clipped_fraction", "sim_rew", "term_frac",
                  "collision_step_fraction", "ep_ret_norm", "bar_success", *["reward_"+k for k in TERMS],
                  "it_per_s", "difficulty", "pi_skipped_updates", "peak_allocated_gib"]
        updates, step_acc = [], 0.
        with (log_dir / "train_log.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            stop_reason = "iterations"
            for it in range(start, args.iterations):
                env.difficulty = min(1., max(0., (it - 300) / task.curriculum_iterations))
                lr = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(it/args.lr_decay_iters, 1.))))
                for opt in (opt_pi, opt_wm, opt_v):
                    for group in opt.param_groups:
                        group["lr"] = lr
                step_acc += 1 / args.utd
                count = int(step_acc)
                step_acc -= count
                collect(count)
                for _ in range(args.wm_batches):
                    wm_loss = wm_error_fn(buffer.sample_windows(args.wm_batch, args.wm_window))
                    opt_wm.zero_grad(set_to_none=True)
                    wm_loss.backward()
                    nn.utils.clip_grad_norm_(wm.parameters(), 1., error_if_nonfinite=True)
                    opt_wm.step()
                update = policy_update(it) if it >= args.wm_only_iters else [0.]*6
                updates.append([float(wm_loss.detach()), *update])
                stop = args.stop_file and Path(args.stop_file).exists()
                timeout = args.max_minutes > 0 and time.monotonic() - run_start >= args.max_minutes * 60
                final = it + 1 == args.iterations or stop or timeout
                if (it + 1) % args.log_every == 0 or final:
                    means = torch.tensor(updates).mean(0).tolist()
                    collected = torch.stack(stats).mean(0).cpu().tolist()
                    ips = len(updates)/(time.monotonic()-last_log)
                    row = [it+1, (time.monotonic()-begin)/3600, sim_steps, *means, *collected,
                           ips, env.difficulty, skipped, torch.cuda.max_memory_allocated()/1024**3]
                    if not all(math.isfinite(x) for x in row):
                        raise FloatingPointError("Nonfinite chase/dodge metrics")
                    if means[2] > 1e4:
                        raise FloatingPointError("Critic divergence threshold exceeded")
                    writer.writerow(row)
                    stream.flush()
                    print(f"it {it+1} | reward {collected[0]:.3f} | wm {means[0]:.4f} | "
                          f"v {means[2]:.4f} | difficulty {env.difficulty:.2f} | {ips:.2f} it/s", flush=True)
                    updates, stats, last_log = [], [], time.monotonic()
                if (it+1) % args.ckpt_every == 0 or final:
                    checkpoint = {
                        "pi": pi.state_dict(), "wm": wm.state_dict(), "critic": critic.state_dict(),
                        "teacher": teacher.state_dict(), "opt_pi": opt_pi.state_dict(),
                        "opt_wm": opt_wm.state_dict(), "opt_v": opt_v.state_dict(),
                        "args": vars(args), "iteration": it+1, "sim_steps": sim_steps,
                        "wall_time": time.monotonic()-begin, "task_cfg": asdict(task), "transfer": initialization,
                    }
                    for name in ("last.pt", f"ckpt_{it+1:06d}.pt"):
                        temporary = log_dir / (name + ".tmp")
                        torch.save(checkpoint, temporary)
                        temporary.replace(log_dir / name)
                if stop or timeout:
                    stop_reason = "stop_file" if stop else "time_limit"
                    break
        save_json("run_summary.json", {
            "start_iteration": start, "completed_iterations": it+1, "requested_iterations": args.iterations,
            "stop_reason": stop_reason, "sim_steps": sim_steps, "pi_skipped_updates": skipped,
            "elapsed_seconds": time.monotonic()-run_start,
        })

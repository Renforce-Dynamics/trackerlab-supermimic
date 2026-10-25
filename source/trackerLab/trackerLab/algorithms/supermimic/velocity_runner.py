"""A separate SuperMimic runner for repeated velocity-command dodge encounters."""

from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from trackerLab.tracker_env.mdp.supermimic import normalize_xi
from trackerLab.tracker_env.mdp.supermimic_velocity_dodge import advance_scenario, next_gait_frame


def adaptive_curriculum_step(difficulty, attempts, successes):
    """Both height groups must pass; easy high bars cannot mask low-bar failure."""
    if len(attempts) != 2 or len(successes) != 2:
        raise ValueError("Expected high-bar and low-bar counts")
    passed = all(n >= 250 and k / n >= 0.65 for n, k in zip(attempts, successes))
    return min(1.0, difficulty + 0.05) if passed else difficulty


def load_chase_initialization(pi, wm, teacher, source, args):
    """Transfer matching tensors only; changed reward starts a new task counter."""
    if source["args"]["task"] != "go2_chase_dodge":
        raise ValueError("--warm_start for velocity/dodge requires a chase/dodge checkpoint")
    for field in ("hist", "goal_frames", "obs_mode", "action_limit"):
        if source["args"].get(field) != getattr(args, field):
            raise ValueError(f"Warm-start {field} must match the chase checkpoint")
    pi.load_chase(source["pi"])
    wm.load_chase(source["wm"])
    teacher.load_state_dict(source["pi"])
    return {
        "kind": "warm_start_new_objective", "source_task": source["args"]["task"],
        "source_iteration": int(source["iteration"]), "task_start_iteration": 0,
        "actor_loaded": True, "world_model_loaded": True,
        "critic_reset": True, "optimizers_reset": True,
        "matching_actor_tensors": len(source["pi"]),
        "matching_world_model_tensors": len(source["wm"]),
    }


def load_velocity_task_initialization(pi, wm, teacher, source, args):
    """Transfer dynamics/policy into a new velocity task; reset task value and optimizers."""
    if source["args"].get("task") != "go2_velocity_dodge":
        raise ValueError("--warm_start_velocity requires a Go2 velocity/dodge checkpoint")
    for field in ("hist", "goal_frames", "obs_mode", "action_limit"):
        if source["args"].get(field) != getattr(args, field):
            raise ValueError(f"Warm-start {field} must match the source checkpoint")
    pi.load_state_dict(source["pi"])
    wm.load_state_dict(source["wm"])
    teacher.load_state_dict({key: value for key, value in source["pi"].items()
                             if not key.startswith(("command_norm.", "command_head."))})
    return {
        "kind": "warm_start_new_velocity_objective", "source_task": source["args"]["task"],
        "source_profile": source["args"].get("task_profile", "standard"),
        "source_iteration": int(source["iteration"]), "task_start_iteration": 0,
        "actor_loaded": True, "world_model_loaded": True,
        "critic_reset": True, "optimizers_reset": True,
        "matching_actor_tensors": len(source["pi"]),
        "matching_world_model_tensors": len(source["wm"]),
    }


def imagined_transition(pi, wm, mdp, motion, S, history, f, xi, c, previous, goal_frames,
                        spawn_difficulty=None):
    """One differentiable control step using the simulator's scenario transition."""
    a = pi(history, motion.goal(f, goal_frames), mdp.task_features(S, c))
    S1 = wm.step(S, a, normalize_xi(xi), c)
    reward_c, c1, _ = advance_scenario(c, S1, mdp.task_cfg, spawn_difficulty)
    f1 = next_gait_frame(f, motion)
    reward = mdp.task_reward(S1, f1, a, previous, reward_c)
    death = mdp.task_soft_termination(S1, reward_c)
    obs = mdp.add_noise(mdp.obs(S1, a, xi, f1, c1))
    history1 = torch.cat((history[..., 1:, :], obs.unsqueeze(-2)), -2)
    return S1, a, history1, f1, c1, reward, death


def train(args, app_launcher, run_start):
    from torch import nn
    import trackerLab.algorithms.supermimic.networks as networks
    from trackerLab.algorithms.supermimic.networks import Critic
    from trackerLab.algorithms.supermimic.chase_networks import TaskPolicy
    from trackerLab.algorithms.supermimic.velocity_networks import VelocityPolicy, VelocityWorldModel
    from trackerLab.algorithms.supermimic.velocity_buffer import VelocityReplayBuffer
    from trackerLab.tracker_env.supermimic.env import make_sim, launch_simulation
    from trackerLab.tracker_env.supermimic.velocity_dodge import VelocityDodgeSim
    from trackerLab.tracker_env.mdp.supermimic import RewardCfg, unflatten_state
    from trackerLab.tracker_env.mdp.supermimic_velocity_dodge import VelocityDodgeMDP, reward_term_names
    from trackerLab.kinematics.supermimic.robot import RobotKinematics
    from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
    from trackerTask.supermimic.robots import get_robot
    from trackerTask.supermimic.go2_velocity_dodge_cfg import get_task_cfg, make_task_cfg

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    device, task = args.device, get_task_cfg(args.task_profile)
    TERMS = reward_term_names(task)
    spec, cfg = get_robot("go2"), make_task_cfg()
    cfg.num_envs, cfg.env_spacing = args.num_envs, args.env_spacing
    cfg.friction_range, cfg.reset_noise = args.friction_range, args.reset_noise
    cfg.joint_offset_range = args.joint_offset_range
    cfg.torso_com_offset_range = tuple(args.com_offset_range)
    kin = RobotKinematics(device, spec)
    motion = MotionLib.from_config(args.motion_config, kin, device)
    mdp = VelocityDodgeMDP(kin, motion, torch.tensor(spec.default_q, device=device),
                           cfg=RewardCfg(**cfg.reward_overrides), obs_mode=args.obs_mode, task_cfg=task)
    goal_dim = spec.goal_frame_dim * args.goal_frames
    pi = VelocityPolicy(mdp.obs_dim, goal_dim, hist=args.hist, action_limit=args.action_limit).to(device)
    wm = VelocityWorldModel(geometry_only=task.autonomous_passage).to(device)
    critic = Critic(spec.state_dim + spec.context_dim + task.task_feature_dim + goal_dim).to(device)
    teacher = TaskPolicy(mdp.obs_dim, goal_dim, hist=args.hist, num_joints=12,
                         action_limit=args.action_limit).to(device)
    source_path = args.resume or args.warm_start_velocity or args.warm_start
    if source_path is None:
        raise ValueError("Velocity/dodge requires --warm_start chase.pt or --resume velocity.pt")
    source = torch.load(source_path, map_location=device, weights_only=False)
    start, sim_steps = 0, 0
    if args.resume:
        if source["args"]["task"] != "go2_velocity_dodge":
            raise ValueError("--resume requires a velocity/dodge checkpoint; use --warm_start for chase")
        if source.get("task_cfg") != asdict(task):
            raise ValueError("Velocity/dodge task configuration differs from the checkpoint")
        for field in ("hist", "goal_frames", "obs_mode", "action_limit"):
            if source["args"].get(field) != getattr(args, field):
                raise ValueError(f"Continuation {field} must match the checkpoint")
        for net, name in ((pi, "pi"), (wm, "wm"), (critic, "critic"), (teacher, "teacher")):
            net.load_state_dict(source[name])
        start, sim_steps = int(source["iteration"]), int(source["sim_steps"])
        initialization = {
            "kind": "continue_velocity_checkpoint", "source_task": "go2_velocity_dodge",
            "source_iteration": start, "task_start_iteration": start,
            "actor_loaded": True, "world_model_loaded": True,
            "critic_reset": False, "optimizers_reset": False, "origin": source.get("transfer"),
        }
    elif args.warm_start_velocity:
        initialization = load_velocity_task_initialization(pi, wm, teacher, source, args)
    else:
        initialization = load_chase_initialization(pi, wm, teacher, source, args)
    if start >= args.iterations:
        raise ValueError("Target iteration must exceed the task checkpoint")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    # Only continuation of this exact task loads optimizer state.
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    if args.resume:
        for name, opt in (("opt_pi", opt_pi), ("opt_wm", opt_wm), ("opt_v", opt_v)):
            opt.load_state_dict(source[name])
    saved_curriculum = source.get("curriculum_state", {}) if args.resume else {}
    del source
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def save_json(name, value):
        (log_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with open(source_path, "rb") as stream:
        initialization.update(source_checkpoint=str(source_path),
                              source_sha256=hashlib.file_digest(stream, "sha256").hexdigest())
    save_json("transfer.json", initialization)
    save_json("args.json", {"arguments": vars(args), "task": asdict(task)})
    save_json("reward_config.json", {
        "terms": TERMS, "positive_weights": task.positive_weights,
        "max_step_reward": task.max_step_reward, "value_bound": task.value_bound(args.gamma),
        "derivation": "sum(nonnegative term maxima) / (1 - gamma); all penalties are nonpositive",
        "source": "Local body-command/repeated-bar extension",
    })
    save_json("motion_library.json", {"names": motion.names, "provenance": motion.provenance,
        "role": "reset state, cyclic phase goal, auxiliary gait style"})
    buffer = VelocityReplayBuffer(args.buffer_steps, args.num_envs, device)
    error_floor = torch.tensor([.003]*3 + [.005]*3 + [.02]*3 + [.05]*3 + [.003]*12 + [.05]*12,
                               device=device)
    bound = task.value_bound(args.gamma)

    def critic_input(S, f, xi, c):
        return torch.cat((mdp.task_critic_features(S, normalize_xi(xi), c),
                          motion.goal(f, args.goal_frames)), -1)

    def value(x):
        raw = critic(x)
        return raw.clamp(max=bound) if args.bounded_critic else raw

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

    def imag_step(S, hist, f, xi, c, previous, spawn_difficulty=None):
        return imagined_transition(pi, wm, mdp, motion, S, hist, f, xi, c, previous,
                                    args.goal_frames, spawn_difficulty)

    wm_error_fn = torch.compile(wm_error) if args.compile else wm_error
    imag_fn = torch.compile(imag_step) if args.compile else imag_step
    skipped = 0

    def policy_update(iteration):
        nonlocal skipped
        s, h = buffer.sample_starts(args.pi_batch, args.hist)
        S = unflatten_state(s["s"])
        f, xi, c, previous = s["f"], s["xi"], s["scenario"], s["a_prev"]
        with torch.no_grad():
            hist = mdp.add_noise(mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"], h["scenario"]))
            teacher_action = teacher(hist, motion.goal(f, args.goal_frames), mdp.task_features(S, c)[..., :18])
        for net in (wm, critic):
            net.requires_grad_(False)
        X = [critic_input(S, f, xi, c).detach()]
        rewards, deaths, actions = [], [], []
        try:
            spawn_difficulty = curriculum_difficulty if gated_curriculum else None
            if args.task_difficulty is not None:
                spawn_difficulty = args.task_difficulty
            for _ in range(args.horizon):
                S, a, hist, f, c, r, d = imag_fn(
                    S, hist, f, xi, c, previous, spawn_difficulty)
                rewards.append(r)
                deaths.append(d)
                actions.append(a)
                X.append(critic_input(S, f, xi, c).detach())
                previous = a
            r, d = torch.stack(rewards), torch.stack(deaths)
            alive = torch.cumprod(1 - d, 0)
            before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
            discount = args.gamma ** torch.arange(args.horizon, device=device)[:, None]
            objective = (discount * before * r).sum(0)
            objective += args.gamma**args.horizon * alive[-1] * value(critic_input(S, f, xi, c))
            anchor_weight = 0.05 * max(0, 1 - iteration / 1500)
            if task.autonomous_passage:
                anchor_weight = 0.0
            loss = -objective.mean() + anchor_weight * (actions[0] - teacher_action).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite velocity/dodge objective")
            opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(pi.parameters(), 1.)
            nn.utils.clip_grad_value_(pi.parameters(), .1)
            command_grad = pi.command_head[-1].weight.grad.norm().detach()
            if torch.isfinite(grad):
                opt_pi.step()
            else:
                skipped += 1
        finally:
            for net in (wm, critic):
                net.requires_grad_(True)
        X = torch.stack(X)
        critic.norm.update(X)
        with torch.no_grad():
            V = value(X)
            raw_values = critic(X)
            R = V[-1]
            targets = []
            for t in range(args.horizon - 1, -1, -1):
                R = r[t].detach() + args.gamma * (1 - d[t].detach()) * (
                    (1 - args.lam) * V[t+1] + args.lam * R)
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
        return [float(loss.detach()), float(vloss.detach()), float(r.detach().mean()/task.max_step_reward),
                float(grad), float(alive[-1].detach().mean()),
                float((raw_values > bound).float().mean()), float(command_grad)]

    sim_cfg = make_sim(device, args, cfg)
    with launch_simulation(sim_cfg, args, app_launcher=app_launcher):
        env = VelocityDodgeSim(sim_cfg, args.num_envs, motion, mdp, cfg, gamma=args.gamma)
        env.difficulty = args.task_difficulty if args.task_difficulty is not None else task.difficulty(start)
        env.reset()
        history = torch.zeros(args.num_envs, args.hist, mdp.obs_dim, device=device)
        stats = []
        encounter_count, encounter_successes = 0, 0
        jump_curriculum = args.task_profile == "jump"
        adaptive_curriculum = task.autonomous_passage
        gated_curriculum = (jump_curriculum or adaptive_curriculum) and args.task_difficulty is None
        curriculum_difficulty = float(saved_curriculum.get("difficulty", 0.0))
        curriculum_events = list(saved_curriculum.get("history", []))
        height_attempts, height_successes = [0, 0], [0, 0]
        if gated_curriculum:
            env.difficulty = curriculum_difficulty
        window_attempts, window_successes = 0, 0
        natural_encounters, failed_encounters, timeout_encounters = 0, 0, 0
        interval_sum, deadline_misses, max_clearance_delay = 0.0, 0, 0.0
        crossing_interval_sum, crossing_intervals = 0.0, 0
        expired_ahead_encounters, lateral_bypass_encounters = 0, 0
        max_bar_pose_error = 0.0
        max_bar_rotation_error, max_encounter_index = 0.0, 0
        max_command_grad, max_orientation_grad = 0.0, 0.0

        @torch.no_grad()
        def collect(n):
            nonlocal history, sim_steps, encounter_count, encounter_successes
            nonlocal window_attempts, window_successes
            nonlocal max_bar_pose_error, max_bar_rotation_error, max_encounter_index
            nonlocal natural_encounters, failed_encounters, timeout_encounters
            nonlocal interval_sum, deadline_misses, max_clearance_delay
            nonlocal crossing_interval_sum, crossing_intervals
            nonlocal expired_ahead_encounters, lateral_bypass_encounters
            for _ in range(n):
                S = env.state()
                o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f, env.scenario))
                history = torch.where((env.ep_step == 0)[:, None, None], o[:, None].expand_as(history),
                                       torch.cat((history[:, 1:], o[:, None]), 1))
                features = mdp.task_features(S, env.scenario)
                pi.update_task_norms(features)
                a = pi(history, motion.goal(env.f, args.goal_frames), features)
                out = env.step((a + args.action_noise * torch.randn_like(a)).clamp(-4, 4))
                buffer.add(out)
                S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
                wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
                wm.out_norm.update(wm.targets(S0, S1))
                wm.bar_norm.update(wm.task_inputs(S0, out["scenario"]))
                wm.orientation_norm.update(wm.orientation_inputs(S0, out["scenario"]))
                sim_steps += args.num_envs
                encounter_count += int(out["encounter_end"].sum())
                encounter_successes += int(out["encounter_success"].sum())
                window_attempts += int(out["encounter_end"].sum())
                window_successes += int(out["encounter_success"].sum())
                if adaptive_curriculum:
                    for bucket, mode in enumerate((1, 2)):
                        selected = out["scenario"][:, 8] == mode
                        height_attempts[bucket] += int((out["encounter_end"] & selected).sum())
                        height_successes[bucket] += int((out["encounter_success"] & selected).sum())
                natural_encounters += int(out["encounter_natural_end"].sum())
                failed_encounters += int(out["encounter_failure"].sum())
                timeout_encounters += int(out["encounter_timeout"].sum())
                natural = out["encounter_natural_end"]
                interval_sum += float((out["encounter_elapsed_s"] * natural).sum())
                deadline_misses += int(out["encounter_deadline_miss"].sum())
                max_clearance_delay = max(max_clearance_delay, float(
                    (out["encounter_clearance_delay_s"] * out["encounter_end"]).max()))
                crossing_interval_sum += float(out["crossing_interval_s"].sum())
                crossing_intervals += int(out["crossing_interval_valid"].sum())
                expired_ahead_encounters += int(out["encounter_expired_ahead"].sum())
                lateral_bypass_encounters += int(out["encounter_lateral_bypass"].sum())
                max_bar_pose_error = max(max_bar_pose_error, float(out["bar_pose_error"].max()))
                max_bar_rotation_error = max(max_bar_rotation_error, float(out["bar_rotation_error"].max()))
                max_encounter_index = max(max_encounter_index, int(out["next_encounter_index"].max()))
                done = out["done"]
                denom = done.sum().clamp_min(1)
                stats.append(torch.stack((
                    out["r"].mean()/task.max_step_reward, out["terminated"].float().sum()/denom,
                    out["collision"].float().mean(), (out["ep_ret_norm"]*done).sum()/denom,
                    out["velocity_error_sq"].mean().sqrt(), out["yaw_error_sq"].mean().sqrt(),
                    *[out["reward_terms"][k].mean()/task.max_step_reward for k in TERMS],
                )))

        print(f"Velocity/dodge {initialization['kind']}: source iteration "
              f"{initialization['source_iteration']}, new task starts at {start}; "
              f"{args.num_envs} envs, difficulty {env.difficulty:.2f}", flush=True)
        collect(max(args.warmup_steps, args.wm_window))
        if jump_curriculum and gated_curriculum:
            initial_success = encounter_successes / max(encounter_count, 1)
            if encounter_count >= 250 and initial_success >= 0.65:
                curriculum_difficulty = 0.05
            env.difficulty = curriculum_difficulty
        begin, last_log = time.monotonic(), time.monotonic()
        header = [
            "iteration", "wall_h", "sim_steps", "wm_loss", "pi_loss", "v_loss", "imag_rew",
            "pi_gradnorm", "imag_survival", "value_clipped_fraction", "command_head_gradnorm", "orientation_adapter_gradnorm",
            "sim_rew", "term_frac", "collision_step_fraction", "ep_ret_norm", "velocity_rmse", "yaw_rmse",
            *["reward_"+k for k in TERMS], "it_per_s", "difficulty", "pi_skipped_updates",
            "attempted_encounters", "completed_encounters", "successful_encounters",
            "failed_encounters", "timeout_encounters", "max_encounter_index",
            "max_bar_pose_error_m", "max_bar_rotation_error", "peak_allocated_gib",
            "mean_completed_interval_s", "deadline_misses", "max_clearance_delay_s",
            "mean_crossing_interval_s", "measured_crossing_intervals",
            "expired_ahead_encounters", "lateral_bypass_encounters",
        ]
        updates, step_acc = [], 0.
        with (log_dir / "train_log.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            stop_reason = "iterations"
            for it in range(start, args.iterations):
                env.difficulty = (args.task_difficulty if args.task_difficulty is not None else
                                  curriculum_difficulty if gated_curriculum else task.difficulty(it))
                lr = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(it/args.lr_decay_iters, 1.))))
                for opt in (opt_pi, opt_wm, opt_v):
                    for group in opt.param_groups:
                        group["lr"] = lr
                step_acc += 1 / args.utd
                count = int(step_acc)
                step_acc -= count
                collect(count)
                if gated_curriculum and (it + 1) % 100 == 0:
                    rate = window_successes / max(window_attempts, 1)
                    previous = curriculum_difficulty
                    if adaptive_curriculum:
                        curriculum_difficulty = adaptive_curriculum_step(
                            curriculum_difficulty, height_attempts, height_successes)
                        env.difficulty = curriculum_difficulty
                    elif window_attempts >= 250 and rate >= 0.65:
                        curriculum_difficulty = min(1.0, curriculum_difficulty + 0.05)
                        env.difficulty = curriculum_difficulty
                    curriculum_events.append({
                        "iteration": it + 1, "difficulty_before": previous,
                        "difficulty_after": curriculum_difficulty,
                        "window_attempts": window_attempts, "window_successes": window_successes,
                        "window_success_rate": rate,
                        "height_attempts": list(height_attempts),
                        "height_successes": list(height_successes),
                    })
                    window_attempts, window_successes = 0, 0
                    height_attempts[:], height_successes[:] = [0, 0], [0, 0]
                for _ in range(args.wm_batches):
                    wm_loss = wm_error_fn(buffer.sample_windows(args.wm_batch, args.wm_window))
                    opt_wm.zero_grad(set_to_none=True)
                    wm_loss.backward()
                    nn.utils.clip_grad_norm_(wm.parameters(), 1., error_if_nonfinite=True)
                    orientation_grad = float(wm.orientation_adapter[-1].weight.grad.norm())
                    opt_wm.step()
                update = policy_update(it) if it >= args.wm_only_iters else [0.]*7
                max_command_grad = max(max_command_grad, update[-1])
                max_orientation_grad = max(max_orientation_grad, orientation_grad)
                updates.append([float(wm_loss.detach()), *update, orientation_grad])
                stop = args.stop_file and Path(args.stop_file).exists()
                timeout = args.max_minutes > 0 and time.monotonic() - run_start >= args.max_minutes * 60
                final = it + 1 == args.iterations or stop or timeout
                if (it + 1) % args.log_every == 0 or final:
                    means = torch.tensor(updates).mean(0).tolist()
                    collected = torch.stack(stats).mean(0).cpu().tolist()
                    ips = len(updates)/(time.monotonic()-last_log)
                    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.
                    row = [it+1, (time.monotonic()-begin)/3600, sim_steps, *means, *collected,
                           ips, env.difficulty, skipped, encounter_count, natural_encounters, encounter_successes,
                           failed_encounters, timeout_encounters, max_encounter_index,
                           max_bar_pose_error, max_bar_rotation_error, peak,
                           interval_sum / max(natural_encounters, 1), deadline_misses,
                           max_clearance_delay, crossing_interval_sum / max(crossing_intervals, 1),
                           crossing_intervals, expired_ahead_encounters, lateral_bypass_encounters]
                    if not all(math.isfinite(x) for x in row):
                        raise FloatingPointError("Nonfinite velocity/dodge metrics")
                    if means[2] > 1e4:
                        raise FloatingPointError("Critic divergence threshold exceeded")
                    writer.writerow(row)
                    stream.flush()
                    print(f"it {it+1} | reward {collected[0]:.3f} | wm {means[0]:.4f} | "
                          f"pi {means[1]:.4f} | v {means[2]:.4f} | command_grad {means[7]:.6f} | "
                          f"encounters {encounter_count} | difficulty {env.difficulty:.2f} | {ips:.2f} it/s",
                          flush=True)
                    updates, stats, last_log = [], [], time.monotonic()
                if (it+1) % args.ckpt_every == 0 or final:
                    checkpoint = {
                        "pi": pi.state_dict(), "wm": wm.state_dict(), "critic": critic.state_dict(),
                        "teacher": teacher.state_dict(), "opt_pi": opt_pi.state_dict(),
                        "opt_wm": opt_wm.state_dict(), "opt_v": opt_v.state_dict(),
                        "args": vars(args), "iteration": it+1, "sim_steps": sim_steps,
                        "wall_time": time.monotonic()-begin, "task_cfg": asdict(task), "transfer": initialization,
                        "curriculum_state": {
                            "difficulty": env.difficulty, "history": curriculum_events,
                        },
                    }
                    for name in ("last.pt", f"ckpt_{it+1:06d}.pt"):
                        temporary = log_dir / (name + ".tmp")
                        torch.save(checkpoint, temporary)
                        temporary.replace(log_dir / name)
                if stop or timeout:
                    stop_reason = "stop_file" if stop else "time_limit"
                    break
        save_json("run_summary.json", {
            "task": args.task, "start_iteration": start, "completed_iterations": it+1,
            "requested_iterations": args.iterations, "stop_reason": stop_reason, "sim_steps": sim_steps,
            "pi_skipped_updates": skipped, "attempted_encounters": encounter_count,
            "completed_encounters": natural_encounters, "failed_encounters": failed_encounters,
            "timeout_encounters": timeout_encounters,
            "successful_encounters": encounter_successes, "max_bar_pose_error_m": max_bar_pose_error,
            "max_encounter_index": max_encounter_index, "max_bar_rotation_error": max_bar_rotation_error,
            "max_command_head_gradnorm": max_command_grad, "max_orientation_adapter_gradnorm": max_orientation_grad,
            "finite_logged_metrics": True, "fixed_difficulty": args.task_difficulty,
            "curriculum": {
                "kind": ("success_gated_per_height" if adaptive_curriculum else
                         "success_gated_jump" if jump_curriculum else "iteration_linear"),
                "final_difficulty": env.difficulty,
                "success_threshold": 0.65 if gated_curriculum else None,
                "minimum_window_attempts": 250 if gated_curriculum else None,
                "history": curriculum_events,
            },
            "policy_update_count": max(0, it + 1 - max(start, args.wm_only_iters)),
            "elapsed_seconds": time.monotonic()-run_start, "initialization": initialization,
            "task_profile": args.task_profile,
            "mean_completed_interval_s": interval_sum / max(natural_encounters, 1),
            "deadline_misses": deadline_misses, "max_clearance_delay_s": max_clearance_delay,
            "mean_crossing_interval_s": crossing_interval_sum / max(crossing_intervals, 1),
            "measured_crossing_intervals": crossing_intervals,
            "expired_ahead_encounters": expired_ahead_encounters,
            "lateral_bypass_encounters": lateral_bypass_encounters,
        })

"""World-model fitting, differentiable policy optimization and training lifecycle."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time


def train(args, app_launcher, run_start):
    import torch
    import torch.nn as nn

    from trackerTask.supermimic.g1_env_cfg import SuperMimicG1TaskCfg
    from trackerLab.algorithms.supermimic.buffer import ReplayBuffer
    from trackerLab.tracker_env.supermimic.env import TrackingSim, launch_simulation, make_sim
    from trackerLab.kinematics.supermimic.g1 import G1Kinematics
    from trackerLab.tracker_env.mdp.supermimic import CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, REWARD_TERMS, invariant_features, normalize_xi, unflatten_state
    from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
    import trackerLab.algorithms.supermimic.networks as networks
    from trackerLab.algorithms.supermimic.networks import Critic, Policy, WorldModel
    from trackerTask.supermimic.robot import default_joint_pos_list

    task_cfg = SuperMimicG1TaskCfg()
    task_cfg.num_envs = args.num_envs
    task_cfg.env_spacing = args.env_spacing
    task_cfg.observation_mode = args.obs_mode
    task_cfg.reset_noise = args.reset_noise
    task_cfg.observation_history = args.hist
    task_cfg.goal_frames = args.goal_frames
    task_cfg.imagination_horizon = args.horizon
    task_cfg.replay_buffer_steps = args.buffer_steps
    task_cfg.world_model_window = args.wm_window
    task_cfg.world_model_updates_per_iteration = args.wm_batches
    task_cfg.world_model_batch_size = args.wm_batch
    task_cfg.policy_batch_size = args.pi_batch
    task_cfg.friction_range = tuple(args.friction_range)
    task_cfg.joint_offset_range = args.joint_offset_range
    task_cfg.torso_com_offset_range = tuple(args.com_offset_range)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = args.device
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(
        {"arguments": vars(args), "task": vars(task_cfg)},
        open(os.path.join(args.log_dir, "args.json"), "w"),
        indent=2,
    )

    kin = G1Kinematics(dev)
    motion = (MotionLib.from_config(args.motion_config, kin, dev) if args.motion_config
              else MotionLib.from_dir(args.motion_dir, kin, dev))
    with (Path(args.log_dir) / "motion_library.json").open("w") as stream:
        json.dump({"names": motion.names, "clip_lengths": motion.clip_len.cpu().tolist(),
                   "num_frames": motion.num_frames, "control_dt": task_cfg.control_dt,
                   "motion_config": args.motion_config, "motion_dir": args.motion_dir}, stream, indent=2)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), obs_mode=args.obs_mode)
    with open(Path(args.log_dir, "reward_config.json"), "w") as reward_file:
        json.dump({
            "source": "SuperMimic Table A.2; vibed a1508d42c9d1699a52e8233ab16d08c488441ab9",
            "config": asdict(mdp.cfg),
            "control_dt": task_cfg.control_dt,
            "max_step_reward": MAX_STEP_REWARD,
            "terms": REWARD_TERMS,
            "logged_contributions": "signed weighted per-step reward / max_step_reward",
            "body_reference": "XY and yaw aligned to robot root; velocities in world frame, following vibed",
            "joint_limits": "90 percent soft limits, following vibed/BeyondMimic",
        }, reward_file, indent=2)
    K, Hh, HZ = args.goal_frames, args.hist, args.horizon
    GOAL_DIM = 73 * K
    CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM

    wm = WorldModel().to(dev)
    pi = Policy(mdp.obs_dim, GOAL_DIM, hist=Hh).to(dev)
    critic = Critic(CRITIC_IN).to(dev)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    pi_skipped_updates = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        missing = {"critic", "opt_pi", "opt_wm", "opt_v"} - ck.keys()
        if missing:
            raise ValueError("Resume needs last.pt with critic and optimizer state; ckpt_*.pt and policy.pt are evaluation exports")
        if ck["args"].get("obs_mode", "paper_obs93") != args.obs_mode:
            raise ValueError("Observation mode must match the resumed checkpoint")
        if ck.get("reward_cfg", asdict(mdp.cfg)) != asdict(mdp.cfg):
            raise ValueError("Reward configuration must match the resumed checkpoint")
        for key in ("hist", "goal_frames"):
            if ck["args"][key] != getattr(args, key):
                raise ValueError(f"{key} must match the resumed checkpoint")
        wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
        opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
        start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
        pi_skipped_updates = ck.get("pi_skipped_updates", 0)
    if start_it >= args.iterations:
        raise ValueError("--iterations must exceed the resumed checkpoint iteration")


    # per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
    ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                           torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip


    def _wm_rollout(s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(args.wm_window):
            S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


    def _wm_err(S_hat, s1, mask):
        std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
        e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
        e = e.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1]


    def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
        a = pi(o_hist, g)
        S1 = wm.step(S, a, xi_n)
        f1 = f + 1
        o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, motion.goal(f1, K)


    def _reward_term(S, f, a, a_prev):
        return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


    def _critic_input(S, f, xi_n, g):
        return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


    if args.compile:
        wm_rollout = torch.compile(_wm_rollout)
        wm_err = torch.compile(_wm_err)
        imag_step = torch.compile(_imag_step)
        reward_term = torch.compile(_reward_term)
        critic_input_c = torch.compile(_critic_input)
    else:
        wm_rollout, wm_err = _wm_rollout, _wm_err
        imag_step, reward_term = _imag_step, _reward_term
        critic_input_c = _critic_input


    def lr_at(it: int) -> float:
        if it >= args.lr_decay_iters:
            return args.lr * 0.03
        return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


    def set_requires_grad(m: nn.Module, flag: bool):
        for p in m.parameters():
            p.requires_grad_(flag)


    # ------------------------------------------------------------------------------------------------
    def wm_update():
        w = buffer.sample_windows(args.wm_batch, args.wm_window)
        S_hat = wm_rollout(w["s"], w["a"], w["xi"])
        loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError("World-model loss is nonfinite")
        opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=True)
        opt_wm.step()
        m = w["mask"][:, -1]
        return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


    def sample_valid_starts(B: int):
        starts, hists, n = [], [], 0
        attempts = 0
        while n < B:
            attempts += 1
            if attempts > 64:
                raise RuntimeError("Replay contains too few frames for the imagination horizon; collect more data or reduce --horizon")
            s, h = buffer.sample_starts(2 * B, Hh)
            f = s["f"]
            ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)


    PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


    def policy_update():
        nonlocal pi_skipped_updates
        if PROFILE:
            torch.cuda.synchronize(); tq0 = time.time()
        s, h = sample_valid_starts(args.pi_batch)
        S = unflatten_state(s["s"])
        xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = mdp.add_noise(o_hist)
        set_requires_grad(wm, False)
        set_requires_grad(critic, False)
        B = args.pi_batch
        g0 = motion.goal(f, K)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for t in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not args.no_critic:
            J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g))
        loss = -J.mean()
        if PROFILE:
            torch.cuda.synchronize(); tq1 = time.time()
        opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if PROFILE:
            torch.cuda.synchronize(); tq2 = time.time()
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(pi.parameters(), 0.1)
        if torch.isfinite(gn):
            opt_pi.step()
        else:
            pi_skipped_updates += 1
        set_requires_grad(wm, True)
        set_requires_grad(critic, True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros(())
        if not args.no_critic:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            critic.norm.update(X)
            with torch.no_grad():
                V = critic(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])  # (H, B)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // args.critic_minibatches
            for _ in range(args.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
                    opt_v.step()
        if PROFILE:
            torch.cuda.synchronize()
            print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
        return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


    # ------------------------------------------------------------------------------------------------
    sim_cfg = make_sim(dev, args, task_cfg)
    with launch_simulation(sim_cfg, args, app_launcher=app_launcher):
        print(f"Creating {args.num_envs} environments.", flush=True)
        env = TrackingSim(
            sim_cfg,
            args.num_envs,
            motion,
            mdp,
            gamma=args.gamma,
            task_cfg=task_cfg,
        )
        buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
        obs_hist = torch.zeros(args.num_envs, Hh, mdp.obs_dim, device=dev)

        @torch.no_grad()
        def collect(n_steps: int, stats: dict):
            nonlocal obs_hist, sim_steps
            for _ in range(n_steps):
                S = env.state()
                o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
                fresh = env.ep_step == 0
                obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
                g = env.motion_manager.goal(K)
                pi.obs_norm.update(o)
                pi.goal_norm.update(g)
                a = pi(obs_hist, g)
                a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
                out = env.step(a)
                buffer.add(out)
                S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
                wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
                wm.out_norm.update(wm.targets(S0, S1))
                sim_steps += args.num_envs
                stats["r"].append(out["r"].mean() / MAX_STEP_REWARD)
                stats["reward_terms"].append(
                    torch.stack([out["reward_terms"][key].mean() for key in REWARD_TERMS]) / MAX_STEP_REWARD
                )
                if "ep_len" in out and bool(out["done"].any()):
                    finished = out["done"]
                    stats["ep_len"].append(out["ep_len"][finished].mean())
                    stats["ep_ret"].append(out["ep_ret_norm"][finished].mean())
                    stats["term"].append(out["term"].float().sum())
                    stats["done"].append(out["done"].float().sum())

        stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": [], "reward_terms": []}
        # Checkpoints omit replay tensors; refill on both a fresh run and resume.
        print(f"Collecting {max(args.warmup_steps, args.wm_window)} warmup steps.", flush=True)
        collect(max(args.warmup_steps, args.wm_window), stats)
        log_path = os.path.join(args.log_dir, "train_log.csv")
        new_log = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        log_header = (
            "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,"
            "sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,peak_allocated_gib,peak_reserved_gib,pi_skipped_updates,"
            + ",".join("reward_" + name for name in REWARD_TERMS)
            + ",actor_norm_count,wm_norm_count\n"
        )
        if not new_log:
            with open(log_path) as existing_log:
                if existing_log.readline() != log_header:
                    raise ValueError("Training CSV schema changed; resume into a new --log_dir")
        log_f = open(log_path, "a")
        if new_log:
            log_f.write(log_header)
        print(f"Training iterations {start_it + 1}..{args.iterations}; policy updates start after {args.wm_only_iters}.", flush=True)
        t_start = time.time() - wall_offset
        t_log = time.time()
        step_acc = 0.0
        log_iters = 0
        stop_reason = "iterations"
        agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        for it in range(start_it, args.iterations):
            lr = lr_at(it)
            for o in (opt_wm, opt_pi, opt_v):
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            step_acc += 1.0 / args.utd
            n = int(step_acc)
            step_acc -= n
            prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
            if prof:
                torch.cuda.synchronize(); tp0 = time.time()
            collect(n, stats)
            if prof:
                torch.cuda.synchronize(); tp1 = time.time()
            for _ in range(args.wm_batches):
                l, e8 = wm_update()
            if prof:
                torch.cuda.synchronize(); tp2 = time.time()
            agg["wm"].append(l), agg["wm8"].append(e8)
            if it >= args.wm_only_iters:
                pl, vl, ir, gn, surv = policy_update()
                if prof:
                    torch.cuda.synchronize(); tp3 = time.time()
                    print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
            log_iters += 1
            time_limit_reached = args.max_minutes > 0 and time.monotonic() - run_start >= args.max_minutes * 60
            stop_requested = args.stop_file is not None and Path(args.stop_file).exists()
            final_iteration = it + 1 == args.iterations or time_limit_reached or stop_requested
            if (it + 1) % args.log_every == 0 or final_iteration:
                mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
                ips = log_iters / (time.time() - t_log)
                t_log = time.time()
                log_iters = 0
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                       mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr,
                       torch.cuda.max_memory_allocated(dev) / 1024**3, torch.cuda.max_memory_reserved(dev) / 1024**3, pi_skipped_updates]  # fmt: skip
                row.extend(torch.stack(stats["reward_terms"]).mean(0).cpu().tolist())
                row.extend([float(pi.obs_norm.count), float(wm.in_norm.count)])
                log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
                log_f.flush()
                print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                      f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s "
                      f"| peak {row[16]:.1f} GiB | skipped {pi_skipped_updates}", flush=True)  # fmt: skip
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            if (it + 1) % args.ckpt_every == 0 or final_iteration:
                ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                      "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                      "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                      "reward_cfg": asdict(mdp.cfg),
                      "pi_skipped_updates": pi_skipped_updates, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
                last = os.path.join(args.log_dir, "last.pt")
                torch.save(ck, last + ".tmp")
                os.replace(last + ".tmp", last)
                small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "reward_cfg", "saved_at")}
                path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
                torch.save(small, path + ".tmp")
                os.replace(path + ".tmp", path)
            if time_limit_reached or stop_requested:
                stop_reason = "stop_file" if stop_requested else "time_limit"
                print(f"Stopping for {stop_reason} at iteration {it + 1}; checkpoint saved.", flush=True)
                break
        log_f.close()
        summary = {
            "start_iteration": start_it,
            "completed_iterations": it + 1,
            "requested_iterations": args.iterations,
            "stop_reason": stop_reason,
            "elapsed_seconds": time.monotonic() - run_start,
            "training_wall_seconds": time.time() - t_start,
            "sim_steps": sim_steps,
            "pi_skipped_updates": pi_skipped_updates,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / 1024**3,
            "checkpoint": str(Path(args.log_dir, "last.pt").resolve()),
        }
        with open(Path(args.log_dir, "run_summary.json"), "w") as summary_file:
            json.dump(summary, summary_file, indent=2)
        print(json.dumps(summary), flush=True)

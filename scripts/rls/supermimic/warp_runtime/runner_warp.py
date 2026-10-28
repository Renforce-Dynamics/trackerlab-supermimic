"""World-model fitting, differentiable policy optimization and training lifecycle."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time


def train(args, app_launcher, run_start):
    from runtime_settings import configure_environment
    configure_environment()
    if args.resume is not None:
        raise ValueError("The optional G1 Warp runtime starts from random initialization")
    import torch
    import torch.nn as nn

    from trackerTask.supermimic.robots import get_robot
    from warp_task_cfg import get_task_cfg
    from trackerLab.algorithms.supermimic.buffer import ReplayBuffer
    from warp_env import TrackingSim, launch_simulation, make_sim
    from trackerLab.kinematics.supermimic.robot import RobotKinematics
    from trackerLab.tracker_env.mdp.supermimic import MAX_STEP_REWARD, MDP, REWARD_TERMS, RewardCfg, invariant_features, normalize_xi, unflatten_state
    from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
    import trackerLab.algorithms.supermimic.networks as networks
    from trackerLab.algorithms.supermimic.networks import Critic, Policy, WorldModel
    from canonical_source_contract import canonical_contracts
    source_contracts = canonical_contracts(networks)
    norm_storage_contract = source_contracts["normalizer"]
    from continuation_audit import (
        adaptation_boundary, file_sha256, model_receipt, optimizer_receipt,
        value_sha256, write_receipt,
    )
    from restore_optimizer_steps import restore_eager_adam_step_counters
    goal_contract = source_contracts["goal"]
    from scratch_runtime_checks import record_initialization, check_goal_tail, check_actor_rollout, record_progress, record_optimizer_health
    spec = get_robot(args.task)
    task_cfg = get_task_cfg(args.task)
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = args.device
    if not torch.cuda.is_available():
        raise RuntimeError("The G1 Warp runtime requires a CUDA device")
    torch.cuda.set_device(dev)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device must support native BF16")
    os.makedirs(args.log_dir, exist_ok=True)
    (Path(args.log_dir) / "running_norm_storage_contract.json").write_text(
        json.dumps(norm_storage_contract, indent=2) + "\n")
    json.dump(
        {"arguments": vars(args), "task": vars(task_cfg)},
        open(os.path.join(args.log_dir, "args.json"), "w"),
        indent=2,
    )

    kin = RobotKinematics(dev, spec)
    from compact_gpu_motion import CompactGpuMotionLib
    if not args.motion_config:
        raise ValueError("Warp migration requires an explicit motion configuration")
    motion = CompactGpuMotionLib(args.motion_root_cache, kin, dev, args.motion_config)
    motion.storage_info['terminal_goal'] = 'Synthetic tail velocity zero; real final-frame velocity retained'
    check_goal_tail(args.log_dir, motion, args.goal_frames)
    (Path(args.log_dir) / 'goal_tail_contract.json').write_text(json.dumps(goal_contract, indent=2) + '\n')
    from static_reference_runtime import (
        install_static_root_fields, finalize_static_reference_fields)
    root_static_report = install_static_root_fields(motion)
    from resident_reference_bodies import install_resident_reference_bodies
    body_residency_report = install_resident_reference_bodies(motion, kin)
    all_reference_contract = finalize_static_reference_fields(motion)
    (Path(args.log_dir) / "static_reference_contract.json").write_text(
        json.dumps(all_reference_contract, indent=2) + "\n")
    (Path(args.log_dir) / 'body_residency.json').write_text(
        json.dumps(body_residency_report, indent=2) + '\n')
    with (Path(args.log_dir) / "motion_library.json").open("w") as stream:
        json.dump({"names": motion.names, "clip_lengths": motion.clip_len.cpu().tolist(),
                   "num_frames": motion.num_frames, "control_dt": task_cfg.control_dt,
                   "provenance": motion.provenance,
                   "storage": getattr(motion, "storage_info", {"backend": "device"}),
                   "motion_config": args.motion_config, "motion_dir": args.motion_dir}, stream, indent=2)
    mdp = MDP(kin, motion, torch.tensor(spec.default_q, device=dev),
              cfg=RewardCfg(**task_cfg.reward_overrides), obs_mode=args.obs_mode)
    from validate_resident_reference import validate_resident_reference
    validate_resident_reference(
        motion, kin, mdp, Path(args.log_dir) / 'body_residency_acceptance.json',
        compile_reward=True, sample_count=4096, gradient_rows=256)
    with open(Path(args.log_dir, "reward_config.json"), "w") as reward_file:
        json.dump({
            "task": spec.name,
            "task_overrides": task_cfg.reward_overrides,
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
    GOAL_DIM = spec.goal_frame_dim * K
    CRITIC_IN = spec.state_dim + spec.critic_extra + GOAL_DIM

    wm = WorldModel(num_joints=spec.num_joints).to(dev)
    pi = Policy(mdp.obs_dim, GOAL_DIM, hist=Hh, num_joints=spec.num_joints,
                action_limit=args.action_limit).to(dev)
    critic = Critic(CRITIC_IN).to(dev)
    from native_bf16_precision import contract as native_precision_contract
    precision_contract = native_precision_contract(
        networks, args, {'wm': wm, 'pi': pi, 'critic': critic},
        normalizer_source=norm_storage_contract)
    (Path(args.log_dir) / "precision_contract.json").write_text(
        json.dumps(precision_contract, indent=2) + "\n")
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    record_initialization(args.log_dir, args, pi, wm, critic, opt_pi, opt_wm, opt_v)
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    pi_skipped_updates = 0
    resume_audit = None
    counters = {
        "world_model_updates_this_run": 0,
        "policy_update_attempts_this_run": 0,
        "policy_optimizer_steps_this_run": 0,
        "critic_optimizer_steps_this_run": 0,
    }
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        if ck["args"].get("task", "g1") != args.task:
            raise ValueError("Robot task must match the resumed checkpoint")
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
        for key, default in (("action_limit", None), ("bounded_critic", False)):
            if ck["args"].get(key, default) != getattr(args, key):
                raise ValueError(f"{key} must match the resumed checkpoint")
        wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
        opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
        models = {"wm": wm, "pi": pi, "critic": critic}
        optimizers = {"wm": opt_wm, "pi": opt_pi, "critic": opt_v}
        source_models = {name: model_receipt(model, ck[name]) for name, model in models.items()}
        optimizer_before = {name: optimizer_receipt(opt) for name, opt in optimizers.items()}
        counters_moved = {
            name: restore_eager_adam_step_counters(opt) for name, opt in optimizers.items()
        }
        source_optimizers = {name: optimizer_receipt(opt) for name, opt in optimizers.items()}
        checkpoint_optimizer_keys = {"wm": "opt_wm", "pi": "opt_pi", "critic": "opt_v"}
        for name in optimizers:
            if optimizer_before[name]["state_sha256"] != source_optimizers[name]["state_sha256"]:
                raise RuntimeError("Adam scalar placement repair changed optimizer values")
            expected = value_sha256(ck[checkpoint_optimizer_keys[name]]["state"])
            source_optimizers[name]["source_parameter_state_sha256"] = expected
            source_optimizers[name]["parameter_state_equal_to_source"] = (
                expected == source_optimizers[name]["parameter_state_sha256"])
            if not source_optimizers[name]["parameter_state_equal_to_source"]:
                raise RuntimeError("Restored optimizer state differs from the source checkpoint")
        start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
        pi_skipped_updates = ck.get("pi_skipped_updates", 0)
        if "rng_state" in ck:
            torch.set_rng_state(ck["rng_state"]["torch_cpu"].cpu())
            torch.cuda.set_rng_state_all([value.cpu() for value in ck["rng_state"]["torch_cuda"]])
        source_config = Path(ck["args"]["motion_config"])
        if not source_config.is_file():
            raise ValueError("Original training motion configuration is required for continuation")
        original_config_hash = file_sha256(source_config)
        if original_config_hash != file_sha256(args.motion_config):
            raise ValueError("Continuation must retain the original training motion membership")
        source_config_data = json.loads(source_config.read_text())
        if [Path(name).stem for name in source_config_data["files"]] != motion.names:
            raise ValueError("Continuation motion cache order differs from original training")
        resume_audit = {
            "schema": "tracker-warp-continuation-resume-v1",
            "source_checkpoint": str(Path(args.resume).resolve()),
            "source_checkpoint_sha256": file_sha256(args.resume),
            "source_iteration": start_it,
            "source_sim_steps": sim_steps,
            "source_models": source_models,
            "source_optimizers": source_optimizers,
            "optimizer_before_scalar_repair": optimizer_before,
            "optimizer_scalar_counters_moved_to_cpu": counters_moved,
            "optimizer_values_unchanged_by_scalar_repair": True,
            "source_missing_sampler_state": "sampler_state" not in ck,
            "source_missing_rng_state": "rng_state" not in ck,
            "recreated_state": ["simulator", "replay", "observation_history"],
            "original_training_membership": {
                "checkpoint_config_path": str(source_config),
                "config_sha256": original_config_hash,
                "ordered_names_sha256": value_sha256(motion.names),
                "clips": len(motion.names),
                "frames": motion.num_frames,
                "cache_provenance": motion.provenance,
                "identical_to_original_config": True,
            },
            "observation_mode": args.obs_mode,
            "goal_tail": "Original MotionLib clamped terminal-frame velocities",
            "source_actor_optimizer_step_count": source_optimizers["pi"]["uniform_step_count"],
            "source_actor_skipped_updates": pi_skipped_updates,
            "adaptation_iterations": args.resume_wm_only_iters,
            "adaptation_boundary_iteration": start_it + args.resume_wm_only_iters,
            "first_actor_iteration": max(args.wm_only_iters, start_it + args.resume_wm_only_iters) + 1,
            "global_iteration_and_lr_schedule_retained": True,
            "counters": dict(counters),
        }
        write_receipt(Path(args.log_dir) / "resume_audit.json", resume_audit)
    if start_it >= args.iterations:
        raise ValueError("--iterations must exceed the resumed checkpoint iteration")


    # Isolated candidate: checkpoint loading is complete before these guards.
    from critic_runtime import frozen_normalizer_plan, make_critic_minibatch_loss, write_contract
    norm_guard_enabled = os.environ.get("SM_FROZEN_NORM_GUARD", "0") == "1"
    critic_compile_enabled = os.environ.get("SM_COMPILE_CRITIC_LOSS", "0") == "1"
    frozen_norms, norm_guard_plan = frozen_normalizer_plan({
        "pi.obs_norm": pi.obs_norm, "pi.goal_norm": pi.goal_norm,
        "wm.in_norm": wm.in_norm, "wm.out_norm": wm.out_norm,
        "critic.norm": critic.norm,
    }, enabled=norm_guard_enabled)
    critic_minibatch_loss = make_critic_minibatch_loss(
        critic, compile_enabled=critic_compile_enabled)
    write_contract(args.log_dir, norm_guard_plan,
                   compile_enabled=critic_compile_enabled)

    # per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
    ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                           torch.full((spec.num_joints,), 3e-3), torch.full((spec.num_joints,), 5e-2))).to(dev)  # fmt: skip


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


    def _wm_window(s, a, xi, s1, mask):
        states = _wm_rollout(s, a, xi)
        loss, tail = _wm_err(states, s1, mask)
        return loss, tail, states


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

    def value_estimate(x):
        value = critic(x)
        # Six positive tracking terms sum to MAX_STEP_REWARD; all other terms are nonpositive.
        # Bound bootstrap values, while fitting the raw critic so overestimates retain gradients.
        if args.bounded_critic:
            value = value.clamp(max=MAX_STEP_REWARD / (1 - args.gamma))
        return value

    graph_scope = os.environ.get("SM_GRAPH_SCOPE", "all")
    if graph_scope not in ("none", "wm", "all"):
        raise ValueError("SM_GRAPH_SCOPE must be none, wm or all")
    (Path(args.log_dir) / "graph_scope_contract.json").write_text(json.dumps({
        "scope": graph_scope, "all_references_separately_contiguous": True,
        "random_draw_expressions_changed": False,
        "rng_bitwise_parity_verified": False,
        "compiler_mode_changed": True,
    }, indent=2) + "\n")
    if args.compile:
        wm_rollout = torch.compile(_wm_rollout, mode="reduce-overhead" if graph_scope in ("wm", "all") else "default")
        wm_err = torch.compile(_wm_err, mode="reduce-overhead" if graph_scope in ("wm", "all") else "default")
        wm_window = torch.compile(_wm_window, mode="reduce-overhead" if graph_scope in ("wm", "all") else "default")
        imag_step = torch.compile(_imag_step, mode="reduce-overhead" if graph_scope == "all" else "default")
        reward_term = torch.compile(_reward_term, mode="reduce-overhead" if graph_scope == "all" else "default")
        critic_input_c = torch.compile(_critic_input, mode="reduce-overhead" if graph_scope == "all" else "default")
    else:
        wm_rollout, wm_err = _wm_rollout, _wm_err
        wm_window = _wm_window
        imag_step, reward_term = _imag_step, _reward_term
        critic_input_c = _critic_input


    from actual_batch_validation import validate_actor_start
    from wm_window_validation import validate_combined_world_model_batch
    graph_validation_enabled = os.environ.get("SM_GRAPH_VALIDATE", "1") == "1"
    graph_validation_status = {"wm": False, "actor": False}
    wm_gate_label = 1
    graph_validation_horizon = int(os.environ.get("SM_GRAPH_VALIDATE_ACTOR_HORIZON", "4"))


    def lr_at(it: int) -> float:
        if it >= args.lr_decay_iters:
            return args.lr * 0.03
        return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


    def set_requires_grad(m: nn.Module, flag: bool):
        for p in m.parameters():
            p.requires_grad_(flag)


    # ------------------------------------------------------------------------------------------------
    def wm_update():
        if args.compile and graph_scope in ("wm", "all"):
            torch.compiler.cudagraph_mark_step_begin()
        w = buffer.sample_windows(args.wm_batch, args.wm_window)
        if graph_validation_enabled and not graph_validation_status["wm"]:
            validate_combined_world_model_batch(
                Path(args.log_dir) / f"graph_world_model_acceptance_{wm_gate_label:06d}.json",
                wm, w, _wm_rollout, _wm_err, _wm_window, wm_window)
            graph_validation_status["wm"] = True
            if args.compile and graph_scope in ("wm", "all"):
                torch.compiler.cudagraph_mark_step_begin()
        loss, e_last, S_hat = wm_window(
            w["s"], w["a"], w["xi"], w["s1"], w["mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError("World-model loss is nonfinite")
        opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=True)
        opt_wm.step()
        counters["world_model_updates_this_run"] += 1
        m = w["mask"][:, -1]
        # Own storage before another CUDA graph generation can replay.
        return ((loss.detach() / args.wm_window).clone(),
                ((e_last.detach() * m).sum() / m.sum().clamp(min=1)).clone())


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
    actor_gate_done = False


    def policy_update():
        nonlocal pi_skipped_updates, actor_gate_done
        if args.compile and graph_scope == "all":
            torch.compiler.cudagraph_mark_step_begin()
        counters["policy_update_attempts_this_run"] += 1
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
        if graph_validation_enabled and graph_scope == "all" and not graph_validation_status["actor"]:
            validate_actor_start(
                Path(args.log_dir) / "graph_actor_start_acceptance.json",
                pi, wm, critic, mdp, motion, S, o_hist, a_prev, f, xi, xi_n, g,
                goal_frames=K, horizon=graph_validation_horizon, gamma=args.gamma,
                bounded_critic=args.bounded_critic, max_step_reward=MAX_STEP_REWARD,
                no_critic=args.no_critic)
            graph_validation_status["actor"] = True
            if args.compile and graph_scope == "all":
                torch.compiler.cudagraph_mark_step_begin()
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
            J = J + (args.gamma**HZ) * c * value_estimate(_critic_input(S, f, xi_n, g))
        loss = -J.mean()
        if PROFILE:
            torch.cuda.synchronize(); tq1 = time.time()
        opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if PROFILE:
            torch.cuda.synchronize(); tq2 = time.time()
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        if not actor_gate_done:
            check_actor_rollout(args.log_dir, it + 1, pi, S_all, flat(As), r, d, alive, J, loss, gn, HZ, B)
            actor_gate_done = True
        nn.utils.clip_grad_value_(pi.parameters(), 0.1)
        if torch.isfinite(gn):
            opt_pi.step()
            counters["policy_optimizer_steps_this_run"] += 1
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
            if "critic.norm" not in frozen_norms:
                critic.norm.update(X)
            with torch.no_grad():
                V = value_estimate(X)
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
                    v_loss = critic_minibatch_loss(Xf, Yf, idx)
                    opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
                    opt_v.step()
                    counters["critic_optimizer_steps_this_run"] += 1
        if PROFILE:
            torch.cuda.synchronize()
            print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
        # Logging retains these beyond the next graph generation.
        return tuple(value.detach().clone() for value in (
            loss, v_loss, (r_sum / HZ / 0.1).mean(), gn, c.mean()))


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
        with (Path(args.log_dir) / "simulator_contract.json").open("w") as stream:
            json.dump(env.contract, stream, indent=2)
        from compiled_collection_runtime import install_compiled_collection
        install_compiled_collection(env, args.log_dir)
        buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, num_joints=spec.num_joints)
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
                if "pi.obs_norm" not in frozen_norms:
                    pi.obs_norm.update(o)
                if "pi.goal_norm" not in frozen_norms:
                    pi.goal_norm.update(g)
                a = pi(obs_hist, g)
                limit = args.action_limit if args.action_limit is not None else 20.0
                a = torch.clamp(a + args.action_noise * torch.randn_like(a), -limit, limit)
                out = env.step(a)
                buffer.add(out)
                S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
                if "wm.in_norm" not in frozen_norms:
                    wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
                if "wm.out_norm" not in frozen_norms:
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

        if args.resume and "sampler_state" in ck:
            sampler = ck["sampler_state"]
            if sampler["reference_config_sha256"] != motion.provenance["source_config_sha256"]:
                print("Sampler state is reset because the reference configuration changed.", flush=True)
            else:
                for name in ("bin_returns", "bin_past_weights"):
                    value = getattr(env.motion_manager, name)
                    if value.shape != sampler[name].shape:
                        raise ValueError("Sampler bin shape changed")
                    value.copy_(sampler[name].to(dev))
        elif args.resume:
            print("Source checkpoint has no sampler state; initialize it for this run.", flush=True)
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
        actor_start = max(args.wm_only_iters, start_it + args.resume_wm_only_iters) + 1
        print(f"Training iterations {start_it + 1}..{args.iterations}; first actor update {actor_start}.", flush=True)
        t_start = time.time() - wall_offset
        t_log = time.time()
        step_acc = 0.0
        log_iters = 0
        stop_reason = "iterations"
        agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        from scratch_speed_probe import ScratchSpeedProbe
        speed_probe = ScratchSpeedProbe(args.log_dir, wm_only_iters=args.wm_only_iters,
                                        actor_warmup=40, measured_updates=1000)
        from detailed_phase_probe import DetailedPhaseProbe
        detailed_probe = DetailedPhaseProbe(args.log_dir, first=20, last=30)
        observed_norms = {'pi.obs': pi.obs_norm, 'pi.goal': pi.goal_norm,
                          'wm.in': wm.in_norm, 'wm.out': wm.out_norm,
                          'critic': critic.norm}
        for it in range(start_it, args.iterations):
            speed_probe.begin(it, counters)
            detailed_probe.begin(it, observed_norms)
            if it in (0, 99, args.wm_only_iters):
                graph_validation_status['wm'] = False
                wm_gate_label = it + 1
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
            speed_probe.start_phase('collect')
            detailed_probe.start('collect')
            collect(n, stats)
            detailed_probe.end('collect')
            speed_probe.end_phase('collect')
            if prof:
                torch.cuda.synchronize(); tp1 = time.time()
            speed_probe.start_phase('world_model')
            for wm_index in range(args.wm_batches):
                detailed_probe.start('wm_' + str(wm_index + 1))
                l, e8 = wm_update()
                detailed_probe.end('wm_' + str(wm_index + 1))
            speed_probe.end_phase('world_model')
            if prof:
                torch.cuda.synchronize(); tp2 = time.time()
            agg["wm"].append(l), agg["wm8"].append(e8)
            if it >= max(args.wm_only_iters, start_it + args.resume_wm_only_iters):
                speed_probe.start_phase('policy_and_critic')
                pl, vl, ir, gn, surv = policy_update()
                speed_probe.end_phase('policy_and_critic')
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
                if it + 1 == args.wm_only_iters + 100:
                    record_optimizer_health(args.log_dir, it + 1,
                        {'pi': pi, 'wm': wm, 'critic': critic},
                        {'pi': opt_pi, 'wm': opt_wm, 'critic': opt_v})
                record_progress(args.log_dir, it + 1, counters, pi, wm, critic)
                print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                      f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s "
                      f"| peak {row[16]:.1f} GiB | skipped {pi_skipped_updates}", flush=True)  # fmt: skip
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            detailed_probe.end_iteration(it, observed_norms)
            speed_probe.end(it, counters)
            if final_iteration:
                detailed_probe.finish()
                speed_probe.finish()
            adaptation_finished = bool(
                args.resume and args.resume_wm_only_iters
                and it + 1 == start_it + args.resume_wm_only_iters
            )
            if ((it + 1) % args.ckpt_every == 0 or final_iteration or adaptation_finished
                    or it + 1 in (args.wm_only_iters, args.wm_only_iters + 100, args.wm_only_iters + 1040)):
                ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                      "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                      "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                      "reward_cfg": asdict(mdp.cfg),
                      "pi_skipped_updates": pi_skipped_updates, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
                ck["sim_backend"] = "mujoco_warp"
                ck["reference_backend"] = motion.storage_info
                ck["sampler_state"] = {
                    "reference_config_sha256": motion.provenance["source_config_sha256"],
                    "bin_returns": env.motion_manager.bin_returns.detach().cpu(),
                    "bin_past_weights": env.motion_manager.bin_past_weights.detach().cpu(),
                }
                ck["rng_state"] = {
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                }
                ck["continuation_counters"] = dict(counters)
                ck["continuation_source_iteration"] = start_it
                boundary_receipt = None
                if adaptation_finished:
                    boundary_receipt = adaptation_boundary(
                        source_models, models, source_optimizers, optimizers, counters)
                    boundary_receipt.update({
                        "schema": "tracker-warp-adaptation-boundary-v1",
                        "source_iteration": start_it,
                        "iteration": it + 1,
                        "world_model_only_iterations": args.resume_wm_only_iters,
                        "lr": lr,
                    })
                last = os.path.join(args.log_dir, "last.pt")
                torch.save(ck, last + ".tmp")
                os.replace(last + ".tmp", last)
                small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "reward_cfg", "saved_at")}
                small["sim_backend"] = ck["sim_backend"]
                small["reference_backend"] = ck["reference_backend"]
                small["reference_identity"] = motion.provenance
                path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
                torch.save(small, path + ".tmp")
                os.replace(path + ".tmp", path)
                if adaptation_finished:
                    adaptation_path = Path(args.log_dir) / "adaptation_last.pt"
                    torch.save(ck, str(adaptation_path) + ".tmp")
                    os.replace(str(adaptation_path) + ".tmp", adaptation_path)
                    boundary_receipt["full_checkpoint"] = str(adaptation_path)
                    boundary_receipt["full_checkpoint_sha256"] = file_sha256(adaptation_path)
                    write_receipt(Path(args.log_dir) / "adaptation_boundary.json", boundary_receipt)
                if resume_audit is not None:
                    resume_audit["latest_saved_iteration"] = it + 1
                    resume_audit["counters"] = dict(counters)
                    resume_audit["actor_optimizer_step_count_at_last_save"] = optimizer_receipt(
                        opt_pi)["uniform_step_count"]
                    resume_audit["actual_actor_updates_since_source"] = (
                        counters["policy_optimizer_steps_this_run"])
                    write_receipt(Path(args.log_dir) / "resume_audit.json", resume_audit)
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
            "continuation_counters": dict(counters),
        }
        with open(Path(args.log_dir, "run_summary.json"), "w") as summary_file:
            json.dump(summary, summary_file, indent=2)
        print(json.dumps(summary), flush=True)

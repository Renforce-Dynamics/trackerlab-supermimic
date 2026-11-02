"""Train SuperMimic's world-model, imagined actor and critic on terrain tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import torch
from bootstrap import configure_paths
REPO = configure_paths()

from configuration import apply_config
from trackerTask.supermimic.terrain import get_task, ROBOT_HEIGHTS, robot_parameters
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerLab.algorithms.supermimic.buffer import ReplayBuffer
from trackerLab.algorithms.supermimic.networks import Critic
import trackerLab.algorithms.supermimic.networks as networks
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy, TerrainWorldModel
from trackerLab.tracker_env.mdp.supermimic import normalize_xi, unflatten_state
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField, TerrainMDP
from trackerLab.tracker_env.supermimic.terrain import TerrainSim
from warp_runtime.restore_optimizer_steps import restore_eager_adam_step_counters


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task_profile", choices=("stairs", "mountain", "parkour"))
    parser.add_argument("--config")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8096)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--buffer-steps", type=int, default=1000)
    parser.add_argument("--utd", type=float, default=.4)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--hist", type=int, default=10)
    parser.add_argument("--wm-window", type=int, default=8)
    parser.add_argument("--wm-batches", type=int, default=10)
    parser.add_argument("--wm-batch", type=int, default=2048)
    parser.add_argument("--pi-batch", type=int, default=4096)
    parser.add_argument("--critic-epochs", type=int, default=5)
    parser.add_argument("--critic-minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--lam", type=float, default=.95)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--wm-only-iters", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--speed-probe-updates", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--solver-iterations", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action="store_true")
    apply_config(parser, argv)
    return parser.parse_args(argv)


def migrate_terrain_checkpoint(checkpoint, model_indices,
                               world_parameter_names, critic_parameter_names):
    """Project checkpoints made with the earlier full terrain context."""
    old_dim = checkpoint["world"]["terrain_norm.mean"].numel()
    new_dim = model_indices.numel()
    if old_dim == new_dim:
        return False
    if old_dim != 187:
        raise ValueError(f"Unsupported terrain checkpoint width {old_dim}")
    index = model_indices.cpu()

    world_state = checkpoint["world"]
    world_state["terrain_norm.mean"] = world_state["terrain_norm.mean"].index_select(0, index)
    world_state["terrain_norm.var"] = world_state["terrain_norm.var"].index_select(0, index)
    adapter = world_state["terrain_adapter.0.weight"]
    base = adapter.shape[1] - old_dim
    world_state["terrain_adapter.0.weight"] = torch.cat(
        (adapter[:, :base], adapter[:, base:].index_select(1, index)), dim=1)

    critic_state = checkpoint["critic"]
    norm_mean = critic_state["norm.mean"]
    base = norm_mean.numel() - old_dim
    critic_state["norm.mean"] = torch.cat(
        (norm_mean[:base], norm_mean[base:].index_select(0, index)))
    critic_state["norm.var"] = torch.cat(
        (critic_state["norm.var"][:base],
         critic_state["norm.var"][base:].index_select(0, index)))
    value_weight = critic_state["net.0.weight"]
    critic_state["net.0.weight"] = torch.cat(
        (value_weight[:, :base], value_weight[:, base:].index_select(1, index)), dim=1)

    for optimizer_name, parameter_names, target_name, prefix_width in (
        ("opt_wm", world_parameter_names, "terrain_adapter.0.weight",
         adapter.shape[1] - old_dim),
        ("opt_v", critic_parameter_names, "net.0.weight", norm_mean.numel() - old_dim),
    ):
        optimizer_state = checkpoint[optimizer_name]
        ids = optimizer_state["param_groups"][0]["params"]
        target_index = parameter_names.index(target_name) if target_name in parameter_names else None
        if target_index is None or target_index >= len(ids):
            continue
        state = optimizer_state["state"].get(ids[target_index], {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if value is not None and value.ndim == 2 and value.shape[1] == prefix_width + old_dim:
                state[key] = torch.cat(
                    (value[:, :prefix_width], value[:, prefix_width:].index_select(1, index)), dim=1)
    checkpoint["_terrain_context_migrated"] = True
    return True


def main():
    args = parse_args()
    task = get_task(args.task)
    if args.task_profile and args.task_profile != task.profile:
        raise SystemExit("--task_profile differs from the selected task")
    if min(args.num_envs, args.iterations, args.horizon, args.hist, args.wm_batches,
           args.wm_batch, args.pi_batch, args.log_every, args.ckpt_every,
           args.critic_epochs, args.critic_minibatches, args.speed_probe_updates) < 1:
        raise SystemExit("Training sizes must be positive")
    if args.buffer_steps < max(args.horizon, args.wm_window) or args.utd <= 0:
        raise SystemExit("Replay buffer must cover imagination and WM windows; UTD must be positive")
    if args.horizon >= args.max_episode_steps or args.wm_only_iters < 0:
        raise SystemExit("Imagination horizon must fit an episode and WM warmup must be nonnegative")
    if args.pi_batch * args.horizon < args.critic_minibatches:
        raise SystemExit("Each critic minibatch must contain at least one sample")
    if args.describe:
        print(json.dumps({
            "task": task.describe(), "algorithm": "supermimic-world-model-imagination-critic",
            "iterations": args.iterations, "perception_rays": 187,
        }, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Terrain training requires CUDA")
    run = args.run_dir.resolve()
    if args.resume and args.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")
    if (run / "last.pt").exists() and not args.resume:
        raise FileExistsError("Use an empty run directory or pass --resume")
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    networks.AMP_DTYPE = torch.bfloat16
    spec, robot_height, _ = robot_parameters(task.robot)
    from trackerTask.supermimic.terrain import generate
    field_data = generate(task.profile, robot_height, args.seed)
    field = HeightField(field_data, robot_height, args.device)
    cpu_model, _, _, _, _, _, _, _, contract = build_model(task.robot, field_data, args.solver_iterations)
    joint_ids = torch.tensor([cpu_model.joint(name).id for name in spec.joint_names])
    qlo = torch.tensor(cpu_model.jnt_range[:, 0], dtype=torch.float32)[joint_ids].to(args.device)
    qhi = torch.tensor(cpu_model.jnt_range[:, 1], dtype=torch.float32)[joint_ids].to(args.device)
    default_q = torch.tensor(spec.default_q, device=args.device)
    mdp = TerrainMDP(field, default_q, qlo, qhi)
    env = TerrainSim(task, mdp, args.num_envs, args.device, args.seed,
                     args.solver_iterations, args.max_episode_steps)
    actor = TerrainPolicy(mdp.obs_dim, mdp.goal_dim, hist=args.hist,
                          num_joints=spec.num_joints).to(args.device)
    world = TerrainWorldModel(field, spec.num_joints).to(args.device)
    critic_dim = spec.state_dim + spec.context_dim + mdp.goal_dim + len(field.model_offsets)
    critic = Critic(critic_dim).to(args.device)
    opt_wm = torch.optim.Adam(world.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(.9, .95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, args.device, spec.num_joints, scenario_dim=2)
    obs_hist = torch.zeros(args.num_envs, args.hist, mdp.obs_dim, device=args.device)
    start_it = 0
    if args.resume:
        ck = torch.load(args.resume.resolve(), map_location=args.device, weights_only=False)
        if ck["task"] != args.task:
            raise ValueError("Resume checkpoint task differs")
        actor.load_state_dict(ck["actor"])
        migrated = migrate_terrain_checkpoint(
            ck, field.model_indices,
            tuple(name for name, _ in world.named_parameters()),
            tuple(name for name, _ in critic.named_parameters()),
        )
        world.load_state_dict(ck["world"])
        critic.load_state_dict(ck["critic"])
        opt_pi.load_state_dict(ck["opt_pi"])
        opt_wm.load_state_dict(ck["opt_wm"])
        opt_v.load_state_dict(ck["opt_v"])
        for optimizer in (opt_wm, opt_pi, opt_v):
            restore_eager_adam_step_counters(optimizer)
        if migrated:
            print("Migrated full-scan terrain context to sparse model/critic context; "
                  "projected optimizer moments were preserved.", flush=True)
        start_it = ck["iteration"]
    elif args.warm_start:
        ck = torch.load(args.warm_start.resolve(), map_location=args.device, weights_only=False)
        if ck.get("robot") != task.robot:
            raise ValueError("Warm-start checkpoint belongs to a different robot")
        actor.load_state_dict(ck["actor"])
        world.load_state_dict(ck["world"])
        critic.load_state_dict(ck["critic"])
    if start_it >= args.iterations:
        raise ValueError("Requested iterations must exceed resumed checkpoint iteration")
    with (run / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump({"arguments": vars(args) | {"run_dir": str(run)},
                   "task": task.describe(), "algorithm": "SuperMimic model-based imagination",
                   "simulator_contract": contract, "robot": spec.name,
                   "num_joints": spec.num_joints, "obs_dim": mdp.obs_dim,
                   "goal_dim": mdp.goal_dim, "critic_dim": critic_dim,
                   "perception": {"heightfield_scan_samples": 187,
                                  "world_model_critic_terrain_samples": len(field.model_offsets),
                                  "source": "analytic terrain-heightfield scan",
                                  "physical_ray_validation": "MuJoCo-Warp validation only",
                                  "rgbd": False}}, stream, indent=2, default=str)
    torch.save(field_data.heights, run / "terrain.pt")

    def wm_window(s, a, xi, s1, mask):
        state, outputs = unflatten_state(s[:, 0]), []
        for k in range(args.wm_window):
            state = world.step(state, a[:, k], normalize_xi(xi[:, k]))
            outputs.append(state)
        predicted = tuple(torch.stack([out[i] for out in outputs], 1) for i in range(6))
        error = world.state_error(predicted, unflatten_state(s1))
        std = torch.maximum(world.out_norm.std(), error.new_full((error.shape[-1],), .01))
        per_step = (error / std).square().mean(-1)
        return (per_step * mask).sum() / s.shape[0]

    def critic_input(state, xi, c):
        return mdp.critic_features(state, xi, c)

    def imag_step(state, history, xi, xi_n, c):
        action = actor(history, mdp.goal(state, c))
        next_state = world.step(state, action, xi_n)
        observation = mdp.add_noise(mdp.obs(next_state, action, xi, c))
        return next_state, action, torch.cat((history[:, 1:], observation[:, None]), 1)

    def reward_term(state, action, previous, c):
        return mdp.reward(state, action, previous, c), mdp.soft_termination(state, c)

    if args.compile:
        wm_window = torch.compile(wm_window, mode="reduce-overhead")
        imag_step = torch.compile(imag_step, mode="reduce-overhead")
        reward_term = torch.compile(reward_term)
        critic_input = torch.compile(critic_input)
        collect_actor = torch.compile(actor, mode="reduce-overhead")
    else:
        collect_actor = actor

    @torch.no_grad()
    def collect(steps):
        nonlocal obs_hist
        outcomes = {key: torch.zeros((), dtype=torch.int64, device=args.device)
                    for key in ("episodes", "successes", "falls", "timeouts")}
        for _ in range(steps):
            state = env.state()
            obs = mdp.add_noise(env.observe())
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], obs[:, None].expand_as(obs_hist),
                                   torch.cat((obs_hist[:, 1:], obs[:, None]), 1))
            goal = mdp.goal(state, env.c)
            actor.obs_norm.update(obs)
            actor.goal_norm.update(goal)
            action = collect_actor(obs_hist, goal)
            action = action + .1 * torch.randn_like(action)
            transition = env.step(action)
            buffer.add(transition)
            outcomes["episodes"] += transition["done"].sum()
            outcomes["successes"] += transition["episode_success"].sum()
            outcomes["falls"] += transition["episode_fall"].sum()
            outcomes["timeouts"] += transition["episode_timeout"].sum()
            s0, s1 = unflatten_state(transition["s"]), unflatten_state(transition["s1"])
            world.in_norm.update(world.inputs(s0, transition["a"], normalize_xi(transition["xi"])))
            world.out_norm.update(world.targets(s0, s1))
            world.update_terrain_norm(s0)
        return outcomes

    def wm_update():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        w = buffer.sample_windows(args.wm_batch, args.wm_window)
        loss = wm_window(w["s"], w["a"], w["xi"], w["s1"], w["mask"])
        opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world.parameters(), 1., error_if_nonfinite=True)
        opt_wm.step()
        return loss.detach().clone()

    def sample_starts(batch):
        starts, histories, count, tries = [], [], 0, 0
        while count < batch:
            tries += 1
            if tries > 64:
                raise RuntimeError("Replay has too few episode-interior starts for imagined rollouts")
            data, hist = buffer.sample_starts(2 * batch, args.hist)
            valid = data["ep_step"] + args.horizon < args.max_episode_steps
            idx = valid.nonzero().flatten()[:batch - count]
            starts.append({key: value[idx] for key, value in data.items()})
            histories.append({key: value[idx] for key, value in hist.items()})
            count += len(idx)
        return ({key: torch.cat([row[key] for row in starts]) for key in starts[0]},
                {key: torch.cat([row[key] for row in histories]) for key in histories[0]})

    def policy_update():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        if args.profile:
            torch.cuda.synchronize()
            profile_start = time.monotonic()
        data, hist = sample_starts(args.pi_batch)
        state = unflatten_state(data["s"])
        xi, c, a_prev = data["xi"], data["c"], data["a_prev"]
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            history_states = unflatten_state(hist["s"])
            obs_hist_local = mdp.add_noise(mdp.obs(
                history_states, hist["a_prev"], hist["xi"], hist["c"]))
        actor_hist = obs_hist_local
        old_wm, old_critic = [p.requires_grad for p in world.parameters()], [p.requires_grad for p in critic.parameters()]
        for parameter in (*world.parameters(), *critic.parameters()):
            parameter.requires_grad_(False)
        initial = state
        rows, actions, previous = [], [], []
        for _ in range(args.horizon):
            previous.append(a_prev)
            state, a_prev, actor_hist = imag_step(state, actor_hist, xi, xi_n, c)
            rows.append(state)
            actions.append(a_prev)
        states = tuple(torch.stack([row[index] for row in rows]) for index in range(6))
        flat_c = c.unsqueeze(0).expand(args.horizon, -1, -1).flatten(0, 1)
        reward, done = reward_term(tuple(value.flatten(0, 1) for value in states),
                                   torch.stack(actions).flatten(0, 1),
                                   torch.stack(previous).flatten(0, 1), flat_c)
        reward, done = reward.view(args.horizon, -1), done.view(args.horizon, -1)
        alive = torch.cumprod(1 - done, 0)
        alive_prev = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        gamma = args.gamma ** torch.arange(args.horizon, device=args.device).float()[:, None]
        objective = (gamma * alive_prev * reward).sum(0)
        critic_x = critic_input(state, xi, c)
        objective = objective + args.gamma ** args.horizon * alive[-1] * critic(critic_x)
        pi_loss = -objective.mean()
        if args.profile:
            torch.cuda.synchronize()
            profile_forward = time.monotonic()
        opt_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        if args.profile:
            torch.cuda.synchronize()
            profile_backward = time.monotonic()
        pi_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        torch.nn.utils.clip_grad_value_(actor.parameters(), .1)
        opt_pi.step()
        for parameter, flag in zip(world.parameters(), old_wm):
            parameter.requires_grad_(flag)
        for parameter, flag in zip(critic.parameters(), old_critic):
            parameter.requires_grad_(flag)
        with torch.no_grad():
            state_seq = tuple(torch.cat((initial[i].unsqueeze(0), states[i].detach()), 0)
                              for i in range(6))
            critic_states = tuple(v.flatten(0, 1) for v in state_seq)
            xi_seq = xi.unsqueeze(0).expand(args.horizon + 1, -1, -1)
            critic_inputs = critic_input(critic_states, xi_seq.flatten(0, 1),
                                         c.unsqueeze(0).expand(args.horizon + 1, -1, -1).flatten(0, 1))
            critic_inputs = critic_inputs.view(args.horizon + 1, args.pi_batch, -1)
            critic.norm.update(critic_inputs)
            values = critic(critic_inputs)
            target, targets = values[-1], []
            for index in range(args.horizon - 1, -1, -1):
                target = reward[index] + args.gamma * (1 - done[index]) * (
                    (1 - args.lam) * values[index + 1] + args.lam * target)
                targets.append(target)
            target = torch.stack(targets[::-1]).detach().flatten()
        inputs, n = critic_inputs[:-1].reshape(-1, critic_dim), target.numel()
        mb = n // args.critic_minibatches
        v_loss = torch.zeros((), device=args.device)
        for _ in range(args.critic_epochs):
            permutation = torch.randperm(n, device=args.device)
            for j in range(args.critic_minibatches):
                indices = permutation[j * mb:(j + 1) * mb if j + 1 < args.critic_minibatches else n]
                v_loss = (critic(inputs[indices]) - target[indices]).square().mean()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
                opt_v.step()
        if args.profile:
            torch.cuda.synchronize()
            print(f"profile imagine={profile_forward - profile_start:.3f}s "
                  f"backward={profile_backward - profile_forward:.3f}s "
                  f"critic={time.monotonic() - profile_backward:.3f}s", flush=True)
        return tuple(value.detach().clone() for value in
                     (pi_loss, v_loss, pi_grad, reward.mean()))

    log_path = run / "train_log.csv"
    collect(max(args.warmup_steps, args.wm_window))
    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(("iteration", "updates_in_interval", "actor_critic_active",
                             "elapsed_seconds", "iterations_per_second",
                             "wm_loss", "actor_loss", "critic_loss", "actor_grad_norm",
                             "imagined_reward", "episodes", "successes", "falls", "timeouts",
                             "success_fraction", "fall_fraction", "terrain_level_mean",
                             "peak_allocated_gib"))
        step_credit = 0.
        measured_start = time.monotonic()
        last_log = measured_start
        aggregates = []
        outcome_window = []
        probe_first_index = max(args.wm_only_iters + 40, start_it)
        probe_started_at = None
        probe_start_completed = probe_first_index + 1
        probe_summary = None
        print(f"task={args.task} iterations={start_it + 1}..{args.iterations} envs={args.num_envs}", flush=True)
        for iteration in range(start_it, args.iterations):
            if iteration == probe_first_index:
                torch.cuda.synchronize()
                probe_started_at = time.monotonic()
            step_credit += 1 / args.utd
            nsteps, step_credit = int(step_credit), step_credit % 1
            outcome_window.append(collect(nsteps))
            wm_loss = torch.zeros((), device=args.device)
            for _ in range(args.wm_batches):
                wm_loss = wm_update()
            if iteration >= args.wm_only_iters:
                pi_loss, v_loss, grad, imag_reward = policy_update()
            else:
                pi_loss = v_loss = grad = imag_reward = torch.zeros((), device=args.device)
            aggregates.append((wm_loss, pi_loss, v_loss, grad, imag_reward))
            final = iteration + 1 == args.iterations
            if (iteration + 1) % args.log_every == 0 or final:
                torch.cuda.synchronize()
                elapsed, window = time.monotonic() - last_log, aggregates
                means = torch.stack([torch.stack(row) for row in window]).mean(0).cpu().tolist()
                ips = len(window) / elapsed
                actor_active = iteration + 1 - len(window) > args.wm_only_iters + 40
                outcomes = {key: int(sum(row[key] for row in outcome_window).item())
                            for key in ("episodes", "successes", "falls", "timeouts")}
                completed_episodes = max(outcomes["episodes"], 1)
                writer.writerow((iteration + 1, len(window), actor_active,
                                 time.monotonic() - measured_start, f"{ips:.6f}",
                                 *[f"{value:.8g}" for value in means],
                                 outcomes["episodes"], outcomes["successes"],
                                 outcomes["falls"], outcomes["timeouts"],
                                 f"{outcomes['successes'] / completed_episodes:.6f}",
                                 f"{outcomes['falls'] / completed_episodes:.6f}",
                                 f"{env.levels.float().mean().item():.6f}",
                                 f"{torch.cuda.max_memory_allocated() / 2**30:.4f}"))
                stream.flush()
                print(f"iteration={iteration + 1} speed={ips:.3f} it/s wm={means[0]:.5f} "
                      f"actor={means[1]:.5f} critic={means[2]:.5f} "
                      f"level={env.levels.float().mean().item():.2f}", flush=True)
                aggregates, outcome_window, last_log = [], [], time.monotonic()
            if probe_started_at is not None and iteration + 1 - probe_first_index >= args.speed_probe_updates:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - probe_started_at
                probe_summary = {
                    "task": args.task,
                    "first_iteration": probe_start_completed,
                    "last_iteration": iteration + 1,
                    "measured_updates": iteration + 1 - probe_first_index,
                    "requested_updates": args.speed_probe_updates,
                    "window_complete": iteration + 1 - probe_first_index == args.speed_probe_updates,
                    "wall_seconds": elapsed,
                    "complete_iterations_per_second":
                        (iteration + 1 - probe_first_index) / elapsed,
                    "world_model_optimizer_updates":
                        (iteration + 1 - probe_first_index) * args.wm_batches,
                    "actor_optimizer_updates": iteration + 1 - probe_first_index,
                    "critic_optimizer_updates": (iteration + 1 - probe_first_index)
                        * args.critic_epochs * args.critic_minibatches,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "tele_historical_reference_iterations_per_second": 2.47115,
                }
                (run / "training_speed_summary.json").write_text(
                    json.dumps(probe_summary, indent=2) + "\n", encoding="utf-8")
                print("TRAINING_SPEED_SUMMARY " + json.dumps(probe_summary), flush=True)
                probe_started_at = None
            stop_requested = (run / "STOP").exists()
            if (iteration + 1) % args.ckpt_every == 0 or final or stop_requested:
                checkpoint = {"task": args.task, "iteration": iteration + 1,
                              "robot": task.robot,
                              "actor": actor.state_dict(), "world": world.state_dict(),
                              "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(),
                              "arguments": vars(args) | {"run_dir": str(run)}}
                torch.save(checkpoint, run / "last.pt.tmp")
                os.replace(run / "last.pt.tmp", run / "last.pt")
            if stop_requested:
                break
    if probe_started_at is not None:
        torch.cuda.synchronize()
        measured_updates = max(0, iteration + 1 - probe_first_index)
        elapsed = time.monotonic() - probe_started_at
        probe_summary = {
            "task": args.task, "first_iteration": probe_start_completed,
            "last_iteration": iteration + 1, "measured_updates": measured_updates,
            "requested_updates": args.speed_probe_updates,
            "window_complete": measured_updates == args.speed_probe_updates,
            "wall_seconds": elapsed,
            "complete_iterations_per_second": measured_updates / max(elapsed, 1e-9),
            "world_model_optimizer_updates": measured_updates * args.wm_batches,
            "actor_optimizer_updates": measured_updates,
            "critic_optimizer_updates": measured_updates * args.critic_epochs * args.critic_minibatches,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "tele_historical_reference_iterations_per_second": 2.47115,
        }
        (run / "training_speed_summary.json").write_text(
            json.dumps(probe_summary, indent=2) + "\n", encoding="utf-8")
    summary = {"task": args.task, "completed_iterations": iteration + 1,
               "requested_iterations": args.iterations,
               "phase": "completed" if iteration + 1 >= args.iterations else "stopped",
               "optimizer_updates": {
                   "world_model": (iteration + 1) * args.wm_batches,
                   "actor": max(0, iteration + 1 - args.wm_only_iters),
                   "critic": max(0, iteration + 1 - args.wm_only_iters) *
                             args.critic_epochs * args.critic_minibatches,
               },
               "elapsed_seconds": time.monotonic() - measured_start,
               "checkpoint": str(run / "last.pt")}
    (run / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

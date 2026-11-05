"""Train command-driven SuperMimic locomotion with an adversarial motion prior."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
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
from trackerLab.tracker_env.mdp.supermimic_terrain import OBJECTIVE_VERSION, HeightField, TerrainMDP
from trackerLab.tracker_env.supermimic.terrain import TerrainSim
from warp_runtime.restore_optimizer_steps import restore_eager_adam_step_counters


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task_profile", choices=("stairs", "mountain", "parkour",
                                                   "velocity", "stealth"))
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
    parser.add_argument("--keep-every", type=int, default=2000)
    parser.add_argument("--speed-probe-updates", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--gait-warm-start", type=Path)
    parser.add_argument("--gait-quality", type=Path)
    parser.add_argument("--amp-motion-config", type=Path)
    parser.add_argument("--amp-weight", type=float, default=.25)
    parser.add_argument("--amp-batch", type=int, default=2048)
    parser.add_argument("--amp-updates", type=int, default=2)
    parser.add_argument("--amp-history", type=int, default=10)
    parser.add_argument("--command-range-mode", choices=("reference", "full"), default="reference")
    parser.add_argument("--stealth-command", type=float, nargs=3)
    parser.add_argument("--style-feature-mode", choices=("full", "proprioceptive"),
                        default="proprioceptive")
    parser.add_argument("--reference-reset-probability", type=float, default=.7)
    parser.add_argument("--foot-slide-weight", type=float, default=.008)
    parser.add_argument("--clearance-ratio", type=float, default=1.)
    parser.add_argument("--action-limit", type=float, default=20.)
    parser.add_argument("--jump-height", type=float, default=0.)
    parser.add_argument("--jump-probability", type=float, default=.15)
    parser.add_argument("--command-resample-steps", type=int, nargs=2, default=(100, 250))
    parser.add_argument("--solver-iterations", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-critic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-optimizers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action="store_true")
    apply_config(parser, argv)
    return parser.parse_args(argv)


def validate_resume_objective(checkpoint, gamma):
    if checkpoint.get("objective_version") != OBJECTIVE_VERSION:
        raise ValueError("Changed terrain objective: use --warm-start in a new run directory")
    if checkpoint["arguments"]["gamma"] != gamma:
        raise ValueError("Resume discount factor differs; use --warm-start for a new objective")


def validate_resume_command_resampling(checkpoint, args):
    saved = checkpoint["arguments"].get("command_resample_steps", (100, 250))
    if tuple(saved) != tuple(args.command_resample_steps):
        raise ValueError("Resume command_resample_steps differs from the saved objective")


def validate_style_references(config_path, robot):
    """Verify immutable campaign pools, or the audited gait pool used in probes."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / config["root"]).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") == "style-campaign-pool-v1":
            from prepare_style_campaign import verify_pool
            verified = verify_pool(root, robot, manifest["style"])
            if config_path != root / "train.json":
                raise ValueError("Training requires the pool's train.json, not held-out references")
            return {"kind": "immutable-style-pool", "style": verified["style"],
                    "manifest": str(manifest_path), "recipe_sha256": verified["recipe_sha256"]}
    if (root / "gait_audit.json").is_file():
        from prepare_gait_references import verify_audit
        audit = verify_audit(config_path, robot)
        return {"kind": "audited-gait-style-pool", "status": audit["status"]}
    raise ValueError("Style references require an immutable style pool or a passed gait audit")


def load_terrain_initialization(actor, world, checkpoint, robot):
    """New task objectives reuse dynamics and policy, with fresh value and optimizers."""
    if checkpoint.get("training_stage") != "terrain-after-qualified-gait":
        raise ValueError("Terrain initialization requires the qualified gait foundation")
    if checkpoint.get("robot") != robot:
        raise ValueError("Warm-start checkpoint belongs to a different robot")
    migrate_terrain_checkpoint(
        checkpoint, world.field.model_indices,
        tuple(name for name, _ in world.named_parameters()), ("net.0.weight",),
    )
    actor.load_state_dict(checkpoint["actor"])
    world.load_state_dict(checkpoint["world"])
    return {"source_task": checkpoint["task"], "source_iteration": checkpoint["iteration"],
            "source_objective_version": checkpoint.get("objective_version", 1),
            "critic_reset": True, "optimizers_reset": True, "task_start_iteration": 0}


def load_gait_foundation(actor, world, checkpoint, quality, robot):
    gait_args = checkpoint.get("args", {})
    if gait_args.get("task") != robot:
        raise ValueError("Gait checkpoint belongs to a different robot")
    if gait_args.get("gait_foundation_version") != 1:
        raise ValueError("A matching SuperMimic gait checkpoint is required")
    if (not quality.get("quality_passed")
            or len(quality.get("held_out_tests", [])) < 2
            or not quality.get("selection", {}).get("quality_passed")
            or not all(row.get("quality_passed") for row in quality.get("held_out_tests", []))):
        raise ValueError("Gait foundation failed the independent physical quality gate")
    source = checkpoint["pi"]
    target = actor.state_dict()
    joints = actor.num_joints
    source_obs_dim = 3 * joints + 12
    target_proprio_dim = actor.obs_dim - actor.scan_dim
    if source["obs_norm.mean"].numel() != source_obs_dim:
        raise ValueError("Gait observation schema differs from relative_orientation")
    if actor.hist != gait_args["hist"]:
        raise ValueError("Gait and terrain observation history lengths differ")

    source_indices = list(range(3 * joints + 6)) + list(range(3 * joints + 6, source_obs_dim))
    target_indices = list(range(3 * joints + 6)) + list(range(3 * joints + 9, target_proprio_dim))
    if len(source_indices) != len(target_indices):
        raise ValueError("Gait and terrain proprioceptive layouts differ")
    source_first, target_first = source["dec.0.weight"], target["dec.0.weight"].clone()
    latent_dim = 32
    source_slot, target_slot = source_obs_dim, target_proprio_dim + 32
    if source_first.shape[1] != latent_dim + actor.hist * source_slot:
        raise ValueError("Unexpected gait decoder input layout")
    for step in range(actor.hist):
        for source_feature, target_feature in zip(source_indices, target_indices):
            target_first[:, latent_dim + step * target_slot + target_feature] = (
                source_first[:, latent_dim + step * source_slot + source_feature])
        target_first[:, latent_dim + step * target_slot + 3 * joints + 6:
                     latent_dim + step * target_slot + 3 * joints + 9] = 0
        target_first[:, latent_dim + step * target_slot + target_proprio_dim:
                     latent_dim + (step + 1) * target_slot] = 0
    target["dec.0.weight"] = target_first
    for name, value in source.items():
        if name.startswith("dec.") and name != "dec.0.weight":
            if name not in target or target[name].shape != value.shape:
                raise ValueError(f"Incompatible gait decoder tensor: {name}")
            target[name] = value
    for source_feature, target_feature in zip(source_indices, target_indices):
        target["obs_norm.mean"][target_feature] = source["obs_norm.mean"][source_feature]
        target["obs_norm.var"][target_feature] = source["obs_norm.var"][source_feature]
    target["obs_norm.count"] = source["obs_norm.count"]
    actor.load_state_dict(target)

    world_state = world.state_dict()
    source_world = checkpoint["wm"]
    common = {name: value for name, value in source_world.items() if name in world_state}
    required = {"net." + name for name in world.net.state_dict()}
    if not required.issubset(common):
        raise ValueError("Gait world-model layers differ from terrain dynamics")
    missing, unexpected = world.load_state_dict(common, strict=False)
    if unexpected or any(not (name.startswith("terrain_norm.") or name.startswith("terrain_adapter."))
                         for name in missing):
        raise ValueError("Gait dynamics warm start did not map to terrain layers")
    return {
        "stage": "terrain-after-qualified-gait",
        "source_robot": robot,
        "source_iteration": checkpoint["iteration"],
        "gait_quality_passed": True,
        "transferred": ["world-model dynamics", "decoder proprioceptive columns",
                        "decoder output layers", "matching observation statistics"],
        "new_for_terrain": ["goal encoder", "height-scan encoder", "terrain adapter",
                            "terrain normalizer", "critic", "optimizers"],
        "critic_reset": True, "optimizers_reset": True, "task_start_iteration": 0,
    }


def migrate_terrain_checkpoint(checkpoint, model_indices,
                               world_parameter_names, critic_parameter_names):
    """Project checkpoints made with the earlier full terrain context."""
    old_dim = checkpoint["world"]["terrain_norm.mean"].numel()
    new_dim = model_indices.numel()
    if old_dim == new_dim:
        return False
    if old_dim != 187:
        raise ValueError(f"Unsupported terrain checkpoint width {old_dim}")
    index = model_indices.to(device=checkpoint["world"]["terrain_norm.mean"].device)

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
    from trackerLab.algorithms.supermimic.amp_style import AMPStyle
    from trackerLab.kinematics.supermimic.robot import RobotKinematics
    from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
    from trackerLab.tracker_env.mdp.supermimic_command import (
        COMMAND_VERSION, SCENARIO_DIM, CommandMDP, advance_command,
    )
    from trackerLab.tracker_env.supermimic.command import CommandSim

    args = parse_args()
    task = get_task(args.task)
    if args.task_profile and args.task_profile != task.profile:
        raise SystemExit("--task_profile differs from the selected task")
    if args.stealth_command is not None and task.profile != "stealth":
        raise SystemExit("--stealth-command is only valid for a stealth task")
    if min(args.num_envs, args.iterations, args.horizon, args.hist, args.wm_batches,
           args.wm_batch, args.pi_batch, args.log_every, args.ckpt_every, args.keep_every,
           args.critic_epochs, args.critic_minibatches, args.speed_probe_updates) < 1:
        raise SystemExit("Training sizes must be positive")
    if args.buffer_steps < max(args.horizon, args.wm_window) or args.utd <= 0:
        raise SystemExit("Replay buffer must cover imagination and WM windows; UTD must be positive")
    if args.horizon >= args.max_episode_steps or args.wm_only_iters < 0:
        raise SystemExit("Imagination horizon must fit an episode and WM warmup must be nonnegative")
    if args.pi_batch * args.horizon < args.critic_minibatches:
        raise SystemExit("Each critic minibatch must contain at least one sample")
    if not 0 < args.gamma < 1 or not 0 <= args.lam <= 1 or args.lr <= 0:
        raise SystemExit("Invalid discount, GAE lambda or learning rate")
    if args.describe:
        print(json.dumps({
            "task": task.describe(), "algorithm": "supermimic-command-amp",
            "iterations": args.iterations, "perception_rays": 187,
            "runtime_motion_reference": False, "style_reference_training_only": True,
        }, indent=2))
        return
    if args.gait_warm_start or args.gait_quality:
        raise SystemExit("Tracking foundations are retired for this command/style objective")
    if args.amp_motion_config is None or not args.amp_motion_config.is_file():
        raise SystemExit("Command locomotion requires an existing --amp-motion-config")
    if args.amp_weight <= 0 or min(args.amp_batch, args.amp_updates) < 1:
        raise SystemExit("Every command task requires a positive AMP weight and update count")
    if args.amp_history < 2 or args.hist < args.amp_history:
        raise SystemExit("Actor history must cover the complete AMP history (at least two frames)")
    if (not 0 <= args.reference_reset_probability <= 1
            or not 0 <= args.jump_probability <= 1 or args.jump_height < 0):
        raise SystemExit("Invalid reference reset or jump-command parameters")
    if (len(args.command_resample_steps) != 2
            or not 100 <= args.command_resample_steps[0] <= args.command_resample_steps[1]
            or args.command_resample_steps[1] > args.max_episode_steps):
        raise SystemExit("Command resampling interval must lie within the episode")
    if not 0 < args.action_limit <= 20:
        raise SystemExit("Action limit must be positive and at most 20")
    if not .5 <= args.clearance_ratio <= 1.2:
        raise SystemExit("Clearance ratio must lie between .5 and 1.2")
    if (isinstance(args.foot_slide_weight, bool)
            or not math.isfinite(args.foot_slide_weight) or args.foot_slide_weight < 0):
        raise SystemExit("Foot-slide weight must be finite and nonnegative")
    reference_receipt = validate_style_references(args.amp_motion_config, task.robot)
    if not torch.cuda.is_available():
        raise RuntimeError("Terrain training requires CUDA")
    run = args.run_dir.resolve()
    if args.resume and args.warm_start:
        raise SystemExit("Choose resume or a command/style warm start")
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
    mdp = CommandMDP(field, default_q, qlo, qhi, gamma=args.gamma,
                     profile=task.profile, clearance_ratio=args.clearance_ratio,
                     action_limit=args.action_limit)
    mdp.set_style_weight(args.amp_weight)
    env = CommandSim(task, mdp, args.num_envs, args.device, args.seed,
                     args.solver_iterations, args.max_episode_steps,
                     jump_height=args.jump_height, jump_probability=args.jump_probability,
                     action_limit=args.action_limit,
                     command_resample_steps=tuple(args.command_resample_steps))
    actor = TerrainPolicy(mdp.obs_dim, mdp.goal_dim, hist=args.hist,
                          num_joints=spec.num_joints, action_limit=args.action_limit).to(args.device)
    world = TerrainWorldModel(field, spec.num_joints).to(args.device)
    critic_dim = spec.state_dim + spec.context_dim + mdp.goal_dim + len(field.model_offsets)
    critic = Critic(critic_dim).to(args.device)
    opt_wm = torch.optim.Adam(world.parameters(), lr=args.lr, fused=args.fused_optimizers)
    opt_pi = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(.9, .95),
                              fused=args.fused_optimizers)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr, fused=args.fused_optimizers)
    kin = RobotKinematics(args.device, spec)
    motion = MotionLib.from_config(str(args.amp_motion_config.resolve()), kin, args.device)
    from trackerLab.algorithms.supermimic.locomotion_quality import LocomotionQuality
    from command_envelope import fixed_command_envelope, reference_envelope
    quality = LocomotionQuality(
        kin, field, contract["motors_effort_velocity_kp_kd_armature"],
        foot_slide_weight=args.foot_slide_weight)
    quality.calibrate(motion)
    command_envelope = None
    if args.stealth_command is not None:
        command_envelope = fixed_command_envelope(args.stealth_command)
        env.set_command_ranges(command_envelope["ranges"])
    elif args.command_range_mode == "reference":
        command_envelope = reference_envelope(
            motion, terrain_navigation=task.profile in ("stairs", "mountain", "parkour"),
            minimum_forward_speed=.08 if task.profile == "stealth" else .15,
            forward_only=task.profile != "stealth")
        env.set_command_ranges(command_envelope["ranges"])
    style = AMPStyle(kin, motion, robot_height, ground_sampler=field.sample,
                     feature_mode=args.style_feature_mode, history_steps=args.amp_history,
                     hidden=(1024, 512), gradient_penalty_weight=5.,
                     gradient_penalty_scope="both", activation="ReLU",
                     optimizer_name="AdamW", weight_decay_coefficient=3e-4,
                     logit_weight_decay_coefficient=.04,
                     minimum_forward_speed=.15 if task.profile == "velocity" else 0.,
                     replay_capacity=48 * args.num_envs).to(args.device)
    grounded = None
    if args.reference_reset_probability:
        import numpy as np
        from trackerLab.kinematics.supermimic.rotations import mat_to_quat
        from prepare_style_references import physical_floor_distances
        values = torch.cat((motion.root_p, mat_to_quat(motion.root_R)[:, [1, 2, 3, 0]],
                            motion.q), -1).cpu().numpy()
        distances = physical_floor_distances(values, task.robot)
        grounded = torch.as_tensor(
            np.isfinite(distances) & (distances >= -.005 * field.scale)
            & (distances <= .04 * field.scale), device=args.device)
    env.set_reference_resets(motion, probability=args.reference_reset_probability,
                             grounded_mask=grounded)
    env.reset()
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, args.device, spec.num_joints,
                          scenario_dim=SCENARIO_DIM)
    obs_hist = torch.zeros(args.num_envs, args.hist, mdp.obs_dim, device=args.device)
    style_hist = style.frame_features(env.state())[:, None].expand(
        -1, style.history_steps, -1).clone()
    start_it = 0
    initialization = None
    if args.resume:
        ck = torch.load(args.resume.resolve(), map_location=args.device, weights_only=False)
        if ck["task"] != args.task:
            raise ValueError("Resume checkpoint task differs")
        if (ck.get("training_stage") != "command-style"
                or ck.get("objective_version") != COMMAND_VERSION):
            raise ValueError("Resume requires the same command/style objective")
        validate_resume_command_resampling(ck, args)
        for name in ("gamma", "hist", "amp_weight", "clearance_ratio",
                     "style_feature_mode", "amp_history", "command_range_mode",
                     "jump_height", "jump_probability", "action_limit", "stealth_command",
                     "foot_slide_weight", "reference_reset_probability"):
            if ck["arguments"].get(name) != getattr(args, name):
                raise ValueError(f"Resume {name} differs from the saved objective")
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
        style.restore(ck["amp_style"])
        for optimizer in (opt_wm, opt_pi, opt_v):
            for group in optimizer.param_groups:
                group["fused"] = args.fused_optimizers
            if args.fused_optimizers:
                for parameter, state in optimizer.state.items():
                    if "step" in state:
                        state["step"] = state["step"].to(parameter.device)
            else:
                restore_eager_adam_step_counters(optimizer)
        if migrated:
            print("Migrated full-scan terrain context to sparse model/critic context; "
                  "projected optimizer moments were preserved.", flush=True)
        start_it = ck["iteration"]
    elif args.warm_start:
        ck = torch.load(args.warm_start.resolve(), map_location=args.device, weights_only=False)
        if (ck.get("training_stage") != "command-style" or ck.get("robot") != task.robot
                or ck.get("objective_version") != COMMAND_VERSION):
            raise ValueError("Warm start needs a matching robot command/style checkpoint")
        if ck["arguments"].get("action_limit", 4.) != args.action_limit:
            raise ValueError("Warm-start action limit differs from the actor's control contract")
        actor.load_state_dict(ck["actor"])
        world.load_state_dict(ck["world"])
        restore_style = ck.get("amp_style", {}).get("expert_hash") == style.expert_hash
        if restore_style:
            style.restore(ck["amp_style"])
        initialization = {"source_task": ck["task"], "source_iteration": ck["iteration"],
                          "actor_world_transferred": True, "critic_reset": True,
                          "actor_world_critic_optimizers_reset": True,
                          "style_optimizer_restored": restore_style,
                          "runtime_motion_reference": False}
    if start_it >= args.iterations:
        raise ValueError("Requested iterations must exceed resumed checkpoint iteration")
    amp_updates_total = style.checkpoint()["updates"]
    with (run / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump({"arguments": vars(args) | {"run_dir": str(run)},
                   "objective_version": COMMAND_VERSION, "initialization": initialization,
                   "success_terminal_value": mdp.success_reward,
                   "task": task.describe(), "algorithm": "SuperMimic command + AMP style",
                   "runtime_motion_reference": False,
                   "training_reference_resets": env.contract.get("reference_resets"),
                   "style_reference_validation": reference_receipt,
                   "command_envelope": command_envelope,
                   "locomotion_quality": {"foot_support": "geometry estimate, not measured force",
                                          "foot_slide_weight": quality.foot_slide_weight,
                                          "sole_height": quality.sole_height.cpu().tolist()},
                   "style": {"motion_config": str(args.amp_motion_config.resolve()),
                             "training_only": True, "weight": args.amp_weight,
                             "feature_mode": args.style_feature_mode,
                             "discriminator_data": "real simulator transitions and expert clips"},
                   "jump_commands": {"height_m": args.jump_height,
                                     "probability": args.jump_probability,
                                     "duration_seconds": .8},
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

    def value_estimate(inputs):
        upper = mdp.max_step_reward / (1 - args.gamma) + mdp.success_reward
        return critic(inputs).clamp(max=upper)

    def imag_step(state, history, xi, xi_n, c):
        action = actor(history, mdp.goal(state, c))
        next_state = world.step(state, action, xi_n)
        next_c = advance_command(c)
        observation = mdp.add_noise(mdp.obs(next_state, action, xi, next_c))
        return next_state, action, torch.cat((history[:, 1:], observation[:, None]), 1), next_c

    def reward_term(state, action, previous, c):
        return mdp.imagined_reward(state, action, previous, c)

    style_reward_fn = style.reward_from_features
    style_frame_fn = style.frame_features
    regularization_reward_fn = quality.reward
    command_outputs_fn = mdp.transition_outputs
    from warp_runtime.critic_runtime import make_critic_minibatch_loss
    critic_loss_fn = make_critic_minibatch_loss(
        critic, compile_enabled=args.compile and args.compile_critic)
    if args.compile:
        wm_window = torch.compile(wm_window, mode="reduce-overhead")
        imag_step = torch.compile(imag_step, mode="reduce-overhead")
        reward_term = torch.compile(reward_term)
        critic_input = torch.compile(critic_input)
        collect_actor = torch.compile(actor, mode="reduce-overhead")
        style_reward_fn = torch.compile(style_reward_fn)
        style_frame_fn = torch.compile(style_frame_fn)
        regularization_reward_fn = torch.compile(regularization_reward_fn)
        command_outputs_fn = torch.compile(command_outputs_fn)
    else:
        collect_actor = actor
    env.locomotion_reward_fn = regularization_reward_fn
    env.command_outputs_fn = command_outputs_fn

    @torch.no_grad()
    def collect(steps):
        nonlocal obs_hist, style_hist
        outcomes = {key: torch.zeros((), dtype=torch.int64, device=args.device)
                    for key in ("episodes", "successes", "falls", "timeouts")}
        for _ in range(steps):
            state = env.state()
            obs = mdp.add_noise(env.observe())
            fresh = env.ep_step == 0
            first_frame = style.frame_features(state)
            style_hist = torch.where(fresh[:, None, None],
                                     first_frame[:, None].expand_as(style_hist), style_hist)
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
            style_hist = torch.cat((style_hist[:, 1:], style.frame_features(s1)[:, None]), 1)
            style.observe_window(style_hist, valid=(~transition["done"]) & (
                transition["ep_step"] >= style.history_steps - 2))
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
            valid = ((data["ep_step"] + args.horizon < args.max_episode_steps)
                     & (data["history_steps"] >= style.history_steps))
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
        actor_hist = obs_hist_local[:, -args.hist:]
        with torch.no_grad():
            style_history = style_frame_fn(history_states)[:, -style.history_steps:]
        frozen = (*world.parameters(), *critic.parameters(), *style.parameters())
        old_flags = [p.requires_grad for p in frozen]
        for parameter in frozen:
            parameter.requires_grad_(False)
        initial = state
        initial_c = c
        rows, actions, previous, commands = [], [], [], []
        for _ in range(args.horizon):
            previous.append(a_prev)
            state, a_prev, actor_hist, c = imag_step(state, actor_hist, xi, xi_n, c)
            rows.append(state)
            actions.append(a_prev)
            commands.append(c)
        states = tuple(torch.stack([row[index] for row in rows]) for index in range(6))
        flat_c = torch.stack(commands).flatten(0, 1)
        reward, done = reward_term(tuple(value.flatten(0, 1) for value in states),
                                   torch.stack(actions).flatten(0, 1),
                                   torch.stack(previous).flatten(0, 1), flat_c)
        preceding = tuple(torch.cat((initial[i][None], states[i][:-1]), 0).flatten(0, 1)
                          for i in range(6))
        reward = reward + regularization_reward_fn(
            preceding, tuple(value.flatten(0, 1) for value in states),
            torch.stack(actions).flatten(0, 1),
            xi.unsqueeze(0).expand(args.horizon, -1, -1).flatten(0, 1))
        style_frames = style_frame_fn(states)
        style_windows = style.rollout_features(style_history, style_frames).flatten(0, 1)
        style_reward = style_reward_fn(style_windows)
        reward = reward + args.amp_weight * style_reward
        reward, done = reward.view(args.horizon, -1), done.view(args.horizon, -1)
        alive = torch.cumprod(1 - done, 0)
        alive_prev = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        gamma = args.gamma ** torch.arange(args.horizon, device=args.device).float()[:, None]
        objective = (gamma * alive_prev * reward).sum(0)
        critic_x = critic_input(state, xi, c)
        objective = objective + args.gamma ** args.horizon * alive[-1] * value_estimate(critic_x)
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
        for parameter, flag in zip(frozen, old_flags):
            parameter.requires_grad_(flag)
        with torch.no_grad():
            state_seq = tuple(torch.cat((initial[i].unsqueeze(0), states[i].detach()), 0)
                              for i in range(6))
            critic_states = tuple(v.flatten(0, 1) for v in state_seq)
            xi_seq = xi.unsqueeze(0).expand(args.horizon + 1, -1, -1)
            command_seq = torch.cat((initial_c.unsqueeze(0), torch.stack(commands)), 0)
            critic_inputs = critic_input(critic_states, xi_seq.flatten(0, 1),
                                         command_seq.flatten(0, 1))
            critic_inputs = critic_inputs.view(args.horizon + 1, args.pi_batch, -1)
            critic.norm.update(critic_inputs)
            values = value_estimate(critic_inputs)
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
                v_loss = critic_loss_fn(inputs, target, indices)
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
                     (pi_loss, v_loss, pi_grad, reward.mean(), style_reward.mean()))

    log_path = run / "train_log.csv"
    collect(max(args.warmup_steps, args.wm_window))
    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(("iteration", "updates_in_interval", "actor_critic_active",
                             "elapsed_seconds", "iterations_per_second",
                             "wm_loss", "actor_loss", "critic_loss", "actor_grad_norm",
                             "imagined_reward", "amp_reward", "amp_loss",
                             "amp_expert_score", "amp_policy_score",
                             "episodes", "successes", "falls", "timeouts",
                             "success_fraction", "fall_fraction", "terrain_level_mean",
                             "peak_allocated_gib"))
        step_credit = 0.
        measured_start = time.monotonic()
        last_log = measured_start
        aggregates = []
        outcome_window = []
        probe_first_index = max(args.wm_only_iters + 40, start_it + 40)
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
            # Fit the prior alongside the actor, after world-model warmup,
            # so the discriminator does not saturate on the initial controller.
            amp_metrics = style.update(batch_size=args.amp_batch, updates=(
                args.amp_updates if iteration >= args.wm_only_iters else 0))
            amp_updates_total += amp_metrics["updates"]
            amp_loss = torch.as_tensor(amp_metrics["loss"], device=args.device).detach()
            if iteration >= args.wm_only_iters:
                pi_loss, v_loss, grad, imag_reward, amp_reward = policy_update()
            else:
                pi_loss = v_loss = grad = imag_reward = amp_reward = torch.zeros((), device=args.device)
            expert_score = torch.as_tensor(amp_metrics["expert_score"], device=args.device)
            policy_score = torch.as_tensor(amp_metrics["policy_score"], device=args.device)
            aggregates.append((wm_loss, pi_loss, v_loss, grad, imag_reward, amp_reward, amp_loss,
                               expert_score, policy_score))
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
                      f"amp={means[5]:.3f} disc={means[7]:.3f}/{means[8]:.3f} "
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
                    "amp_optimizer_updates": (iteration + 1 - probe_first_index) * args.amp_updates,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "tele_historical_reference_iterations_per_second": 2.47115,
                }
                (run / "training_speed_summary.json").write_text(
                    json.dumps(probe_summary, indent=2) + "\n", encoding="utf-8")
                print("TRAINING_SPEED_SUMMARY " + json.dumps(probe_summary), flush=True)
                probe_started_at = None
            stop_requested = (run / "STOP").exists()
            keep = (iteration + 1) % args.keep_every == 0
            if (iteration + 1) % args.ckpt_every == 0 or keep or final or stop_requested:
                checkpoint = {"task": args.task, "iteration": iteration + 1,
                              "training_stage": "command-style",
                              "objective_version": COMMAND_VERSION,
                              "robot": task.robot,
                              "command_envelope": command_envelope,
                              "quality_sole_height": quality.sole_height,
                              "actor": actor.state_dict(), "world": world.state_dict(),
                              "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(),
                              "amp_style": style.checkpoint(),
                              "arguments": vars(args) | {"run_dir": str(run),
                                  "amp_motion_config": str(args.amp_motion_config.resolve())}}
                torch.save(checkpoint, run / "last.pt.tmp")
                os.replace(run / "last.pt.tmp", run / "last.pt")
                if keep:
                    milestone = run / f"iteration_{iteration + 1:06d}.pt"
                    shutil.copy2(run / "last.pt", milestone.with_suffix(".pt.tmp"))
                    os.replace(milestone.with_suffix(".pt.tmp"), milestone)
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
            "amp_optimizer_updates": measured_updates * args.amp_updates,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "tele_historical_reference_iterations_per_second": 2.47115,
        }
        (run / "training_speed_summary.json").write_text(
            json.dumps(probe_summary, indent=2) + "\n", encoding="utf-8")
    summary = {"task": args.task, "completed_iterations": iteration + 1,
               "objective_version": COMMAND_VERSION,
               "requested_iterations": args.iterations,
               "phase": "completed" if iteration + 1 >= args.iterations else "stopped",
               "optimizer_updates": {
                   "world_model": (iteration + 1) * args.wm_batches,
                   "actor": max(0, iteration + 1 - args.wm_only_iters),
                   "critic": max(0, iteration + 1 - args.wm_only_iters) *
                             args.critic_epochs * args.critic_minibatches,
                   "amp": amp_updates_total,
               },
               "elapsed_seconds": time.monotonic() - measured_start,
               "checkpoint": str(run / "last.pt")}
    (run / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

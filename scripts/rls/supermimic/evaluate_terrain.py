"""Evaluate physical task returns, including checkpoint-calibrated locomotion penalties.

AMP is reported separately as a diagnostic score; physical task returns are
not the combined imagined task-plus-AMP training objective.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from bootstrap import configure_paths
REPO = configure_paths()
from export_command_policy import command_training_metadata
from trackerTask.supermimic.terrain import get_task, generate, robot_parameters
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerLab.algorithms.supermimic.stealth_upper_body import UpperBodyConstraint
import trackerLab.algorithms.supermimic.networks as networks
from trackerLab.tracker_env.mdp.supermimic import unflatten_state
from trackerLab.tracker_env.mdp.supermimic_terrain import OBJECTIVE_VERSION, HeightField, TerrainMDP
from trackerLab.tracker_env.supermimic.terrain import TerrainSim
from trackerLab.tracker_env.supermimic.command import CONTROL_TRACE_FIELDS, CONTROL_TRACE_PHASES


WASD_RELEASE_REGRESSION = (
    ("forward", 3.),
    ("idle", 3.),
    ("backward", 3.),
    ("forward_left", 3.),
    ("idle", 3.),
    ("backward_right", 3.),
    ("idle", 3.),
)


def command_sequence_plan(name, modes, *, dt=.02, stage_seconds=3.):
    """Validate and discretize the deterministic keyboard-release regression."""
    if name is None:
        return None
    if name != "wasd-release-regression":
        raise ValueError("command_sequence must be 'wasd-release-regression'")
    if (isinstance(dt, bool) or not isinstance(dt, (int, float))
            or not math.isfinite(dt) or dt <= 0):
        raise ValueError("control dt must be finite and positive")
    if (isinstance(stage_seconds, bool) or not isinstance(stage_seconds, (int, float))
            or not math.isfinite(stage_seconds) or stage_seconds < 3.):
        raise ValueError("command sequence stages must each last at least three seconds")
    names = [mode["name"] for mode in modes]
    if len(names) != len(set(names)) or any(name not in names for name, _ in WASD_RELEASE_REGRESSION):
        raise ValueError("WASD release regression requires every named mode in the checkpoint")
    steps = math.ceil(stage_seconds / dt)
    return {
        "name": name,
        "control_dt_s": float(dt),
        "stage_duration_s": steps * float(dt),
        "minimum_requested_stage_duration_s": float(stage_seconds),
        "total_steps": steps * len(WASD_RELEASE_REGRESSION),
        "entry_clock_behavior": "Starts at ordinary physical reset; key changes preserve the clock",
        "stages": [
            {"mode": mode, "steps": steps, "start_step": index * steps,
             "end_step": (index + 1) * steps}
            for index, (mode, _) in enumerate(WASD_RELEASE_REGRESSION)
        ],
        "lengthen_steps": "Physical resets still follow ordinary episode termination",
    }


def evaluation_episode_limit(checkpoint_limit, requested_steps, sequence_plan):
    """Keep a scheduled command regression inside one physical episode."""
    for name, value in (("checkpoint_limit", checkpoint_limit),
                        ("requested_steps", requested_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return (max(checkpoint_limit, requested_steps + 1)
            if sequence_plan is not None else checkpoint_limit)


def validate_command_selection(command_mode, mode_layout, command_sequence,
                               checkpoint_mode, mode_names, *, stage_seconds=3., dt=.02):
    """Validate mutually exclusive held, cycle, and scheduled keyboard evaluation."""
    if mode_layout not in ("random", "cycle"):
        raise ValueError("mode_layout must be 'random' or 'cycle'")
    keyboard_checkpoint = checkpoint_mode in ("keyboard", "wasd")
    if command_mode is not None:
        if not keyboard_checkpoint:
            raise ValueError("A held command mode requires a keyboard-trained checkpoint")
        if command_mode not in mode_names:
            raise ValueError(f"Unknown held command mode {command_mode!r}")
        if mode_layout != "random" or command_sequence is not None:
            raise ValueError("Choose exactly one held command, cycle layout, or command sequence")
    if mode_layout == "cycle" and not keyboard_checkpoint:
        raise ValueError("A cycle layout requires a keyboard-trained checkpoint")
    if command_sequence is not None:
        if not keyboard_checkpoint or checkpoint_mode != "wasd":
            raise ValueError("The WASD release sequence requires a WASD command/style checkpoint")
        if command_mode is not None or mode_layout != "random":
            raise ValueError("Choose exactly one held command, cycle layout, or command sequence")
    return command_sequence_plan(
        command_sequence, [{"name": name} for name in mode_names],
        dt=dt, stage_seconds=stage_seconds)


def freeze_recorded_commands(previous, current, active):
    """Hold each first-attempt command on exactly the frames whose poses are held."""
    previous = np.asarray(previous)
    current = np.asarray(current)
    active = np.asarray(active, dtype=bool)
    if previous.shape != current.shape or previous.ndim != 2 or active.shape != previous.shape[:1]:
        raise ValueError("Recorded command arrays/mask must align as (envs, commands)")
    return np.where(active[:, None], current, previous)


def keyboard_transition_sums(mode_ids, valid, tracking, velocity, angular_velocity,
                             command, clearance, sole_slip, support, mode_count):
    """Aggregate requested and physical transition quantities without environment stepping."""
    if mode_ids.ndim != 1 or valid.shape != mode_ids.shape:
        raise ValueError("mode IDs and validity must be aligned one-dimensional tensors")
    if any(value.shape[0] != len(mode_ids) for value in
           (tracking, velocity, angular_velocity, command, clearance, sole_slip, support)):
        raise ValueError("Keyboard transition measurements must align by environment")
    if mode_count < 1 or bool(((mode_ids < 0) | (mode_ids >= mode_count)).any()):
        raise ValueError("Keyboard mode ID outside mode table")
    selected = valid.nonzero(as_tuple=False).flatten()
    result = velocity.new_zeros((mode_count, 14))
    if not len(selected):
        return result
    ids = mode_ids[selected]
    planar_command = command[selected, 1:3]
    planar_velocity = velocity[selected, :2]
    command_speed = torch.linalg.vector_norm(planar_command, dim=-1)
    cross_track = ((planar_command[:, 0] * planar_velocity[:, 1]
                    - planar_command[:, 1] * planar_velocity[:, 0]).abs()
                   / command_speed.clamp_min(1e-8))
    cross_track = torch.where(
        command_speed > 1e-8, cross_track,
        torch.linalg.vector_norm(planar_velocity, dim=-1))
    values = torch.stack((
        torch.ones_like(command_speed),
        tracking[selected].to(velocity.dtype),
        velocity[selected, 0],
        velocity[selected, 1],
        angular_velocity[selected, 2],
        torch.linalg.vector_norm(planar_velocity, dim=-1),
        cross_track,
        command_speed,
        ((planar_velocity * planar_command).sum(-1)
         / command_speed.clamp_min(.01)),
        clearance[selected],
        sole_slip[selected],
        support[selected].to(velocity.dtype),
        (planar_velocity - planar_command).square().sum(-1),
        (angular_velocity[selected, 2] - command[selected, 3]).square(),
    ), -1)
    result.scatter_add_(0, ids[:, None].expand(-1, values.shape[-1]), values)
    return result


def keyboard_episode_sums(mode_ids, done, fallen, success, mode_count):
    """Return whole-episode totals and optional stable-mode attribution.

    Pass stable per-environment mode IDs for fixed/cycle layouts. Random
    resampling and multi-stage command sequences must pass ``None``.
    """
    if done.ndim != 1 or fallen.shape != done.shape or success.shape != done.shape:
        raise ValueError("Episode outcome flags must be aligned one-dimensional tensors")
    outcomes = torch.stack((
        done.to(torch.float32),
        (done & fallen).to(torch.float32),
        (done & success).to(torch.float32),
        (done & ~fallen & ~success).to(torch.float32),
    ), -1)
    total = outcomes.sum(0)
    if mode_ids is None:
        return total, None
    if mode_ids.shape != done.shape or mode_count < 1:
        raise ValueError("Stable episode mode IDs must match the outcome batch")
    result = outcomes.new_zeros((mode_count, 4))
    result.scatter_add_(0, mode_ids[:, None].expand(-1, 4), outcomes)
    return total, result


def _episode_summary(total, per_mode, modes, attribution):
    names = ("episodes", "falls", "successes", "timeouts")
    totals = {name: int(value.item()) for name, value in zip(names, total)}
    rows = None if per_mode is None else [
        {"name": mode["name"], **{
            name: int(value.item()) for name, value in zip(names, values)
        }}
        for mode, values in zip(modes, per_mode)
    ]
    return {"attribution": attribution, "total": totals, "per_keyboard_mode": rows}


class FirstEpisodeRecorder:
    """Keep the first attempt, including its pre-reset terminal pose, on each lane."""

    def __init__(self, qpos, scans, root_qpos, joint_qpos):
        self.root_qpos, self.joint_qpos = root_qpos, joint_qpos
        self.poses = [np.asarray(qpos).copy()]
        self.scans = [np.asarray(scans).copy()]
        self.count = len(qpos)
        self.active = np.ones(self.count, dtype=bool)
        self.terminal_steps = np.full(self.count, -1, dtype=np.int32)
        self.outcomes = np.full(self.count, "horizon", dtype="<U8")

    def add(self, flat_state, scans, done, success, fall):
        if not self.active.any():
            return
        pose, scan = self.poses[-1].copy(), self.scans[-1].copy()
        root = self.root_qpos
        pose[self.active, root:root + 7] = flat_state[self.active, :7]
        pose[np.ix_(self.active, self.joint_qpos)] = flat_state[self.active, 13:13 + len(self.joint_qpos)]
        scan[self.active] = scans[self.active]
        self.poses.append(pose)
        self.scans.append(scan)
        ended = self.active & done
        self.terminal_steps[ended] = len(self.poses) - 1
        self.outcomes[ended] = np.where(success[ended], "success", np.where(fall[ended], "fall", "timeout"))
        self.active[ended] = False

    def save(self, path, **metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, qpos=np.stack(self.poses), scan=np.stack(self.scans),
                            terminal_steps=self.terminal_steps, outcomes=self.outcomes,
                            control_dt=np.float64(.02), **metadata)


def command_runtime_options(checkpoint):
    """Restore runtime settings; training command subsets remain metadata only."""
    arguments = checkpoint["arguments"]
    return {
        "jump_height": arguments.get("jump_height", 0.),
        "jump_probability": arguments.get("jump_probability", .15),
        "action_limit": arguments.get("action_limit", 4.),
        "command_resample_steps": tuple(arguments.get("command_resample_steps", (100, 250))),
        "stealth_keyboard_mode": arguments.get("stealth_control_mode") in ("keyboard", "wasd"),
        "self_collisions": arguments.get("self_collisions", False),
        "collision_mask_version": arguments.get("collision_mask_version", 1),
        "physical_domain_randomization":
            arguments.get("physical_domain_randomization", False),
        "grounded_start": arguments.get("grounded_start", False),
        "standing_joint_noise": arguments.get("standing_joint_noise", 0.),
        **({"waist_pd_scale": arguments["waist_pd_scale"]}
           if arguments.get("waist_pd_scale", 1.) != 1. else {}),
        **({"command_mode_set": "wasd"} if arguments.get("stealth_control_mode") == "wasd" else {}),
    }


def control_trace_metadata(record_ids, joint_names):
    return {
        "version": 1, "recorded_env_ids": list(record_ids),
        "joint_names": list(joint_names), "phases": dict(CONTROL_TRACE_PHASES),
        "invalid_rows": "Initial, terminal and later-attempt rows are NaN with control_trace_valid=false",
        "scope": ("Frozen pre-reset snapshots: last-substep force buffers and poststep state. "
                  "Forces are not recomputed at poststep q/qd; these fields do not provide "
                  "a same-time PD reconstruction or a four-substep force history."),
    }


def control_trace_frame(transition, record_ids, active):
    """Mask frozen transition snapshots; never read post-reset simulator buffers."""
    trace = transition.get("control_trace")
    if trace is None:
        raise ValueError("Control trace requires a frozen pre-reset transition snapshot")
    captured_ids = trace["recorded_env_ids"].detach().cpu().numpy()
    if not np.array_equal(captured_ids, np.asarray(record_ids)):
        raise ValueError("Frozen control trace IDs differ from the recorder order")
    if trace.get("phases") != CONTROL_TRACE_PHASES:
        raise ValueError("Frozen control trace phase metadata differs from the capture contract")
    active = np.asarray(active, dtype=bool)
    if active.shape != captured_ids.shape:
        raise ValueError("Control trace activity must match the recorded IDs")
    valid = active & ~transition["done"][record_ids].detach().cpu().numpy()
    result = {name: trace[name].detach().cpu().numpy().copy() for name in CONTROL_TRACE_FIELDS}
    for value in result.values():
        value[~valid] = np.nan
    return {**result, "control_trace_valid": valid}


def attach_locomotion_quality(env, kin, field, motors, checkpoint,
                              upper_body_constraint=None):
    """Use the saved training calibration for both physical rewards and metrics."""
    from trackerLab.algorithms.supermimic.locomotion_quality import LocomotionQuality
    quality = LocomotionQuality(
        kin, field, motors,
        foot_slide_weight=checkpoint.get("arguments", {}).get("foot_slide_weight", .008),
        feet_close_weight=checkpoint.get("arguments", {}).get("feet_close_weight", .008),
        foot_slip_mode=checkpoint.get("arguments", {}).get("foot_slip_mode", "legacy"))
    sole_height = torch.as_tensor(
        checkpoint["quality_sole_height"], device=kin.device, dtype=quality.sole_height.dtype)
    if sole_height.shape != quality.sole_height.shape or not bool(torch.isfinite(sole_height).all()):
        raise ValueError("Checkpoint sole calibration must contain one finite height per foot")
    quality.sole_height = sole_height.detach().clone()
    duration = checkpoint.get("arguments", {}).get("entry_duration_s", 0.)
    if upper_body_constraint is None or duration:
        env.locomotion_reward_fn = quality.reward
    else:
        def reward(previous_state, state, action, xi):
            return quality.reward(previous_state, state, action, xi) + (
                upper_body_constraint.posture_reward(state)
            )
        env.locomotion_reward_fn = reward
    if upper_body_constraint is not None and duration:
        from trackerLab.tracker_env.mdp.supermimic_command import entry_blend
        def entry_reward(state, c):
            if upper_body_constraint.entry_duration_s:
                return upper_body_constraint.posture_reward(state, entry_time=c[..., 6])
            return upper_body_constraint.posture_reward(state, entry_blend(c, duration))
        env.entry_upper_body_reward_fn = entry_reward
    return quality


def command_evaluation_summary(command_sum, command_count, motion_sum, style_sum,
                               quality_sum, command_envelope, keyboard_mode_sums=None,
                               episode_mode_sums=None, episode_total_sums=None,
                               episode_attribution=None):
    metrics = (command_sum / command_count.clamp_min(1)).sqrt().cpu().tolist()
    result = {
        "runtime_motion_reference": False,
        "command_metrics": dict(zip(
            ("velocity_rmse_scale_m_s", "yaw_rate_rmse_rad_s", "clearance_rmse_scale_m"),
            metrics,
        )),
        "command_metric_samples": int(command_count.item()),
        "motion_metrics": {
            **dict(zip(("mean_requested_planar_speed_mps", "mean_projected_velocity_along_command_mps"),
                       (motion_sum / command_count.clamp_min(1)).cpu().tolist())),
            "samples": int(command_count.item()),
        },
        "task_return_definition": {
            "components": ["command_task_reward", "locomotion_regularization", "terminal_reward"],
            "aggregation": "undiscounted sum per completed physical episode",
            "includes_amp_style_reward": False,
            "matches_combined_training_objective": False,
            "amp_role": "Separate diagnostic scoring; excluded from physical task returns",
        },
        "style_metrics": {
            "mean_reward_rate": float((style_sum[0] / style_sum[1].clamp_min(1)).item()),
            "samples": int(style_sum[1].item()),
            "role": "offline scoring only; actor never reads scorer or motion data",
            "independent_style_validation": False,
        },
        "style_quality": "Requires video/physical review; training discriminator score alone is insufficient",
        "command_envelope": command_envelope,
        "keyboard_mode_metric_definitions": {
            "body_velocity": "Mean actual vx/vy in the current robot body frame",
            "body_yaw_rate": "Mean actual body-frame wz",
            "mean_planar_speed": "Mean norm of actual body-frame vx/vy",
            "mean_planar_drift": "Mean absolute cross-track planar speed relative to moving "
                                 "commands; for zero-translation commands, actual planar speed",
            "desired_planar_speed": "Mean norm of the effective body-frame velocity command",
            "mean_clearance": "Mean root z above sampled terrain",
            "mean_support_sole_slip": "Checkpoint locomotion-quality metric; physical sole "
                                      "geometry when foot_slip_mode is g1_sole",
            "supported_fraction": "Fraction of valid transitions with at least one supported foot",
        },
        "gait_metrics": {
            **dict(zip(("support_foot_slip_scale_m_s", "support_foot_tilt_squared",
                        "joint_acceleration_squared", "close_feet_fraction"),
                       (quality_sum[:4] / quality_sum[4].clamp_min(1)).cpu().tolist())),
            "support": "geometry estimate; independent of discriminator outputs",
            "samples": int(quality_sum[4].item()),
        },
    }
    if keyboard_mode_sums is not None:
        episode_modes = keyboard_mode_sums["modes"]
        episode_rows = (_episode_summary(
            episode_total_sums, episode_mode_sums, episode_modes, episode_attribution)
            if episode_total_sums is not None else None)
        per_mode_episodes = (
            episode_rows["per_keyboard_mode"]
            if episode_rows is not None else None)
        result["keyboard_mode_metrics"] = [
            {
                "name": mode["name"],
                "command": mode["command"],
                "samples": int(values[0].item()),
                "tracking_fraction": float((values[1] / values[0].clamp_min(1)).item()),
                "body_vx_m_s": float((values[2] / values[0].clamp_min(1)).item()),
                "body_vy_m_s": float((values[3] / values[0].clamp_min(1)).item()),
                "body_yaw_rate_rad_s": float((values[4] / values[0].clamp_min(1)).item()),
                "mean_planar_speed_m_s": float((values[5] / values[0].clamp_min(1)).item()),
                "mean_planar_drift_m_s": float((values[6] / values[0].clamp_min(1)).item()),
                "desired_planar_speed_m_s": float((values[7] / values[0].clamp_min(1)).item()),
                "mean_projected_velocity_m_s": float(
                    (values[8] / values[0].clamp_min(1)).item()),
                "mean_clearance_m": float((values[9] / values[0].clamp_min(1)).item()),
                "mean_support_sole_slip_m_s":
                    float((values[10] / values[0].clamp_min(1)).item()),
                "supported_fraction": float((values[11] / values[0].clamp_min(1)).item()),
                "velocity_rmse_m_s": float(
                    (values[12] / values[0].clamp_min(1)).sqrt().item()),
                "yaw_rate_rmse_rad_s": float(
                    (values[13] / values[0].clamp_min(1)).sqrt().item()),
                "episodes": None if per_mode_episodes is None else per_mode_episodes[index]["episodes"],
                "falls": None if per_mode_episodes is None else per_mode_episodes[index]["falls"],
                "successes": None if per_mode_episodes is None else per_mode_episodes[index]["successes"],
            }
            for index, (mode, values) in enumerate(zip(episode_modes, keyboard_mode_sums["values"]))
        ]
        if episode_rows is not None:
            result["keyboard_episode_metrics"] = episode_rows
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--trajectory-output", type=Path)
    parser.add_argument("--command-mode", help="Hold one named keyboard mode for the entire evaluation")
    parser.add_argument("--mode-layout", choices=("random", "cycle"), default="random")
    parser.add_argument("--command-sequence", choices=("wasd-release-regression",))
    parser.add_argument("--command-stage-seconds", type=float, default=3.)
    parser.add_argument("--record-env-ids", type=int, nargs="+",
                        help="Environment IDs whose first attempts are recorded")
    parser.add_argument("--record-control-trace", action="store_true")
    args = parser.parse_args()
    if args.record_control_trace and args.trajectory_output is None:
        parser.error("Control traces require --trajectory-output")
    if args.num_envs < 6 or args.steps < 1:
        parser.error("Evaluation needs at least six environments and one step")
    record_ids = list(range(6)) if args.record_env_ids is None else args.record_env_ids
    if (len(record_ids) != len(set(record_ids))
            or any(index < 0 or index >= args.num_envs for index in record_ids)):
        parser.error("Recorded environment IDs must be unique and within num-envs")
    device = "cuda:0"
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    command_mode = ck.get("training_stage") == "command-style"
    if getattr(args, "record_control_trace", False) and not command_mode:
        parser.error("Control traces currently require a command-style checkpoint")
    from trackerLab.tracker_env.mdp.supermimic_command import stealth_keyboard_modes
    checkpoint_control_mode = ck["arguments"].get("stealth_control_mode", "fixed")
    training_metadata = command_training_metadata(ck)
    # Evaluation always uses the canonical table, regardless of the training subset.
    mode_names = ([row[0] for row in stealth_keyboard_modes(checkpoint_control_mode)]
                  if checkpoint_control_mode in ("keyboard", "wasd") else [])
    try:
        sequence_plan = validate_command_selection(
            args.command_mode, args.mode_layout, args.command_sequence,
            checkpoint_control_mode, mode_names, stage_seconds=args.command_stage_seconds)
    except ValueError as error:
        parser.error(str(error))
    if sequence_plan is not None and args.steps < sequence_plan["total_steps"]:
        parser.error(
            f"Sequence requires at least {sequence_plan['total_steps']} steps; got {args.steps}")
    task = get_task(ck["task"])
    spec, height, motors = robot_parameters(task.robot)
    field_data = generate(task.profile, height, args.seed)
    field = HeightField(field_data, height, device)
    solver_iterations = ck["arguments"].get("solver_iterations", 50)
    model, *_ = build_model(
        task.robot, field_data, solver_iterations,
        self_collisions=ck["arguments"].get("self_collisions", False),
        collision_mask_version=ck["arguments"].get("collision_mask_version", 1),
        waist_pd_scale=ck["arguments"].get("waist_pd_scale", 1.))
    joint_ids = [model.joint(name).id for name in spec.joint_names]
    limits = torch.tensor(model.jnt_range[joint_ids], device=device, dtype=torch.float32)
    if command_mode:
        from trackerLab.tracker_env.mdp.supermimic_command import CommandMDP, COMMAND_VERSION
        from trackerLab.tracker_env.supermimic.command import CommandSim
        mdp = CommandMDP(
            field, torch.tensor(spec.default_q, device=device), limits[:, 0], limits[:, 1],
            gamma=ck["arguments"].get("gamma", .99), profile=task.profile,
            clearance_ratio=ck["arguments"].get("clearance_ratio", 1.),
            action_limit=ck["arguments"].get("action_limit", 4.),
            entry_duration_s=ck["arguments"].get("entry_duration_s", 0.),
            control_mode="wasd" if ck["arguments"].get("stealth_control_mode") == "wasd" else "legacy",
            reward_control_mode=training_metadata["command_reward_mode"],
            wasd_reward_version=ck["arguments"].get("wasd_reward_version", 1),
        )
        mdp.set_style_weight(ck["arguments"].get("amp_weight", .5))
        simulation = CommandSim
        evaluation_version = COMMAND_VERSION
    else:
        mdp = TerrainMDP(field, torch.tensor(spec.default_q, device=device), limits[:, 0], limits[:, 1],
                         gamma=ck["arguments"].get("gamma", .99))
        simulation = TerrainSim
        evaluation_version = OBJECTIVE_VERSION
    runtime_options = command_runtime_options(ck) if command_mode else {
        "collision_mask_version": ck["arguments"].get("collision_mask_version", 1),
    }
    evaluation_episode_steps = evaluation_episode_limit(
        ck["arguments"]["max_episode_steps"], args.steps, sequence_plan)
    env = simulation(task, mdp, args.num_envs, device, args.seed, solver_iterations,
                     max_episode_steps=evaluation_episode_steps, **runtime_options)
    if command_mode and ck.get("command_envelope"):
        if ck["arguments"].get("stealth_control_mode") == "wasd":
            env.set_wasd_keyboard_mode()
        elif ck["arguments"].get("stealth_control_mode") == "keyboard":
            env.set_stealth_keyboard_mode()
        else:
            from command_envelope import validate_envelope
            env.set_command_ranges(validate_envelope(ck["command_envelope"])["ranges"])
    if args.command_mode:
        if not command_mode or ck["arguments"].get("stealth_control_mode") not in ("keyboard", "wasd"):
            raise ValueError("A held command mode requires a keyboard-trained checkpoint")
        env.set_fixed_keyboard_mode(args.command_mode)
    if args.mode_layout == "cycle":
        if not command_mode or ck["arguments"].get("stealth_control_mode") not in ("keyboard", "wasd"):
            raise ValueError("A cycle layout requires a keyboard-trained checkpoint")
        env.set_keyboard_evaluation_cycle()
    if sequence_plan is not None:
        env.set_fixed_keyboard_mode(sequence_plan["stages"][0]["mode"])
    env.curriculum = False
    levels = torch.arange(args.num_envs, device=device) % 6
    env.reset(levels=levels)
    actor = TerrainPolicy(mdp.obs_dim, mdp.goal_dim, hist=ck["arguments"]["hist"],
                          num_joints=spec.num_joints,
                          action_limit=ck["arguments"].get("action_limit", 4.)
                          if command_mode else None).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    upper_body_constraint = None
    upper_body_config = ck.get("stealth_upper_body_constraint")
    if upper_body_config is not None:
        upper_body_constraint = UpperBodyConstraint.from_dict(
            upper_body_config,
        ).to(device).eval()
    rate_limiter_config = ck.get("upper_body_rate_limit")
    action_constraint = upper_body_constraint
    if rate_limiter_config is not None:
        if upper_body_constraint is not None:
            raise ValueError("Checkpoint cannot combine static arms and a separate arm rate limiter")
        from trackerLab.algorithms.supermimic.stealth_action_filter import UpperBodyRateLimiter
        action_constraint = UpperBodyRateLimiter.from_dict(rate_limiter_config).to(device).eval()
    if ck.get("stealth_waist_constraint") is not None:
        from trackerLab.algorithms.supermimic.stealth_waist_entry import compose_entry_waist_constraint
        action_constraint = compose_entry_waist_constraint(
            action_constraint, ck["stealth_waist_constraint"], device=device).eval()
    style = None
    style_sum = torch.zeros(2, device=device)
    if command_mode:
        from trackerLab.algorithms.supermimic.amp_style import AMPStyle
        from trackerLab.kinematics.supermimic.robot import RobotKinematics
        kin = RobotKinematics(device, spec)
        style = AMPStyle.for_scoring(kin, ck["amp_style"], ground_sampler=field.sample)
        quality = attach_locomotion_quality(
            env, kin, field, env.contract["motors_effort_velocity_kp_kd_armature"], ck,
            upper_body_constraint=upper_body_constraint,
        )
        quality_sum = torch.zeros(5, device=device)
        style_history = style.frame_features(env.state())[:, None].expand(
            -1, style.history_steps, -1).clone()
    recorder = None
    if args.trajectory_output:
        free_id = np.flatnonzero(model.jnt_type == 0)[0]
        recorder = FirstEpisodeRecorder(
            env.qpos[record_ids].cpu().numpy(),
            field.features(env.state())[record_ids].cpu().numpy(),
            int(model.jnt_qposadr[free_id]), model.jnt_qposadr[joint_ids],
        )
        first_commands = (mdp.effective_command(env.c) if command_mode else env.c)[record_ids].cpu().numpy()
        recorded_commands_last = first_commands.copy()
        first_context = env.xi[record_ids].cpu().numpy()
        recorded_commands = [first_commands.copy()]
        recorded_control = None
        if args.record_control_trace:
            env.set_control_trace_ids(record_ids)
            trace_metadata = control_trace_metadata(record_ids, spec.joint_names)
            initial_trace = {
                name: np.full((len(record_ids), spec.num_joints), np.nan, dtype=np.float32)
                for name in CONTROL_TRACE_FIELDS}
            initial_trace["control_trace_valid"] = np.zeros(len(record_ids), dtype=bool)
            recorded_control = {name: [value] for name, value in initial_trace.items()}
    history = torch.zeros(args.num_envs, actor.hist, mdp.obs_dim, device=device)
    counts = {name: torch.zeros(6, device=device) for name in
              ("episodes", "successes", "falls", "timeouts", "return_sum", "progress_sum")}
    progress = torch.zeros(args.num_envs, device=device)
    command_sum = torch.zeros(3, device=device)
    command_count = torch.zeros((), device=device)
    motion_sum = torch.zeros(2, device=device)
    keyboard_mode_sums = None
    if ck["arguments"].get("stealth_control_mode") in ("keyboard", "wasd"):
        modes = ck.get("command_envelope", {}).get("command_modes", [])
        from trackerLab.tracker_env.mdp.supermimic_command import stealth_keyboard_modes
        expected = stealth_keyboard_modes(ck["arguments"]["stealth_control_mode"])
        if (len(modes) != len(expected)
                or any(mode["name"] != row[0] or mode["command"] != list(row[1:4])
                       for mode, row in zip(modes, expected))):
            raise ValueError("Keyboard mode metadata differs from the selected policy contract")
        keyboard_mode_sums = {
            "modes": modes,
            "values": [torch.zeros(14, device=device) for _ in modes],
        }
    stable_episode_mode_ids = None
    episode_attribution = None
    episode_total_sums = torch.zeros(4, device=device)
    episode_mode_sums = None
    if keyboard_mode_sums is not None:
        modes = keyboard_mode_sums["modes"]
        if sequence_plan is not None:
            episode_attribution = "multi_stage_command_sequence; episodes pooled"
        elif args.command_mode:
            fixed_index = next(
                index for index, mode in enumerate(modes) if mode["name"] == args.command_mode)
            stable_episode_mode_ids = torch.full(
                (args.num_envs,), fixed_index, device=device, dtype=torch.long)
            episode_mode_sums = torch.zeros((len(modes), 4), device=device)
            episode_attribution = "fixed_command_mode_per_environment"
        elif args.mode_layout == "cycle":
            stable_episode_mode_ids = torch.arange(
                args.num_envs, device=device, dtype=torch.long).remainder(len(modes))
            episode_mode_sums = torch.zeros((len(modes), 4), device=device)
            episode_attribution = "environment_id_modulo_cycle_mode; stable across resets"
        else:
            episode_attribution = "random_mode_resampling; episodes pooled without mode attribution"
    upper_body_sums = torch.zeros(3, device=device) if upper_body_constraint else None
    sequence_stage_index = 0
    for step_index in range(args.steps):
        if (sequence_plan is not None and sequence_stage_index + 1 < len(sequence_plan["stages"])
                and step_index == sequence_plan["stages"][sequence_stage_index]["end_step"]):
            sequence_stage_index += 1
            # This resamples only the twist and preserves c[..., 6] for entry.
            env.set_fixed_keyboard_mode(sequence_plan["stages"][sequence_stage_index]["mode"])
        if command_mode:
            first_frame = style.frame_features(env.state())
            style_history = torch.where((env.ep_step == 0)[:, None, None],
                                        first_frame[:, None].expand_as(style_history), style_history)
        observation = mdp.add_noise(env.observe())
        history = torch.where((env.ep_step == 0)[:, None, None],
                              observation[:, None].expand_as(history),
                              torch.cat((history[:, 1:], observation[:, None]), 1))
        action = actor(history, mdp.goal(env.state(), env.c))
        if action_constraint is not None:
            n = spec.num_joints
            previous_action = history[:, -1, 2 * n + 6:3 * n + 6]
            older_action = history[:, -2, 2 * n + 6:3 * n + 6]
            action = action_constraint(
                action, previous_action, older_action, entry_time=env.c[..., 6] + .02)
        tr = env.step(action)
        if command_mode:
            p, R, v, w, *_ = unflatten_state(tr["s1"])
            local_v = (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
            local_w = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
            evaluated_command = mdp.effective_command(tr["c1"])
            velocity_error = ((local_v[:, :2] - evaluated_command[:, 1:3]) / field.scale).square().sum(-1)
            yaw_error = (local_w[:, 2] - evaluated_command[:, 3]).square()
            from trackerLab.tracker_env.mdp.supermimic_command import target_clearance
            clearance_error = ((p[:, 2] - field.sample(p[:, :2])
                                - target_clearance(evaluated_command)) / field.scale).square()
            valid = ~tr["done"] & (tr["ep_step"] >= 10)
            command_sum += torch.stack((velocity_error[valid].sum(), yaw_error[valid].sum(),
                                        clearance_error[valid].sum()))
            command_count += valid.sum()
            motion_sum += torch.stack((
                tr["command_metrics"]["command_speed_mps"][valid].sum(),
                tr["command_metrics"]["commanded_progress_mps"][valid].sum(),
            ))
            if keyboard_mode_sums is not None:
                mode_ids = tr["stealth_mode_id"]
                clearance = p[:, 2] - field.sample(p[:, :2])
                measured = quality.measurements(
                    unflatten_state(tr["s1"]), unflatten_state(tr["s"]))
                support = measured["support_heuristic"].amax(-1) > 0
                mode_sums = keyboard_transition_sums(
                    mode_ids, valid, tr["command_metrics"]["command_tracking"],
                    local_v, local_w, evaluated_command, clearance,
                    measured["support_foot_slip_scale_m_s"] * field.scale, support,
                    len(keyboard_mode_sums["modes"]))
                for mode_index, values in enumerate(keyboard_mode_sums["values"]):
                    values += mode_sums[mode_index]
                episode_totals, episode_modes = keyboard_episode_sums(
                    mode_ids if stable_episode_mode_ids is not None else None,
                    tr["done"], tr["episode_fall"], tr["episode_success"],
                    len(keyboard_mode_sums["modes"]))
                episode_total_sums += episode_totals
                if episode_mode_sums is not None:
                    episode_mode_sums += episode_modes
            style_history = torch.cat((style_history[:, 1:],
                style.frame_features(unflatten_state(tr["s1"]))[:, None]), 1)
            rate = style.reward_from_features(style.window_features(style_history))
            style_valid = valid & (tr["ep_step"] >= style.history_steps - 2)
            style_sum += torch.stack((rate[style_valid].sum(), style_valid.sum()))
            measured = quality.measurements(unflatten_state(tr["s1"]), unflatten_state(tr["s"]))
            quality_sum += torch.stack((
                measured["support_foot_slip_scale_m_s"][valid].sum(),
                measured["support_foot_tilt_squared"][valid].sum(),
                measured["joint_acceleration_squared"][valid].sum(),
                (measured["foot_separation_scale_m"][valid] < .08).float().sum()
                if quality.biped else valid.sum() * 0.,
                valid.sum(),
            ))
            if upper_body_constraint is not None:
                state = unflatten_state(tr["s1"])
                indices = upper_body_constraint.upper_indices
                target = (upper_body_constraint.reference_at(tr["c1"][..., 6])
                          if upper_body_constraint.entry_duration_s else
                          upper_body_constraint.reference_q)
                pose_error = state[4].index_select(-1, indices) - target
                arm_speed = state[5].index_select(-1, indices).abs()
                upper_body_sums += torch.stack((
                    pose_error[valid].abs().sum(),
                    arm_speed[valid].sum(),
                    valid.sum(),
                ))
        if recorder is not None and recorder.active.any():
            recorded_active = recorder.active.copy()
            recorder.add(
                tr["s1"][record_ids].cpu().numpy(),
                field.features(unflatten_state(tr["s1"][record_ids])).cpu().numpy(),
                tr["done"][record_ids].cpu().numpy(), tr["episode_success"][record_ids].cpu().numpy(),
                tr["episode_fall"][record_ids].cpu().numpy(),
            )
            command = tr.get("c1", tr["c"])
            if command_mode:
                command = mdp.effective_command(command)
            current_commands = command[record_ids].cpu().numpy()
            recorded_commands_last = freeze_recorded_commands(
                recorded_commands_last, current_commands, recorded_active)
            recorded_commands.append(recorded_commands_last.copy())
            if recorded_control is not None:
                trace = control_trace_frame(tr, record_ids, recorded_active)
                for name, value in trace.items():
                    recorded_control[name].append(value)
        position = unflatten_state(tr["s1"])[0][:, 0] / field.scale
        progress = torch.maximum(progress, position)
        done = tr["done"].float()
        values = {"episodes": done, "successes": tr["episode_success"].float(),
                  "falls": tr["episode_fall"].float(), "timeouts": tr["episode_timeout"].float(),
                  "return_sum": tr["episode_return"], "progress_sum": progress * done}
        for name, value in values.items():
            counts[name].scatter_add_(0, levels, value)
        progress = torch.where(tr["done"], torch.zeros_like(progress), progress)
    lanes = []
    for level in range(6):
        row = {name: float(value[level].item()) for name, value in counts.items()}
        denominator = max(row["episodes"], 1)
        row.update(level=level, success_fraction=row["successes"] / denominator,
                   fall_fraction=row["falls"] / denominator,
                   mean_return=row["return_sum"] / denominator,
                   mean_max_progress_scale_m=row["progress_sum"] / denominator)
        lanes.append(row)
    result = {"task": task.id, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_objective_version": ck.get("objective_version", 1),
              "evaluation_objective_version": evaluation_version,
              "trained_iterations": ck["iteration"], "evaluation_seed": args.seed,
              "physical_domain_randomization":
                  env.contract.get("physical_domain_randomization"),
              "entry_duration_s": ck["arguments"].get("entry_duration_s", 0.),
              "waist_pd_scale": ck["arguments"].get("waist_pd_scale", 1.),
              "foot_slip_mode": ck["arguments"].get("foot_slip_mode", "legacy"),
              "upper_body_rate_limit": rate_limiter_config,
              "stealth_control_mode": ck["arguments"].get("stealth_control_mode", "fixed"),
              **training_metadata,
              "wasd_reward_version": ck["arguments"].get("wasd_reward_version", 1),
              "fixed_command_mode": args.command_mode,
              "command_mode_layout": args.mode_layout,
              "command_sequence": sequence_plan,
              "evaluation_max_episode_steps": evaluation_episode_steps,
              "terrain_seed_held_out": args.seed != ck["arguments"]["seed"],
              "physics_steps": args.steps, "num_envs": args.num_envs,
              "policy_exploration_noise": False, "observation_noise": True, "lanes": lanes}
    if command_mode:
        result.update(command_evaluation_summary(
            command_sum, command_count, motion_sum, style_sum, quality_sum,
            ck.get("command_envelope"), keyboard_mode_sums,
            episode_mode_sums, episode_total_sums, episode_attribution))
    if upper_body_sums is not None:
        count = upper_body_sums[2].clamp_min(1) * len(upper_body_constraint.upper_joint_names)
        result["upper_body_metrics"] = {
            "mean_absolute_pose_error_rad": float((upper_body_sums[0] / count).item()),
            "mean_absolute_joint_speed_rad_s": float((upper_body_sums[1] / count).item()),
            "tracked_joints": list(upper_body_constraint.upper_joint_names),
            "static_reference": upper_body_config["reference_q"],
        }
    if recorder is not None:
        recorder.save(args.trajectory_output, seed=np.int64(args.seed),
                      task=np.array(task.id), iteration=np.int64(ck["iteration"]),
                      commands=first_commands, context=first_context,
                      command_sequence=np.stack(recorded_commands),
                      recorded_env_ids=np.array(record_ids, dtype=np.int64),
                      waist_pd_scale=np.float64(ck["arguments"].get("waist_pd_scale", 1.)),
                      self_collisions=np.bool_(ck["arguments"].get("self_collisions", False)),
                      collision_mask_version=np.int64(ck["arguments"].get("collision_mask_version", 1)),
                      **({**{name: np.stack(values) for name, values in recorded_control.items()},
                          "control_trace_metadata_json": np.array(json.dumps(trace_metadata, sort_keys=True))}
                         if recorded_control is not None else {}),
                      solver_iterations=np.int64(solver_iterations))
        if recorded_control is not None:
            result["control_trace_scope"] = trace_metadata["scope"]
            result["control_trace_metadata"] = trace_metadata
        result["demo_trajectory"] = str(args.trajectory_output.resolve())
        result["demo_selection"] = (
            f"First episode of environment IDs {record_ids}; no outcome filtering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

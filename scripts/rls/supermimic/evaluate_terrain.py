"""Evaluate physical task returns, including checkpoint-calibrated locomotion penalties.

AMP is reported separately as a diagnostic score; physical task returns are
not the combined imagined task-plus-AMP training objective.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.terrain import get_task, generate, robot_parameters
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerLab.algorithms.supermimic.stealth_upper_body import UpperBodyConstraint
import trackerLab.algorithms.supermimic.networks as networks
from trackerLab.tracker_env.mdp.supermimic import unflatten_state
from trackerLab.tracker_env.mdp.supermimic_terrain import OBJECTIVE_VERSION, HeightField, TerrainMDP
from trackerLab.tracker_env.supermimic.terrain import TerrainSim


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
    """Restore command timing and control settings without training references."""
    arguments = checkpoint["arguments"]
    return {
        "jump_height": arguments.get("jump_height", 0.),
        "jump_probability": arguments.get("jump_probability", .15),
        "action_limit": arguments.get("action_limit", 4.),
        "command_resample_steps": tuple(arguments.get("command_resample_steps", (100, 250))),
        "stealth_keyboard_mode": arguments.get("stealth_control_mode") == "keyboard",
        "self_collisions": arguments.get("self_collisions", False),
        "physical_domain_randomization":
            arguments.get("physical_domain_randomization", False),
    }


def attach_locomotion_quality(env, kin, field, motors, checkpoint,
                              upper_body_constraint=None):
    """Use the saved training calibration for both physical rewards and metrics."""
    from trackerLab.algorithms.supermimic.locomotion_quality import LocomotionQuality
    quality = LocomotionQuality(
        kin, field, motors,
        foot_slide_weight=checkpoint.get("arguments", {}).get("foot_slide_weight", .008),
        feet_close_weight=checkpoint.get("arguments", {}).get("feet_close_weight", .008))
    sole_height = torch.as_tensor(
        checkpoint["quality_sole_height"], device=kin.device, dtype=quality.sole_height.dtype)
    if sole_height.shape != quality.sole_height.shape or not bool(torch.isfinite(sole_height).all()):
        raise ValueError("Checkpoint sole calibration must contain one finite height per foot")
    quality.sole_height = sole_height.detach().clone()
    if upper_body_constraint is None:
        env.locomotion_reward_fn = quality.reward
    else:
        def reward(previous_state, state, action, xi):
            return quality.reward(previous_state, state, action, xi) + (
                upper_body_constraint.posture_reward(state)
            )
        env.locomotion_reward_fn = reward
    return quality


def command_evaluation_summary(command_sum, command_count, motion_sum, style_sum,
                               quality_sum, command_envelope, keyboard_mode_sums=None):
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
        "gait_metrics": {
            **dict(zip(("support_foot_slip_scale_m_s", "support_foot_tilt_squared",
                        "joint_acceleration_squared", "close_feet_fraction"),
                       (quality_sum[:4] / quality_sum[4].clamp_min(1)).cpu().tolist())),
            "support": "geometry estimate; independent of discriminator outputs",
            "samples": int(quality_sum[4].item()),
        },
    }
    if keyboard_mode_sums is not None:
        result["keyboard_mode_metrics"] = [
            {
                "name": mode["name"],
                "command": mode["command"],
                "samples": int(values[0].item()),
                "tracking_fraction": float((values[1] / values[0].clamp_min(1)).item()),
                "velocity_rmse_m_s": float((values[2] / values[0].clamp_min(1)).sqrt().item()),
                "yaw_rate_rmse_rad_s": float((values[3] / values[0].clamp_min(1)).sqrt().item()),
                "mean_projected_velocity_m_s": float((values[4] / values[0].clamp_min(1)).item()),
            }
            for mode, values in zip(keyboard_mode_sums["modes"], keyboard_mode_sums["values"])
        ]
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
    args = parser.parse_args()
    if args.num_envs < 6 or args.steps < 1:
        parser.error("Evaluation needs at least six environments and one step")
    device = "cuda:0"
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    command_mode = ck.get("training_stage") == "command-style"
    task = get_task(ck["task"])
    spec, height, motors = robot_parameters(task.robot)
    field_data = generate(task.profile, height, args.seed)
    field = HeightField(field_data, height, device)
    solver_iterations = ck["arguments"].get("solver_iterations", 50)
    model, *_ = build_model(
        task.robot, field_data, solver_iterations,
        self_collisions=ck["arguments"].get("self_collisions", False))
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
        )
        mdp.set_style_weight(ck["arguments"].get("amp_weight", .5))
        simulation = CommandSim
        evaluation_version = COMMAND_VERSION
    else:
        mdp = TerrainMDP(field, torch.tensor(spec.default_q, device=device), limits[:, 0], limits[:, 1],
                         gamma=ck["arguments"].get("gamma", .99))
        simulation = TerrainSim
        evaluation_version = OBJECTIVE_VERSION
    runtime_options = command_runtime_options(ck) if command_mode else {}
    env = simulation(task, mdp, args.num_envs, device, args.seed, solver_iterations,
                     max_episode_steps=ck["arguments"]["max_episode_steps"], **runtime_options)
    if command_mode and ck.get("command_envelope"):
        if ck["arguments"].get("stealth_control_mode") == "keyboard":
            env.set_stealth_keyboard_mode()
        else:
            from command_envelope import validate_envelope
            env.set_command_ranges(validate_envelope(ck["command_envelope"])["ranges"])
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
    style = None
    style_sum = torch.zeros(2, device=device)
    if command_mode:
        from trackerLab.algorithms.supermimic.amp_style import AMPStyle
        from trackerLab.kinematics.supermimic.robot import RobotKinematics
        kin = RobotKinematics(device, spec)
        style = AMPStyle.for_scoring(kin, ck["amp_style"], ground_sampler=field.sample)
        quality = attach_locomotion_quality(
            env, kin, field, motors, ck, upper_body_constraint=upper_body_constraint,
        )
        quality_sum = torch.zeros(5, device=device)
        style_history = style.frame_features(env.state())[:, None].expand(
            -1, style.history_steps, -1).clone()
    recorder = None
    if args.trajectory_output:
        free_id = np.flatnonzero(model.jnt_type == 0)[0]
        recorder = FirstEpisodeRecorder(
            env.qpos[:6].cpu().numpy(), field.features(env.state())[:6].cpu().numpy(),
            int(model.jnt_qposadr[free_id]), model.jnt_qposadr[joint_ids],
        )
        first_commands = env.c[:6].cpu().numpy()
        first_context = env.xi[:6].cpu().numpy()
        recorded_commands = [first_commands.copy()]
    history = torch.zeros(args.num_envs, actor.hist, mdp.obs_dim, device=device)
    counts = {name: torch.zeros(6, device=device) for name in
              ("episodes", "successes", "falls", "timeouts", "return_sum", "progress_sum")}
    progress = torch.zeros(args.num_envs, device=device)
    command_sum = torch.zeros(3, device=device)
    command_count = torch.zeros((), device=device)
    motion_sum = torch.zeros(2, device=device)
    keyboard_mode_sums = None
    if ck["arguments"].get("stealth_control_mode") == "keyboard":
        modes = ck.get("command_envelope", {}).get("command_modes", [])
        if len(modes) != 5:
            raise ValueError("Keyboard mode metadata does not match the five-mode policy contract")
        keyboard_mode_sums = {
            "modes": modes,
            "values": [torch.zeros(5, device=device) for _ in modes],
        }
    upper_body_sums = torch.zeros(3, device=device) if upper_body_constraint else None
    for _ in range(args.steps):
        if command_mode:
            first_frame = style.frame_features(env.state())
            style_history = torch.where((env.ep_step == 0)[:, None, None],
                                        first_frame[:, None].expand_as(style_history), style_history)
        observation = mdp.add_noise(env.observe())
        history = torch.where((env.ep_step == 0)[:, None, None],
                              observation[:, None].expand_as(history),
                              torch.cat((history[:, 1:], observation[:, None]), 1))
        action = actor(history, mdp.goal(env.state(), env.c))
        if upper_body_constraint is not None:
            n = spec.num_joints
            previous_action = history[:, -1, 2 * n + 6:3 * n + 6]
            action = upper_body_constraint(action, previous_action)
        tr = env.step(action)
        if command_mode:
            p, R, v, w, *_ = unflatten_state(tr["s1"])
            local_v = (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
            local_w = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
            velocity_error = ((local_v[:, :2] - tr["c"][:, 1:3]) / field.scale).square().sum(-1)
            yaw_error = (local_w[:, 2] - tr["c"][:, 3]).square()
            from trackerLab.tracker_env.mdp.supermimic_command import target_clearance
            clearance_error = ((p[:, 2] - field.sample(p[:, :2])
                                - target_clearance(tr["c1"])) / field.scale).square()
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
                for mode_index, values in enumerate(keyboard_mode_sums["values"]):
                    selected = valid & (mode_ids == mode_index)
                    values += torch.stack((
                        selected.sum().float(),
                        tr["command_metrics"]["command_tracking"][selected].float().sum(),
                        velocity_error[selected].sum(),
                        yaw_error[selected].sum(),
                        tr["command_metrics"]["commanded_progress_mps"][selected].sum(),
                    ))
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
                pose_error = state[4].index_select(-1, indices) - upper_body_constraint.reference_q
                arm_speed = state[5].index_select(-1, indices).abs()
                upper_body_sums += torch.stack((
                    pose_error[valid].abs().sum(),
                    arm_speed[valid].sum(),
                    valid.sum(),
                ))
        if recorder is not None and recorder.active.any():
            recorder.add(
                tr["s1"][:6].cpu().numpy(),
                field.features(unflatten_state(tr["s1"][:6])).cpu().numpy(),
                tr["done"][:6].cpu().numpy(), tr["episode_success"][:6].cpu().numpy(),
                tr["episode_fall"][:6].cpu().numpy(),
            )
            recorded_commands.append(tr.get("c1", tr["c"])[:6].cpu().numpy())
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
              "terrain_seed_held_out": args.seed != ck["arguments"]["seed"],
              "physics_steps": args.steps, "num_envs": args.num_envs,
              "policy_exploration_noise": False, "observation_noise": True, "lanes": lanes}
    if command_mode:
        result.update(command_evaluation_summary(
            command_sum, command_count, motion_sum, style_sum, quality_sum,
            ck.get("command_envelope"), keyboard_mode_sums))
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
                      solver_iterations=np.int64(solver_iterations))
        result["demo_trajectory"] = str(args.trajectory_output.resolve())
        result["demo_selection"] = "First episode of environment IDs 0..5, one per difficulty; no outcome filtering"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

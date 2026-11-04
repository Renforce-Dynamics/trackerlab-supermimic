"""Evaluate complete gait clips, including body/foot errors and preselected demos."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bootstrap import configure_paths
REPO = configure_paths()
from gait_runtime import TrackingSim, make_sim, task_settings
from trackerTask.supermimic.gait import FOUNDATION_VERSION, quality_passes
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import ROBOT_HEIGHTS
from trackerLab.algorithms.supermimic.networks import Policy
import trackerLab.algorithms.supermimic.networks as networks
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.kinematics.supermimic.rotations import mat_to_quat, yaw_mat
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp.supermimic import MAX_STEP_REWARD, RewardCfg, unflatten_state
from trackerLab.tracker_env.mdp.supermimic_gait import GaitMDP


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--trials-per-clip", type=int, default=16)
    parser.add_argument("--trajectory-output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ck["args"]
    if train_args.get("gait_foundation_version") != FOUNDATION_VERSION:
        raise ValueError("Checkpoint is not a current gait foundation")
    robot, device = train_args["task"], "cuda:0"
    spec = get_robot(robot)
    kin = RobotKinematics(device, spec)
    motion = MotionLib.from_config(str(args.motion_config), kin, device)
    cfg = task_settings(robot)
    mdp = GaitMDP(kin, motion, torch.tensor(spec.default_q, device=device),
                  cfg=RewardCfg(**cfg.reward_overrides), obs_mode=train_args["obs_mode"])
    actor = Policy(mdp.obs_dim, spec.goal_frame_dim * train_args["goal_frames"],
                   hist=train_args["hist"], num_joints=spec.num_joints,
                   action_limit=train_args["action_limit"]).to(device)
    actor.load_state_dict(ck["pi"])
    actor.eval()
    count, trials = len(motion.names), args.trials_per_clip
    if trials < 1:
        parser.error("Trial count must be positive")
    env = TrackingSim(make_sim(device, task_cfg=cfg), count * trials, motion, mdp,
                      gamma=train_args["gamma"], eval_mode=True, task_cfg=cfg)
    env.reset(frame_ids=motion.clip_start.repeat_interleave(trials))
    histories = torch.zeros(count * trials, actor.hist, mdp.obs_dim, device=device)
    names = ("body_error", "joint_squared", "tilt_error", "foot_squared", "slip", "samples", "contacts")
    sums = {name: torch.zeros(count * trials, device=device) for name in names}
    selected = torch.arange(count, device=device) * trials
    recorded, alive = [], []
    if args.trajectory_output:
        initial = unflatten_state(env.state())
        final_states = torch.cat((initial[0], mat_to_quat(initial[1]), initial[4]), -1)[selected]
        recorded.append(final_states.cpu().numpy())
        alive.append((~env.done_eval[selected]).cpu().numpy())
    scale = ROBOT_HEIGHTS[robot] / .76
    for step in range(int(motion.clip_len.max().item())):
        state = env.state()
        active = ~env.done_eval
        if not active.any():
            break
        observation = mdp.add_noise(mdp.obs(state, env.a_prev, env.xi, env.f))
        histories = torch.where((env.ep_step == 0)[:, None, None],
                                observation[:, None].expand_as(histories),
                                torch.cat((histories[:, 1:], observation[:, None]), 1))
        action = actor(histories, motion.goal(env.f, train_args["goal_frames"]))
        transition = env.step(action)
        state = unflatten_state(transition["s1"])
        if args.trajectory_output:
            flat = torch.cat((state[0], mat_to_quat(state[1]), state[4]), -1)[selected]
            final_states = torch.where(active[selected, None], flat, final_states)
            recorded.append(final_states.cpu().numpy())
            alive.append(active[selected].cpu().numpy())
        frame = transition["f"] + 1
        body = kin.forward(*state)
        reference_root = motion.root_p[frame]
        alignment = yaw_mat(state[1] @ motion.root_R[frame].transpose(-1, -2))
        anchor = torch.cat((state[0][:, :2], reference_root[:, 2:3]), -1)
        reference_body = anchor[:, None] + (
            alignment[:, None] @ (motion.body_p[frame] - reference_root[:, None])[..., None]).squeeze(-1)
        errors = mdp.gait_errors(state, frame, body)
        _, foot_velocity = mdp.foot_state(body)
        stance = mdp.reference_stance[frame]
        values = {
            "body_error": ((body[0] - reference_body) / scale).norm(dim=-1).mean(-1),
            "joint_squared": errors["joint_tracking"],
            "tilt_error": torch.acos((state[1][:, 2] * motion.root_R[frame][:, 2]).sum(-1).clamp(-1, 1)),
            "foot_squared": errors["foot_tracking"],
            "slip": (foot_velocity[..., :2].norm(dim=-1) * stance).sum(-1) / scale,
            "samples": torch.ones_like(env.ep_return),
            "contacts": stance.sum(-1).float(),
        }
        for name, value in values.items():
            sums[name] += torch.where(active, value, 0)
        if args.trajectory_output:
            flat = torch.cat((state[0], mat_to_quat(state[1]), state[4]), -1)[selected]
            final_states = torch.where(active[selected, None], flat, final_states)
    per_motion, all_metrics = {}, []
    for clip, name in enumerate(motion.names):
        group = slice(clip * trials, (clip + 1) * trials)
        total = {key: float(value[group].sum()) for key, value in sums.items()}
        denominator = max(total["samples"], 1)
        metrics = {
            "full_clip_survival": float((env.done_eval[group] & ~env.failed_eval[group]).float().mean()),
            "mean_body_position_error_scale_m": total["body_error"] / denominator,
            "joint_rmse_rad": (total["joint_squared"] / denominator) ** .5,
            "root_tilt_error_rad": total["tilt_error"] / denominator,
            "foot_position_rmse_scale_m": (total["foot_squared"] / denominator) ** .5,
            "stance_foot_slip_scale_m_s": total["slip"] / max(total["contacts"], 1),
        }
        per_motion[name] = {
            **metrics, "quality_passed": quality_passes(metrics),
            "failed": bool(env.failed_eval[clip * trials]),
            "normalized_return": float((env.ep_return[group] / (
                env.ep_step[group].clamp_min(1) * MAX_STEP_REWARD)).mean()),
            "trials": trials,
        }
        all_metrics.append(metrics)
    metrics = {key: sum(row[key] for row in all_metrics) / len(all_metrics) for key in all_metrics[0]}
    result = {
        "stage": "reference-gait-foundation", "robot": robot, "iteration": ck["iteration"],
        "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "motion_config": str(args.motion_config.resolve()), "metrics": metrics,
        "quality_passed": all(row["quality_passed"] for row in per_motion.values()),
        "per_motion": per_motion, "reward_config": asdict(mdp.cfg),
        "protocol": f"Complete reference clips from frame zero; {trials} observation-noise trials per clip",
    }
    if args.trajectory_output:
        frames = np.stack(recorded)
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trajectory_output, root_p=frames[..., :3], root_quat=frames[..., 3:7],
            q=frames[..., 7:], alive=np.stack(alive), names=np.array(motion.names),
            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy(),
            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(),
            ref_q=motion.q.cpu().numpy(), iteration=np.int64(ck["iteration"]),
            task=np.array(robot), simulator=np.array("MuJoCo-Warp"),
            reference_label=np.array("Audited locomotion reference"),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

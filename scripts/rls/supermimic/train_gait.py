"""Train a robot's SuperMimic gait foundation from audited locomotion references."""

import argparse
import json
from pathlib import Path
import sys
import time

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.gait import FOUNDATION_VERSION, ROBOTS
from trackerTask.supermimic.terrain import ROBOT_HEIGHTS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=ROBOTS, required=True)
    parser.add_argument("--motion_config", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--num_envs", type=int, default=8096)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    for name, default in (
        ("buffer_steps", 1000), ("horizon", 32), ("goal_frames", 10), ("hist", 10),
        ("wm_window", 8), ("wm_batches", 10), ("wm_batch", 2048), ("pi_batch", 4096),
        ("critic_epochs", 5), ("critic_minibatches", 4), ("warmup_steps", 64),
        ("wm_only_iters", 500), ("lr_decay_iters", 10000), ("ckpt_every", 2000), ("log_every", 25),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_minutes", type=float, default=0)
    parser.add_argument("--stop_file")
    args = parser.parse_args(argv)
    scale = ROBOT_HEIGHTS[args.task] / .76
    defaults = dict(
        env_spacing=0., obs_mode="relative_orientation", reset_noise=True,
        friction_range=(.3, 1.2), joint_offset_range=.01,
        com_offset_range=(.02 * scale, .025 * scale, .025 * scale),
        allow_tf32=False, amp="bf16", utd=.4, lr=2e-4, gamma=.99, lam=.95,
        action_noise=.1, action_limit=4., bounded_critic=True, no_critic=False,
        keep_optimizer_checkpoints=True,
        motion_dir=None, gait_foundation_version=FOUNDATION_VERSION,
    )
    vars(args).update(defaults)
    if min(args.num_envs, args.iterations, args.horizon, args.goal_frames, args.hist,
           args.wm_window, args.wm_batches, args.wm_batch, args.pi_batch,
           args.critic_epochs, args.critic_minibatches, args.ckpt_every, args.log_every) < 1:
        parser.error("Training dimensions must be positive")
    if args.buffer_steps < max(args.wm_window, args.horizon) or args.wm_only_iters < 0:
        parser.error("Replay must cover both training windows and warmup must be nonnegative")
    root = Path(args.log_dir).resolve()
    if (root / "last.pt").exists() or (root / "train_log.csv").exists():
        if not args.resume:
            parser.error("Use a new output directory, including for resume")
    args.log_dir = str(root)
    args.motion_config = str(Path(args.motion_config).resolve())
    if args.resume:
        args.resume = str(Path(args.resume).resolve())
    args.stop_file = str(Path(args.stop_file).resolve()) if args.stop_file else str(root / "STOP")
    return args


def main():
    args = parse_args()
    import torch
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("Gait foundation training requires CUDA")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck["args"].get("gait_foundation_version") != FOUNDATION_VERSION:
            raise ValueError("Gait foundations resume only from the same gait objective")
        del ck
    from prepare_gait_references import verify_audit
    verify_audit(args.motion_config, args.task)
    from gait_runtime import TrackingSim, launch_simulation, make_sim, task_settings
    from trackerLab.tracker_env.mdp.supermimic_gait import GaitMDP
    from trackerLab.algorithms.supermimic.runner import train
    train(args, None, time.monotonic(),
          simulation_api=(TrackingSim, launch_simulation, make_sim),
          task_cfg=task_settings(args.task), mdp_cls=GaitMDP)


if __name__ == "__main__":
    main()

"""TrackerLab SuperMimic training entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
from bootstrap import configure_paths
configure_paths()

from isaaclab.app import AppLauncher
from configuration import apply_config
from trackerTask.supermimic.robots import TASKS

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=TASKS, default="g1")
parser.add_argument("--config", help="JSON training preset; explicit CLI flags take precedence")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--max_minutes", type=float, default=0.0, help="save and exit after this run time; 0 disables the limit")
parser.add_argument("--stop_file", type=str, default=None, help="save and exit when this file appears")
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--action_limit", type=float, default=None,
                    help="Optional smooth policy action bound, shared by real and imagined rollouts")
parser.add_argument("--bounded_critic", action=argparse.BooleanOptionalAction, default=False,
                    help="Cap critic bootstrap at the maximum discounted tracking reward")
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
motions = parser.add_mutually_exclusive_group()
motions.add_argument("--motion_dir", type=str, default=str(REPO / "data" / "retargeted" / "lafan1_g1"))
motions.add_argument("--motion_config", type=str, help="TrackerLab reference-motion JSON in data/configs")
parser.add_argument("--log_dir", type=str, default=str(REPO / "logs" / "supermimic"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--warm_start", type=str, default=None, help="Initialize a new downstream task from gait or chase weights")
parser.add_argument("--warm_start_velocity", type=str, default=None,
                    help="Initialize a new velocity objective from a velocity/dodge checkpoint")
parser.add_argument("--task_difficulty", type=float, default=None,
                    help="Fixed velocity/dodge difficulty in [0,1]; omit for its training curriculum")
parser.add_argument("--task_profile", choices=["standard", "rapid", "jump", "adaptive"], default="standard",
                    help="Scenario curriculum for go2_velocity_dodge")
parser.add_argument("--obs_mode", type=str, default=None,
                    choices=["paper_obs93", "relative_ori99", "proprioception", "relative_orientation"])
parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--friction_range", type=float, nargs=2, default=(0.3, 1.2))
parser.add_argument("--joint_offset_range", type=float, default=0.01)
parser.add_argument("--com_offset_range", type=float, nargs=3, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cuda:0")
apply_config(parser)
args = parser.parse_args()
args.obs_mode = args.obs_mode or ("relative_ori99" if args.task == "g1" else "relative_orientation")
args.com_offset_range = args.com_offset_range or (
    (0.025, 0.05, 0.05) if args.task in ("g1", "a3") else
    (0.01, 0.02, 0.02) if args.task == "t1" else (0.01, 0.01, 0.01))
if args.action_limit is not None and (not math.isfinite(args.action_limit) or not 0 < args.action_limit <= 20):
    parser.error("--action_limit must be finite, positive and at most the simulator limit of 20")
if args.task != "g1" and not args.motion_config:
    parser.error("This task requires --motion_config produced by prepare_resources.py")
if args.task != "g1" and args.obs_mode in ("paper_obs93", "relative_ori99"):
    parser.error("Use --obs_mode relative_orientation or proprioception for this robot")
if args.buffer_steps < args.wm_window or args.wm_window < 1:
    parser.error("--buffer_steps must cover a positive --wm_window")
if min(args.num_envs, args.iterations, args.horizon, args.hist, args.goal_frames, args.wm_batches,
       args.wm_batch, args.pi_batch, args.critic_epochs, args.log_every, args.ckpt_every) < 1:
    parser.error("environment, rollout, batch and logging sizes must be positive")
if args.max_minutes < 0:
    parser.error("--max_minutes must be nonnegative")
if args.utd <= 0 or args.horizon * args.pi_batch < args.critic_minibatches or args.critic_minibatches < 1:
    parser.error("invalid update ratio or critic minibatch count")
if not 0 < args.gamma < 1 or not 0 <= args.lam <= 1 or args.lr <= 0 or args.lr_decay_iters < 1:
    parser.error("invalid discount, GAE or learning-rate configuration")
if args.warmup_steps < 0 or args.wm_only_iters < 0:
    parser.error("warmup sizes must be nonnegative")
if sum(bool(value) for value in (args.resume, args.warm_start, args.warm_start_velocity)) > 1:
    parser.error("Choose only one of --resume, --warm_start or --warm_start_velocity")
if args.warm_start and args.task not in ("go2_chase_dodge", "go2_velocity_dodge"):
    parser.error("--warm_start is supported by go2_chase_dodge and go2_velocity_dodge")
if args.task in ("go2_chase_dodge", "go2_velocity_dodge") and (
        args.action_limit != 4.0 or args.obs_mode != "relative_orientation" or args.no_critic):
    parser.error(f"{args.task} requires action_limit=4, relative_orientation and the critic")
if args.task_difficulty is not None and (
        args.task != "go2_velocity_dodge" or not math.isfinite(args.task_difficulty) or
        not 0 <= args.task_difficulty <= 1):
    parser.error("--task_difficulty requires go2_velocity_dodge and a finite value in [0,1]")
if args.task_profile != "standard" and args.task != "go2_velocity_dodge":
    parser.error("--task_profile requires go2_velocity_dodge")
if args.warm_start_velocity and args.task != "go2_velocity_dodge":
    parser.error("--warm_start_velocity requires go2_velocity_dodge")
if args.task == "go2_velocity_dodge" and not (args.warm_start or args.warm_start_velocity or args.resume):
    parser.error("go2_velocity_dodge requires a compatible warm-start or resume checkpoint")
for name in ("motion_dir", "motion_config", "log_dir", "resume", "warm_start", "warm_start_velocity", "stop_file", "config"):
    if getattr(args, name):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
if (Path(args.log_dir) / "last.pt").exists() or (Path(args.log_dir) / "train_log.csv").exists():
    parser.error("Use a new --log_dir for each run, including --resume, to preserve previous artifacts")
run_start = time.monotonic()
app_launcher = AppLauncher(args)


if args.task == "go2_velocity_dodge":
    from trackerLab.algorithms.supermimic.velocity_runner import train
elif args.task == "go2_chase_dodge":
    from trackerLab.algorithms.supermimic.chase_runner import train
else:
    from trackerLab.algorithms.supermimic.runner import train

train(args, app_launcher, run_start)

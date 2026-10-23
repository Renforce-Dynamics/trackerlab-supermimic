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

parser = argparse.ArgumentParser()
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
parser.add_argument("--obs_mode", type=str, default="relative_ori99", choices=["paper_obs93", "relative_ori99"])
parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--friction_range", type=float, nargs=2, default=(0.3, 1.2))
parser.add_argument("--joint_offset_range", type=float, default=0.01)
parser.add_argument("--com_offset_range", type=float, nargs=3, default=(0.025, 0.05, 0.05))
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cuda:0")
apply_config(parser)
args = parser.parse_args()
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
for name in ("motion_dir", "motion_config", "log_dir", "resume", "stop_file", "config"):
    if getattr(args, name):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
if (Path(args.log_dir) / "last.pt").exists() or (Path(args.log_dir) / "train_log.csv").exists():
    parser.error("Use a new --log_dir for each run, including --resume, to preserve previous artifacts")
run_start = time.monotonic()
app_launcher = AppLauncher(args)


from trackerLab.algorithms.supermimic.runner import train

train(args, app_launcher, run_start)

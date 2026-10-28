"""Portable G1 scratch training using the accepted native BF16 Warp runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get(
    "TRACKERLAB_BASE_REPO", Path(__file__).resolve().parents[4])).expanduser().resolve()
for package in ("trackerLab", "trackerTask"):
    sys.path.insert(0, str(REPO / "source" / package))
os.environ.setdefault("SUPERMIMIC_REPO_ROOT", str(REPO))
from runtime_settings import configure_environment
runtime_environment = configure_environment()
from configuration import apply_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("g1",), default="g1")
parser.add_argument("--config", help="JSON training preset; explicit CLI flags take precedence")
parser.add_argument("--dry-run", action="store_true",
                    help="Print resolved configuration without loading PyTorch, physics or data")
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
parser.add_argument("--motion_storage", choices=("device",), default="device",
                    help="Integrity-checked reference roots and body fields resident on CUDA")
parser.add_argument("--motion_cache_dir", type=str, default=None,
                    help="Mmap cache directory; defaults to <log_dir>/motion_cache")
parser.add_argument("--motion_cache_mb", type=float, default=128.0,
                    help="Maximum retained mmap reference batches on the execution device, in MiB")
parser.add_argument("--motion_transfer_rows", type=int, default=8192,
                    help="Maximum CPU staging rows per mmap-to-device copy")
parser.add_argument("--log_dir", type=str, default=str(REPO / "logs" / "g1_warp_bf16"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--obs_mode", type=str, default=None,
                    choices=["paper_obs93", "relative_ori99", "proprioception", "relative_orientation"])
parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--friction_range", type=float, nargs=2, default=(0.3, 1.2))
parser.add_argument("--joint_offset_range", type=float, default=0.01)
parser.add_argument("--com_offset_range", type=float, nargs=3, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--kit_args", default="", help="Compatibility argument; Kit is never loaded")
parser.add_argument("--motion_root_cache", required=True)
parser.add_argument("--resume_wm_only_iters", type=int, default=0)
parser.add_argument("--warp_model", default=None)
parser.add_argument("--warp_nconmax", type=int, default=128)
parser.add_argument("--warp_njmax", type=int, default=500)
parser.add_argument("--warp_solver_iterations", type=int, default=50)
parser.add_argument("--warp_solver_ls_iterations", type=int, default=50)
parser.set_defaults(headless=True, device="cuda:0")
apply_config(parser)
args = parser.parse_args()
if args.resume is not None:
    parser.error('This named experiment starts from random initialization')
if not args.motion_config:
    parser.error("--motion_config is required for the integrity-checked root83 cache")
if args.task != "g1" or args.amp != "bf16":
    parser.error("This runtime supports G1 with native BF16 MLPs")
args.obs_mode = args.obs_mode or ("relative_ori99" if args.task == "g1" else "relative_orientation")
if args.obs_mode != "relative_ori99" or args.goal_frames != 10:
    parser.error("The qualified normalizer fixture requires relative_ori99 and 10 goal frames")
if args.motion_storage != "device":
    parser.error("This Warp runtime requires the device-resident root83 cache")
if not args.device.startswith("cuda"):
    parser.error("This Warp runtime requires a CUDA device")
args.com_offset_range = args.com_offset_range or (
    (0.025, 0.05, 0.05) if args.task == "g1" else (0.01, 0.01, 0.01))
if args.action_limit is not None and (not math.isfinite(args.action_limit) or not 0 < args.action_limit <= 20):
    parser.error("--action_limit must be finite, positive and at most the simulator limit of 20")
if args.buffer_steps < args.wm_window or args.wm_window < 1:
    parser.error("--buffer_steps must cover a positive --wm_window")
if min(args.num_envs, args.iterations, args.horizon, args.hist, args.goal_frames, args.wm_batches,
       args.wm_batch, args.pi_batch, args.critic_epochs, args.log_every, args.ckpt_every) < 1:
    parser.error("environment, rollout, batch and logging sizes must be positive")
if args.max_minutes < 0:
    parser.error("--max_minutes must be nonnegative")
if (not math.isfinite(args.motion_cache_mb) or args.motion_cache_mb < 0
        or args.motion_transfer_rows < 1):
    parser.error("motion cache size must be finite/nonnegative and transfer rows positive")
if args.utd <= 0 or args.horizon * args.pi_batch < args.critic_minibatches or args.critic_minibatches < 1:
    parser.error("invalid update ratio or critic minibatch count")
if not 0 < args.gamma < 1 or not 0 <= args.lam <= 1 or args.lr <= 0 or args.lr_decay_iters < 1:
    parser.error("invalid discount, GAE or learning-rate configuration")
if args.warmup_steps < 0 or args.wm_only_iters < 0 or args.resume_wm_only_iters < 0:
    parser.error("warmup sizes must be nonnegative")
for name in ("motion_dir", "motion_config", "motion_cache_dir", "motion_root_cache",
             "log_dir", "resume", "stop_file", "config"):
    if getattr(args, name):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
if (Path(args.log_dir) / "last.pt").exists() or (Path(args.log_dir) / "train_log.csv").exists():
    parser.error("Use a new --log_dir for each run, including --resume, to preserve previous artifacts")
if args.dry_run:
    print(json.dumps({"arguments": vars(args), "environment": runtime_environment,
                      "repository": str(REPO), "training_started": False}, indent=2))
    raise SystemExit(0)
run_start = time.monotonic()
app_launcher = None


from runner_warp import train

train(args, app_launcher, run_start)

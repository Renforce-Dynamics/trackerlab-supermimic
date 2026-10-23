"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python scripts/rls/supermimic/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
from bootstrap import configure_paths
configure_paths()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=("g1", "go2"), default=None,
                    help="Optional assertion; robot is inferred from checkpoint arguments")
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
motions = parser.add_mutually_exclusive_group()
motions.add_argument("--motion_dir", type=str, default=str(REPO / "data" / "retargeted" / "lafan1_g1"))
motions.add_argument("--motion_config", type=str)
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cuda:0")
args = parser.parse_args()
if args.every < 1:
    parser.error("--every must be positive")
app_launcher = AppLauncher(args)

import numpy as np
import torch

from trackerTask.supermimic.robots import get_robot, get_task_cfg
import trackerLab.algorithms.supermimic.networks as networks
from trackerLab.tracker_env.supermimic.env import TrackingSim, launch_simulation, make_sim
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.tracker_env.mdp.supermimic import MAX_STEP_REWARD, MDP, REWARD_TERMS, RewardCfg
from trackerLab.metrics.supermimic import summarize_motion_returns, write_json
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.algorithms.supermimic.networks import Policy
from trackerLab.kinematics.supermimic.rotations import mat_to_quat

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
dev = args.device
initial_config = (torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"] if args.ckpt
                  else json.loads((Path(args.run_dir) / "args.json").read_text())["arguments"])
task_name = initial_config.get("task", "g1")
if args.task is not None and args.task != task_name:
    raise ValueError("Requested task differs from checkpoint")
args.task = task_name
if args.motion_config is None and "--motion_dir" not in sys.argv:
    args.motion_config = initial_config.get("motion_config")
    args.motion_dir = initial_config.get("motion_dir", args.motion_dir)
if task_name == "go2" and not args.motion_config:
    raise ValueError("Go2 evaluation requires --motion_config declaring the source frame rate")
spec = get_robot(task_name)
kin = RobotKinematics(dev, spec)
motion = (MotionLib.from_config(args.motion_config, kin, dev) if args.motion_config
          else MotionLib.from_dir(args.motion_dir, kin, dev))
_obs_mode = initial_config.get("obs_mode", "paper_obs93")
mdp = MDP(kin, motion, torch.tensor(spec.default_q, device=dev), obs_mode=_obs_mode)
C = len(motion.names)
task_cfg = get_task_cfg(task_name)
task_cfg.num_envs = C


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    a_ = ck["args"]
    if a_.get("task", "g1") != task_name:
        raise ValueError("Evaluation cannot mix robot tasks")
    K, Hh = a_["goal_frames"], a_["hist"]
    mdp.set_obs_mode(a_.get("obs_mode", "paper_obs93"))
    mdp.cfg = RewardCfg(**ck.get("reward_cfg", {}))
    mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
    mdp.soft_lo, mdp.soft_hi = mid - half * mdp.cfg.soft_limit_factor, mid + half * mdp.cfg.soft_limit_factor
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(mdp.obs_dim, spec.goal_frame_dim * K, hist=Hh, num_joints=spec.num_joints,
                action_limit=a_.get("action_limit")).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end.clone()
    ret = torch.zeros(C, device=dev)
    term_ret = torch.zeros(C, len(REWARD_TERMS), device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    evaluation_start = time.monotonic()
    control_step = 0
    max_control_steps = int((motion.clip_len - 1).max().item())
    locomotion = torch.zeros(C, 5, device=dev)

    def record_state(state, alive):
        traj["root_p"].append(state[0].cpu())
        traj["root_quat"].append(mat_to_quat(state[1]).cpu())
        traj["q"].append(state[4].cpu())
        traj["f"].append(env.f.cpu())
        traj["alive"].append(alive.cpu())

    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if record:
            record_state(S, alive)
        out = env.step(a)
        S1 = env.state()
        ref = motion.ref_state(env.f)
        measured = torch.stack((
            S1[2][:, 0], ref[2][:, 0],
            ((S1[2] - ref[2]) ** 2).sum(-1),
            ((S1[0] - ref[0]) ** 2).sum(-1),
            ((S1[4] - ref[4]) ** 2).mean(-1),
        ), -1)
        locomotion += torch.where(alive[:, None], measured, 0)
        ret += torch.where(alive, out["r"], 0.0)
        contributions = torch.stack([out["reward_terms"][key] for key in REWARD_TERMS], -1)
        term_ret += torch.where(alive[:, None], contributions, 0.0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
        control_step += 1
        if control_step % 1000 == 0:
            print(
                f"Evaluation step {control_step}/{max_control_steps}; "
                f"finished {int(env.done_eval.sum().item())}/{C}; "
                f"failed {int(env.failed_eval.sum().item())}; "
                f"elapsed {time.monotonic() - evaluation_start:.1f}s",
                flush=True,
            )
        if control_step >= max_control_steps and not bool(env.done_eval.all()):
            raise RuntimeError("Evaluation exceeded the longest reference clip")
    summary = summarize_motion_returns(
        motion.names,
        ret.cpu().tolist(),
        (motion.clip_len - 1).cpu().tolist(),
        env.failed_eval.cpu().tolist(),
        max_normalized_return=MAX_STEP_REWARD,
    )
    active_steps = env.ep_step.cpu().tolist()
    survival_returns = [
        float(total) / (max(int(steps), 1) * MAX_STEP_REWARD)
        for total, steps in zip(ret.cpu().tolist(), active_steps)
    ]
    total_active_steps = max(sum(active_steps), 1)
    per_motion = summary["per_motion"]
    term_norm = term_ret / ((motion.clip_len - 1) * MAX_STEP_REWARD)[:, None]
    if not torch.allclose(term_ret.sum(-1), ret, rtol=2e-5, atol=2e-4):
        raise AssertionError("Reward components do not sum to the evaluated return")
    term_values = term_norm.cpu().tolist()
    for i, name in enumerate(motion.names):
        per_motion[name]["survival_normalized_return"] = survival_returns[i]
        per_motion[name]["survival_steps"] = int(active_steps[i])
        per_motion[name]["reward_components"] = dict(zip(REWARD_TERMS, term_values[i]))
        values = (locomotion[i] / max(active_steps[i], 1)).cpu().tolist()
        per_motion[name]["tracking"] = {
            "mean_forward_speed_m_s": values[0], "reference_forward_speed_m_s": values[1],
            "speed_axis": "world X; not heading-aligned",
            "root_velocity_rmse_m_s": values[2] ** 0.5,
            "root_position_rmse_m": values[3] ** 0.5,
            "joint_position_rmse_rad": values[4] ** 0.5,
            "measured_seconds": active_steps[i] * 0.02,
        }
    res = {
        "iteration": int(ck["iteration"]),
        "evaluation_seconds": time.monotonic() - evaluation_start,
        "protocol": {
            "task": task_name,
            "dataset": f"{task_name} motion library; {C} clips from {args.motion_config or args.motion_dir}",
            "data_provenance": motion.provenance,
            "start": "first frame of every full clip",
            "policy": "deterministic",
            "friction": 1.0,
            "domain_randomization": False,
            "observation_noise": False,
            "max_step_reward": MAX_STEP_REWARD,
            "termination": f"reference root height error > {mdp.cfg.term_height} m or tilt component error > {mdp.cfg.term_tilt}",
            "normalization": "reward_sum / ((clip_len - 1) * max_step_reward); zero reward after termination",
            "observation_mode": a_.get("obs_mode", "paper_obs93"),
            "mlp_dtype": a_.get("amp", "none"),
            "simulator": "IsaacLab / Isaac Sim / PhysX",
            "runtime_versions": {
                name: importlib.metadata.version(name) for name in ("isaaclab", "isaacsim", "torch", "mujoco")
            },
            "reset_noise": False,
            "policy_action_limit": a_.get("action_limit"),
            "reward_cfg": asdict(mdp.cfg),
        },
        "norm_return": summary["clip_weighted_return"],
        "norm_return_framew": summary["frame_weighted_return"],
        "survival_norm_return": sum(survival_returns) / C,
        "survival_norm_return_stepw": float(ret.sum().item()) / (total_active_steps * MAX_STEP_REWARD),
        "n_failed": summary["failed_count"],
        "coverage": summary["coverage"],
        "reward_components": dict(zip(REWARD_TERMS, term_norm.mean(0).cpu().tolist())),
        "reward_components_framew": dict(zip(
            REWARD_TERMS,
            (term_ret.sum(0) / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)).cpu().tolist(),
        )),
        "failed": summary["failed_motions"],
        "per_clip": {
            name: round(values["normalized_return"], 4)
            for name, values in summary["per_motion"].items()
        },
        "per_motion": summary["per_motion"],
    }
    if record:
        # Include the terminal pose even when the last active clip ends.
        record_state(env.state(), ~env.done_eval.clone())
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy(),
                            iteration=np.array(int(ck["iteration"])), task=np.array(task_name),
                            control_frequency_hz=np.array(50))  # fmt: skip
    return res


sim_cfg = make_sim(dev, args, task_cfg)
with launch_simulation(sim_cfg, args, app_launcher=app_launcher):
    env = TrackingSim(sim_cfg, C, motion, mdp, eval_mode=True, task_cfg=task_cfg)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        res = evaluate(env, ck, args.record)
        write_json(str(Path(args.ckpt).with_suffix("")) + "_eval.json", res)
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        sys.exit(0)
    out_csv = os.path.join(args.run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,coverage,failed\n")
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
        todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
        # Select by checkpoint order so custom save intervals work as documented.
        selected = set(ckpts[::args.every])
        todo = [c for c in todo if c in selected]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(env, ck)
            line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{res['coverage']:.4f},{'|'.join(res['failed'])}\n"
            open(out_csv, "a").write(line)
            write_json(c.replace(".pt", "_eval.json"), res)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)

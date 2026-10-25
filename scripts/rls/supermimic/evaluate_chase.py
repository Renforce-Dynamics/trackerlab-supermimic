"""Evaluate task checkpoints on deterministic flat, duck and jump trials."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from bootstrap import configure_paths
configure_paths()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--motion_config", type=Path)
parser.add_argument("--difficulty", type=float, default=1.)
parser.add_argument("--trials_per_mode", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cuda:0")
args = parser.parse_args()
app_launcher = AppLauncher(args)

import numpy as np
import torch
import trackerLab.algorithms.supermimic.networks as networks
from trackerLab.algorithms.supermimic.networks import Critic
from trackerLab.algorithms.supermimic.chase_networks import TaskPolicy
from trackerLab.tracker_env.supermimic.env import make_sim, launch_simulation
from trackerLab.tracker_env.supermimic.chase_dodge import ChaseDodgeSim
from trackerLab.tracker_env.mdp.supermimic import RewardCfg, normalize_xi
from trackerLab.tracker_env.mdp.supermimic_chase_dodge import ChaseDodgeMDP, TERMS, scenario_state
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.kinematics.supermimic.rotations import mat_to_quat
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.go2_chase_dodge_cfg import make_task_cfg, ChaseDodgeCfg

torch.manual_seed(20261023)
torch.backends.cuda.matmul.allow_tf32 = True
ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
config = ck["args"]
if config["task"] != "go2_chase_dodge":
    raise ValueError("Expected a chase/dodge checkpoint")
networks.AMP_DTYPE = torch.bfloat16 if config["amp"] == "bf16" else None
spec = get_robot("go2")
task = ChaseDodgeCfg()
cfg = make_task_cfg()
cfg.num_envs = args.trials_per_mode * 3
kin = RobotKinematics(args.device, spec)
motion = MotionLib.from_config(str(args.motion_config or config["motion_config"]), kin, args.device)
mdp = ChaseDodgeMDP(kin, motion, torch.tensor(spec.default_q, device=args.device),
                   cfg=RewardCfg(**cfg.reward_overrides), obs_mode=config["obs_mode"])
goal_dim = spec.goal_frame_dim * config["goal_frames"]
pi = TaskPolicy(mdp.obs_dim, goal_dim, hist=config["hist"], num_joints=12, action_limit=config["action_limit"]).to(args.device)
pi.load_state_dict(ck["pi"])
pi.eval()
critic = Critic(spec.state_dim + spec.context_dim + task.task_feature_dim + goal_dim).to(args.device)
critic.load_state_dict(ck["critic"])
args.output.mkdir(parents=True, exist_ok=True)
sim_cfg = make_sim(args.device, task_cfg=cfg)
with launch_simulation(sim_cfg, args, app_launcher=app_launcher), torch.no_grad():
    env = ChaseDodgeSim(sim_cfg, cfg.num_envs, motion, mdp, cfg, eval_mode=True)
    env.difficulty = args.difficulty
    env.reset(frame_ids=torch.zeros(cfg.num_envs, dtype=torch.long, device=args.device))
    modes = torch.arange(cfg.num_envs, device=args.device) % 3
    env.scenario[:, 8] = modes
    env.episode_mode[:] = modes
    env.scenario[:, 7] = torch.where(modes == 1, .65 - .31 * args.difficulty, .025 + .095 * args.difficulty)
    env._write_bar(env.scenario)
    selected = [0, 1, 2]
    trajectory = {k: [] for k in ("root_p", "root_quat", "q", "scenario", "alive", "contact_force")}
    rewards = torch.zeros(cfg.num_envs, len(TERMS), device=args.device)
    speed_error = torch.zeros(cfg.num_envs, device=args.device)
    distance = torch.zeros_like(speed_error)
    active_steps = torch.zeros_like(speed_error)
    raw_values = []
    hist = None
    started = time.monotonic()
    for step in range(task.episode_steps):
        S = env.state()
        active = ~env.done_eval.clone()
        o = mdp.obs(S, env.a_prev, env.xi, env.f, env.scenario)
        hist = o[:, None].repeat(1, config["hist"], 1) if hist is None else torch.cat((hist[:, 1:], o[:, None]), 1)
        g = motion.goal(env.f, config["goal_frames"])
        a = pi(hist, g, mdp.task_features(S, env.scenario))
        x = torch.cat((mdp.task_critic_features(S, normalize_xi(env.xi), env.scenario), g), -1)
        raw_values.append(critic(x)[active].cpu())
        trajectory["root_p"].append(S[0][selected].cpu().numpy())
        trajectory["root_quat"].append(mat_to_quat(S[1])[selected].cpu().numpy())
        trajectory["q"].append(S[4][selected].cpu().numpy())
        trajectory["scenario"].append(env.scenario[selected].cpu().numpy())
        trajectory["alive"].append(active[selected].cpu().numpy())
        out = env.step(a)
        trajectory["contact_force"].append(out["bar_contact_force"][selected].cpu().numpy())
        target, velocity, _ = scenario_state(out["scenario1"])
        S1 = env.state()
        speed_error += ((S1[2][:, :2] - velocity)**2).sum(-1) * active
        distance += (S1[0][:, :2] - target).norm(dim=-1) * active
        active_steps += active
        rewards += torch.stack([out["reward_terms"][k] for k in TERMS], -1) * active[:, None]
    per_mode = {}
    for mode, name in enumerate(("flat", "duck", "jump")):
        mask = modes == mode
        success = ~env.ever_fallen if mode == 0 else env.passed & ~env.collided & ~env.ever_fallen
        denominator = active_steps[mask].sum().clamp_min(1)
        per_mode[name] = {
            "trials": int(mask.sum()), "successes": int(success[mask].sum()),
            "success_rate": float(success[mask].float().mean()),
            "collision_rate": float(env.collided[mask].float().mean()),
            "fall_rate": float(env.ever_fallen[mask].float().mean()),
            "passed_bar_rate": float(env.passed[mask].float().mean()) if mode else None,
            "velocity_rmse_m_s": float((speed_error[mask].sum()/denominator).sqrt()),
            "mean_target_distance_m": float(distance[mask].sum()/denominator),
            "full_episode_return": float(rewards[mask].sum(-1).mean()/(task.episode_steps*.1)),
            "reward_terms": {k: float(rewards[mask, i].mean()/(task.episode_steps*.1)) for i, k in enumerate(TERMS)},
        }
    v = torch.cat(raw_values)
    result = {
        "task": "go2_chase_dodge", "iteration": ck["iteration"], "difficulty": args.difficulty,
        "per_mode": per_mode, "raw_value_quantiles": dict(zip(
            ("min", "p50", "p90", "p99", "max"), torch.quantile(v, torch.tensor([0., .5, .9, .99, 1.])).tolist())),
        "raw_value_above_10_fraction": float((v > 10).float().mean()),
        "evaluation_seconds": time.monotonic()-started,
        "protocol": {"simulator": "IsaacLab / PhysX", "episode_seconds": task.episode_steps*.02,
                     "seed": 20261023, "action_noise": False, "observation_noise": False,
                     "physical_collision_threshold_N": 1.0, "task_config": asdict(task),
                     "success": "flat: no fall; bars: crossed with no contact above threshold and no fall",
                     "scope": "Local task inspired by the linked video; original author's training settings are unpublished",
                     "recorded_trials": selected, "selection": "first trial of each mode, selected before evaluation"},
    }
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "trajectories.npz", **{k: np.stack(v) for k, v in trajectory.items()},
                        iteration=np.array(ck["iteration"]), names=np.array(["flat", "duck", "jump"]),
                        difficulty=np.array(args.difficulty))
    print(json.dumps(result, indent=2), flush=True)

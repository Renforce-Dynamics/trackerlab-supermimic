"""Evaluate a terrain actor in physics on six held-out difficulty lanes."""

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
    task = get_task(ck["task"])
    spec, height, _ = robot_parameters(task.robot)
    field_data = generate(task.profile, height, args.seed)
    field = HeightField(field_data, height, device)
    solver_iterations = ck["arguments"].get("solver_iterations", 50)
    model, *_ = build_model(task.robot, field_data, solver_iterations)
    joint_ids = [model.joint(name).id for name in spec.joint_names]
    limits = torch.tensor(model.jnt_range[joint_ids], device=device, dtype=torch.float32)
    mdp = TerrainMDP(field, torch.tensor(spec.default_q, device=device), limits[:, 0], limits[:, 1],
                     gamma=ck["arguments"].get("gamma", .99))
    env = TerrainSim(task, mdp, args.num_envs, device, args.seed, solver_iterations,
                     max_episode_steps=ck["arguments"]["max_episode_steps"])
    env.curriculum = False
    levels = torch.arange(args.num_envs, device=device) % 6
    env.reset(levels=levels)
    actor = TerrainPolicy(mdp.obs_dim, mdp.goal_dim, hist=ck["arguments"]["hist"],
                          num_joints=spec.num_joints).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    recorder = None
    if args.trajectory_output:
        free_id = np.flatnonzero(model.jnt_type == 0)[0]
        recorder = FirstEpisodeRecorder(
            env.qpos[:6].cpu().numpy(), field.features(env.state())[:6].cpu().numpy(),
            int(model.jnt_qposadr[free_id]), model.jnt_qposadr[joint_ids],
        )
        first_commands = env.c[:6].cpu().numpy()
        first_context = env.xi[:6].cpu().numpy()
    history = torch.zeros(args.num_envs, actor.hist, mdp.obs_dim, device=device)
    counts = {name: torch.zeros(6, device=device) for name in
              ("episodes", "successes", "falls", "timeouts", "return_sum", "progress_sum")}
    progress = torch.zeros(args.num_envs, device=device)
    for _ in range(args.steps):
        observation = mdp.add_noise(env.observe())
        history = torch.where((env.ep_step == 0)[:, None, None],
                              observation[:, None].expand_as(history),
                              torch.cat((history[:, 1:], observation[:, None]), 1))
        action = actor(history, mdp.goal(env.state(), env.c))
        tr = env.step(action)
        if recorder is not None and recorder.active.any():
            recorder.add(
                tr["s1"][:6].cpu().numpy(),
                field.features(unflatten_state(tr["s1"][:6])).cpu().numpy(),
                tr["done"][:6].cpu().numpy(), tr["episode_success"][:6].cpu().numpy(),
                tr["episode_fall"][:6].cpu().numpy(),
            )
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
              "evaluation_objective_version": OBJECTIVE_VERSION,
              "trained_iterations": ck["iteration"], "evaluation_seed": args.seed,
              "terrain_seed_held_out": args.seed != ck["arguments"]["seed"],
              "physics_steps": args.steps, "num_envs": args.num_envs,
              "policy_exploration_noise": False, "observation_noise": True, "lanes": lanes}
    if recorder is not None:
        recorder.save(args.trajectory_output, seed=np.int64(args.seed),
                      task=np.array(task.id), iteration=np.int64(ck["iteration"]),
                      commands=first_commands, context=first_context,
                      solver_iterations=np.int64(solver_iterations))
        result["demo_trajectory"] = str(args.trajectory_output.resolve())
        result["demo_selection"] = "First episode of environment IDs 0..5, one per difficulty; no outcome filtering"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

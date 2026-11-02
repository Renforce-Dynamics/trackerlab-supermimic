"""Evaluate a terrain actor in physics on six held-out difficulty lanes."""

import argparse
import json
from pathlib import Path

import torch

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.terrain import get_task, generate, robot_parameters
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
import trackerLab.algorithms.supermimic.networks as networks
from trackerLab.tracker_env.mdp.supermimic import unflatten_state
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField, TerrainMDP
from trackerLab.tracker_env.supermimic.terrain import TerrainSim


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=43)
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
    model, *_ = build_model(task.robot, field_data)
    joint_ids = [model.joint(name).id for name in spec.joint_names]
    limits = torch.tensor(model.jnt_range[joint_ids], device=device, dtype=torch.float32)
    mdp = TerrainMDP(field, torch.tensor(spec.default_q, device=device), limits[:, 0], limits[:, 1])
    env = TerrainSim(task, mdp, args.num_envs, device, args.seed,
                     max_episode_steps=ck["arguments"]["max_episode_steps"])
    env.curriculum = False
    levels = torch.arange(args.num_envs, device=device) % 6
    env.reset(levels=levels)
    actor = TerrainPolicy(mdp.obs_dim, mdp.goal_dim, hist=ck["arguments"]["hist"],
                          num_joints=spec.num_joints).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
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
              "trained_iterations": ck["iteration"], "evaluation_seed": args.seed,
              "terrain_seed_held_out": args.seed != ck["arguments"]["seed"],
              "physics_steps": args.steps, "num_envs": args.num_envs,
              "policy_exploration_noise": False, "observation_noise": True, "lanes": lanes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

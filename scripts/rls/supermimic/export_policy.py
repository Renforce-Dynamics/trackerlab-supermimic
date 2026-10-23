"""Export a SuperMimic policy and a validated portable TorchScript module."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[3]
from bootstrap import configure_paths
configure_paths()
from trackerLab.algorithms.supermimic.networks import Policy
from trackerLab.tracker_env.mdp.supermimic import observation_dim
from trackerTask.supermimic.robots import get_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    spec = get_robot(config.get("task", "g1"))
    n = spec.num_joints
    obs_dim = observation_dim(config.get("obs_mode", "paper_obs93"), n)
    goal_dim = config["goal_frames"] * spec.goal_frame_dim
    policy = Policy(obs_dim, goal_dim, hist=config["hist"], num_joints=n,
                    action_limit=config.get("action_limit")).eval()
    policy.load_state_dict(checkpoint["pi"])
    if not all(bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
        raise FloatingPointError("Policy tensors contain nonfinite values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.output_dir / "policy.pt"
    torch.save({
        "pi": policy.state_dict(),
        "args": config,
        "iteration": checkpoint["iteration"],
        "task": spec.name,
        "wall_time": checkpoint["wall_time"],
        "sim_steps": checkpoint["sim_steps"],
        "reward_cfg": checkpoint.get("reward_cfg", {}),
    }, policy_path)
    examples = (torch.zeros(1, config["hist"], obs_dim), torch.zeros(1, goal_dim))
    with torch.no_grad():
        scripted = torch.jit.trace(policy, examples)
        scripted.save(str(args.output_dir / "policy.ts"))
        reloaded = torch.jit.load(str(args.output_dir / "policy.ts")).eval()
        torch.manual_seed(17)
        errors = []
        for batch in (1, 7):
            observations = torch.randn(batch, config["hist"], obs_dim)
            goals = torch.randn(batch, goal_dim)
            expected = policy(observations, goals)
            actual = reloaded(observations, goals)
            if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
                raise AssertionError("Serialized policy differs from the original FP32 policy")
            errors.append(float((actual - expected).abs().max()))
    metadata = {
        "task": spec.name,
        "iteration": checkpoint["iteration"],
        "observation_mode": config.get("obs_mode", "paper_obs93"),
        "inputs": {
            "observation_history": ["batch", config["hist"], obs_dim],
            "goal": ["batch", goal_dim],
        },
        "output": ["batch", n],
        "history_order": "oldest to newest",
        "observation_fields": [
            {"slice": [0, n], "value": "joint_position - default_joint_position - joint_offset"},
            {"slice": [n, 2*n], "value": "joint_velocity in rad/s"},
            {"slice": [2*n, 2*n+3], "value": "root angular velocity in root coordinates, rad/s"},
            {"slice": [2*n+3, 2*n+6], "value": "unit gravity direction in root coordinates"},
            {"slice": [2*n+6, 3*n+6], "value": "previous executed dimensionless action"},
        ],
        "normalization": "Running observation and goal normalizers are included in both files.",
        "torchscript_dtype": "float32",
        "checkpoint_evaluation_mlp_dtype": config.get("amp", "none"),
        "control_frequency_hz": 50,
        "joint_names": spec.joint_names,
        "default_joint_positions": spec.default_q,
        "action_scale": spec.action_scale,
        "policy_action_limit": config.get("action_limit"),
        "joint_target_formula": "default_joint_position + joint_offset + clip(action, -20, 20) * action_scale",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "torchscript_validation_batches": [1, 7],
        "torchscript_max_absolute_errors": errors,
        "validation_scope": "Serialization and dynamic batch inference. Motion scores come from the native bf16 checkpoint evaluation.",
    }
    with args.checkpoint.open("rb") as stream:
        metadata["source_checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if obs_dim == 3*n+12:
        metadata["observation_fields"].append({
            "slice": [3*n+6, 3*n+12], "value": "reference root orientation relative to the robot root, 6D"
        })
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# SuperMimic Policy\n\n"
        f"Checkpoint iteration: {checkpoint['iteration']}.\n\n"
        "`policy.pt` contains policy weights, normalizers and training arguments.\n"
        "`policy.ts` is a portable FP32 TorchScript module. Its arguments are\n"
        f"observation history `(B, {config['hist']}, {obs_dim})` and motion goal `(B, {goal_dim})`.\n"
        f"It returns {n} dimensionless actions in the joint order in `metadata.json`.\n\n"
        "Build observations and goals with the matching `MDP.obs` and `MotionLib.goal`\n"
        "implementations. Normalization is included in the exported model.\n"
        "Raw observations are in SI units with no extra input scaling. At reset, repeat\n"
        "the initial observation across the history; thereafter append each new observation.\n"
        f"The goal contains {config['goal_frames']} future reference frames with {spec.goal_frame_dim} features per frame:\n"
        "reference displacement (3), rotation columns (6), local linear/angular velocity\n"
        f"(3+3), joint position ({n}), and joint velocity scaled by 0.1 ({n}).\n"
        "Use the reference's current heading frame and the matching `MotionLib.goal`.\n"
        "For deterministic simulation evaluation, joint offsets are zero.\n"
        "The adjacent experiment report contains native-checkpoint evaluation results.\n\n"
        "```python\nimport torch\n"
        "policy = torch.jit.load('policy.ts', map_location='cpu').eval()\n"
        "with torch.no_grad():\n"
        "    action = policy(observation_history, motion_goal)\n```\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

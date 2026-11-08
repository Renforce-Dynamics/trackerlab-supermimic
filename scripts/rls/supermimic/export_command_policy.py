"""Export only the command actor; motion clips and discriminator stay in training."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from bootstrap import configure_paths
configure_paths()
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerLab.algorithms.supermimic.stealth_upper_body import UpperBodyConstraint
from trackerLab.algorithms.supermimic import networks
from trackerLab.tracker_env.mdp.supermimic_command import (
    COMMAND_VERSION, GOAL_FIELDS, wasd_keyboard_metadata,
)
from trackerTask.supermimic.robots import get_robot


def command_training_metadata(checkpoint):
    """Describe saved training settings without narrowing runtime commands."""
    arguments = checkpoint["arguments"]
    reward_mode = arguments.get("command_reward_mode")
    if reward_mode is None:
        reward_mode = "wasd" if arguments.get("stealth_control_mode") == "wasd" else "legacy"
    if reward_mode not in ("legacy", "wasd"):
        raise ValueError("command_reward_mode must be 'legacy' or 'wasd'")
    training_modes = arguments.get("command_training_modes")
    return {
        "command_reward_mode": reward_mode,
        "training_command_modes": None if training_modes is None else list(training_modes),
    }


class ConstrainedCommandPolicy(nn.Module):
    def __init__(self, policy, constraint, num_joints, entry_duration_s=0.):
        super().__init__()
        self.policy, self.constraint, self.num_joints = policy, constraint, num_joints
        self.command_clock_scale = max(.8, entry_duration_s)

    def forward(self, observation_history, goal):
        action = self.policy(observation_history, goal)
        n = self.num_joints
        previous = observation_history[:, -1, 2 * n + 6:3 * n + 6]
        older = observation_history[:, -2, 2 * n + 6:3 * n + 6]
        return self.constraint(
            action, previous, older, entry_time=goal[:, 6] * self.command_clock_scale + .02)


def export(checkpoint_path, output):
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("training_stage") != "command-style"
            or checkpoint.get("objective_version") != COMMAND_VERSION):
        raise ValueError("Expected a command/style checkpoint")
    training_metadata = command_training_metadata(checkpoint)
    spec = get_robot(checkpoint["robot"])
    n, hist = spec.num_joints, checkpoint["arguments"]["hist"]
    obs_dim, goal_dim = 3 * n + 196, len(GOAL_FIELDS)
    networks.AMP_DTYPE = None
    action_limit = checkpoint["arguments"].get("action_limit", 4.)
    actor = TerrainPolicy(obs_dim, goal_dim, hist=hist, num_joints=n,
                          action_limit=action_limit).eval()
    actor.load_state_dict(checkpoint["actor"])
    policy = actor
    constraint = None
    upper_body_config = checkpoint.get("stealth_upper_body_constraint")
    if upper_body_config is not None:
        constraint = UpperBodyConstraint.from_dict(upper_body_config).eval()
        policy = ConstrainedCommandPolicy(
            actor, constraint, n, checkpoint["arguments"].get("entry_duration_s", 0.)).eval()
    rate_limiter_config = checkpoint.get("upper_body_rate_limit")
    if rate_limiter_config is not None:
        if upper_body_config is not None:
            raise ValueError("Checkpoint cannot combine static arms and a separate arm rate limiter")
        from trackerLab.algorithms.supermimic.stealth_action_filter import UpperBodyRateLimiter
        constraint = UpperBodyRateLimiter.from_dict(rate_limiter_config).eval()
        policy = ConstrainedCommandPolicy(
            actor, constraint, n, checkpoint["arguments"].get("entry_duration_s", 0.)).eval()
    waist_config = checkpoint.get("stealth_waist_constraint")
    if waist_config is not None:
        from trackerLab.algorithms.supermimic.stealth_waist_entry import compose_entry_waist_constraint
        constraint = compose_entry_waist_constraint(constraint, waist_config, device="cpu").eval()
        policy = ConstrainedCommandPolicy(
            actor, constraint, n, checkpoint["arguments"].get("entry_duration_s", 0.)).eval()
    if any(not bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
        raise FloatingPointError("Actor contains nonfinite values")
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        example = (torch.zeros(1, hist, obs_dim), torch.zeros(1, goal_dim))
        scripted = torch.jit.trace(policy, example)
        scripted.save(str(output / "policy.ts"))
        loaded = torch.jit.load(str(output / "policy.ts"))
        errors = []
        for batch in (1, 7):
            observations, goal = torch.randn(batch, hist, obs_dim), torch.randn(batch, goal_dim)
            expected, actual = policy(observations, goal), loaded(observations, goal)
            if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
                raise AssertionError("Portable actor differs from eager actor")
            errors.append(float((expected - actual).abs().max()))
    actor_payload = {
        "actor": actor.state_dict(), "robot": spec.name, "hist": hist,
        "obs_dim": obs_dim, "goal_dim": goal_dim, "action_limit": action_limit,
        "objective_version": COMMAND_VERSION,
        **training_metadata,
    }
    if upper_body_config is not None:
        actor_payload["stealth_upper_body_constraint"] = upper_body_config
    if waist_config is not None:
        actor_payload["stealth_waist_constraint"] = waist_config
    if rate_limiter_config is not None:
        actor_payload["upper_body_rate_limit"] = rate_limiter_config
    if checkpoint["arguments"].get("entry_duration_s", 0.):
        actor_payload["entry_duration_s"] = checkpoint["arguments"]["entry_duration_s"]
    if checkpoint["arguments"].get("stealth_control_mode") == "wasd":
        actor_payload["stealth_control_mode"] = "wasd"
    torch.save(actor_payload, output / "actor.pt")
    metadata = {
        "task": checkpoint["task"], "robot": spec.name, "iteration": checkpoint["iteration"],
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "inputs": {"observation_history": ["batch", hist, obs_dim],
                   "command_goal": ["batch", goal_dim]},
        "goal_fields": GOAL_FIELDS,
        "observation_fields": [
            ["joint_position-default_joint_position-joint_offset", n],
            ["joint_velocity*0.1", n], ["body_angular_velocity*0.25", 3],
            ["projected_gravity", 3], ["previous_executed_action", n],
            ["body_linear_velocity/robot_height_scale", 3],
            ["terrain_height_scan", 187],
        ],
        "observation_builder": "CommandMDP.obs",
        "command_builder": "CommandMDP.goal",
        "history_order": "oldest to newest; repeat initial frame on reset",
        "height_scan_shape": [17, 11], "control_hz": 50,
        "output": ["batch", n], "joint_names": spec.joint_names,
        "default_joint_positions": spec.default_q, "action_scale": spec.action_scale,
        "action_limit": action_limit,
        "action_formula": "clip(default_q + joint_offset + clip(action,-action_limit,action_limit)*action_scale, joint_limits)",
        "runtime_reference_motion": False, "runtime_discriminator": False,
        "runtime_world_model": False, "normalization_included": True,
        "trained_command_envelope": checkpoint.get("command_envelope"),
        **training_metadata,
        "keyboard_control": (
            wasd_keyboard_metadata()
            if checkpoint["arguments"].get("stealth_control_mode") == "wasd" else None),
        "stealth_upper_body_constraint": upper_body_config,
        "stealth_waist_constraint": waist_config,
        "waist_control_scope": (
            "Explicit source-derived waist target ramp followed by persistent crouched hold; "
            "waist channels are constrained, legs remain policy controlled"
            if waist_config is not None else None),
        "upper_body_rate_limit": rate_limiter_config,
        "physical_domain_randomization":
            checkpoint["arguments"].get("physical_domain_randomization", False),
        "entry_duration_s": checkpoint["arguments"].get("entry_duration_s", 0.),
        "arm_entry_profile": (
            "Embedded fixed skill-entry arm waypoints; evaluated from the existing goal clock"
            if upper_body_config and upper_body_config.get("entry_times_s") else None),
        "entry_command": (
            "CommandMDP.effective_command: quintic velocity/yaw and height blend; "
            "clock starts at skill entry and continues across command changes"
        ),
        "validation_max_error": errors,
        "deployment_requirement": "Local height map and a body-velocity estimate; simulator truth is used in training.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()

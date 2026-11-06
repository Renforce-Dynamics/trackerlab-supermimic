"""Verify G1 physical parameter randomization and constrained actions on CUDA."""

import argparse
import json
from pathlib import Path

import torch

from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.command_domain_randomization import command_domain_scales
from trackerLab.algorithms.supermimic.stealth_upper_body import (
    UpperBodyConstraint, calibrate_upper_body_reference,
)
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp.supermimic_command import CommandMDP
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField
from trackerLab.tracker_env.supermimic.command import CommandSim
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerTask.supermimic.terrain import generate, get_task, robot_parameters


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(89)
    device = "cuda:0"
    task = get_task("locomotion/g1/stealth")
    spec, height, _ = robot_parameters("g1")
    field = HeightField(generate("stealth", height, 89), height, device)
    model, *_ = build_model("g1", field.terrain, 50, self_collisions=True)
    joints = [model.joint(name).id for name in spec.joint_names]
    limits = torch.as_tensor(model.jnt_range[joints], device=device, dtype=torch.float32)
    mdp = CommandMDP(
        field, torch.tensor(spec.default_q, device=device),
        limits[:, 0], limits[:, 1], profile="stealth",
        clearance_ratio=.92, action_limit=20.,
    )
    env = CommandSim(
        task, mdp, num_envs=6, device=device, seed=89,
        stealth_keyboard_mode=True, self_collisions=True,
        action_limit=20., physical_domain_randomization=True,
    )
    scales = command_domain_scales(env.xi, spec.num_joints)
    torch.testing.assert_close(env.body_mass, env.base_body_mass * scales[:, 0, None])
    torch.testing.assert_close(env.body_inertia, env.base_body_inertia * scales[:, 0, None, None])
    actuators = env._actuator_ids
    torch.testing.assert_close(
        env.actuator_gainprm[:, actuators, 0],
        env.base_actuator_gainprm[:, actuators, 0] * scales[:, 1, None],
    )
    torch.testing.assert_close(
        env.actuator_biasprm[:, actuators, 1],
        env.base_actuator_biasprm[:, actuators, 1] * scales[:, 1, None],
    )
    torch.testing.assert_close(
        env.actuator_biasprm[:, actuators, 2],
        env.base_actuator_biasprm[:, actuators, 2] * scales[:, 2, None],
    )
    torch.testing.assert_close(
        env.actuator_forcerange[:, actuators],
        env.base_actuator_forcerange[:, actuators] * scales[:, 3, None, None],
    )
    subtree = env.wp.to_torch(env.model.body_subtreemass)
    nominal_mass = float(model.body_mass.sum())
    torch.testing.assert_close(subtree[:, env.root_id], nominal_mass * scales[:, 0])
    context = env.xi.clone()
    env.reset()
    torch.testing.assert_close(env.xi[:, 1:5], context[:, 1:5])
    torch.testing.assert_close(env.xi[:, -3:], context[:, -3:])
    torch.testing.assert_close(env.body_mass, env.base_body_mass * scales[:, 0, None])

    kin = RobotKinematics(device, spec)
    motion = MotionLib.from_config(str(args.motion_config.resolve()), kin, device)
    reference = calibrate_upper_body_reference(motion, spec.joint_names, method="medoid")
    constraint = UpperBodyConstraint(
        spec.joint_names, spec.default_q, spec.action_scale, reference,
        action_limit=20.,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = TerrainPolicy(
        mdp.obs_dim, mdp.goal_dim, hist=checkpoint["arguments"]["hist"],
        num_joints=spec.num_joints, action_limit=20.,
    ).to(device).eval()
    actor.load_state_dict(checkpoint["actor"])
    history = env.observe()[:, None].expand(-1, actor.hist, -1).clone()
    max_step = torch.zeros((), device=device)
    falls = 0
    for _ in range(100):
        observation = mdp.add_noise(env.observe())
        history = torch.where(
            (env.ep_step == 0)[:, None, None], observation[:, None].expand_as(history),
            torch.cat((history[:, 1:], observation[:, None]), dim=1),
        )
        action = constraint(actor(history, mdp.goal(env.state(), env.c)), env.a_prev)
        delta = (action - env.a_prev) * env.action_scale
        max_step = torch.maximum(max_step, delta[:, constraint.upper_indices].abs().max())
        transition = env.step(action)
        falls += int(transition["episode_fall"].sum())
        if not bool(torch.isfinite(transition["s1"]).all()):
            raise AssertionError("Nonfinite physical state under randomized constrained actions")
    if float(max_step) > .02401:
        raise AssertionError("Upper-body command changed beyond its per-frame bound")
    result = {
        "verification": "passed",
        "num_envs": 6, "control_steps": 100,
        "model_parameters_verified": [
            "body_mass", "body_inertia", "body_subtreemass",
            "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
        ],
        "domain_scales": scales.cpu().tolist(),
        "fixed_domain_reset_parity": True,
        "max_upper_joint_target_step_rad": float(max_step),
        "falls": falls,
        "qualification": "bounded physical smoke; motion quality requires longer training and review",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

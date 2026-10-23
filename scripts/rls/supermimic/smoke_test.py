"""Validate task kinematics, PhysX transitions and replay on reference data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
from bootstrap import configure_paths
configure_paths()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("g1", "go2"), default="g1")
parser.add_argument("--motion_dir", default=str(REPO / "data" / "retargeted" / "lafan1_g1"))
parser.add_argument("--motion_config")
parser.add_argument("--output", default=str(REPO / "logs" / "supermimic_smoke" / "smoke.json"))
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--obs_mode", choices=["paper_obs93", "relative_ori99", "relative_orientation", "proprioception"])
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cuda:0")
args = parser.parse_args()
if args.task == "go2" and not args.motion_config:
    parser.error("Go2 requires --motion_config declaring its source frame rate")
launcher = AppLauncher(args)

import mujoco
import numpy as np
import torch

from trackerLab.algorithms.supermimic.buffer import ReplayBuffer
from trackerLab.tracker_env.supermimic.env import TrackingSim, launch_simulation, make_sim
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.tracker_env.mdp.supermimic import MAX_STEP_REWARD, MDP, REWARD_TERMS, RewardCfg, flatten_state
from trackerLab.metrics.supermimic import write_json
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerTask.supermimic.robots import get_robot, get_task_cfg
from trackerLab.kinematics.supermimic.rotations import mat_to_quat, quat_to_mat

torch.manual_seed(0)
torch.set_num_threads(8)
dev = args.device
spec = get_robot(args.task)
cfg = get_task_cfg(args.task)
cfg.num_envs, cfg.reset_noise = 4, args.reset_noise
kin = RobotKinematics(dev, spec)
motion = (MotionLib.from_config(args.motion_config, kin, dev) if args.motion_config
          else MotionLib.from_dir(args.motion_dir, kin, dev))
mdp = MDP(kin, motion, torch.tensor(spec.default_q, device=dev),
          cfg=RewardCfg(**cfg.reward_overrides), obs_mode=args.obs_mode or cfg.observation_mode)
report = {
    "task": args.task,
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(), "clips": len(motion.names),
    "frames_50hz": motion.num_frames, "observation_dim": mdp.obs_dim,
}


def compare_mujoco():
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    ids = [model.body(name).id for name in spec.tracked_bodies]
    jids = [model.joint(name).id for name in spec.joint_names]
    qids, vids = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
    frames = torch.linspace(0, motion.num_frames - 1, 12, device=dev).long()
    states = motion.ref_state(frames)
    predicted = [value.detach().cpu().numpy() for value in kin.forward(*states)]
    p, R, v, w, q, qd = [value.cpu().numpy() for value in states]
    quats = mat_to_quat(states[1]).cpu().numpy()
    errors = np.zeros(4)
    epsilon = 1e-6
    for i in range(len(frames)):
        data.qpos[:3], data.qpos[3:7], data.qpos[qids] = p[i], quats[i], q[i]
        data.qvel[:3], data.qvel[3:6], data.qvel[vids] = v[i], R[i].T @ w[i], qd[i]
        mujoco.mj_forward(model, data)
        pos = data.xpos[ids].copy()
        rot = data.xmat[ids].reshape(-1, 3, 3).copy()
        errors[0] = max(errors[0], np.max(np.abs(predicted[0][i] - pos)))
        errors[1] = max(errors[1], np.max(np.abs(predicted[1][i] - rot)))
        q0 = data.qpos.copy()
        future = []
        for dt in (-epsilon, epsilon):
            data.qpos[:] = q0
            mujoco.mj_integratePos(model, data.qpos, data.qvel, dt)
            mujoco.mj_forward(model, data)
            future.append((data.xpos[ids].copy(), data.xmat[ids].reshape(-1, 3, 3).copy()))
        linvel = (future[1][0] - future[0][0]) / (2 * epsilon)
        dR = (future[1][1] - future[0][1]) / (2 * epsilon)
        skew = dR @ rot.transpose(0, 2, 1)
        angvel = np.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), -1)
        errors[2] = max(errors[2], np.max(np.abs(predicted[2][i] - linvel)))
        errors[3] = max(errors[3], np.max(np.abs(predicted[3][i] - angvel)))
    assert np.all(errors < [2e-5, 2e-5, 2e-4, 2e-4]), errors
    report["mujoco_fk_max_errors_pos_rot_vel_angvel"] = errors.tolist()
    differentiable = [tensor.detach().requires_grad_(True) for tensor in states]
    output = kin.forward(*differentiable)
    sum(tensor.square().mean() for tensor in output).backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in differentiable)
    report["fk_gradients_finite"] = True


with launch_simulation(make_sim(dev, task_cfg=cfg), args, app_launcher=launcher) as sim:
    compare_mujoco()
    env = TrackingSim(sim.cfg, 4, motion, mdp, task_cfg=cfg)
    assert torch.count_nonzero(env.bin_returns) == 0
    assert abs(env.max_bin_return - MAX_STEP_REWARD / (1.0 - env.gamma)) < 1e-6
    initial_probs = torch.softmax(-25.0 * env.bin_returns / env.max_bin_return, 0)
    assert torch.allclose(initial_probs, torch.full_like(initial_probs, 1.0 / motion.num_bins))
    assert env.max_episode_steps == 500
    ids = torch.arange(4, device=dev)
    # Equal sustained reward has equal reset value at short and long time limits.
    saved_bins = env.reset_bin.clone()
    env.reset_bin[:3] = torch.arange(3, device=dev)
    env.ep_step[:3] = torch.tensor([1, 100, 1], device=dev)
    reward = MAX_STEP_REWARD * 0.5
    lengths = env.ep_step[:3].float()
    env.ep_return[:3] = reward * lengths
    env.ep_discounted_return[:3] = reward * (1 - env.gamma**lengths) / (1 - env.gamma)
    env._update_bin_returns(ids[:3], torch.tensor([False, False, True, False], device=dev))
    expected = torch.tensor([reward / (1 - env.gamma), reward / (1 - env.gamma), reward], device=dev) * 0.2
    assert torch.allclose(env.bin_returns[:3], expected, atol=1e-5), (env.bin_returns[:3], expected)
    env.bin_returns.zero_()
    env.reset_bin.copy_(saved_bins)
    report["reset_sampler_uniform_initialization"] = True
    report["reset_sampler_truncation_bootstrap"] = True
    frames = motion.clip_start[torch.arange(4, device=dev) % len(motion.names)] + 20
    env.reset(ids, frames)
    if args.reset_noise:
        actual, reference = env.state(), motion.ref_state(frames)
        position_delta = (actual[0] - reference[0]).abs()
        assert bool((position_delta <= torch.tensor(cfg.reset_position, device=dev) + 1e-4).all())
        assert float(position_delta.max()) > 1e-4
        assert bool((actual[4] <= kin.q_hi + 1e-4).all())
        assert bool((actual[4] >= kin.q_lo - 1e-4).all())
        report["reset_noise_position_max"] = position_delta.max(0).values.cpu().tolist()
        context = env.xi.clone()
        env.eval_mode = True
        env.reset_noise = False
        env.reset(ids, frames)
        torch.testing.assert_close(env.state()[0], reference[0], atol=1e-4, rtol=1e-5)
        env.eval_mode = False
        env.xi.copy_(context)
        env.reset_noise = True
        env.reset(ids, frames)
        report["reset_noise_training_only"] = True
    predicted = kin.forward(*env.state())
    body_ids = [env.robot.body_names.index(name) for name in spec.tracked_bodies]
    body_pos = env.robot.data.body_link_pos_w[:, body_ids] - env.env_origins[:, None]
    body_rot = quat_to_mat(env.robot.data.body_link_quat_w[:, body_ids])
    pos_error = (predicted[0] - body_pos).abs().max().item()
    rot_error = (predicted[1] - body_rot).abs().max().item()
    report["physx_fk_position_error_m"] = pos_error
    report["physx_fk_rotation_matrix_error"] = rot_error
    assert pos_error < 2e-3 and rot_error < 2e-3, (pos_error, rot_error)
    before = flatten_state(env.state()).clone()
    env.reset(torch.tensor([1], device=dev), frames[1:2] + 3)
    after = flatten_state(env.state())
    assert torch.allclose(before[[0, 2, 3]], after[[0, 2, 3]], atol=1e-5)
    env._set_domain_randomization(torch.tensor([1], device=dev))
    materials = env.robot.root_physx_view.get_material_properties()
    coms = env.robot.root_physx_view.get_coms()
    report["physx_material_shape"] = list(materials.shape)
    report["physx_com_shape"] = list(coms.shape)
    expected_mu = env.xi[:, 0].cpu()[:, None].expand_as(materials[..., 0])
    assert torch.allclose(materials[..., 0].cpu(), expected_mu, atol=1e-5)
    expected_com = env._default_coms[:, env._torso_body_id, :3].cpu() + env.xi[:, -3:].cpu()
    assert torch.allclose(coms[:, env._torso_body_id, :3].cpu(), expected_com, atol=1e-5)

    replay = ReplayBuffer(32, 4, dev, num_joints=spec.num_joints)
    terminations = truncations = 0
    for step in range(args.steps):
        if step == 0:
            env.reset(ids[:1], motion.clip_end[:1] - 1)
        if step == 4:
            root = env.robot.data.root_link_pose_w[1:2].clone()
            root[:, 2] += 2
            env.robot.write_root_link_pose_to_sim(root, ids[1:2])
            env.sim.forward()
            env.scene.update(env.physics_dt)
        if step == 8:
            env.reset(ids[2:3], frames[2:3])
            env.ep_step[2] = env.max_episode_steps - 1
        action = (motion.q[env.f] - env.default_q - env.xi[:, 1:-3]) / env.action_scale
        result = env.step(action)
        assert torch.isfinite(result["s1"]).all() and torch.isfinite(result["r"]).all()
        torch.testing.assert_close(
            sum(result["reward_terms"][key] for key in REWARD_TERMS), result["r"], atol=1e-6, rtol=1e-5,
        )
        if step == 0:
            assert result["done"][0]
            assert result["f"][0] == motion.clip_end[0] - 1
            assert not torch.allclose(result["s1"][0], flatten_state(env.state())[0])
        if step == 8:
            assert result["truncated"][2] and not result["terminated"][2]
            assert env.ep_step[2] == 0
            report["training_timeout_truncation"] = True
        replay.add(result)
        terminations += int(result["terminated"].sum())
        truncations += int(result["truncated"].sum())
    assert terminations > 0 and truncations > 0, (terminations, truncations)
    sampled = replay.sample_windows(16, 8)
    expected_mask = torch.cumprod(
        torch.cat((torch.ones(16, 1, device=dev), (~sampled["done"][:, :-1]).float()), 1), 1,
    )
    assert torch.equal(sampled["mask"], expected_mask)
    start, history = replay.sample_starts(16, 10)
    assert torch.equal(start["f"], history["f"][:, -1])
    assert torch.all(history["f"][:, 1:] >= history["f"][:, :-1])
    assert torch.all(history["f"][:, 1:] - history["f"][:, :-1] <= 1)
    report.update(
        vector_steps=args.steps, env_steps=args.steps * 4,
        terminations=terminations, truncations=truncations, replay_validated=True,
        partial_reset_validated=True, domain_randomization_validated=True,
        terminal_transition_preserved=True, status="passed",
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)

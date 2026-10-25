"""Robot dimensions and data contracts must stay isolated across tasks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerTask.supermimic.robots import get_robot
from trackerLab.algorithms.supermimic.buffer import ReplayBuffer
from trackerLab.algorithms.supermimic.networks import Policy, WorldModel
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp.supermimic import (
    MDP, flatten_state, unflatten_state, invariant_features, normalize_xi, observation_dim,
)
from prepare_kine2go import convert_array


class SimpleKinematics:
    def __init__(self, robot):
        self.num_joints, self.joint_names = robot.num_joints, robot.joint_names
        self.q_lo, self.q_hi = torch.full((self.num_joints,), -3.), torch.full((self.num_joints,), 3.)

    def forward(self, p, R, v, w, q, qd):
        return p[..., None, :], R[..., None, :, :], v[..., None, :], w[..., None, :]


class RobotTaskTests(unittest.TestCase):
    def test_kine2go_raw_joint_groups_and_quaternion_conversion(self):
        raw = np.zeros((3, 61), dtype=np.float32)
        raw[:, 6:18] = np.arange(12)
        raw[:, 48:51] = [.1, .2, .3]
        raw[:, 51:55] = [0., 1., 0., 0.]
        values = convert_array(raw, get_robot("go2").joint_names)
        np.testing.assert_array_equal(values[0, 7:], [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
        np.testing.assert_array_equal(values[0, 3:7], [1., 0., 0., 0.])
        np.testing.assert_array_equal(values[:, :3], raw[:, 48:51])
        raw[0, 51:55] = 0.
        with self.assertRaisesRegex(ValueError, "unit norm"):
            convert_array(raw, get_robot("go2").joint_names)

    def test_bounded_policy_keeps_gradients_and_exported_behavior(self):
        pi = Policy(48, 78, hist=2, num_joints=12, action_limit=4.)
        with torch.no_grad():
            pi.dec[-1].bias.fill_(5.)
        obs, goal = torch.randn(2, 2, 48), torch.randn(2, 78)
        output = pi(obs, goal)
        self.assertTrue(bool((output.abs() < 4).all()))
        output.sum().backward()
        self.assertTrue(bool((pi.dec[-1].bias.grad > 0).all()))
        with torch.no_grad():
            traced = torch.jit.trace(pi.eval(), (obs, goal))
            torch.testing.assert_close(traced(obs, goal), pi(obs, goal))

    def test_both_robots_share_algorithm_with_independent_shapes_and_gradients(self):
        torch.set_num_threads(2)
        for name in ("g1", "go2", "a3", "t1"):
            with self.subTest(task=name):
                spec = get_robot(name)
                n = spec.num_joints
                state = (torch.zeros(2, 3), torch.eye(3).expand(2, 3, 3),
                         torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, n), torch.zeros(2, n))
                flat = flatten_state(state)
                self.assertEqual(flat.shape[-1], spec.state_dim)
                for original, restored in zip(state, unflatten_state(flat)):
                    torch.testing.assert_close(original, restored)
                self.assertEqual(invariant_features(state).shape[-1], spec.state_dim)
                obs_dim = observation_dim("relative_orientation", n)
                wm = WorldModel(hidden=(32, 32), num_joints=n)
                pi = Policy(obs_dim, spec.goal_frame_dim*2, hist=2, num_joints=n)
                a = pi(torch.randn(2, 2, obs_dim), torch.randn(2, spec.goal_frame_dim*2))
                self.assertEqual(a.shape, (2, n))
                xi = normalize_xi(torch.zeros(2, spec.context_dim))
                next_state = wm.step(state, a, xi)
                self.assertEqual(flatten_state(next_state).shape, flat.shape)
                wm.state_error(next_state, state).square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in wm.parameters()))
                buffer = ReplayBuffer(8, 2, "cpu", num_joints=n)
                self.assertEqual(buffer.data["a"].shape[-1], n)
                self.assertEqual(buffer.data["s"].shape[-1], spec.state_dim)
                self.assertEqual(buffer.data["xi"].shape[-1], spec.context_dim)

    def test_go2_csv_contract_and_g1_rejection(self):
        spec = get_robot("go2")
        kin = SimpleKinematics(spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.zeros((101, 19))
            values[:, 2], values[:, 6] = .3, 1
            values[:, 7:] = spec.default_q
            np.savetxt(root / "trot.csv", values, delimiter=",")
            config = {"version": 1, "root": ".", "source_fps": 50,
                      "quaternion_order": "xyzw", "joint_names": list(spec.joint_names)}
            path = root / "motion.json"
            path.write_text(json.dumps(config))
            motion = MotionLib.from_config(path, kin, "cpu")
            self.assertEqual(motion.q.shape, (101, 12))
            f = torch.tensor([0, 99])
            self.assertEqual(motion.goal(f, 10).shape, (2, 390))
            mdp = MDP(kin, motion, torch.tensor(spec.default_q), obs_mode="relative_orientation")
            self.assertEqual(mdp.obs(motion.ref_state(f), torch.zeros(2, 12), torch.zeros(2, 16), f).shape, (2, 48))
            with self.assertRaisesRegex(ValueError, "joint order"):
                MotionLib.from_config(path, SimpleKinematics(get_robot("g1")), "cpu")
            config["joint_names"] = list(reversed(spec.joint_names))
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "joint order"):
                MotionLib.from_config(path, kin, "cpu")

    def test_g1_defaults_and_aliases_are_preserved(self):
        spec = get_robot()
        self.assertEqual((spec.state_dim, spec.context_dim, spec.goal_frame_dim), (71, 33, 73))
        self.assertEqual(observation_dim("paper_obs93"), 93)
        self.assertEqual(observation_dim("relative_ori99"), 99)
        with self.assertRaises(ValueError):
            observation_dim("relative_ori99", 12)
        with self.assertRaises(ValueError):
            get_robot("typo")

"""Reference-frame boundaries, partial reset state and data configuration contracts."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

import numpy as np
import torch

from trackerLab.managers.supermimic import SuperMimicMotionManager
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerTask.supermimic.robot import G1_JOINT_NAMES


class Kinematics:
    def forward(self, p, R, v, w, q, qd):
        return p[:, None], R[:, None], v[:, None], w[:, None]


class MotionPipelineTests(unittest.TestCase):
    def motion(self, directory):
        root = Path(directory)
        for name, frames in (("a.csv", 67), ("b.csv", 97)):
            raw = np.zeros((frames, 36))
            raw[:, 0] = np.arange(frames) / 30
            raw[:, 2], raw[:, 6] = 0.8, 1
            np.savetxt(root / name, raw, delimiter=",")
        return MotionLib.from_dir(str(root), Kinematics(), "cpu")

    def test_partial_reset_and_clip_end_are_owned_by_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            motion = self.motion(directory)
            manager = SuperMimicMotionManager(motion, 3, "cpu", 0.99, 0.1)
            manager.reset(torch.arange(3), torch.tensor([0, 10, int(motion.clip_start[1])]))
            previous = manager.frames.clone()
            manager.reset(torch.tensor([1]), motion.clip_end[:1] - 1)
            self.assertTrue(torch.equal(manager.frames[[0, 2]], previous[[0, 2]]))
            manager.compute()
            self.assertEqual(manager.frames[1], motion.clip_end[0])
            manager.compute()
            self.assertEqual(manager.frames[1], motion.clip_end[0])
            self.assertEqual(manager.motion_ids.tolist(), [0, 0, 1])
            self.assertTrue(torch.equal(manager.goal(10), motion.goal(manager.frames, 10)))
            expected_times = (manager.frames - motion.clip_start[manager.motion_ids]) * 0.02
            torch.testing.assert_close(manager._motion_buffer._motion_times, expected_times)

    def test_reset_sampler_tail_value_and_physical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            motion = self.motion(directory)
            manager = SuperMimicMotionManager(motion, 3, "cpu", 0.99, 0.1)
            manager.reset_bin[:] = torch.arange(3)
            steps = torch.tensor([1, 100, 1])
            reward = 0.05
            discounted = reward * (1 - 0.99 ** steps.float()) / 0.01
            manager.update_returns(torch.arange(3), torch.tensor([False, False, True]),
                                   steps, reward * steps, discounted)
            torch.testing.assert_close(manager.bin_returns[:3], torch.tensor([1., 1., .01]),
                                       atol=2e-6, rtol=2e-6)
            _, frames = manager.sample_frames(torch.arange(1024))
            self.assertTrue(bool((frames < motion.clip_end[motion.clip_id[frames]]).all()))

    def test_config_load_matches_csv_and_rejects_joint_order_change(self):
        with tempfile.TemporaryDirectory() as directory:
            motion = self.motion(directory)
            config = {"version": 1, "root": ".", "source_fps": 30,
                      "quaternion_order": "xyzw", "joint_names": G1_JOINT_NAMES}
            path = Path(directory) / "motion.json"
            path.write_text(json.dumps(config))
            loaded = MotionLib.from_config(str(path), Kinematics(), "cpu")
            for field in ("root_p", "root_R", "root_v", "root_w", "q", "qd", "body_p", "clip_len"):
                self.assertTrue(torch.equal(getattr(motion, field), getattr(loaded, field)), field)
            config["joint_names"] = list(reversed(G1_JOINT_NAMES))
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "joint order"):
                MotionLib.from_config(str(path), Kinematics(), "cpu")

    def test_malformed_motion_configs_report_validation_errors(self):
        base = {"version": 1, "root": ".", "source_fps": 30,
                "quaternion_order": "xyzw", "joint_names": G1_JOINT_NAMES}
        cases = [[], {}, {**base, "root": None}, {**base, "source_fps": True},
                 {**base, "source_fps": 0}, {**base, "source_fps": float("nan")},
                 {**base, "files": [["bad.csv"]]}, {**base, "files": ["../bad.csv"]},
                 {**base, "files": ["..\\bad.csv"]}, {**base, "files": ["a.csv", "a.csv"]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            for config in cases:
                with self.subTest(config=config):
                    path.write_text(json.dumps(config))
                    with self.assertRaises(ValueError):
                        MotionLib.from_config(str(path), Kinematics(), "cpu")

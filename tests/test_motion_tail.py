"""Recorded endpoints and synthetic goal tails have distinct velocity semantics."""

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

from trackerLab.kinematics.supermimic.rotations import mat_to_6d, quat_to_mat, yaw_mat
from trackerLab.motion_buffer.motion_lib.supermimic import CONTROL_DT, MotionLib


class RootOnlyKinematics:
    def __init__(self, num_joints):
        self.num_joints = num_joints

    def forward(self, p, R, v, w, q, qd):
        return p[:, None], R[:, None], v[:, None], w[:, None]


class MotionTailTests(unittest.TestCase):
    def make_motion(self, directory, num_joints=29, lengths=(30, 48), fps=50):
        files, sources = [], []
        for clip, frames in enumerate(lengths):
            t = np.arange(frames, dtype=np.float64) / fps
            raw = np.zeros((frames, 7 + num_joints), dtype=np.float64)
            # Distinct clips expose accidental reads from the following clip.
            raw[:, :3] = (
                np.array([10.0 * clip, -3.0 * clip, 0.8])
                + t[:, None] * np.array([0.7, -0.3, 0.2]) * (clip + 1)
            )
            angle = 0.25 * clip + (0.5 + 0.2 * clip) * t
            raw[:, 5] = np.sin(angle / 2)
            raw[:, 6] = np.cos(angle / 2)
            raw[:, 7:] = (
                clip + np.arange(num_joints)[None, :] * 0.01
                + t[:, None] * np.linspace(0.1, 0.6, num_joints)[None, :]
            )
            path = Path(directory) / f"clip_{clip}.csv"
            np.savetxt(path, raw, delimiter=",")
            files.append(str(path))
            sources.append(torch.tensor(raw, dtype=torch.float32))
        motion = MotionLib(files, RootOnlyKinematics(num_joints), "cpu", src_fps=fps)
        return motion, sources, files

    @staticmethod
    def recorded_goal_features(motion, frames, horizon):
        """Original reference features, before synthetic-tail velocity masking."""
        future = motion.future_frames(frames, horizon)
        heading_t = yaw_mat(motion.root_R[frames]).transpose(-1, -2).unsqueeze(1)
        position = (
            heading_t
            @ (motion.root_p[future] - motion.root_p[frames].unsqueeze(1)).unsqueeze(-1)
        ).squeeze(-1)
        rotation = mat_to_6d(heading_t @ motion.root_R[future])
        return torch.cat(
            (
                position, rotation, motion.root_v_local[future], motion.root_w_local[future],
                motion.q[future], 0.1 * motion.qd[future],
            ),
            dim=-1,
        )

    def test_actual_endpoint_keeps_velocity_and_only_later_goals_hold_still(self):
        for joints in (12, 29):
            with self.subTest(joints=joints), tempfile.TemporaryDirectory() as directory:
                motion, _, _ = self.make_motion(directory, num_joints=joints)
                frames = motion.clip_end - 1
                actual = motion.goal(frames, 4).reshape(2, 4, 15 + 2 * joints)
                recorded = self.recorded_goal_features(motion, frames, 4)

                # The first goal is the recorded final frame, with real motion.
                torch.testing.assert_close(actual[:, 0], recorded[:, 0], rtol=0, atol=0)
                for values in (actual[:, 0, 9:12], actual[:, 0, 12:15], actual[:, 0, 15 + joints:]):
                    self.assertTrue(bool((values.norm(dim=-1) > 0).all()))

                # Held frames retain the endpoint pose, including root orientation.
                for feature_slice in (slice(0, 9), slice(15, 15 + joints)):
                    torch.testing.assert_close(
                        actual[:, 1:, feature_slice], recorded[:, 1:, feature_slice],
                        rtol=0, atol=0,
                    )
                self.assertEqual(torch.count_nonzero(actual[:, 1:, 9:15]).item(), 0)
                self.assertEqual(torch.count_nonzero(actual[:, 1:, 15 + joints:]).item(), 0)

    def test_goal_at_endpoint_has_zero_future_velocity_and_preserves_ref_state(self):
        with tempfile.TemporaryDirectory() as directory:
            motion, _, _ = self.make_motion(directory)
            frames = motion.clip_end.clone()
            before = tuple(value.clone() for value in motion.ref_state(frames))
            actual = motion.goal(frames, 5).reshape(2, 5, 73)
            torch.testing.assert_close(actual[:, :, :3], torch.zeros(2, 5, 3), rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(actual[:, :, 9:15]).item(), 0)
            self.assertEqual(torch.count_nonzero(actual[:, :, 44:]).item(), 0)
            for expected, observed in zip(before, motion.ref_state(frames)):
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            for values in (before[2], before[3], before[5]):
                self.assertTrue(bool((values.norm(dim=-1) > 0).all()))

    def test_mixed_clip_batch_masks_each_clip_independently(self):
        for joints in (12, 29):
            with self.subTest(joints=joints), tempfile.TemporaryDirectory() as directory:
                motion, _, _ = self.make_motion(directory, num_joints=joints, lengths=(30, 58))
                frames = torch.tensor([
                    3, int(motion.clip_end[0]) - 1, int(motion.clip_end[0]),
                    int(motion.clip_start[1]) + 5, int(motion.clip_end[1]) - 3,
                    int(motion.clip_end[1]),
                ])
                horizon = 6
                actual = motion.goal(frames, horizon).reshape(len(frames), horizon, 15 + 2 * joints)
                expected = self.recorded_goal_features(motion, frames, horizon)
                ends = motion.clip_end[motion.clip_id[frames]]
                beyond = frames[:, None] + torch.arange(1, horizon + 1) > ends[:, None]
                expected[:, :, 9:15][beyond] = 0
                expected[:, :, 15 + joints:][beyond] = 0
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                future = motion.future_frames(frames, horizon)
                self.assertTrue(bool((future <= ends[:, None]).all()))
                torch.testing.assert_close(
                    motion.clip_id[future], motion.clip_id[frames, None].expand_as(future),
                    rtol=0, atol=0,
                )

    def test_interior_goals_are_unchanged_for_multiple_horizons(self):
        for joints in (12, 29):
            with self.subTest(joints=joints), tempfile.TemporaryDirectory() as directory:
                motion, _, _ = self.make_motion(directory, num_joints=joints)
                frames = motion.clip_start + 2
                for horizon in (1, 8, 10):
                    with self.subTest(horizon=horizon):
                        expected = self.recorded_goal_features(motion, frames, horizon)
                        actual = motion.goal(frames, horizon).reshape(expected.shape)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tail_queries_do_not_mutate_source_tensors_or_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            motion, _, files = self.make_motion(directory)
            tensor_before = {
                name: value.clone() for name, value in vars(motion).items()
                if isinstance(value, torch.Tensor)
            }
            file_before = {path: Path(path).read_bytes() for path in files}
            interior_before = motion.goal(motion.clip_start, 10).clone()
            for horizon in (1, 8, 32):
                motion.goal(motion.clip_end, horizon)
                motion.goal(motion.clip_end - 1, horizon)
            for name, expected in tensor_before.items():
                torch.testing.assert_close(getattr(motion, name), expected, rtol=0, atol=0, msg=name)
            for path, expected in file_before.items():
                self.assertEqual(Path(path).read_bytes(), expected)
            torch.testing.assert_close(
                motion.goal(motion.clip_start, 10), interior_before, rtol=0, atol=0,
            )

    def test_50hz_preserves_every_source_frame_and_endpoint_velocity(self):
        lengths = (2, 3, 30, 48, 58, 101)
        for joints in (12, 29):
            with self.subTest(joints=joints), tempfile.TemporaryDirectory() as directory:
                motion, sources, _ = self.make_motion(
                    directory, num_joints=joints, lengths=lengths, fps=50,
                )
                self.assertEqual(motion.clip_len.tolist(), list(lengths))
                self.assertEqual(motion.num_frames, sum(lengths))
                for clip, raw in enumerate(sources):
                    with self.subTest(clip=clip, frames=len(raw)):
                        start, end = int(motion.clip_start[clip]), int(motion.clip_end[clip])
                        self.assertEqual(end - start + 1, len(raw))
                        torch.testing.assert_close(motion.root_p[start:end + 1], raw[:, :3], rtol=0, atol=0)
                        torch.testing.assert_close(motion.q[start:end + 1], raw[:, 7:], rtol=0, atol=0)
                        quat_wxyz = torch.cat((raw[:, 6:7], raw[:, 3:6]), dim=-1)
                        torch.testing.assert_close(
                            motion.root_R[start:end + 1], quat_to_mat(quat_wxyz), rtol=0, atol=0,
                        )
                        torch.testing.assert_close(
                            motion.root_v[end], (raw[-1, :3] - raw[-2, :3]) / CONTROL_DT,
                            rtol=0, atol=0,
                        )
                        torch.testing.assert_close(
                            motion.qd[end], (raw[-1, 7:] - raw[-2, 7:]) / CONTROL_DT,
                            rtol=0, atol=0,
                        )

    def test_non_50hz_sources_still_resample_to_control_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            motion, sources, _ = self.make_motion(
                directory, num_joints=12, lengths=(31, 61), fps=30,
            )
            self.assertEqual(motion.clip_len.tolist(), [51, 101])
            for clip, raw in enumerate(sources):
                start, end = int(motion.clip_start[clip]), int(motion.clip_end[clip])
                torch.testing.assert_close(motion.root_p[start], raw[0, :3], rtol=0, atol=0)
                torch.testing.assert_close(motion.root_p[end], raw[-1, :3], rtol=0, atol=0)
                torch.testing.assert_close(motion.q[end], raw[-1, 7:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

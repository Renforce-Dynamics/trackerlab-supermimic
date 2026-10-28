"""CPU checks of root-cache integrity and the canonical goal-tail behavior."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "scripts/rls/supermimic/warp_runtime"
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
sys.path.insert(0, str(RUNTIME))
from bootstrap import configure_paths
configure_paths()

from compact_gpu_motion import CompactGpuMotionLib, ROOT_LAYOUT
from resident_reference_bodies import install_resident_reference_bodies
from static_reference_runtime import (
    BODY_FIELDS, ROOT_FIELDS, assert_static_reference_identity,
    finalize_static_reference_fields, install_static_root_fields,
)
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.kinematics.supermimic.rotations import mat_to_6d, yaw_mat


class StubKinematics:
    num_joints = 29
    num_bodies = 1
    joint_names = tuple(f"joint_{index}" for index in range(29))
    tracked_idx = (0,)

    def __init__(self):
        self.forward_calls = 0

    def forward(self, p, rotation, v, w, q, qd):
        self.forward_calls += 1
        return p[:, None], rotation[:, None], v[:, None], w[:, None]


class WarpRootCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.kinematics = StubKinematics()
        matrix = np.zeros((8, 83), dtype=np.float32)
        fields, offset = [], 0
        for name, shape in ROOT_LAYOUT:
            width = int(np.prod(shape)) if shape else 1
            fields.append({"name": name, "offset": offset, "shape": list(shape)})
            if name == "root_R":
                matrix[:, offset:offset + width] = np.eye(3).reshape(1, 9)
            if name in ("root_p", "q"):
                matrix[:, offset:offset + width] = np.arange(8)[:, None] + 0.01 * np.arange(width)[None, :]
            if name in ("qd", "root_v", "root_w", "root_v_local", "root_w_local", "root_gz"):
                matrix[:, offset:offset + width] = 1
            offset += width
        self.binary = self.root / "root.f32"
        matrix.tofile(self.binary)
        self.config = self.root / "motion.json"
        self.config_value = {
            "version": 1, "root": ".", "files": ["a.csv", "b.csv"], "source_fps": 50,
            "joint_names": list(self.kinematics.joint_names), "quaternion_order": "xyzw",
        }
        self.config.write_text(json.dumps(self.config_value), encoding="utf-8")
        self.manifest = {
            "schema": "supermimic-root-cache-v1",
            "names": ["a", "b"], "clip_lengths": [4, 4], "num_frames": 8,
            "source_config_sha256": hashlib.sha256(self.config.read_bytes()).hexdigest(),
            "groups": {"root": {
                "file": "root.f32", "width": 83, "fields": fields,
                "bytes": self.binary.stat().st_size,
                "sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            }},
        }
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def load(self):
        return CompactGpuMotionLib(
            self.root, self.kinematics, "cpu", self.config, allow_cpu_for_tests=True,
        )

    @staticmethod
    def recorded_features(motion, frames, horizon):
        future = motion.future_frames(frames, horizon)
        heading_t = yaw_mat(motion.root_R[frames]).transpose(-1, -2).unsqueeze(1)
        position = (
            heading_t
            @ (motion.root_p[future] - motion.root_p[frames].unsqueeze(1)).unsqueeze(-1)
        ).squeeze(-1)
        return torch.cat((
            position, mat_to_6d(heading_t @ motion.root_R[future]),
            motion.root_v_local[future], motion.root_w_local[future],
            motion.q[future], motion.qd[future] * 0.1,
        ), dim=-1)

    def test_clip_mapping_and_computed_body_fields(self):
        motion = self.load()
        frames = torch.tensor([0, 3, 4, 7])
        self.assertEqual(motion.clip_id[frames].tolist(), [0, 0, 1, 1])
        self.assertEqual(motion.names, ["a", "b"])
        self.assertEqual(motion.num_frames, 8)
        self.assertEqual(motion.num_joints, 29)
        for body, root in (("body_p", "root_p"), ("body_R", "root_R"),
                           ("body_v", "root_v"), ("body_w", "root_w")):
            torch.testing.assert_close(getattr(motion, body)[frames], getattr(motion, root)[frames, None])
        self.assertEqual(motion.storage_info["runtime_reference_host_to_device_bytes"], 0)
        self.assertFalse(motion.storage_info["reference_host_gather_required"])

    def test_goal_tail_preserves_recorded_endpoint_and_pose(self):
        motion = self.load()
        frames = torch.tensor([2, 6])
        actual = motion.goal(frames, 3).reshape(2, 3, 73)
        recorded = self.recorded_features(motion, frames, 3)
        velocity_columns = list(range(9, 15)) + list(range(44, 73))
        torch.testing.assert_close(actual[:, 0], recorded[:, 0], rtol=0, atol=0)
        self.assertTrue(bool((actual[:, 0, velocity_columns] != 0).all()))
        self.assertTrue(bool((actual[:, 1:, velocity_columns] == 0).all()))
        torch.testing.assert_close(actual[:, :, :9], recorded[:, :, :9], rtol=0, atol=0)
        torch.testing.assert_close(actual[:, :, 15:44], recorded[:, :, 15:44], rtol=0, atol=0)
        torch.testing.assert_close(motion.qd[[3, 7]], torch.ones(2, 29), rtol=0, atol=0)
        torch.testing.assert_close(motion.root_v_local[[3, 7]], torch.ones(2, 3), rtol=0, atol=0)

    def test_interior_goal_matches_recorded_features(self):
        motion = self.load()
        frames = torch.tensor([0, 4])
        actual = motion.goal(frames, 2).reshape(2, 2, 73)
        torch.testing.assert_close(actual, self.recorded_features(motion, frames, 2), rtol=0, atol=0)

    def test_endpoint_goal_has_only_synthetic_velocity_and_keeps_ref_state(self):
        motion = self.load()
        frames = torch.tensor([3, 7])
        before = tuple(value.clone() for value in motion.ref_state(frames))
        resident_before = motion._resident_root.clone()
        source_before = self.binary.read_bytes()
        result = motion.goal(frames, 2).reshape(2, 2, 73)
        self.assertTrue(bool((result[:, :, 9:15] == 0).all()))
        self.assertTrue(bool((result[:, :, 44:73] == 0).all()))
        torch.testing.assert_close(result[:, :, :3], torch.zeros(2, 2, 3), rtol=0, atol=0)
        for expected, actual in zip(before, motion.ref_state(frames)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(motion._resident_root, resident_before, rtol=0, atol=0)
        self.assertEqual(self.binary.read_bytes(), source_before)

    def test_mixed_batch_respects_each_clip_endpoint(self):
        motion = self.load()
        frames = torch.tensor([0, 2, 3, 4, 6, 7])
        actual = motion.goal(frames, 4).reshape(6, 4, 73)
        expected = self.recorded_features(motion, frames, 4)
        ends = motion.clip_end[motion.clip_id[frames]]
        tail = frames[:, None] + torch.arange(1, 5) > ends[:, None]
        expected[:, :, 9:15][tail] = 0
        expected[:, :, 44:73][tail] = 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        future = motion.future_frames(frames, 4)
        torch.testing.assert_close(
            motion.clip_id[future], motion.clip_id[frames][:, None].expand_as(future),
            rtol=0, atol=0,
        )

    def test_runtime_reuses_canonical_goal_without_duplicate_monkeypatch(self):
        motion = self.load()
        frames = torch.tensor([2, 3, 6, 7])
        torch.testing.assert_close(
            motion.goal(frames, 3), MotionLib.goal(motion, frames, 3), rtol=0, atol=0,
        )
        self.assertEqual(MotionLib.goal.__module__, "trackerLab.motion_buffer.motion_lib.supermimic")
        for path in RUNTIME.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = [alias.name for alias in node.names]
                    if isinstance(node, ast.ImportFrom):
                        modules.append(node.module or "")
                    self.assertNotIn("goal_tail_runtime", modules, path.name)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self.assertFalse(
                            isinstance(target, ast.Attribute) and target.attr == "goal"
                            and isinstance(target.value, ast.Name) and target.value.id == "MotionLib",
                            path.name,
                        )

    def test_cpu_mode_requires_explicit_test_permission(self):
        with self.assertRaisesRegex(ValueError, "require CUDA"):
            CompactGpuMotionLib(self.root, self.kinematics, "cpu", self.config)

    def test_corrupt_binary_rejected(self):
        with self.binary.open("r+b") as stream:
            stream.write(b"xxxx")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load()

    def test_truncated_binary_rejected(self):
        self.binary.write_bytes(self.binary.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, "size"):
            self.load()

    def test_wrong_field_layout_rejected(self):
        for field, value in (("offset", 5), ("shape", [9]), ("name", "unexpected")):
            with self.subTest(field=field):
                original = self.manifest["groups"]["root"]["fields"][1][field]
                self.manifest["groups"]["root"]["fields"][1][field] = value
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "layout"):
                    self.load()
                self.manifest["groups"]["root"]["fields"][1][field] = original

    def test_wrong_row_width_rejected(self):
        self.manifest["groups"]["root"]["width"] = 82
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "nine float32 root fields"):
            self.load()

    def test_manifest_frame_count_and_short_clips_rejected(self):
        self.manifest["num_frames"] = 7
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Frame count"):
            self.load()
        self.manifest["num_frames"] = 8
        self.manifest["clip_lengths"] = [1, 7]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "at least two frames"):
            self.load()

    def test_reordered_clips_rejected(self):
        self.manifest["names"].reverse()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "order"):
            self.load()

    def test_changed_config_rejected(self):
        self.config.write_text(json.dumps(self.config_value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load()

    def test_binary_path_cannot_escape_cache(self):
        self.manifest["groups"]["root"]["file"] = "../outside.f32"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "inside"):
            self.load()

    def install_resident(self, motion, *, max_reference_bytes=4096):
        root_report = install_static_root_fields(motion, chunk_rows=3, allow_cpu=True)
        body_report = install_resident_reference_bodies(
            motion, self.kinematics, chunk_rows=3, reserve_bytes=0,
            max_reference_bytes=max_reference_bytes, allow_cpu_for_tests=True,
        )
        static_report = finalize_static_reference_fields(motion)
        return root_report, body_report, static_report

    def test_static_residency_preserves_all_bits_and_removes_query_time_fk(self):
        motion = self.load()
        roots_before = {name: getattr(motion, name).clone() for name in ROOT_FIELDS}
        source_before = self.binary.read_bytes()
        frames = torch.tensor([0, 2, 3, 4, 6, 7])
        goals_before = motion.goal(frames, 4).clone()
        original_bodies = self.kinematics.forward(*(
            roots_before[name] for name in ("root_p", "root_R", "root_v", "root_w", "q", "qd")
        ))
        bodies_before = {
            name: value.clone() for name, value in zip(BODY_FIELDS, original_bodies)
        }
        calls_before = self.kinematics.forward_calls
        rng_before = torch.get_rng_state().clone()
        root_report, body_report, static_report = self.install_resident(motion)
        self.assertEqual(self.kinematics.forward_calls - calls_before, 3)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(root_report["all_values_bitwise_equal"])
        self.assertTrue(body_report["full_corpus_copy_bitwise_equal"])
        self.assertEqual(body_report["body_values_checked"], 8 * 18)
        self.assertEqual(body_report["resident_reference_bytes"], 8 * (83 + 18) * 4)
        self.assertEqual(static_report["root_and_body_reference_bytes"], 8 * (83 + 18) * 4)
        self.assertIsNone(motion._resident_root)
        self.assertFalse(body_report["reference_fk_at_runtime"])
        self.assertFalse(body_report["runtime_host_reference_gathers"])
        self.assertFalse(motion.storage_info["reference_host_gather_required"])
        self.assertEqual(motion.storage_info["runtime_reference_host_to_device_bytes"], 0)
        for name, before in {**roots_before, **bodies_before}.items():
            with self.subTest(field=name):
                actual = getattr(motion, name)
                self.assertIsInstance(actual, torch.Tensor)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.data_ptr() % 16, 0)
                self.assertTrue(torch.equal(actual.view(torch.int32), before.contiguous().view(torch.int32)))
        provider = motion._gpu_reference_bodies
        self.assertEqual(provider.forward_calls, 0)
        self.assertFalse(provider.cache)
        with patch.object(self.kinematics, "forward", side_effect=AssertionError("Query-time FK")):
            with patch.object(provider, "query", side_effect=AssertionError("Old reference provider used")):
                with patch.object(np, "memmap", side_effect=AssertionError("Query-time host cache read")):
                    for name, before in bodies_before.items():
                        torch.testing.assert_close(getattr(motion, name)[frames], before[frames], rtol=0, atol=0)
                    torch.testing.assert_close(motion.goal(frames, 4), goals_before, rtol=0, atol=0)
                    assert_static_reference_identity(motion)
        self.assertEqual(self.binary.read_bytes(), source_before)

    def test_static_identity_guard_detects_replaced_roots_bodies_and_metadata(self):
        motion = self.load()
        self.install_resident(motion)
        for name in ("root_p", "body_v", "clip_end"):
            with self.subTest(field=name):
                original = getattr(motion, name)
                setattr(motion, name, original.clone())
                with self.assertRaisesRegex(RuntimeError, name):
                    assert_static_reference_identity(motion)
                setattr(motion, name, original)
                assert_static_reference_identity(motion)

    def test_static_identity_guard_detects_inplace_reference_mutation(self):
        motion = self.load()
        self.install_resident(motion)
        motion.root_v.add_(0.5)
        with self.assertRaisesRegex(RuntimeError, "root_v"):
            assert_static_reference_identity(motion)

    def test_residency_budget_rejects_without_publishing_body_field_changes(self):
        motion = self.load()
        install_static_root_fields(motion, chunk_rows=3, allow_cpu=True)
        fields_before = {name: getattr(motion, name) for name in (*ROOT_FIELDS, *BODY_FIELDS)}
        info_before = dict(motion.storage_info)
        calls_before = self.kinematics.forward_calls
        total_reference_bytes = 8 * (83 + 18) * 4
        with self.assertRaisesRegex(MemoryError, "max_reference_bytes"):
            install_resident_reference_bodies(
                motion, self.kinematics, chunk_rows=3, reserve_bytes=0,
                max_reference_bytes=total_reference_bytes - 1, allow_cpu_for_tests=True,
            )
        self.assertEqual(self.kinematics.forward_calls, calls_before)
        self.assertEqual(motion.storage_info, info_before)
        self.assertFalse(hasattr(motion, "_resident_body_reference"))
        self.assertFalse(hasattr(motion, "_resident_body_reference_report"))
        for name, before in fields_before.items():
            self.assertIs(getattr(motion, name), before, name)
        # A subsequent admitted initialization can still complete cleanly.
        report = install_resident_reference_bodies(
            motion, self.kinematics, chunk_rows=3, reserve_bytes=0,
            max_reference_bytes=total_reference_bytes, allow_cpu_for_tests=True,
        )
        self.assertEqual(report["resident_reference_bytes"], total_reference_bytes)
        finalize_static_reference_fields(motion)
        assert_static_reference_identity(motion)

    def test_nonfinite_fk_rejects_without_publishing_body_fields(self):
        motion = self.load()
        install_static_root_fields(motion, chunk_rows=3, allow_cpu=True)
        fields_before = {name: getattr(motion, name) for name in BODY_FIELDS}
        original_forward = self.kinematics.forward

        def nonfinite_forward(*states):
            outputs = tuple(value.clone() for value in original_forward(*states))
            outputs[0][0, 0, 0] = float("nan")
            return outputs

        with patch.object(self.kinematics, "forward", side_effect=nonfinite_forward):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
                install_resident_reference_bodies(
                    motion, self.kinematics, chunk_rows=3, reserve_bytes=0,
                    max_reference_bytes=4096, allow_cpu_for_tests=True,
                )
        self.assertFalse(hasattr(motion, "_resident_body_reference"))
        for name, before in fields_before.items():
            self.assertIs(getattr(motion, name), before, name)


if __name__ == "__main__":
    unittest.main()

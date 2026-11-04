"""CPU contracts for original BVH conversion; optional local two-frame G1 IK."""

from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
import retarget_lafan_aiming as aiming


FIXTURE = """HIERARCHY
ROOT Hips
{
 OFFSET 100 100 100
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Spine
 {
  OFFSET 0 10 0
  CHANNELS 3 Xrotation Yrotation Zrotation
  End Site
  {
   OFFSET 0 5 0
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.033333
1 100 3 30 20 10 0 0 0
4 100 6 60 40 20 0 0 0
"""


@unittest.skipUnless(importlib.util.find_spec("bvh"), "Install bvh==0.3")
class BvhTests(unittest.TestCase):
    def test_parser_uses_channel_order_and_replaces_root_offset(self):
        motion = aiming.parse_bvh(FIXTURE)
        self.assertEqual(motion.names, ("Hips", "Spine"))
        np.testing.assert_array_equal(motion.parents, (-1, 0))
        np.testing.assert_allclose(motion.positions[0, 0], (1, 100, 3))
        np.testing.assert_allclose(motion.positions[0, 1], (0, 10, 0))
        np.testing.assert_allclose(motion.rotations[0, 0],
                                   Rotation.from_euler("ZXY", (30, 20, 10), degrees=True).as_matrix())
        self.assertFalse(np.allclose(motion.rotations[0, 0],
                                    Rotation.from_euler("zxy", (30, 20, 10), degrees=True).as_matrix()))

    def test_slice_is_inclusive_start_exclusive_stop(self):
        motion = aiming.parse_bvh(FIXTURE, start_frame=1, max_frames=1)
        self.assertEqual(motion.start_frame, 1)
        self.assertEqual(motion.total_frames, 2)
        np.testing.assert_allclose(motion.positions[0, 0], (4, 100, 6))

    def test_parser_rejects_nonfinite_or_missing_motion_and_bad_fps(self):
        for text in (FIXTURE.replace("1 100 3", "nan 100 3"),
                     FIXTURE.replace("Frames: 2", "Frames: 3"),
                     FIXTURE.replace("0.033333", "0.01")):
            with self.assertRaises(ValueError):
                aiming.parse_bvh(text)
        for count in (0, -1, 501):
            with self.assertRaises(ValueError):
                aiming.parse_bvh(FIXTURE, max_frames=count)

    def test_slerp_occurs_at_50hz_without_extrapolating(self):
        motion = aiming.parse_bvh(FIXTURE)
        p, r, t = aiming.resample_local(motion)
        np.testing.assert_allclose(t, (0, .02))
        np.testing.assert_allclose(p[1, 0], (2.8, 100, 4.8))
        relative = Rotation.from_quat(motion.quaternions[0, 0]).inv() * Rotation.from_quat(
            motion.quaternions[1, 0])
        expected = Rotation.from_quat(motion.quaternions[0, 0]) * Rotation.from_rotvec(
            relative.as_rotvec() * .6)
        np.testing.assert_allclose(r[1, 0], expected.as_matrix(), atol=1e-12)
        self.assertLessEqual(t[-1], 1 / 30)

    def test_fk_rotates_child_offset_in_parent_frame(self):
        motion = aiming.parse_bvh(FIXTURE)
        p, r = aiming.forward_kinematics(motion.positions, motion.rotations, motion.parents)
        np.testing.assert_allclose(p[:, 1], p[:, 0] + np.einsum(
            "nij,j->ni", r[:, 0], (0, 10, 0)))


class GeometryTests(unittest.TestCase):
    def test_y_up_cm_maps_to_z_up_m_with_proper_rotation(self):
        p, _ = aiming.human_to_robot(np.array(((100., 0, 0), (0, 100., 0), (0, 0, 100.))),
                                     np.broadcast_to(np.eye(3), (3, 3, 3)))
        np.testing.assert_allclose(p, ((1, 0, 0), (0, 0, 1), (0, -1, 0)))
        self.assertAlmostEqual(np.sum(np.cross(aiming.WORLD_BASIS[0],
                                               aiming.WORLD_BASIS[1]) * aiming.WORLD_BASIS[2]), 1.)

    def test_anatomical_basis_keeps_real_lafan_pelvis_upright(self):
        human = Rotation.from_euler("ZYX", (91.911438, 5.797277, 88.877003),
                                    degrees=True).as_matrix()
        robot = np.einsum("ij,jk,kl->il", aiming.WORLD_BASIS, human, aiming.BODY_BASIS)
        self.assertGreater(robot[2, 2], .99)
        np.testing.assert_allclose(robot[:, 0],
                                   np.einsum("ij,jk,k->i", aiming.WORLD_BASIS,
                                             human, np.array((0., 1, 0))))
        np.testing.assert_allclose(np.einsum("ji,jk->ik", robot, robot), np.eye(3), atol=1e-14)

    def test_mapping_uses_upper_arm_joint_not_clavicle(self):
        self.assertEqual(aiming.SOURCE_ROLES[8], "LeftArm")
        self.assertEqual(aiming.SOURCE_ROLES[11], "RightArm")
        with self.assertRaisesRegex(ValueError, "Missing original"):
            aiming.role_indices(("Hips",))

    def test_toe_basis_retains_pitch_and_roll(self):
        robot = Rotation.from_euler("XYZ", (20., -30., 5.), degrees=True).as_matrix()
        source_toe = np.einsum("ij,jk->ik", robot, aiming.FOOT_BASIS.T)
        converted = np.einsum("ij,jk->ik", source_toe, aiming.FOOT_BASIS)
        np.testing.assert_allclose(converted, robot, atol=1e-14)
        self.assertLess(converted[2, 2], .9)

    def test_obstacles_alignment_preserves_vertical_excursions(self):
        from bootstrap import configure_paths
        configure_paths()
        values = np.zeros((3, 9))
        values[:, 2] = (.5, .8, 1.2)
        values[:, 6] = 1.
        with patch("prepare_gait_references.physical_floor_distances",
                   side_effect=(np.array((-.1, .2, .6)), np.array((.003, .303, .703)))):
            result, report = aiming.align_floor(values, "obstacles")
        np.testing.assert_allclose(np.diff(result[:, 2]), np.diff(values[:, 2]))
        self.assertAlmostEqual(report["root_height_shift_m"], .103)
        np.testing.assert_allclose(values[:, 2], (.5, .8, 1.2))

    def test_morphology_preserves_all_segment_lengths_and_hand_pose(self):
        rest = np.array((
            (0, 0, .76), (0, 0, .80),
            (0, .10, .64), (0, .10, .34), (0, .10, .04),
            (0, -.10, .64), (0, -.10, .34), (0, -.10, .04),
            (0, .15, 1.0), (0, .15, .8), (0, .15, .6),
            (0, -.15, 1.0), (0, -.15, .8), (0, -.15, .6),
        ), dtype=float)
        source = rest * 2
        source[9:11] = ((.3, .3, 2), (.6, .3, 2))
        source[12:14] = ((.3, -.3, 2), (.6, -.3, 2))
        rotations = np.broadcast_to(aiming.BODY_BASIS.T, (14, 3, 3)).copy()
        target, orient, scale = aiming.morphology_targets(source, rotations, rest)
        self.assertAlmostEqual(scale, .5)
        for a, b, c in aiming.CHAINS:
            for parent, child in ((a, b), (b, c)):
                self.assertAlmostEqual(aiming.vector_norm(target[child] - target[parent]),
                                       aiming.vector_norm(rest[child] - rest[parent]))
        self.assertGreater(target[10, 0], .35)
        self.assertGreater(target[13, 0], .35)
        np.testing.assert_allclose(orient[0], np.eye(3))

    def test_take_identity_groups_subjects_and_holds_out_take_one(self):
        a = aiming.source_identity("aiming1_subject1.bvh", "aiming")
        b = aiming.source_identity("aiming1_subject4.bvh", "aiming")
        self.assertEqual(a["source_group"], b["source_group"])
        self.assertEqual(a["split"], "test")
        self.assertEqual(aiming.source_identity("aiming2_subject5.bvh", "aiming")["split"], "train")
        self.assertEqual(aiming.source_identity("obstacles3_subject2.bvh", "obstacles")["source_group"],
                         "obstacles3")
        with self.assertRaises(ValueError):
            aiming.source_identity("walk1_subject1.bvh", "aiming")

    def test_output_validation_checks_quaternion_and_joint_limits(self):
        values = np.zeros((2, 9))
        values[:, 6] = 1
        limits = np.array(((-1, 1), (-2, 2)))
        self.assertTrue(aiming.validate_output(values, limits)["finite"])
        for column, number in ((7, 2.), (6, .5), (0, np.nan)):
            bad = values.copy()
            bad[0, column] = number
            with self.assertRaises(ValueError):
                aiming.validate_output(bad, limits)


class LocalRealIKTests(unittest.TestCase):
    def test_two_real_aiming_frames_and_provenance(self):
        source = REPO / "data/raw/lafan1-original/aiming1_subject1.bvh"
        if not source.is_file() or not all(importlib.util.find_spec(m) for m in (
                "bvh", "mujoco", "mink", "daqp", "torch")):
            self.skipTest("Local original BVH, robot assets and IK dependencies required")
        from bootstrap import configure_paths
        configure_paths()
        from trackerTask.supermimic.robots import get_robot
        if not Path(get_robot("g1").mjcf_path).is_file():
            self.skipTest("G1 assets required")
        before = aiming.digest(source)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "aiming1_subject1.csv"
            env = os.environ.copy()
            if sys.platform == "win32":
                env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
            process = subprocess.run([
                sys.executable, "-B", str(Path(aiming.__file__)),
                "--bvh", str(source), "--output", str(output), "--max-frames", "2",
            ], capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60, env=env)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            values = np.loadtxt(output, delimiter=",", ndmin=2)
            self.assertEqual(values.shape, (2, 36))
            row = report["files"][0]
            self.assertEqual(row["source_sha256"], before)
            self.assertEqual(row["sha256"], aiming.digest(output))
            self.assertEqual(row["source_frame_range"], [0, 2])
            self.assertEqual(row["source_fps"], 30.)
            self.assertEqual(row["output_fps"], 50.)
            self.assertEqual(row["split"], "test")
            self.assertLess(row["ground_alignment"]["physical_foot_floor_distance_m"]["min"], .01)
            self.assertGreaterEqual(row["ground_alignment"]["physical_foot_floor_distance_m"]["min"], 0)
        self.assertEqual(aiming.digest(source), before)


if __name__ == "__main__":
    unittest.main()

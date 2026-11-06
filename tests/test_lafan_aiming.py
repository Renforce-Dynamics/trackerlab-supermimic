"""CPU contracts for original BVH conversion; optional local two-frame G1 IK."""

from pathlib import Path
from dataclasses import replace
import importlib.util
import inspect
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

    def test_original_hand_axes_use_zero_end_site_fallback_without_side_flip(self):
        source = REPO / "data/raw/lafan1-original/aiming2_subject3.bvh"
        if not source.is_file():
            self.skipTest("Original LAFAN1 BVH required")
        motion = aiming.parse_bvh(source.read_text(encoding="utf-8-sig"), max_frames=1)
        axes, evidence = aiming.source_hand_axes(motion)
        for side, axis in enumerate(axes):
            name = ("LeftHand", "RightHand")[side]
            joint = motion.names.index(name)
            self.assertEqual(motion.end_sites[joint], ((0., 0., 0.),))
            np.testing.assert_allclose(
                axis, motion.offsets[joint] / np.linalg.norm(motion.offsets[joint]))
            self.assertGreater(axis[0], .99)
            self.assertFalse(evidence[name]["palm_roll_observed"])

    def test_nonzero_hand_end_sites_define_each_forward_axis(self):
        motion = aiming.parse_bvh(FIXTURE)
        names = ("LeftForeArm", "LeftHand", "RightForeArm", "RightHand")
        offsets = np.array(((0, 0, 0), (2, 0, 0), (0, 0, 0), (3, 0, 0)), dtype=float)
        custom = replace(
            motion, names=names, parents=np.array((-1, 0, -1, 2)), offsets=offsets,
            positions=np.broadcast_to(offsets, (2, 4, 3)).copy(),
            rotations=np.broadcast_to(np.eye(3), (2, 4, 3, 3)).copy(),
            quaternions=np.broadcast_to((0., 0., 0., 1.), (2, 4, 4)).copy(),
            channels=((), (), (), ()),
            end_sites=((), ((1., 1., 0.),), (), ((1., -1., 0.),)),
        )
        axes, evidence = aiming.source_hand_axes(custom)
        np.testing.assert_allclose(axes[0], (2**-.5, 2**-.5, 0))
        np.testing.assert_allclose(axes[1], (2**-.5, -2**-.5, 0))
        self.assertEqual(evidence["LeftHand"]["basis"], "terminal End Site")
        with self.assertRaisesRegex(ValueError, "End Site"):
            aiming.source_hand_axes(replace(custom, end_sites=()))
        with self.assertRaisesRegex(ValueError, "direct child"):
            aiming.source_hand_axes(replace(custom, parents=np.array((-1, 2, -1, 2))))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            aiming.source_hand_axes(replace(
                custom, offsets=np.zeros((4, 3)),
                end_sites=((), ((0., 0., 0.),), (), ((0., 0., 0.),))))


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

    def test_hand_directions_are_covariant_under_world_rotation(self):
        rng = np.random.default_rng(20261106)
        rotations = Rotation.random(12, random_state=rng).as_matrix().reshape(6, 2, 3, 3)
        local_axes = np.array(((1., .02, 0.), (1., -.01, .01)))
        world_rotation = Rotation.from_rotvec((.4, -.2, .7)).as_matrix()
        original = aiming.hand_forward_directions(rotations, local_axes)
        transformed = aiming.hand_forward_directions(
            np.einsum("ij,nsjk->nsik", world_rotation, rotations), local_axes)
        np.testing.assert_allclose(
            transformed, np.einsum("ij,nsj->nsi", world_rotation, original), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(original, axis=-1), 1., atol=1e-12)

    def test_hand_direction_ignores_unobservable_local_roll_and_axis_scale(self):
        rotations = Rotation.from_euler("xyz", ((.4, -.2, .6), (-.1, .9, -.2))).as_matrix()
        axes = np.array(((1., .2, .1), (1., -.3, .2)))
        unit_axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        twists = Rotation.from_rotvec(unit_axes * np.array((1.2, -.6))[:, None]).as_matrix()
        np.testing.assert_allclose(
            aiming.hand_forward_directions(rotations @ twists, axes * 10.),
            aiming.hand_forward_directions(rotations, axes), atol=1e-12)
        for bad in (np.zeros((2, 3)), np.full((2, 3), np.nan), np.zeros((3, 3))):
            with self.assertRaises(ValueError):
                aiming.hand_forward_directions(rotations, bad)

    def test_optional_hand_orientation_is_disabled_by_default(self):
        self.assertEqual(inspect.signature(aiming.retarget_motion).parameters[
            "hand_orientation"].default, "none")
        self.assertEqual(inspect.signature(aiming.convert).parameters[
            "hand_orientation"].default, "none")
        aiming.validate_hand_options("none", 1.)
        aiming.validate_hand_options("forward", .5)
        for mode, cost in (("guessed", 1.), ("forward", 0.), ("none", float("nan"))):
            with self.subTest(mode=mode, cost=cost), self.assertRaises(ValueError):
                aiming.validate_hand_options(mode, cost)

    def test_cli_forwards_explicit_hand_mode_and_weight(self):
        with patch.object(aiming, "convert") as convert:
            aiming.main([
                "--bvh", "aiming2_subject3.bvh", "--output", "candidate.csv",
                "--hand-orientation", "forward", "--hand-orientation-cost", ".6",
            ])
        self.assertEqual(convert.call_args.kwargs,
                         {"hand_orientation": "forward", "hand_orientation_cost": .6})

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
            self.assertEqual(row["hand_orientation"]["mode"], "none")
            self.assertEqual(row["hand_orientation"]["cost"], 0.)
        self.assertEqual(aiming.digest(source), before)


@unittest.skipUnless(all(importlib.util.find_spec(name)
                        for name in ("mujoco", "mink", "torch", "daqp")),
                     "Local MuJoCo and Mink required")
class HandTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bootstrap import configure_paths
        configure_paths()
        import mink
        import mujoco
        from trackerTask.supermimic.robots import get_robot

        if not Path(get_robot("g1").mjcf_path).is_file():
            raise unittest.SkipTest("Local G1 model assets required")
        from retarget_gait_references import flat_model

        cls.mink, cls.mujoco = mink, mujoco
        cls.model, cls.spec, height, qids, *_ = flat_model("g1")
        cls.data = mujoco.MjData(cls.model)
        cls.data.qpos[:3] = (0, 0, height)
        cls.data.qpos[qids] = cls.spec.default_q
        mujoco.mj_forward(cls.model, cls.data)

    def test_robot_axes_follow_mesh_principal_axis_on_both_sides(self):
        axes, evidence = aiming.robot_hand_axes(self.model)
        for axis in axes:
            np.testing.assert_allclose(np.linalg.norm(axis), 1., atol=1e-12)
            self.assertGreater(axis[0], .8)
        self.assertLess(axes[0, 1], 0.)
        self.assertGreater(axes[1, 1], 0.)
        for side in ("left", "right"):
            self.assertGreater(evidence[side]["principal_variance_m2"],
                               evidence[side]["next_variance_m2"])
            self.assertGreater(evidence[side]["principal_axis_to_wrist_axis_deg"], 10.)

    def test_direction_task_jacobian_matches_finite_difference_and_keeps_roll_free(self):
        body_name = "left_wrist_yaw_link"
        body_id = self.model.body(body_name).id
        local_axis = aiming.robot_hand_axes(self.model)[0][0]
        task = aiming.make_hand_direction_task(body_name, local_axis, 1.)
        configuration = self.mink.Configuration(self.model, self.data.qpos.copy())
        rotation = configuration.data.xmat[body_id].reshape(3, 3)
        current = rotation @ local_axis
        task.set_target(Rotation.from_rotvec((.2, -.1, .3)).apply(current))
        analytic = task.compute_jacobian(configuration)
        numeric = np.empty_like(analytic)
        epsilon = 1e-6
        q0 = configuration.q.copy()
        for dof in range(self.model.nv):
            tangent = np.zeros(self.model.nv)
            tangent[dof] = epsilon
            plus, minus = q0.copy(), q0.copy()
            self.mujoco.mj_integratePos(self.model, plus, tangent, 1.)
            self.mujoco.mj_integratePos(self.model, minus, tangent, -1.)
            configuration.update(plus)
            error_plus = task.compute_error(configuration)
            configuration.update(minus)
            error_minus = task.compute_error(configuration)
            numeric[:, dof] = (error_plus - error_minus) / (2 * epsilon)
        configuration.update(q0)
        np.testing.assert_allclose(analytic, numeric, atol=2e-7, rtol=2e-6)

        frame_jacobian = configuration.get_frame_jacobian(body_name, "body")
        angular_world = rotation @ frame_jacobian[3:]
        roll_velocity = np.linalg.lstsq(angular_world, current, rcond=None)[0]
        np.testing.assert_allclose(analytic @ roll_velocity, np.zeros(3), atol=1e-7)

    def test_unset_and_antipodal_targets_fail_explicitly(self):
        body_name = "right_wrist_yaw_link"
        task = aiming.make_hand_direction_task(body_name, (1., 0., 0.), 1.)
        configuration = self.mink.Configuration(self.model, self.data.qpos.copy())
        with self.assertRaisesRegex(ValueError, "not been set"):
            task.compute_error(configuration)
        direction = configuration.data.xmat[self.model.body(body_name).id].reshape(3, 3)[:, 0]
        task.set_target(-direction)
        with self.assertRaisesRegex(ValueError, "Antipodal"):
            task.compute_error(configuration)

    @unittest.skipUnless(importlib.util.find_spec("bvh"), "Install bvh==0.3")
    def test_real_optional_ik_reduces_forward_error_and_disabled_path_ignores_hands(self):
        source = REPO / "data/raw/lafan1-original/aiming2_subject3.bvh"
        if not source.is_file():
            self.skipTest("Original LAFAN1 BVH required")
        motion = aiming.parse_bvh(source.read_text(encoding="utf-8-sig"), 340, 4)
        baseline, baseline_report = aiming.retarget_motion(motion)
        changed_rotations = motion.rotations.copy()
        changed_quaternions = motion.quaternions.copy()
        hands = [motion.names.index(side + "Hand") for side in ("Left", "Right")]
        changed_rotations[:, hands] = np.eye(3)
        changed_quaternions[:, hands] = (0., 0., 0., 1.)
        changed = replace(motion, rotations=changed_rotations,
                          quaternions=changed_quaternions, end_sites=())
        with patch.object(aiming, "source_hand_axes", side_effect=AssertionError(
                "Disabled path must not calibrate hand orientation")):
            unchanged, _ = aiming.retarget_motion(changed, hand_orientation="none")
        np.testing.assert_array_equal(baseline, unchanged)
        corrected, report = aiming.retarget_motion(motion, hand_orientation="forward")
        self.assertEqual(baseline_report["hand_orientation"]["cost"], 0.)
        self.assertEqual(report["validation"]["joint_limit_violation_max_rad"], 0.)
        self.assertFalse(report["hand_orientation"]["palm_roll_observed"])
        self.assertLess(max(report["hand_orientation"]["forward_error_deg_mean"].values()), 1.)
        self.assertGreater(np.max(abs(baseline[:, 7:] - corrected[:, 7:])), .1)
        legacy = REPO / (
            "data/style-research-20261104/g1-stealth-single-v3/source/"
            "aiming2_subject3__f00340-00460.csv")
        if legacy.is_file():
            saved = np.loadtxt(legacy, delimiter=",", max_rows=len(baseline))
            # Floor-height smoothing uses the clip length; compare the IK pose
            # and horizontal root against the longer archived conversion.
            columns = [0, 1, *range(3, baseline.shape[1])]
            np.testing.assert_allclose(baseline[:, columns], saved[:, columns],
                                       atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()

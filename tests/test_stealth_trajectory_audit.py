"""Offline trajectory-audit checks; synthetic numpy metrics and optional local G1 FK."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
import audit_stealth_trajectory as audit


def record(frames=12, lanes=2):
    qpos = np.zeros((frames, lanes, 36))
    qpos[..., 3] = 1.
    return {"qpos": qpos, "control_dt": np.array(.02), "terminal_steps": np.full(lanes, -1),
            "task": np.array("locomotion/g1/stealth"), "iteration": np.array(32000),
            "seed": np.array(101), "recorded_env_ids": np.arange(lanes) + 10}


class NumpyGaitTests(unittest.TestCase):
    def test_wxyz_pitch_and_backward_lean_signs_under_yaw(self):
        for pitch in (-.3, 0., .25):
            q = np.array([np.cos(pitch/2), 0., np.sin(pitch/2), 0.])
            rotation = audit.quat_matrix(q)
            yaw = audit.quat_matrix([np.cos(.7), 0., 0., np.sin(.7)])
            rotation = yaw @ rotation
            heading = rotation[:2, 0] / np.linalg.norm(rotation[:2, 0])
            measured, lean = audit.pitch_and_lean(rotation, heading)
            self.assertAlmostEqual(measured, np.rad2deg(pitch))
            self.assertAlmostEqual(lean, -np.rad2deg(pitch))
        np.testing.assert_allclose(audit.quat_matrix([1., 0., 0., 0.]), np.eye(3))
        np.testing.assert_allclose(audit.quat_matrix(-q), audit.quat_matrix(q))

    def test_quaternion_validation(self):
        for q in ([0., 0., 0., 0.], [1., 0., 0., .1], [1., 0., 0.],
                  [float("nan"), 0., 0., 0.]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                audit.quat_matrix(q)

    def test_bouts_reject_blips_and_threshold_equality(self):
        heights = np.array([0., .021, 0., .02, .02, .02, .02, .02, 0.] + [.03]*5 + [0.])
        bouts = audit.lifted_bouts(heights, .02)
        self.assertEqual(len(bouts), 1)
        self.assertEqual((bouts[0]["first_sample"], bouts[0]["last_sample"]), (9, 13))
        self.assertAlmostEqual(bouts[0]["duration_s"], .08)
        self.assertFalse(bouts[0]["left_censored"])
        self.assertFalse(bouts[0]["right_censored"])

    def test_bouts_expose_censored_edges_without_inventing_duration(self):
        self.assertEqual(audit.lifted_bouts(np.full(4, .03), .02), [])
        bout, = audit.lifted_bouts(np.full(5, .03), .02)
        self.assertTrue(bout["left_censored"])
        self.assertTrue(bout["right_censored"])
        self.assertAlmostEqual(bout["duration_s"], .08)
        left, right = audit.lifted_bouts(np.r_[np.full(5, .03), 0., np.full(5, .03)], .02)
        self.assertTrue(left["left_censored"])
        self.assertFalse(left["right_censored"])
        self.assertFalse(right["left_censored"])
        self.assertTrue(right["right_censored"])

    def sole_input(self, frames=21):
        position = np.zeros((frames, 2, 3))
        position[..., 2] = .005
        rotation = np.broadcast_to(np.eye(3), (frames, 2, 3, 3)).copy()
        centers = np.zeros((2, 2, 3))
        centers[:, :, 0] = [-.05, .12]
        return position, rotation, centers, np.full((2, 2), .005)

    def test_sliding_planted_vs_alternating_lifted_feet(self):
        position, rotation, centers, radii = self.sole_input()
        position[..., 0] = np.arange(21)[:, None] * .02 * .2
        sliding = audit.sole_metrics(position, rotation, centers, radii, .02)
        self.assertAlmostEqual(sliding["both_feet_down_proportion"], 1.)
        self.assertAlmostEqual(sliding["both_feet_down_duration_s"], .4)
        for foot in sliding["feet"].values():
            self.assertEqual(foot["sustained_lift_count"], 0)
            self.assertAlmostEqual(foot["stance_pole_material_speed_m_s"]["mean"], .2)
        position[..., 0] = 0.
        position[1:10, 0, 2] += .05
        position[11:20, 1, 2] += .05
        alternating = audit.sole_metrics(position, rotation, centers, radii, .02)
        self.assertLess(alternating["both_feet_down_proportion"], .1)
        for foot in alternating["feet"].values():
            self.assertEqual(foot["sustained_lift_count"], 1)
            self.assertAlmostEqual(foot["whole_foot_clearance_m"]["max"], .05)
            self.assertAlmostEqual(foot["whole_foot_clearance_m"]["p95"], .05)
            self.assertAlmostEqual(foot["stance_pole_material_speed_m_s"]["max"], 0.)

    def test_whole_foot_uses_lowest_collider_not_highest_toe(self):
        position, rotation, centers, radii = self.sole_input()
        centers[:, 1, 2] = .1
        result = audit.sole_metrics(position, rotation, centers, radii, .02)
        for foot in result["feet"].values():
            self.assertEqual(foot["whole_foot_clearance_m"]["max"], 0.)
            self.assertEqual(foot["sustained_lift_count"], 0)
        self.assertEqual(result["both_feet_down_proportion"], 1.)

    def test_rotating_sphere_material_speed_is_not_spatial_pole_speed(self):
        frames, dt, radius, angular_speed = 101, .001, .05, 2.
        position = np.zeros((frames, 2, 3))
        position[..., 2] = radius
        theta = np.arange(frames)*dt*angular_speed
        q = np.stack((np.cos(theta/2), np.zeros(frames), np.sin(theta/2), np.zeros(frames)), axis=-1)
        rotation = np.repeat(audit.quat_matrix(q)[:, None], 2, axis=1)
        result = audit.sole_metrics(position, rotation, np.zeros((2, 1, 3)), np.full((2, 1), radius), dt)
        for foot in result["feet"].values():
            # The spatial bottom is stationary, but different material passes it.
            self.assertAlmostEqual(foot["stance_pole_material_speed_m_s"]["mean"], radius*angular_speed, places=6)

    def test_airborne_slip_is_null_and_json_finite(self):
        position, rotation, centers, radii = self.sole_input(frames=2)
        position[..., 2] = 1.
        result = audit.sole_metrics(position, rotation, centers, radii, .02)
        self.assertIsNone(result["feet"]["left"]["stance_pole_material_speed_m_s"])
        self.assertEqual(result["both_feet_down_proportion"], 0.)
        json.dumps(result, allow_nan=False)


class RecordTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "trajectory.npz"

    def load(self, contents=None, **kwargs):
        np.savez(self.path, **(record() if contents is None else contents))
        return audit.load_record(self.path, **kwargs)

    def test_first_terminal_inclusive_and_window_excludes_reset_tail(self):
        contents = record()
        contents["terminal_steps"][0] = 7
        contents["qpos"][8:, 0] = np.nan
        poses, dt, receipt = self.load(contents, start_second=.04, end_second=1.)
        self.assertEqual(poses.shape, (6, 36))
        self.assertEqual(receipt["frame_range_inclusive"], [2, 7])
        self.assertEqual(receipt["environment_id"], 10)
        self.assertEqual(receipt["first_terminal_step"], 7)
        self.assertEqual(receipt["observed_interval_s"], [.04, .14])
        self.assertEqual(dt, .02)

    def test_short_terminal_and_short_windows_rejected(self):
        contents = record()
        contents["terminal_steps"][0] = 0
        with self.assertRaisesRegex(ValueError, "two samples"):
            self.load(contents)
        for kwargs in ({"start_second": .21}, {"start_second": 1.},
                       {"start_second": .02, "end_second": .03}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "two samples"):
                self.load(**kwargs)

    def test_invalid_lane_and_interval(self):
        for kwargs in ({"lane": -1}, {"lane": 2}, {"lane": 1.2}, {"lane": True},
                       {"start_second": -1}, {"start_second": float("nan")},
                       {"start_second": [.02]}, {"end_second": [.1]},
                       {"end_second": float("inf")}, {"end_second": 0.},
                       {"start_second": .1, "end_second": .05}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.load(**kwargs)

    def test_input_validation(self):
        changes = (
            ("control_dt", np.array(0.)), ("control_dt", np.array(-.02)),
            ("control_dt", np.array(float("nan"))), ("control_dt", np.array([.02])),
            ("control_dt", np.array(float("inf"))), ("control_dt", np.array(True)),
            ("qpos", np.zeros((1, 2, 36))), ("qpos", np.zeros((2, 0, 36))),
            ("qpos", np.zeros((12, 2, 35))), ("qpos", np.zeros((12, 2, 36))),
            ("qpos", np.full((12, 2, 36), np.inf)),
            ("terminal_steps", np.array([12, -1])), ("terminal_steps", np.array([-2, -1])),
            ("terminal_steps", np.array([2., -1.])), ("terminal_steps", np.array([-1])),
            ("recorded_env_ids", np.array([0, 0])), ("recorded_env_ids", np.array([-1, 0])),
            ("recorded_env_ids", np.array([0., 1.])), ("seed", np.array(float("nan"))),
            ("iteration", np.array([32000])), ("task", np.array("locomotion/g1/stairs")),
        )
        for key, value in changes:
            with self.subTest(key=key, shape=value.shape), self.assertRaises(ValueError):
                self.load({**record(), key: value})
        contents = record()
        del contents["terminal_steps"]
        with self.assertRaisesRegex(ValueError, "requires"):
            self.load(contents)

    def test_terminal_truncation_cannot_extend_a_lift_bout(self):
        contents = record()
        contents["terminal_steps"][0] = 3
        poses, dt, _ = self.load(contents)
        self.assertEqual(audit.lifted_bouts(np.full(len(poses), .04), dt), [])


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Optional local CPU MuJoCo")
class G1IntegrationTests(unittest.TestCase):
    def test_original_xml_order_fk_receipt_and_cli(self):
        from bootstrap import configure_paths
        configure_paths()
        from trackerTask.supermimic.robots import get_robot
        import mujoco
        spec = get_robot("g1")
        if not Path(spec.mjcf_path).is_file():
            self.skipTest("Optional configured G1 assets absent; no downloads")
        model, bodies, _, _, geometry = audit.g1_geometry()
        pose = model.qpos0.copy()
        pose[:3] = (0., 0., .76)
        for name, value in zip(spec.joint_names, spec.default_q):
            pose[model.joint(name).qposadr[0]] = value
        contents = record(frames=251, lanes=1)
        contents["qpos"][:] = pose
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            output = Path(directory) / "audit.json"
            np.savez(path, **contents)
            with patch.object(mujoco, "mj_step", side_effect=AssertionError("physics step")), \
                    patch.object(mujoco, "mj_forward", side_effect=AssertionError("dynamics")), \
                    patch.object(mujoco, "mj_kinematics", wraps=mujoco.mj_kinematics) as fk:
                result = audit.audit(path)
            self.assertEqual(fk.call_count, 251)
            self.assertEqual(result["trajectory_sha256"], audit.sha256(path))
            self.assertEqual(geometry["xml_sha256"], audit.sha256(spec.mjcf_path))
            self.assertEqual(geometry["mujoco_version"], mujoco.__version__)
            self.assertEqual(geometry["qpos_order"][3:7], ["root_qw", "root_qx", "root_qy", "root_qz"])
            for name in spec.joint_names:
                self.assertEqual(geometry["qpos_order"][model.joint(name).qposadr[0]], name)
            self.assertEqual(result["metrics"]["entry"]["observed_samples"], 243)
            self.assertAlmostEqual(result["metrics"]["entry"]["root"]["pitch_deg"]["mean"], 0.)
            self.assertEqual(len(bodies), 4)
            late = audit.audit(path, start_second=4.9)
            self.assertEqual(late["metrics"]["entry"]["observed_samples"], 0)
            self.assertIsNone(late["metrics"]["entry"]["torso"]["pitch_deg"])
            json.dumps(late, allow_nan=False)
            completed = subprocess.run(
                [sys.executable, "-B", str(Path(audit.__file__)), "--trajectory", str(path),
                 "--output", str(output), "--lane", "0", "--start-second", ".02", "--end-second", ".08"],
                cwd=REPO, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            parsed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(parsed["samples"], 4)
            self.assertEqual(parsed["source"]["frame_range_inclusive"], [1, 4])
            self.assertNotIn("NaN", output.read_text(encoding="utf-8"))

    def test_expert_xyzw_named_order_is_explicit_source_comparison(self):
        from bootstrap import configure_paths
        configure_paths()
        from trackerTask.supermimic.robots import get_robot
        spec = get_robot("g1")
        if not Path(spec.mjcf_path).is_file():
            self.skipTest("Optional configured G1 assets absent; no downloads")
        model, bodies, centers, radii, geometry = audit.g1_geometry()
        pose = model.qpos0.copy()
        pose[:3] = (0., 0., .76)
        pose[3:7] = [np.cos(.1), 0., -np.sin(.1), 0.]
        for index, name in enumerate(spec.joint_names):
            pose[model.joint(name).qposadr[0]] = .001*index
        poses = np.repeat(pose[None], 6, axis=0)
        direct = audit.measure_poses(poses, .02, 0, model, bodies, centers, radii)
        names = list(reversed(spec.joint_names))
        csv = np.column_stack((poses[:, :3], poses[:, [4, 5, 6, 3]],
                               poses[:, [model.joint(name).qposadr[0] for name in names]]))
        config = {"version": 1, "root": ".", "source_fps": 50, "quaternion_order": "xyzw",
                  "joint_names": names, "files": ["backward.csv"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            clip = path.with_name("backward.csv")
            np.savetxt(clip, csv, delimiter=",", fmt="%.17g")
            path.write_text(json.dumps(config), encoding="utf-8")
            compared = audit.expert_comparison(path, model, bodies, centers, radii, geometry)
            self.assertEqual(compared["config_sha256"], audit.sha256(path))
            self.assertEqual(compared["clips"][0]["sha256"], audit.sha256(clip))
            self.assertEqual(compared["clips"][0]["metrics"], direct["metrics"])
            self.assertIn("not a standing-to-motion transition", compared["scope"])
            self.assertGreater(compared["clips"][0]["metrics"]["entry"]["root"][
                "signed_backward_lean_deg"]["mean"], 0.)
            for update in ({"quaternion_order": "wxyz"}, {"files": ["../bad.csv"]},
                           {"files": ["backward.csv", "backward.csv"]}, {"source_fps": 0},
                           {"files": [".."]}, {"joint_names": names[:-1]}, {"joint_names": [None]*29}):
                path.write_text(json.dumps({**config, **update}), encoding="utf-8")
                with self.subTest(update=update), self.assertRaises(ValueError):
                    audit.expert_comparison(path, model, bodies, centers, radii, geometry)


if __name__ == "__main__":
    unittest.main()

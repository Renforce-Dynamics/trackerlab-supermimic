"""Focused CPU tests for source selection and explicitly kinematic waist previews."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"
sys.path.insert(0, str(SCRIPTS))
import render_stealth_entry_views as views


def record(frames=501):
    q = np.zeros((frames, 2, 36))
    q[:, :, 2:4] = [.76, 1.]
    q[:, :, 0] = np.arange(frames)[:, None] * .001
    return {"qpos": q, "control_dt": np.array(.02), "terminal_steps": np.array([-1, 75]),
            "task": np.array("locomotion/g1/stealth"), "iteration": np.array(5000),
            "seed": np.array(101), "recorded_env_ids": np.array([17, 29])}


class EntrySelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trajectory.npz"
        self.values = record()
        np.savez(self.path, **self.values)

    def test_six_seconds_preserves_25fps_and_first_attempt(self):
        poses, indices, meta = views.load_trajectory(self.path)
        self.assertEqual(len(indices), 150)
        np.testing.assert_array_equal(indices, np.arange(150) * 2)
        np.testing.assert_array_equal(poses, self.values["qpos"][:, 0])
        self.assertEqual(meta["environment_id"], 17)
        self.assertEqual(meta["iteration"], 5000)
        self.assertAlmostEqual(len(indices) / views.FPS, 6.)
        self.assertLess(indices[-1] * meta["control_dt"], 6.)

    def test_first_terminal_stops_before_any_reset(self):
        poses, indices, meta = views.load_trajectory(self.path, lane=1)
        self.assertEqual(len(poses), 76)
        self.assertLessEqual(len(indices) / views.FPS, 75 * .02)
        self.assertLessEqual(indices[-1], 75)
        self.assertEqual(meta["environment_id"], 29)
        self.assertEqual(meta["first_terminal_step"], 75)

    def test_bad_cadence_nonfinite_and_missing_terminal_are_rejected(self):
        for field, value in (("control_dt", np.array(.03)),
                             ("control_dt", np.array(np.nan)),
                             ("terminal_steps", np.array([np.nan, 1])),
                             ("qpos", self.values["qpos"] * np.nan)):
            with self.subTest(field=field):
                contents = {**self.values, field: value}
                np.savez(self.path, **contents)
                with self.assertRaises(ValueError):
                    views.load_trajectory(self.path)
        contents = dict(self.values)
        del contents["terminal_steps"]
        np.savez(self.path, **contents)
        with self.assertRaises(ValueError):
            views.load_trajectory(self.path)

    def test_invalid_lane_and_seconds_rejected(self):
        for lane in (-1, 2, True):
            with self.subTest(lane=lane), self.assertRaises(ValueError):
                views.load_trajectory(self.path, lane=lane)
        for seconds in (0, -1, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                views.load_trajectory(self.path, seconds=seconds)

    def test_heading_uses_root_wxyz_and_tracks_turn(self):
        for degrees in (0., 90., -45., 179.):
            pose = self.values["qpos"][0, 0].copy()
            angle = np.deg2rad(degrees) / 2
            pose[3:7] = [np.cos(angle), 0., 0., np.sin(angle)]
            self.assertAlmostEqual(views.heading_degrees(pose), degrees)

    def test_saved_motor_and_collision_settings_are_retained_for_rendering(self):
        np.savez(self.path, **self.values, waist_pd_scale=np.array(3.),
                 self_collisions=np.array(True), collision_mask_version=np.array(2))
        _, _, metadata = views.load_trajectory(self.path)
        self.assertEqual(metadata["waist_pd_scale"], 3.)
        self.assertTrue(metadata["self_collisions"])
        self.assertEqual(metadata["collision_mask_version"], 2)
        np.savez(self.path, **self.values, waist_pd_scale=np.array(float("nan")))
        with self.assertRaises(ValueError):
            views.load_trajectory(self.path)

    def test_existing_output_directory_is_never_overwritten(self):
        output = self.path.parent / "already-exists"
        output.mkdir()
        protected = output / "entry-front-side.mp4"
        protected.write_bytes(b"preserve this file")
        with patch.dict(sys.modules, {"imageio_ffmpeg": None, "mujoco": None, "PIL": None}), \
                patch.object(views, "load_trajectory", side_effect=AssertionError("source loaded")), \
                patch.object(views, "nominal_model", side_effect=AssertionError("model loaded")), \
                patch.object(views.subprocess, "Popen", side_effect=AssertionError("encoder opened")):
            with self.assertRaises(FileExistsError) as error:
                views.render(self.path, output)
        self.assertEqual(error.exception.filename, str(output.resolve()))
        self.assertEqual(protected.read_bytes(), b"preserve this file")
        self.assertEqual(set(output.iterdir()), {protected})

    def test_existing_output_file_is_preserved_without_render_dependencies(self):
        output = self.path.parent / "already-exists"
        output.write_bytes(b"preserve this file")
        with patch.dict(sys.modules, {"imageio_ffmpeg": None, "mujoco": None, "PIL": None}):
            with self.assertRaises(FileExistsError):
                views.render(self.path, output)
        self.assertEqual(output.read_bytes(), b"preserve this file")


class WaistTargetTests(unittest.TestCase):
    def test_quintic_replaces_exactly_three_waist_entries(self):
        names = list(views.WAIST_NAMES) + [f"other_{i}" for i in range(26)]
        addresses = [14, 18, 21]
        mapping = dict(zip(views.WAIST_NAMES, addresses))
        model = SimpleNamespace(joint=lambda name: SimpleNamespace(qposadr=[mapping[name]]))
        poses = np.random.default_rng(4).normal(size=(101, 36))
        original = poses.copy()
        reference = {
            "schema": "g1-stealth-waist-entry-v1",
            "constraint": {"joint_names": names, "default_q": [0.] * 29,
                           "reference_q": dict(zip(views.WAIST_NAMES, [.1, -.1, .5])),
                           "ramp_duration_s": 1.},
        }
        changed, info = views.waist_targets(poses, .02, reference, model)
        untouched = [index for index in range(36) if index not in addresses]
        np.testing.assert_array_equal(poses, original)
        np.testing.assert_array_equal(changed[:, untouched], poses[:, untouched])
        np.testing.assert_allclose(changed[0, addresses], [0., 0., 0.])
        np.testing.assert_allclose(changed[25, addresses], [.05, -.05, .25])
        np.testing.assert_allclose(changed[50:, addresses], np.tile([.1, -.1, .5], (51, 1)))
        self.assertEqual(info["changed_qpos_addresses"], addresses)

    def test_quintic_agrees_with_planned_cpu_class(self):
        import torch
        from bootstrap import configure_paths
        configure_paths()
        from trackerLab.algorithms.supermimic.stealth_waist_entry import WaistEntryConstraint
        names = list(views.WAIST_NAMES) + [f"other_{i}" for i in range(26)]
        mapping = dict(zip(views.WAIST_NAMES, [19, 20, 21]))
        model = SimpleNamespace(joint=lambda name: SimpleNamespace(qposadr=[mapping[name]]))
        config = {"joint_names": names, "default_q": [0.] * 29, "action_scale": [.5] * 29,
                  "reference_q": dict(zip(views.WAIST_NAMES, [-.03, -.02, .52])),
                  "ramp_duration_s": 1.02}
        poses = np.zeros((151, 36))
        changed, _ = views.waist_targets(
            poses, .02, {"schema": "g1-stealth-waist-entry-v1", "constraint": config}, model)
        constraint = WaistEntryConstraint.from_dict(config).cpu()
        expected = constraint.reference_at(torch.arange(151, dtype=torch.float32, device="cpu") * .02)
        np.testing.assert_allclose(changed[:, [19, 20, 21]], expected.numpy(), atol=1e-6)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Local CPU MuJoCo unavailable")
class PoseReplayTests(unittest.TestCase):
    def test_mujoco_front_view_requires_heading_plus_180(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/>'
            '</body></worldbody></mujoco>')
        data, camera = mujoco.MjData(model), mujoco.MjvCamera()
        scene, option = mujoco.MjvScene(model, maxgeom=100), mujoco.MjvOption()
        mujoco.mj_forward(model, data)
        camera.distance, camera.elevation = 2.8, -8.
        for heading in (0., 45., 90.):
            camera.azimuth = heading + 180.
            mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            forward = np.array([np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))])
            self.assertGreater(np.dot(scene.camera[0].pos[:2], forward), 2.7)

    def test_pose_assignment_does_not_step_physics(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/>'
            '</body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        pose = model.qpos0.copy()
        pose[:3] = [1., 2., .8]
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("physics step")), \
                patch.object(mujoco, "mj_forward", wraps=mujoco.mj_forward) as fk:
            views.replay_pose(model, data, pose, .4)
        self.assertEqual(fk.call_count, 1)
        np.testing.assert_array_equal(data.qpos, pose)
        self.assertAlmostEqual(data.time, .4)


if __name__ == "__main__":
    unittest.main()

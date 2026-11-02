"""Native task discovery, terrain morphology and model contracts."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/perception"))
from common import REPO
from trackerTask.native.locomotion.perception.registry import TASKS, ppo_config, robot_parameters
from trackerLab.tracker_env.perception.terrain import generate


class PerceptionTests(unittest.TestCase):
    def test_all_existing_robots_have_all_three_profiles(self):
        from trackerTask.supermimic.robots import TRACKING_TASKS
        self.assertEqual(len(TASKS), 3 * len(TRACKING_TASKS))
        self.assertEqual(tuple(TASKS), tuple(
            f"locomotion/{robot}/{profile}"
            for robot in TRACKING_TASKS for profile in ("stairs", "mountain", "parkour")
        ))
        self.assertNotIn("locomotion/microduck/stairs", TASKS)
        for robot in TRACKING_TASKS:
            spec, height, motors = robot_parameters(robot)
            self.assertEqual(len(motors), spec.num_joints)
            self.assertGreater(height, 0)
            self.assertTrue(np.isfinite(np.asarray(motors)).all())
            self.assertTrue((np.asarray(motors)[:, :4] > 0).all())
            for profile in ("stairs", "mountain", "parkour"):
                self.assertIn(f"locomotion/{robot}/{profile}", TASKS)

    def test_seeded_fields_start_flat_and_have_six_levels(self):
        for profile in ("stairs", "mountain", "parkour"):
            terrain = generate(profile, .76, 42)
            np.testing.assert_array_equal(terrain.heights, generate(profile, .76, 42).heights)
            self.assertEqual(len(terrain.centers), 6)
            self.assertTrue(np.isfinite(terrain.normalized).all())
            self.assertGreaterEqual(terrain.normalized.min(), 0)
            self.assertLessEqual(terrain.normalized.max(), 1)
            self.assertGreater(terrain.heights.max(), .05)
            xs = np.linspace(terrain.x_min, terrain.x_max, terrain.heights.shape[1])
            self.assertEqual(np.count_nonzero(terrain.heights[:, xs < 1.25]), 0)
            flat_y = np.argmin(np.abs(np.linspace(-12, 12, terrain.heights.shape[0]) + 10))
            np.testing.assert_array_equal(terrain.heights[flat_y], 0)

    def test_morphology_scales_positions_and_heights(self):
        for profile in ("stairs", "mountain", "parkour"):
            big, small = generate(profile, .76, 4), generate(profile, .38, 4)
            np.testing.assert_allclose(small.heights, big.heights * .5)
            self.assertEqual(small.goal_x, big.goal_x * .5)

    def test_stairs_are_monotone_and_mountains_change_with_seed(self):
        field = generate("stairs", .76)
        row = field.heights[np.argmin(np.abs(np.linspace(-12, 12, field.heights.shape[0]) + 10))]
        self.assertTrue((np.diff(row) >= 0).all())
        self.assertFalse(np.array_equal(field.heights, generate("stairs", .76, 2).heights))
        self.assertFalse(np.array_equal(generate("mountain", .76, 1).heights,
                                       generate("mountain", .76, 2).heights))
        self.assertFalse(np.array_equal(generate("parkour", .76, 1).heights,
                                       generate("parkour", .76, 2).heights))

    def test_unknown_profile_and_bad_morphology_rejected(self):
        for profile, height in (("typo", .76), ("stairs", 0), ("mountain", float("nan"))):
            with self.assertRaises(ValueError):
                generate(profile, height)

    def test_cli_describe_loads_without_gpu_or_isaac(self):
        result = subprocess.run([sys.executable, str(REPO / "scripts/rls/perception/train.py"),
                                 "--task", "locomotion/g1/stairs", "--describe"],
                                text=True, capture_output=True, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["task"]["algorithm"], "rsl-rl-ppo")
        self.assertEqual(value["ppo"]["num_steps_per_env"], 24)
        self.assertEqual(ppo_config()["algorithm"]["num_learning_epochs"], 5)

    def test_models_when_assets_are_prepared(self):
        from trackerLab.tracker_env.perception.model import build_model
        checked = 0
        for robot in ("g1", "go2", "a3", "t1", "bumi", "pi_plus"):
            spec, height, _ = robot_parameters(robot)
            if not Path(spec.mjcf_path).exists():
                continue
            with self.subTest(robot=robot):
                model, spec, _, qids, vids, _, contract = build_model(robot, generate("stairs", height))
                self.assertEqual(model.nu, spec.num_joints)
                self.assertEqual(len(qids), spec.num_joints)
                self.assertEqual(len(vids), spec.num_joints)
                self.assertEqual(contract["solver_iterations"], 50)
                self.assertEqual(model.nhfield, 1)
                np.testing.assert_allclose(model.qpos0[qids], 0)
                checked += 1
        if not checked:
            self.skipTest("Pinned geometry is prepared separately")


if __name__ == "__main__":
    unittest.main()

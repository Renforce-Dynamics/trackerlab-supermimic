import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class MicroDuckNativeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/rls/microduck/train.py"
        spec = importlib.util.spec_from_file_location("microduck_native_train", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_recipe_is_explicitly_native_and_fresh(self):
        task, cfg = self.module.load_settings()
        self.assertEqual(task["runtime_task"], "Mjlab-Velocity-Flat-MicroDuck")
        self.assertEqual(task["algorithm"], "PPO / RSL-RL")
        self.assertEqual(len(task["revision"]), 40)
        self.assertEqual((cfg["num_envs"], cfg["iterations"]), (4096, 10000))
        self.assertTrue(cfg["fresh_initialization"])
        self.assertEqual(cfg["logger"], "tensorboard")

    def test_smoke_is_separate(self):
        _, smoke = self.module.load_settings(smoke=True)
        self.assertEqual((smoke["num_envs"], smoke["iterations"]), (64, 5))
        _, full = self.module.load_settings()
        self.assertEqual(full["iterations"], 10000)

    def test_native_progress_uses_zero_based_index(self):
        path = Path(__file__).resolve().parents[1] / "scripts/rls/microduck/supervise.py"
        spec = importlib.util.spec_from_file_location("microduck_native_supervise", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        status = module.parse_progress("\x1b[1m Learning iteration 499/10000 \x1b[0m")
        self.assertEqual(status["completed_updates"], 500)
        self.assertEqual(status["reported_iteration_index"], 499)
        self.assertIsNone(module.parse_progress("Episode_Reward/nan_state: 0.0"))

    def test_evaluation_uses_native_base_link_velocity(self):
        path = Path(__file__).resolve().parents[1] / "scripts/rls/microduck/evaluate.py"
        spec = importlib.util.spec_from_file_location("microduck_native_evaluate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = SimpleNamespace(
            root_link_lin_vel_b=np.array([[0.5, -0.2, 9.0], [0.0, 0.0, -9.0]]),
            root_link_ang_vel_b=np.array([[3.0, 4.0, 0.7], [5.0, 6.0, -0.2]]),
        )
        command = np.array([[0.3, -0.1, 0.4], [0.0, 0.0, -0.2]])
        linear, angular = module.tracking_squared_errors(state, command)
        np.testing.assert_allclose(linear, [[0.04, 0.01], [0.0, 0.0]])
        np.testing.assert_allclose(angular, [0.09, 0.0], atol=1e-14)


if __name__ == "__main__":
    unittest.main()

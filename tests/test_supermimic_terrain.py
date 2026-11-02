"""CPU contracts for the SuperMimic terrain downstream path."""

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerTask.supermimic.registry import TASK_REGISTRY
from trackerTask.supermimic.terrain import ROBOTS, PROFILES, TASKS, generate
from trackerLab.algorithms.supermimic.buffer import ReplayBuffer
from trackerLab.algorithms.supermimic.networks import WorldModel
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy, TerrainWorldModel
from trackerLab.tracker_env.mdp.supermimic import normalize_xi
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField, TerrainMDP, terminal_masks


class SuperMimicTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exactly_eighteen_registered_supermimic_tasks(self):
        expected = tuple(f"locomotion/{robot}/{profile}"
                         for robot in ROBOTS for profile in PROFILES)
        self.assertEqual(tuple(TASKS), expected)
        self.assertEqual(sum(entry.runtime_task == "terrain" for entry in TASK_REGISTRY.values()), 18)
        self.assertEqual(len(TASKS), 18)
        for name, task in TASKS.items():
            self.assertEqual(task.id, name)
            self.assertEqual(task.describe()["algorithm"], "supermimic")

    def test_plan_has_one_ten_thousand_iteration_job_per_task(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts/rls/supermimic/supervise_terrain.py"),
             "--run-dir", str(REPO / "unused-plan"), "--plan"],
            capture_output=True, text=True, check=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(len(plan), 18)
        self.assertEqual([row["task"] for row in plan], list(TASKS))
        self.assertTrue(all(row["iterations"] == 10000 for row in plan))

    def test_profile_preset_is_loaded_before_cli_overrides(self):
        config = REPO / "scripts/rls/supermimic/configs/terrain_stairs.json"
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts/rls/supermimic/train_terrain.py"),
             "--task", "locomotion/g1/stairs", "--config", str(config),
             "--run-dir", "unused-config-probe", "--iterations", "23", "--describe"],
            capture_output=True, text=True, check=True,
        )
        value = json.loads(result.stdout)
        self.assertEqual(value["iterations"], 23)
        self.assertEqual(value["task"]["profile"], "stairs")

    def test_seeded_profiles_scale_with_robot_height(self):
        for profile in PROFILES:
            full = generate(profile, .76, 42)
            repeated = generate(profile, .76, 42)
            half = generate(profile, .38, 42)
            np.testing.assert_array_equal(full.heights, repeated.heights)
            np.testing.assert_allclose(half.heights, full.heights * .5)
            self.assertEqual(len(full.centers), 6)
            self.assertEqual(half.goal_x, full.goal_x * .5)
            self.assertTrue(np.isfinite(full.normalized).all())
            self.assertGreater(float(np.ptp(full.heights)), .05)

    def test_heightfield_sampler_and_sensor_observation_have_matching_contract(self):
        data = generate("stairs", .76, 3)
        field = HeightField(data, .76, "cpu")
        x = torch.linspace(data.x_min, data.x_max, len(data.heights[0]))
        y = torch.linspace(data.y_min, data.y_max, len(data.heights))
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack((xx, yy), -1)
        sampled = field.sample(points)
        torch.testing.assert_close(sampled, field.values, atol=2e-5, rtol=0)
        root = torch.tensor([[.2, -2., .9]], requires_grad=True)
        state = (root, torch.eye(3).unsqueeze(0), torch.zeros(1, 3), torch.zeros(1, 3),
                 torch.zeros(1, 12), torch.zeros(1, 12))
        scan = field.features(state)
        self.assertEqual(scan.shape, (1, 187))
        scan.square().sum().backward()
        self.assertTrue(torch.isfinite(root.grad).all())
        self.assertGreater(root.grad.abs().sum().item(), 0)

    def test_world_model_perception_branch_and_reward_gradients(self):
        data = generate("mountain", .76, 7)
        field = HeightField(data, .76, "cpu")
        default = torch.zeros(12)
        mdp = TerrainMDP(field, default, torch.full((12,), -2.), torch.full((12,), 2.))
        batch = 3
        p = torch.tensor([[.5, -10., .85], [.6, -6., .9], [.7, -2., 1.0]], requires_grad=True)
        R = torch.eye(3).expand(batch, 3, 3)
        S = (p, R, torch.zeros(batch, 3), torch.zeros(batch, 3),
             torch.zeros(batch, 12), torch.zeros(batch, 12))
        xi = torch.zeros(batch, 16)
        context = torch.zeros(batch, 2)
        observation = mdp.obs(S, torch.zeros(batch, 12), xi, context)
        self.assertEqual(observation.shape[-1], 238)
        self.assertEqual(mdp.goal(S, context).shape, (batch, 6))
        wm = TerrainWorldModel(field, 12, hidden=(64, 64))
        wm.update_terrain_norm(S)
        next_state = wm.step(S, torch.zeros(batch, 12), normalize_xi(xi))
        reward = mdp.reward(next_state, torch.zeros(batch, 12), torch.zeros(batch, 12), context)
        self.assertTrue(torch.isfinite(reward).all())
        reward.sum().backward(retain_graph=True)
        self.assertTrue(torch.isfinite(p.grad).all())
        error = WorldModel.state_error(next_state, S).square().mean()
        error.backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in wm.parameters()))

    def test_heightfield_uses_mujoco_main_diagonal_triangles(self):
        data = replace(generate("mountain", .76), heights=np.array([[0., 0.], [0., 1.]],
                                                                 dtype=np.float32),
                       x_min=0., x_max=1., y_min=0., y_max=1.)
        field = HeightField(data, .76, "cpu")
        points = torch.tensor([[.75, .25], [.25, .75], [.5, .5]])
        torch.testing.assert_close(field.sample(points), torch.tensor([.25, .25, .5]))

    def test_replay_buffer_keeps_task_scenario_for_windows_and_histories(self):
        buffer = ReplayBuffer(8, 2, "cpu", num_joints=12, scenario_dim=2)
        self.assertEqual(buffer.data["c"].shape, (8, 2, 2))
        self.assertEqual(buffer.fields["c"][0], 2)

    def test_actor_receives_scan_gradients_and_keeps_sensor_normalization(self):
        actor = TerrainPolicy(238, 6, hist=2, num_joints=12)
        observations = torch.randn(3, 2, 238, requires_grad=True)
        action = actor(observations, torch.randn(3, 6))
        self.assertEqual(action.shape, (3, 12))
        self.assertEqual(actor.obs_norm.mean.shape, (238,))
        action.square().sum().backward()
        self.assertTrue(torch.isfinite(observations.grad).all())
        self.assertGreater(observations.grad[..., -187:].abs().sum().item(), 0)

    def test_success_fall_timeout_masks_are_mutually_exclusive(self):
        goal = torch.tensor([False, True, True, False])
        fallen = torch.tensor([False, False, True, False])
        timeout = torch.tensor([True, True, True, True])
        success, done, truncated = terminal_masks(goal, fallen, timeout)
        self.assertEqual(success.tolist(), [False, True, False, False])
        self.assertEqual(done.tolist(), [True] * 4)
        self.assertEqual(truncated.tolist(), [True, False, False, True])


if __name__ == "__main__":
    unittest.main()

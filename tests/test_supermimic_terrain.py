"""CPU contracts for the SuperMimic terrain downstream path."""

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

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
from trackerLab.tracker_env.mdp.supermimic_terrain import (
    FALL_PENALTY, MAX_STEP_REWARD, OBJECTIVE_VERSION, HeightField, TerrainMDP,
    success_terminal_value, terminal_masks, update_terrain_levels,
)
from train_terrain import (
    load_terrain_initialization, migrate_terrain_checkpoint, validate_resume_objective,
)
from supervise_terrain import choose_checkpoint, evaluation_metrics, trusted_checkpoint
from retrain_terrain import build_retrain_plan


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
        self.assertEqual(field.model_features(state).shape, (1, 54))
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
        self.assertEqual(mdp.critic_features(S, xi, context).shape[-1],
                         13 + 2 * 12 + 12 + 4 + 6 + 54)
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

    def test_full_scan_checkpoint_migrates_weights_norms_and_adam_moments(self):
        indices = torch.arange(0, 187, dtype=torch.long)[torch.tensor(
            [index % 2 == 0 for index in range(187)])][:54]
        world_weight = torch.arange(256 * 320, dtype=torch.float32).view(256, 320)
        critic_weight = torch.arange(256 * 246, dtype=torch.float32).view(256, 246)
        old_world_moment = world_weight.clone()
        old_critic_moment = critic_weight.clone()
        checkpoint = {
            "world": {
                "terrain_norm.mean": torch.arange(187, dtype=torch.float32),
                "terrain_norm.var": torch.ones(187),
                "terrain_adapter.0.weight": world_weight.clone(),
            },
            "critic": {
                "norm.mean": torch.arange(246, dtype=torch.float32),
                "norm.var": torch.ones(246),
                "net.0.weight": critic_weight.clone(),
            },
            "opt_wm": {
                "param_groups": [{"params": [10, 11]}],
                "state": {11: {"exp_avg": old_world_moment, "exp_avg_sq": old_world_moment}},
            },
            "opt_v": {
                "param_groups": [{"params": [20]}],
                "state": {20: {"exp_avg": old_critic_moment, "exp_avg_sq": old_critic_moment}},
            },
        }
        self.assertTrue(migrate_terrain_checkpoint(
            checkpoint, indices, ("net.0.weight", "terrain_adapter.0.weight"),
            ("net.0.weight",)))
        torch.testing.assert_close(checkpoint["world"]["terrain_norm.mean"], indices.float())
        torch.testing.assert_close(checkpoint["world"]["terrain_adapter.0.weight"][:, :133],
                                   world_weight[:, :133])
        torch.testing.assert_close(checkpoint["world"]["terrain_adapter.0.weight"][:, 133:],
                                   world_weight[:, 133:].index_select(1, indices))
        self.assertEqual(checkpoint["opt_wm"]["state"][11]["exp_avg"].shape, (256, 187))
        self.assertEqual(checkpoint["opt_v"]["state"][20]["exp_avg"].shape, (256, 113))

    def test_success_fall_timeout_masks_are_mutually_exclusive(self):
        goal = torch.tensor([False, True, True, False])
        fallen = torch.tensor([False, False, True, False])
        timeout = torch.tensor([True, True, True, True])
        success, done, truncated = terminal_masks(goal, fallen, timeout)
        self.assertEqual(success.tolist(), [False, True, False, False])
        self.assertEqual(done.tolist(), [True] * 4)
        self.assertEqual(truncated.tolist(), [True, False, False, True])

    def test_curriculum_demotes_late_falls_and_keeps_timeouts_at_current_level(self):
        levels = torch.tensor([3, 3, 3, 0, 5])
        success = torch.tensor([True, False, False, False, False])
        fallen = torch.tensor([False, True, False, True, True])
        updated = update_terrain_levels(levels, success, fallen)
        self.assertEqual(updated.tolist(), [4, 2, 3, 0, 4])

    def test_soft_terminal_reward_matches_success_and_fall_signals(self):
        data = generate("stairs", .76, 42)
        field = HeightField(data, .76, "cpu")
        default = torch.zeros(12)
        mdp = TerrainMDP(field, default, torch.full((12,), -2.), torch.full((12,), 2.))
        center = float(data.centers[0])
        xy = torch.tensor([[data.goal_x + 1., center],
                           [3., center],
                           [data.goal_x, center]], requires_grad=True)
        ground = field.sample(xy.detach())
        z = torch.stack((ground[0] + .76, ground[1] + .1, ground[2] + .76))
        p = torch.cat((xy, z[:, None]), -1)
        R = torch.eye(3).expand(3, 3, 3)
        S = (p, R, torch.zeros(3, 3), torch.zeros(3, 3),
             torch.zeros(3, 12), torch.zeros(3, 12))
        c = torch.tensor([[center, .7]]).expand(3, 2)

        reward = mdp.soft_terminal_reward(S, c)
        self.assertGreater(reward[0].item(), mdp.success_reward * .9)
        self.assertLess(reward[1].item(), -FALL_PENALTY * .9)
        reward[2].backward()
        self.assertGreater(xy.grad[2, 0].item(), 0)
        self.assertTrue(torch.isfinite(xy.grad).all())

        actions = torch.zeros(3, 12)
        imagined, done = mdp.imagined_reward(S, actions, actions, c)
        torch.testing.assert_close(imagined, mdp.reward(S, actions, actions, c) + reward)
        torch.testing.assert_close(done, mdp.soft_termination(S, c))
        self.assertTrue(((done >= 0) & (done <= 1)).all())
        hard_success = torch.tensor([True, False])
        hard_fall = torch.tensor([False, True])
        torch.testing.assert_close(reward[:2], mdp.terminal_reward(hard_success, hard_fall),
                                   atol=1e-3, rtol=1e-3)

    def test_success_payout_dominates_delaying_with_maximum_running_reward(self):
        for gamma in (.9, .99, .999):
            success = success_terminal_value(gamma)
            self.assertGreater(success, MAX_STEP_REWARD / (1 - gamma))
            for delay in (1, 32, 500, 1000):
                delayed = (MAX_STEP_REWARD * (1 - gamma ** delay) / (1 - gamma)
                           + gamma ** delay * success)
                self.assertGreater(success, delayed)
        for gamma in (0, 1, -1, 1.1):
            with self.assertRaises(ValueError):
                success_terminal_value(gamma)

    def test_resume_rejects_old_objective_and_changed_discount(self):
        with self.assertRaisesRegex(ValueError, "warm-start"):
            validate_resume_objective({"arguments": {"gamma": .99}}, .99)
        ck = {"objective_version": OBJECTIVE_VERSION, "arguments": {"gamma": .99}}
        validate_resume_objective(ck, .99)
        with self.assertRaisesRegex(ValueError, "discount"):
            validate_resume_objective(ck, .95)

    def test_warm_start_transfers_only_policy_and_dynamics(self):
        actor, world = Mock(), Mock()
        world.field.model_indices = torch.arange(54)
        world.named_parameters.return_value = ()
        ck = {"task": "locomotion/g1/stairs", "robot": "g1", "iteration": 10000,
              "actor": {"policy": "weights"}, "world": {"terrain_norm.mean": torch.zeros(54)},
              "critic": {"obsolete_value": "must not load"}}
        receipt = load_terrain_initialization(actor, world, ck, "g1")
        actor.load_state_dict.assert_called_once_with(ck["actor"])
        world.load_state_dict.assert_called_once_with(ck["world"])
        self.assertTrue(receipt["critic_reset"])
        self.assertTrue(receipt["optimizers_reset"])
        self.assertEqual(receipt["task_start_iteration"], 0)
        with self.assertRaisesRegex(ValueError, "different robot"):
            load_terrain_initialization(actor, world, ck, "go2")

    def test_checkpoint_selection_balances_difficulty_and_requires_validation(self):
        result = {"lanes": [{"episodes": n, "successes": s, "falls": n - s}
                            for n, s in [(100, 100)] + [(10, 0)] * 5]}
        metrics = evaluation_metrics(result)
        self.assertAlmostEqual(metrics["macro_success"], 1 / 6)
        self.assertAlmostEqual(metrics["success_fraction"], 2 / 3)
        candidates = [
            {"iteration": 2000, "metrics": {"macro_success": .8, "macro_fall": .1}},
            {"iteration": 10000, "metrics": {"macro_success": .1, "macro_fall": .8}},
        ]
        self.assertEqual(choose_checkpoint(candidates)["iteration"], 2000)
        job = {"best_checkpoint": "best.pt",
               "validation_metrics": {"min_seed_macro_success": .1}}
        self.assertIsNone(trusted_checkpoint(job, .5))
        job["validation_metrics"]["min_seed_macro_success"] = .8
        self.assertEqual(trusted_checkpoint(job, .5), Path("best.pt"))
        result["lanes"][0]["episodes"] = 0
        with self.assertRaisesRegex(ValueError, "six difficulty"):
            evaluation_metrics(result)

    def test_retrain_plan_retains_good_task_and_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory)
            good = baseline / "a3/stairs"
            good.mkdir(parents=True)
            (good / "last.pt").touch()
            result = {"task": "locomotion/a3/stairs", "trained_iterations": 10000,
                      "lanes": [{"episodes": 10, "successes": 9, "falls": 1}] * 6}
            (good / "evaluation.json").write_text(json.dumps(result), encoding="utf-8")
            partial = baseline / "bumi/stairs"
            partial.mkdir(parents=True)
            (partial / "last.pt").touch()
            plan = build_retrain_plan(baseline)
            self.assertEqual(list(plan["retained"]), ["locomotion/a3/stairs"])
            self.assertEqual(len(plan["jobs"]), 17)
            self.assertEqual(plan["initial_checkpoints"]["locomotion/a3/mountain"],
                             str(good / "last.pt"))
            self.assertEqual(plan["initial_checkpoints"]["locomotion/bumi/stairs"],
                             str(partial / "last.pt"))
            self.assertNotIn("locomotion/pi_plus/stairs", plan["initial_checkpoints"])
            self.assertEqual(json.loads((good / "evaluation.json").read_text()), result)


if __name__ == "__main__":
    unittest.main()

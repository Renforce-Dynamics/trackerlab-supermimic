"""Autonomous bar passage: label independence, clearance gradients and cadence."""

from dataclasses import asdict
import math
import unittest

import torch

from test_velocity_task import Kinematics, Motion, state
from test_rapid_task import twist_state
from trackerTask.supermimic.go2_velocity_dodge_cfg import AdaptiveDodgeCfg, get_task_cfg
from trackerLab.tracker_env.mdp import supermimic_velocity_dodge as signals
from trackerLab.algorithms.supermimic.velocity_networks import VelocityWorldModel
from trackerLab.algorithms.supermimic.velocity_runner import adaptive_curriculum_step
from evaluate_velocity import checkpoint_task, encounter_metrics


class AdaptiveTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self):
        return signals.VelocityDodgeMDP(
            Kinematics(), Motion(), torch.zeros(12), obs_mode="relative_orientation",
            task_cfg=AdaptiveDodgeCfg())

    def test_both_height_groups_vary_immediately_and_keep_1_to_2_second_cadence(self):
        cfg = get_task_cfg("adaptive")
        self.assertAlmostEqual(cfg.max_step_reward, .1)
        self.assertAlmostEqual(cfg.value_bound(.99), 10)
        S = state(2048)
        for difficulty in (0.0, 0.5, 1.0):
            c = signals.make_scenario(S, difficulty, torch.arange(1, 2049), cfg)
            self.assertEqual(set(c[:, 8].tolist()), {1., 2.})
            self.assertGreaterEqual(c[:, 10].min().item(), 1.0)
            self.assertLessEqual(c[:, 10].max().item(), 2.0)
            self.assertLessEqual(c[:, 7].max().item(), cfg.max_bar_speed)
            for bucket in (1, 2):
                self.assertGreater(c[c[:, 8] == bucket, 5].std().item(), .002)
            S_end = twist_state(S, c, c[:, 10])
            _, following, ended = signals.advance_scenario(
                signals.offset_scenario(c, c[:, 10] - cfg.control_dt), S_end, cfg)
            self.assertTrue(bool(ended.all()))
            torch.testing.assert_close(following[:, 14], torch.ones(2048))
        sector = torch.floor(torch.remainder(c[:, 6] + math.pi, 2 * math.pi) / (math.pi / 4))
        self.assertEqual(sector.unique().numel(), 8)

    def test_action_labels_do_not_change_policy_features_reward_or_dynamics(self):
        model, S = self.model(), state(32)
        c = signals.make_scenario(S, .5, torch.arange(1, 33), model.task_cfg)
        swapped = c.clone()
        swapped[:, 8] = 3 - swapped[:, 8]
        torch.testing.assert_close(model.task_features(S, c), model.task_features(S, swapped))
        a = torch.randn(32, 12)
        f = torch.zeros(32, dtype=torch.long)
        torch.testing.assert_close(model.task_reward(S, f, a, a, c),
                                   model.task_reward(S, f, a, a, swapped))
        wm = VelocityWorldModel(hidden=(32, 32), geometry_only=True)
        torch.testing.assert_close(wm.task_inputs(S, c), wm.task_inputs(S, swapped))
        for left, right in zip(wm.step(S, a, torch.zeros(32, 16), c),
                               wm.step(S, a, torch.zeros(32, 16), swapped)):
            torch.testing.assert_close(left, right)
        torch.testing.assert_close(model.desired_height(S, c), c[:, 12])

    def test_clearance_has_upward_gradient_for_low_bars_and_downward_for_high_bars(self):
        model = self.model()
        for height, expected_sign in ((.08, 1), (.32, -1)):
            S = state()
            S = (S[0].requires_grad_(), *S[1:])
            c = signals.make_scenario(S, 0., torch.tensor([1]), model.task_cfg).detach()
            c[:, 3:6] = torch.tensor([0., 0., height])
            c[:, 6], c[:, 9], c[:, 8] = 0, 0, 1
            reward = model.passage_clearance(S, c)
            gradient = torch.autograd.grad(reward.sum(), S[0])[0]
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(expected_sign * gradient[0, 2].item(), .01)

    def test_rewards_bounded_and_all_terms_accounted(self):
        torch.manual_seed(71)
        model, S = self.model(), state(512)
        S[0][:, 2] = torch.rand(512) * .65 + .1
        c = signals.make_scenario(S, .8, torch.arange(1, 513), model.task_cfg)
        c[:, 9] = torch.rand(512) * c[:, 10]
        a, previous = torch.randn(512, 12), torch.randn(512, 12)
        reward, terms = model.task_reward(
            S, torch.zeros(512, dtype=torch.long), a, previous, c, terms=True)
        torch.testing.assert_close(reward, sum(terms.values()))
        self.assertIn("passage_clearance", terms)
        self.assertTrue(bool((reward <= .1 + 1e-6).all()))
        self.assertTrue(bool(torch.isfinite(reward).all()))
        clearance = model.passage_clearance(S, c)
        self.assertTrue(bool(((clearance >= -1e-6) & (clearance <= 1+1e-6)).all()))

    def test_curriculum_requires_success_for_both_height_groups(self):
        self.assertEqual(adaptive_curriculum_step(0, [1000, 1000], [990, 0]), 0)
        self.assertEqual(adaptive_curriculum_step(.2, [1000, 100], [900, 100]), .2)
        self.assertAlmostEqual(adaptive_curriculum_step(.2, [1000, 1000], [650, 700]), .25)
        self.assertEqual(adaptive_curriculum_step(1, [1000, 1000], [1000, 1000]), 1)

    def test_saved_profile_restores_and_high_low_evaluation_buckets_are_preserved(self):
        cfg = AdaptiveDodgeCfg()
        profile, restored = checkpoint_task({
            "args": {"task": "go2_velocity_dodge", "task_profile": "adaptive"},
            "task_cfg": asdict(cfg),
        })
        self.assertEqual(profile, "adaptive")
        self.assertEqual(cfg, restored)
        self.assertTrue(restored.autonomous_passage)
        self.assertEqual(encounter_metrics([])["successful_observed_maneuvers"], {})


if __name__ == "__main__":
    unittest.main()

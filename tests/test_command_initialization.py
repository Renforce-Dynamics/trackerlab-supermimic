"""Command warm-start migration preserves useful outputs and releases arm learning."""

from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from train_terrain import adapt_warm_start_actor
from trackerLab.algorithms.supermimic import networks
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerTask.supermimic.robots import get_robot


class CommandInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        cls.addClassCleanup(setattr, networks, "AMP_DTYPE", networks.AMP_DTYPE)
        torch.set_num_threads(2)
        networks.AMP_DTYPE = None

    def make_actor(self):
        spec = get_robot("g1")
        actor = TerrainPolicy(3 * spec.num_joints + 196, 12, hist=2,
                              num_joints=spec.num_joints, action_limit=20.)
        with torch.no_grad():
            actor.dec[-1].weight.normal_(std=.1)
            actor.dec[-1].bias.copy_(torch.linspace(-10., 10., spec.num_joints))
            for norm in (actor.obs_norm, actor.goal_norm):
                norm.mean.uniform_(-.2, .2)
                norm.var.fill_(.5)
                norm.count.fill_(200000512.)
        return spec, actor

    def test_released_arm_heads_start_at_default_and_other_joint_outputs_are_identical(self):
        spec, actor = self.make_actor()
        observation = torch.randn(3, 2, actor.obs_dim)
        goal = torch.randn(3, 12)
        with torch.no_grad():
            before = actor(observation, goal)
        original = {name: value.clone() for name, value in actor.state_dict().items()}
        receipt = adapt_warm_start_actor(
            actor, spec.joint_names, reset_arms=True, normalizer_prior_count=1e6)
        names = receipt["arm_output_rows_reset"]
        arms = [spec.joint_names.index(name) for name in names]
        legs_waist = [i for i in range(spec.num_joints) if i not in arms]
        self.assertEqual(len(arms), 14)
        with torch.no_grad():
            after = actor(observation, goal)
        torch.testing.assert_close(after[:, arms], torch.zeros(3, 14), rtol=0, atol=0)
        torch.testing.assert_close(after[:, legs_waist], before[:, legs_waist], rtol=0, atol=0)
        for key, value in actor.state_dict().items():
            if key.endswith(".count"):
                continue
            if key in {f"dec.{len(actor.dec)-1}.weight", f"dec.{len(actor.dec)-1}.bias"}:
                continue
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        loss = actor(observation, goal)[:, arms].sum()
        loss.backward()
        self.assertGreater(float(actor.dec[-1].weight.grad[arms].abs().sum()), 0.)

    def test_reopened_statistics_keep_initial_outputs_and_accept_new_commands(self):
        spec, actor = self.make_actor()
        observation, goal = torch.randn(3, 2, actor.obs_dim), torch.randn(3, 12)
        with torch.no_grad():
            expected = actor(observation, goal)
        old_mean = actor.goal_norm.mean.clone()
        receipt = adapt_warm_start_actor(
            actor, spec.joint_names, normalizer_prior_count=1024.)
        with torch.no_grad():
            torch.testing.assert_close(actor(observation, goal), expected, rtol=0, atol=0)
        self.assertEqual(receipt["arm_output_rows_reset"], [])
        self.assertEqual(actor.goal_norm.count.item(), 1024.)
        actor.goal_norm.update(torch.full((1024, 12), -.2))
        self.assertEqual(actor.goal_norm.count.item(), 2048.)
        self.assertFalse(torch.equal(actor.goal_norm.mean, old_mean))
        norm_before = actor.obs_norm.count.item()
        actor.obs_norm.update(torch.zeros(7, actor.obs_dim))
        self.assertEqual(actor.obs_norm.count.item(), norm_before + 7)

    def test_default_migration_is_byte_preserving_and_invalid_prior_is_rejected(self):
        spec, actor = self.make_actor()
        before = {name: value.clone() for name, value in actor.state_dict().items()}
        self.assertEqual(adapt_warm_start_actor(actor, spec.joint_names),
                         {"arm_output_rows_reset": [], "normalizer_prior_counts": {}})
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for count in (-1., float("nan"), float("inf"), 2e8, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                adapt_warm_start_actor(actor, spec.joint_names, normalizer_prior_count=count)


if __name__ == "__main__":
    unittest.main()

"""Chase transfer and differentiable velocity/dodge adapter regressions."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.chase_networks import TaskPolicy, TaskWorldModel
from trackerLab.algorithms.supermimic.velocity_networks import VelocityPolicy, VelocityWorldModel
from trackerLab.algorithms.supermimic.velocity_runner import (
    load_chase_initialization, load_velocity_task_initialization, imagined_transition,
)
from trackerLab.algorithms.supermimic.networks import Critic
from trackerLab.tracker_env.mdp.supermimic_velocity_dodge import (
    VelocityDodgeMDP, make_scenario, yaw_rotation,
)
from trackerLab.tracker_env.mdp.supermimic import normalize_xi
from trackerTask.supermimic.robots import get_robot


class Kinematics:
    robot = get_robot("go2")
    q_lo, q_hi = torch.full((12,), -3.), torch.full((12,), 3.)

    def forward(self, p, R, v, w, q, qd):
        n = len(self.robot.tracked_bodies)
        return (p.unsqueeze(-2).expand(*p.shape[:-1], n, 3),
                R.unsqueeze(-3).expand(*R.shape[:-2], n, 3, 3),
                v.unsqueeze(-2).expand(*v.shape[:-1], n, 3),
                w.unsqueeze(-2).expand(*w.shape[:-1], n, 3))


class Motion:
    q = torch.zeros(8, 12)
    root_R = torch.eye(3).repeat(8, 1, 1)
    clip_id = torch.zeros(8, dtype=torch.long)
    clip_start, clip_end = torch.tensor([0]), torch.tensor([7])

    def goal(self, f, frames):
        return self.q.new_zeros((*f.shape, frames * 39))


class VelocityNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(32)

    def assert_finite_nonzero(self, tensor):
        self.assertIsNotNone(tensor)
        self.assertTrue(bool(torch.isfinite(tensor).all()))
        self.assertGreater(tensor.abs().sum().item(), 0.)

    def assert_source_tensors_preserved(self, source, target):
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    def actor_pair(self):
        chase = TaskPolicy(48, 78, hist=2, num_joints=12, action_limit=4.)
        chase.obs_norm.update(.3 + .8 * torch.randn(16, 2, 48))
        chase.goal_norm.update(.2 + .6 * torch.randn(16, 78))
        chase.task_norm.update(.1 + .7 * torch.randn(16, 18))
        with torch.no_grad():
            nn.init.normal_(chase.task_head[-1].weight, std=.04)
            nn.init.normal_(chase.task_head[-1].bias, std=.02)
            nn.init.normal_(chase.dec[-1].weight, std=.02)
        velocity = VelocityPolicy(48, 78, hist=2)
        result = velocity.load_chase(chase.state_dict())
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(all(key.startswith(("command_norm.", "command_head.")) for key in result.missing_keys))
        self.assert_source_tensors_preserved(chase, velocity)
        return chase, velocity

    def world_pair(self):
        chase = TaskWorldModel(hidden=(32, 32))
        chase.in_norm.update(.1 + .9 * torch.randn(32, chase.in_dim))
        chase.out_norm.update(.002 + .01 * torch.randn(32, 36))
        chase.bar_norm.update(.2 + .8 * torch.randn(32, 8))
        with torch.no_grad():
            nn.init.normal_(chase.net[-1].weight, std=.02)
            nn.init.normal_(chase.bar_adapter[-1].weight, std=.01)
            nn.init.normal_(chase.bar_adapter[-1].bias, std=.01)
        velocity = VelocityWorldModel(hidden=(32, 32))
        result = velocity.load_chase(chase.state_dict())
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(all(key.startswith(("orientation_norm.", "orientation_adapter."))
                            for key in result.missing_keys))
        self.assert_source_tensors_preserved(chase, velocity)
        return chase, velocity

    def state(self):
        p = torch.tensor([[.1, -.2, .29], [-.2, .1, .30], [.3, -.1, .28]])
        R = yaw_rotation(torch.tensor([0., .4, -.7]))
        return (p, R, .1 * torch.randn(3, 3), .1 * torch.randn(3, 3),
                .1 * torch.randn(3, 12), .1 * torch.randn(3, 12))

    def equivalent_scenarios(self):
        old = torch.tensor([
            [2., 0., .8, 0., 3., .1, -.8, .34, 0., .2, .8, .29],
            [2., 0., .8, 0., 2., .2, -1., .34, 1., .5, .8, .29],
            [2., 0., .8, 0., 4., .3, -1.2, .12, 2., .8, .8, .29],
        ])
        new = torch.zeros(3, 15)
        new[:, 0], new[:, 3:5], new[:, 5] = old[:, 2], old[:, 4:6], old[:, 7]
        new[:, 7], new[:, 8], new[:, 9] = -old[:, 6], old[:, 8], old[:, 9]
        new[:, 10], new[:, 11], new[:, 12], new[:, 13] = 4., old[:, 10], old[:, 11], 1.
        return old, new

    def test_actor_exactly_preserves_chase_for_same_legacy_inputs(self):
        chase, velocity = self.actor_pair()
        obs, goal, legacy = torch.randn(3, 2, 48), torch.randn(3, 78), torch.randn(3, 18)
        expected = chase(obs, goal, legacy)
        for extra in (torch.randn(3, 6), torch.full((3, 6), 50.)):
            features = torch.cat((legacy, extra), -1)
            torch.testing.assert_close(velocity(obs, goal, features), expected, atol=0, rtol=0)
        velocity.command_norm.update(torch.randn(20, 6))
        torch.testing.assert_close(velocity(obs, goal, features), expected, atol=0, rtol=0)
        self.assertTrue(bool((expected.abs() <= 4).all()))
        self.assert_source_tensors_preserved(chase, velocity)

    def test_world_exactly_preserves_chase_for_equivalent_physical_bar(self):
        chase, velocity = self.world_pair()
        S, a, xi = self.state(), torch.randn(3, 12), torch.randn(3, 16)
        old, new = self.equivalent_scenarios()
        torch.testing.assert_close(velocity.task_inputs(S, new), chase.task_inputs(S, old), atol=0, rtol=0)
        expected = chase.step(S, a, xi, old)
        actual = velocity.step(S, a, xi, new)
        for original, loaded in zip(expected, actual):
            torch.testing.assert_close(loaded, original, atol=0, rtol=0)
        self.assert_source_tensors_preserved(chase, velocity)

    def test_actor_zero_branch_learns_and_activated_commands_affect_actions(self):
        _, velocity = self.actor_pair()
        obs, goal = torch.randn(3, 2, 48), torch.randn(3, 78)
        features = torch.randn(3, 24, requires_grad=True)
        weights = torch.arange(1., 13.)
        (velocity(obs, goal, features) * weights).sum().backward()
        self.assert_finite_nonzero(velocity.command_head[-1].weight.grad)
        torch.testing.assert_close(features.grad[:, 18:], torch.zeros(3, 6), atol=0, rtol=0)
        velocity.zero_grad(set_to_none=True)
        features.grad = None
        with torch.no_grad():
            nn.init.normal_(velocity.command_head[-1].weight, std=.08)
        alternate = features.detach().clone()
        alternate[:, 18] += 1.
        original = velocity(obs, goal, features)
        changed = velocity(obs, goal, alternate)
        self.assertGreater((original - changed).abs().max().item(), 1e-7)
        (original * weights).sum().backward()
        self.assert_finite_nonzero(features.grad[:, 18:])
        self.assert_finite_nonzero(velocity.command_head[0].weight.grad)
        self.assertTrue(bool((original.abs() <= 4).all()))

    def test_world_orientation_branch_has_learning_and_input_gradients(self):
        _, velocity = self.world_pair()
        S, a, xi = self.state(), torch.randn(3, 12), torch.randn(3, 16)
        _, c = self.equivalent_scenarios()
        c[:, 8], c[:, 7], c[:, 9] = 1., 0., 0.
        c.requires_grad_()
        prediction = velocity.step(S, a, xi, c)
        sum(x.square().sum() for x in prediction).backward()
        self.assert_finite_nonzero(velocity.orientation_adapter[-1].weight.grad)
        torch.testing.assert_close(c.grad[:, 6], torch.zeros(3), atol=0, rtol=0)
        velocity.zero_grad(set_to_none=True)
        c.grad = None
        with torch.no_grad():
            nn.init.normal_(velocity.orientation_adapter[-1].weight, std=.1)
        turned = c.detach().clone()
        turned[:, 6] = 1.
        # Static center and velocity leave the legacy branch inputs unchanged.
        torch.testing.assert_close(velocity.task_inputs(S, c), velocity.task_inputs(S, turned), atol=0, rtol=0)
        actual, alternate = velocity.step(S, a, xi, c), velocity.step(S, a, xi, turned)
        self.assertGreater(sum((x-y).abs().sum().item() for x, y in zip(actual, alternate)), 1e-6)
        sum(x.square().sum() for x in actual).backward()
        self.assert_finite_nonzero(c.grad[:, 6])
        self.assert_finite_nonzero(velocity.orientation_adapter[0].weight.grad)

    def test_warm_start_reports_source_and_requires_complete_compatible_weights(self):
        chase_pi, pi = self.actor_pair()
        chase_wm, wm = self.world_pair()
        teacher = TaskPolicy(48, 78, hist=2, num_joints=12, action_limit=4.)
        args = SimpleNamespace(hist=2, goal_frames=2, obs_mode="relative_orientation", action_limit=4.)
        source = {"args": {**vars(args), "task": "go2_chase_dodge"}, "iteration": 8000,
                  "pi": chase_pi.state_dict(), "wm": chase_wm.state_dict()}
        result = load_chase_initialization(pi, wm, teacher, source, args)
        self.assertEqual(result["kind"], "warm_start_new_objective")
        self.assertEqual(result["source_iteration"], 8000)
        self.assertEqual(result["task_start_iteration"], 0)
        self.assertTrue(result["critic_reset"])
        self.assertTrue(result["optimizers_reset"])
        self.assert_source_tensors_preserved(chase_pi, teacher)
        with self.assertRaisesRegex(ValueError, "hist"):
            load_chase_initialization(pi, wm, teacher, source,
                                      SimpleNamespace(**{**vars(args), "hist": 3}))
        missing = dict(source["pi"])
        del missing["dec.0.weight"]
        with self.assertRaisesRegex(ValueError, "dec.0.weight"):
            pi.load_chase(missing)
        extra = dict(source["wm"])
        extra["unknown.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unknown.weight"):
            wm.load_chase(extra)

    def test_velocity_task_warm_start_preserves_actor_and_world_model_but_resets_task_value(self):
        _, source_pi = self.actor_pair()
        _, source_wm = self.world_pair()
        pi = VelocityPolicy(48, 78, hist=2)
        wm = VelocityWorldModel(hidden=(32, 32))
        teacher = TaskPolicy(48, 78, hist=2, num_joints=12, action_limit=4.)
        args = SimpleNamespace(hist=2, goal_frames=2, obs_mode="relative_orientation", action_limit=4.)
        source = {"args": {**vars(args), "task": "go2_velocity_dodge", "task_profile": "rapid"},
                  "iteration": 10000, "pi": source_pi.state_dict(), "wm": source_wm.state_dict()}
        result = load_velocity_task_initialization(pi, wm, teacher, source, args)
        self.assertEqual(result["kind"], "warm_start_new_velocity_objective")
        self.assertEqual(result["source_profile"], "rapid")
        self.assertEqual(result["task_start_iteration"], 0)
        self.assertTrue(result["critic_reset"])
        self.assertTrue(result["optimizers_reset"])
        self.assert_source_tensors_preserved(source_pi, pi)
        self.assert_source_tensors_preserved(source_wm, wm)
        for key, value in teacher.state_dict().items():
            self.assertTrue(torch.equal(value, source_pi.state_dict()[key]), key)
        with self.assertRaisesRegex(ValueError, "obs_mode"):
            load_velocity_task_initialization(pi, wm, teacher, source,
                                              SimpleNamespace(**{**vars(args), "obs_mode": "proprioception"}))

    def test_velocity_checkpoint_roundtrip_preserves_new_branches_and_norms(self):
        _, pi = self.actor_pair()
        _, wm = self.world_pair()
        S, a, xi = self.state(), torch.randn(3, 12), torch.randn(3, 16)
        _, c = self.equivalent_scenarios()
        obs, goal, features = torch.randn(3, 2, 48), torch.randn(3, 78), torch.randn(3, 24)
        pi.update_task_norms(features)
        wm.orientation_norm.update(wm.orientation_inputs(S, c))
        with torch.no_grad():
            nn.init.normal_(pi.command_head[-1].weight, std=.05)
            nn.init.normal_(wm.orientation_adapter[-1].weight, std=.05)
        loaded_pi = VelocityPolicy(48, 78, hist=2)
        loaded_wm = VelocityWorldModel(hidden=(32, 32))
        loaded_pi.load_state_dict(pi.state_dict())
        loaded_wm.load_state_dict(wm.state_dict())
        self.assert_source_tensors_preserved(pi, loaded_pi)
        self.assert_source_tensors_preserved(wm, loaded_wm)
        torch.testing.assert_close(loaded_pi(obs, goal, features), pi(obs, goal, features), atol=0, rtol=0)
        for x, y in zip(loaded_wm.step(S, a, xi, c), wm.step(S, a, xi, c)):
            torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_imagination_keeps_actor_gradients_across_a_respawn(self):
        _, pi = self.actor_pair()
        _, wm = self.world_pair()
        motion = Motion()
        mdp = VelocityDodgeMDP(Kinematics(), motion, torch.zeros(12), obs_mode="relative_orientation")
        S = self.state()
        c = make_scenario(S, 1., torch.tensor([1, 2, 3]))
        c[:, 9] = c[:, 10] - .02
        previous, f, xi = torch.zeros(3, 12), torch.zeros(3, dtype=torch.long), torch.zeros(3, 16)
        xi[:, 0] = 1.
        history = mdp.obs(S, previous, xi, f, c)[:, None].repeat(1, 2, 1)
        critic = Critic(37 + 16 + 24 + 78)
        wm.requires_grad_(False)
        critic.requires_grad_(False)
        before = {name: value.clone() for name, value in wm.named_buffers()}
        objective = torch.zeros(3)
        for _ in range(4):
            S, a, history, f, c, reward, death = imagined_transition(
                pi, wm, mdp, motion, S, history, f, xi, c, previous, 2)
            objective = objective + reward * (1 - death)
            previous = a
        self.assertTrue(bool((c[:, 14] >= 1).all()))
        terminal = critic(torch.cat((mdp.task_critic_features(S, normalize_xi(xi), c),
                                    motion.goal(f, 2)), -1))
        (-objective.mean() - .1 * terminal.mean()).backward()
        self.assert_finite_nonzero(pi.command_head[-1].weight.grad)
        self.assert_finite_nonzero(pi.dec[0].weight.grad)
        self.assertTrue(all(p.grad is None for p in wm.parameters()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        for name, value in wm.named_buffers():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

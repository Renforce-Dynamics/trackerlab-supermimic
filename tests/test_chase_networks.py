"""CPU checks for gait-preserving, differentiable chase task adapters."""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.chase_networks import TaskPolicy, TaskWorldModel
from trackerLab.algorithms.supermimic.networks import Policy, WorldModel
from trackerLab.kinematics.supermimic.rotations import so3_exp, yaw_mat
from trackerLab.tracker_env.mdp.supermimic_chase_dodge import advance_scenario


class ChaseNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(17)

    def assert_finite_nonzero(self, value):
        self.assertIsNotNone(value)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertGreater(value.abs().sum().item(), 0.0)

    def assert_base_loaded(self, base, task, result, added_prefixes):
        self.assertEqual(result.unexpected_keys, [])
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(key.startswith(added_prefixes) for key in result.missing_keys))
        for key, original in base.state_dict().items():
            self.assertTrue(torch.equal(original, task.state_dict()[key]), key)

    def policy_pair(self):
        base = Policy(48, 390, hist=10, num_joints=12, action_limit=4.0)
        base.obs_norm.update(0.2 + 0.8 * torch.randn(8, 10, 48))
        base.goal_norm.update(-0.1 + 1.2 * torch.randn(8, 390))
        task = TaskPolicy()
        result = task.load_gait(base.state_dict())
        self.assert_base_loaded(base, task, result, ("task_head.", "task_norm."))
        return base, task

    def policy_inputs(self):
        return torch.randn(3, 10, 48), torch.randn(3, 390), torch.randn(3, 18)

    def world_pair(self):
        base = WorldModel(hidden=(32, 32), num_joints=12)
        with torch.no_grad():
            nn.init.normal_(base.net[-1].weight, std=0.02)
            nn.init.normal_(base.net[-1].bias, std=0.01)
        base.in_norm.update(0.2 + 0.8 * torch.randn(16, base.in_dim))
        base.out_norm.update(0.03 + 0.1 * torch.randn(16, 36))
        task = TaskWorldModel(hidden=(32, 32))
        result = task.load_gait(base.state_dict())
        self.assert_base_loaded(base, task, result, ("bar_adapter.", "bar_norm."))
        return base, task

    def world_inputs(self):
        S = (
            torch.tensor([[0.3, -0.4, 0.3], [0.0, 0.2, 0.29], [-0.2, 0.1, 0.31]]),
            so3_exp(0.3 * torch.randn(3, 3)),
            0.4 * torch.randn(3, 3), 0.2 * torch.randn(3, 3),
            0.2 * torch.randn(3, 12), 0.3 * torch.randn(3, 12),
        )
        c = torch.tensor([
            [2.0, 0.0, 0.8, 0.2, 3.0, 0.0, -1.2, 0.34, 0.0, 0.2, 0.5, 0.29],
            [2.0, 0.0, 0.8, 0.2, 2.0, 0.1, -1.0, 0.34, 1.0, 0.3, 0.5, 0.29],
            [2.0, 0.0, 0.8, 0.2, 2.5, 0.2, -0.8, 0.12, 2.0, 0.4, 0.5, 0.29],
        ])
        return S, torch.randn(3, 12), torch.randn(3, 16), c

    def test_policy_initial_actions_exactly_match_gait(self):
        base, task = self.policy_pair()
        obs, goal, features = self.policy_inputs()
        with torch.no_grad():
            expected = base(obs, goal)
            for condition in (features, features * 5.0 + 2.0):
                actual = task(obs, goal, condition)
                self.assertEqual(actual.shape, (3, 12))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            task.task_norm.update(features)
            torch.testing.assert_close(task(obs, goal, features), expected, rtol=0, atol=0)

    def test_zero_policy_head_has_finite_learning_gradients(self):
        _, task = self.policy_pair()
        obs, goal, features = self.policy_inputs()
        features.requires_grad_()
        weights = torch.arange(1.0, 13.0)
        (task(obs, goal, features) * weights).sum().backward()
        self.assert_finite_nonzero(task.task_head[-1].weight.grad)
        self.assert_finite_nonzero(task.task_head[-1].bias.grad)
        # Zero final weights initially block gradients to earlier task layers.
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features)))
        self.assertTrue(bool((task.task_head[0].weight.grad == 0).all()))

    def test_policy_conditions_affect_actions_and_receive_gradients(self):
        _, task = self.policy_pair()
        with torch.no_grad():
            nn.init.normal_(task.task_head[-1].weight, std=0.1)
        obs, goal, features = self.policy_inputs()
        obs.requires_grad_()
        features.requires_grad_()
        actual = task(obs, goal, features)
        alternate = task(obs, goal, features + 1.5)
        self.assertGreater((actual - alternate).abs().max().item(), 1e-7)
        self.assertTrue(bool((actual.abs() <= 4.0).all()))
        (actual * torch.arange(1.0, 13.0)).sum().backward()
        for grad in (
            features.grad, obs.grad, task.task_head[0].weight.grad,
            task.task_head[-1].weight.grad,
        ):
            self.assert_finite_nonzero(grad)

    def test_policy_action_bound_and_task_checkpoint_roundtrip(self):
        _, task = self.policy_pair()
        obs, goal, features = self.policy_inputs()
        task.task_norm.update(features)
        with torch.no_grad():
            task.dec[-1].weight.zero_()
            task.dec[-1].bias.copy_(torch.linspace(-80.0, 80.0, 12))
            nn.init.normal_(task.task_head[-1].weight, std=0.1)
            expected = task(obs, goal, features)
            self.assertTrue(bool((expected.abs() <= 4.0).all()))
            self.assertEqual(expected[0, 0].item(), -4.0)
            self.assertEqual(expected[0, -1].item(), 4.0)
            restored = TaskPolicy()
            restored.load_state_dict(task.state_dict())
            torch.testing.assert_close(
                restored(obs, goal, features), expected, rtol=0, atol=0,
            )
        for name, value in task.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_world_initial_step_exactly_matches_nonzero_gait_dynamics(self):
        base, task = self.world_pair()
        S, a, xi, c = self.world_inputs()
        task.bar_norm.update(task.task_inputs(S, c))
        with torch.no_grad():
            expected = base.step(S, a, xi)
            actual = task.step(S, a, xi, c)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(expected, S)))
        for original, extended in zip(expected, actual):
            torch.testing.assert_close(original, extended, rtol=0, atol=0)
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, task.state_dict()[name]), name)

    def test_bar_features_use_elapsed_time_yaw_and_relative_velocity(self):
        _, task = self.world_pair()
        S, _, _, c = self.world_inputs()
        p = torch.tensor([[1.0, 2.0, 0.3]]).expand(3, 3)
        R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]]).expand(3, 3, 3)
        v = torch.tensor([[0.25, -0.5, 0.1]]).expand(3, 3)
        S = (p, R, v, *S[3:])
        c[:, 4:8] = torch.tensor([3.0, 4.0, -0.5, 0.5])
        c[:, 9] = 2.0
        expected = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, -1.0, 0.2, 0.5, 0.75, -0.1, 1.0, 0.0],
            [2.0, -1.0, 0.2, 0.5, 0.75, -0.1, 0.0, 1.0],
        ])
        torch.testing.assert_close(task.task_inputs(S, c), expected)
        advanced = task.task_inputs(S, advance_scenario(c, dt=1.0))
        torch.testing.assert_close(advanced[1:, 1], torch.full((2,), -0.5))
        shifted = c.clone()
        shifted[:, :4] += 10.0
        torch.testing.assert_close(task.task_inputs(S, shifted), expected)

    def test_zero_world_adapter_has_finite_learning_gradients(self):
        _, task = self.world_pair()
        S, a, xi, c = self.world_inputs()
        c.requires_grad_()
        prediction = task.step(S, a, xi, c)
        sum(value.square().sum() for value in prediction).backward()
        self.assert_finite_nonzero(task.bar_adapter[-1].weight.grad)
        self.assert_finite_nonzero(task.bar_adapter[-1].bias.grad)
        self.assertTrue(torch.equal(c.grad, torch.zeros_like(c)))
        self.assertTrue(bool((task.bar_adapter[0].weight.grad == 0).all()))

    def test_world_conditions_affect_dynamics_and_receive_gradients(self):
        _, task = self.world_pair()
        with torch.no_grad():
            nn.init.normal_(task.bar_adapter[-1].weight, std=0.05)
        S, a, xi, c = self.world_inputs()
        S = tuple(value.detach().requires_grad_() for value in S)
        a.requires_grad_()
        c.requires_grad_()
        alternate = c.detach().clone()
        alternate[:, 4] += 1.0
        actual = task.step(S, a, xi, c)
        changed = task.step(S, a, xi, alternate)
        difference = sum((x - y).abs().sum() for x, y in zip(actual, changed))
        self.assertGreater(difference.item(), 1e-6)
        sum(value.square().sum() for value in actual).backward()
        for grad in (
            c.grad[1:, 4:8], c.grad[1:, 9], a.grad, S[0].grad, S[1].grad,
            task.bar_adapter[0].weight.grad, task.bar_adapter[-1].weight.grad,
        ):
            self.assert_finite_nonzero(grad)

    def test_world_residual_is_applied_before_increment_denormalization(self):
        base, task = self.world_pair()
        S, a, xi, c = self.world_inputs()
        with torch.no_grad():
            task.bar_adapter[-1].bias[0] = 0.5
            task.bar_adapter[-1].bias[12] = -0.3
            expected = base.step(S, a, xi)
            actual = task.step(S, a, xi, c)
            local_delta = torch.zeros_like(S[0])
            local_delta[:, 0] = 0.5 * base.out_norm.std()[0]
            position_delta = (yaw_mat(S[1]) @ local_delta.unsqueeze(-1)).squeeze(-1)
            joints = expected[4].clone()
            joints[:, 0] -= 0.3 * base.out_norm.std()[12]
            torch.testing.assert_close(actual[0], expected[0] + position_delta)
            torch.testing.assert_close(actual[4], joints)
            for index in (1, 2, 3, 5):
                torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)

    def test_world_rollout_gradients_and_checkpoint_roundtrip(self):
        _, task = self.world_pair()
        S, a, xi, c = self.world_inputs()
        task.bar_norm.update(task.task_inputs(S, c))
        with torch.no_grad():
            nn.init.normal_(task.bar_adapter[-1].weight, std=0.03)
        restored = TaskWorldModel(hidden=(32, 32))
        restored.load_state_dict(task.state_dict())
        for x, y in zip(task.step(S, a, xi, c), restored.step(S, a, xi, c)):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        original_norms = {
            name: value.clone() for name, value in task.named_buffers()
        }
        c.requires_grad_()
        a.requires_grad_()
        for _ in range(3):
            S = task.step(S, a, xi, c)
            c = advance_scenario(c)
        sum(value.square().sum() for value in S).backward()
        self.assert_finite_nonzero(a.grad)
        self.assert_finite_nonzero(task.bar_adapter[-1].weight.grad)
        for name, value in task.named_buffers():
            self.assertTrue(torch.equal(value, original_norms[name]), name)


if __name__ == "__main__":
    unittest.main()

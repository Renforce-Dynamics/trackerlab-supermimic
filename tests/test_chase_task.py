"""Task reward bounds and replay timing are part of the algorithm contract."""

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerTask.supermimic.robots import get_robot
from trackerLab.tracker_env.mdp.supermimic_chase_dodge import (
    ChaseDodgeMDP, advance_scenario, scenario_state,
)
from trackerLab.tracker_env.mdp.supermimic import RewardCfg
from trackerLab.algorithms.supermimic.chase_buffer import ChaseReplayBuffer


class Kinematics:
    robot = get_robot("go2")
    q_lo, q_hi = torch.full((12,), -3.), torch.full((12,), 3.)

    def forward(self, p, R, v, w, q, qd):
        count = len(self.robot.tracked_bodies)
        return (p.unsqueeze(-2).expand(*p.shape[:-1], count, 3),
                R.unsqueeze(-3).expand(*R.shape[:-2], count, 3, 3),
                v.unsqueeze(-2).expand(*v.shape[:-1], count, 3),
                w.unsqueeze(-2).expand(*w.shape[:-1], count, 3))


class ChaseTaskTests(unittest.TestCase):
    def test_scenario_time_and_bar_velocity_match(self):
        c = torch.tensor([[0., 0., .8, .2, 5., .1, -1., .34, 1., 0., .7, .29]])
        c1 = advance_scenario(c, 1.5)
        self.assertEqual(float(c[0, 9]), 0.)
        _, _, bar = scenario_state(c1)
        torch.testing.assert_close(bar, torch.tensor([[3.5, .1, .34]]))
        dt = .001
        target0, velocity, _ = scenario_state(c1)
        target1 = scenario_state(advance_scenario(c1, dt))[0]
        torch.testing.assert_close((target1-target0)/dt, velocity, atol=.0003, rtol=.001)

    def test_reward_upper_bound_and_task_gradients(self):
        torch.manual_seed(11)
        B = 128
        motion = SimpleNamespace(q=torch.zeros(B, 12), root_R=torch.eye(3).expand(B, 3, 3))
        mdp = ChaseDodgeMDP(Kinematics(), motion, torch.zeros(12), cfg=RewardCfg(), obs_mode="relative_orientation")
        p = torch.randn(B, 3, requires_grad=True)
        S = (p, torch.eye(3).expand(B, 3, 3), torch.randn(B, 3),
             torch.randn(B, 3), torch.randn(B, 12), torch.randn(B, 12))
        c = torch.zeros(B, 12)
        c[:, 2], c[:, 4], c[:, 7], c[:, 8], c[:, 10], c[:, 11] = .8, 2., .34, 1., .5, .29
        f, a = torch.arange(B), torch.randn(B, 12)
        reward, terms = mdp.task_reward(S, f, a, torch.zeros_like(a), c, terms=True)
        torch.testing.assert_close(reward, sum(terms.values()))
        self.assertTrue(bool((reward <= .1 + 1e-7).all()))
        reward.sum().backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all()))
        self.assertGreater(float(p.grad.abs().sum()), 0.)
        self.assertEqual(mdp.task_features(S, c).shape, (B, 18))

    def test_flat_scenarios_have_zero_bar_penalty(self):
        B = 2
        motion = SimpleNamespace(q=torch.zeros(B, 12), root_R=torch.eye(3).expand(B, 3, 3))
        mdp = ChaseDodgeMDP(Kinematics(), motion, torch.zeros(12), obs_mode="relative_orientation")
        S = (torch.zeros(B, 3), torch.eye(3).expand(B, 3, 3),
             torch.zeros(B, 3), torch.zeros(B, 3), torch.zeros(B, 12), torch.zeros(B, 12))
        c = torch.zeros(B, 12)
        torch.testing.assert_close(mdp.collision_probability(S, c), torch.zeros(B))

    def test_scenario_history_stays_within_episode_after_ring_wrap(self):
        buffer = ChaseReplayBuffer(4, 1, "cpu")
        for index in range(9):
            data = {k: torch.zeros_like(v[0]) for k, v in buffer.data.items()}
            ep = index % 3
            data["ep_step"].fill_(ep)
            data["scenario"].fill_(index//3)
            data["scenario1"].fill_(index//3)
            data["done"].fill_(ep == 2)
            buffer.add(data)
        start, history = buffer.sample_starts(64, 10)
        torch.testing.assert_close(history["scenario"][:, :, 0], start["scenario"][:, :1].expand(-1, 10))


if __name__ == "__main__":
    unittest.main()

"""Sole swing shaping must reject dragging, held-leg and flight shortcuts."""

from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from trackerLab.algorithms.supermimic.stealth_gait import command_gait_penalty


def penalty(heights, velocities, *, command=None, elapsed=3.):
    if command is None:
        command = heights.new_tensor([0., .25, 0., 0., .70, 0., elapsed]).expand(
            *heights.shape[:-2], 7)
    return command_gait_penalty(
        heights, velocities, command, scale=1., weight=.02,
        foot_slide_weight=.04, entry_duration_s=.60)


def test_single_moving_swing_is_preferred_to_dragging_or_a_static_held_leg():
    heights = torch.zeros(4, 2, 4)
    heights[1:3, 1] = .04
    heights[3] = .05
    velocity = torch.zeros(4, 2, 4, 3)
    velocity[0, ..., 0] = .25
    velocity[1, 1, :, 0] = .5
    velocity[3, ..., 0] = .25
    reward = penalty(heights, velocity)
    assert reward[1] > reward[0] and reward[1] > reward[2]
    assert reward[2] <= -.02
    assert reward[3] < reward[0]-.04*.25
    assert (reward <= 0).all()


def test_slip_cutoff_cannot_improve_total_reward_even_at_large_drag_speed():
    for speed in (.1, .25, 1., 4.):
        for elapsed in (0., .1, .3, .6, 3.):
            heights = torch.tensor([.019999, .020001])[:, None, None].expand(2, 2, 4)
            velocity = torch.zeros(2, 2, 4, 3)
            velocity[..., 0] = speed
            gait = penalty(heights, velocity, elapsed=elapsed)
            total = gait + torch.tensor([-.04*speed, 0.])
            assert total[1] <= total[0]+1e-7


def test_idle_has_no_gait_cost_and_pivots_are_enabled():
    heights = torch.zeros(4, 2, 4)
    velocity = torch.zeros(4, 2, 4, 3)
    c = torch.tensor([[0., 0., 0., yaw, .7, 0., 3.]
                      for yaw in (0., .55, -.55, 0.)])
    c[3, 1] = -.20
    reward = penalty(heights, velocity, command=c)
    assert reward[0] == 0 and (reward[1:] < 0).all()
    torch.testing.assert_close(reward[1], reward[2])


def test_sole_lift_has_finite_ascent_gradient_with_a_supported_other_foot():
    height = torch.tensor([[[0.]*4, [.01]*4]], requires_grad=True)
    velocity = torch.zeros(1, 2, 4, 3)
    velocity[:, 1, :, 0] = .4
    gh, gv = torch.autograd.grad(penalty(height, velocity.requires_grad_()).sum(),
                                (height, velocity))
    assert torch.isfinite(gh).all() and torch.isfinite(gv).all()
    assert gh[:, 1].sum() > 0
    assert gh[:, 0].sum() < 0


def test_whole_sole_and_zero_startup_are_respected():
    heights = torch.full((2, 2, 4), .04)
    heights[:, 0] = 0.
    heights[0, 1, 0] = 0.
    velocity = torch.zeros(2, 2, 4, 3)
    velocity[:, 1, :, 0] = .4
    reward = penalty(heights, velocity)
    assert reward[1] > reward[0]
    torch.testing.assert_close(penalty(heights, velocity, elapsed=0.), torch.zeros(2))


def test_pure_foot_rotation_does_not_receive_translational_swing_relief():
    heights = torch.zeros(2, 2, 4)
    heights[:, 1] = .04
    velocity = torch.zeros(2, 2, 4, 3)
    velocity[1, 1, :2, 0] = .5
    velocity[1, 1, 2:, 0] = -.5
    torch.testing.assert_close(penalty(heights, velocity)[0], penalty(heights, velocity)[1])

"""Command-gated sole swing/support shaping without a prescribed gait clock."""

import torch

from trackerLab.tracker_env.mdp.supermimic_command import entry_blend


def command_gait_penalty(heights, velocities, command, *, scale, weight,
                         foot_slide_weight, entry_duration_s, version=1,
                         swing_height=.04):
    """Nonpositive shaping; heights (...,2,P), velocities (...,2,P,3).

    Whole-sole clearance and opposite support increase moving-foot relief.
    Static single-leg holds receive no swing relief. Flight
    pays the slip cost it otherwise avoids, in addition to a larger base cost.
    Alternation remains learned from physics and the motion prior.
    """
    clearance = heights.amin(-1)
    if version == 3:
        # A planted sole is fully supported; the tail preserves landing gradients.
        gap = torch.relu(clearance / scale - .004)
        width = gap.new_tensor(.006)
        support = (width / torch.hypot(gap, width)).square()
    else:
        support = torch.sigmoid((.008 * scale - clearance) / (.004 * scale))
    point_speed_squared = velocities[..., :2].square().sum(-1) / scale**2
    mean_velocity = velocities[..., :2].mean(-2) / scale
    swing = 1. - torch.exp(-mean_velocity.square().sum(-1) / .15**2)
    left, right = support.unbind(-1)
    swing_left, swing_right = swing.unbind(-1)
    single_swing = left * (1. - right) * swing_right + right * (1. - left) * swing_left
    flight = (1. - left) * (1. - right)
    maximum_speed = torch.sqrt(point_speed_squared.amax((-1, -2)) + 1e-8)
    cost = weight * (1. - single_swing + 4. * flight)
    if version in (2, 3):
        # BeyondAMP's velocity-weighted clearance objective, normalized to a
        # whole-sole target and bounded to preserve the command value ceiling.
        error = clearance / scale - swing_height
        height_cost = (error / torch.hypot(error, error.new_tensor(swing_height))).square()
        speed_gate = torch.tanh(2. * torch.linalg.vector_norm(mean_velocity, dim=-1))
        cost = cost + 2. * weight * (height_cost * speed_gate).sum(-1)
    airborne = (heights >= .02 * scale).all(-1).all(-1)
    moving = ((torch.linalg.vector_norm(command[..., 1:3], dim=-1) / scale > .1)
              | (command[..., 3].abs() > .1))
    flight_cost = foot_slide_weight * maximum_speed * airborne.to(cost.dtype)
    return -(cost * entry_blend(command, entry_duration_s) + flight_cost) * moving.to(cost.dtype)

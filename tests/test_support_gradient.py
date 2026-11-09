"""Isolated production support gradients and optional configured-G1 checks."""

from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.locomotion_quality import LocomotionQuality
from trackerTask.supermimic.gait import FOOT_BODIES


class Plane:
    scale = 1.

    def sample(self, xy):
        return xy[..., 0] * 0


def point_quality(differentiate_support=False):
    quality = LocomotionQuality.__new__(LocomotionQuality)
    quality.field = Plane()
    quality.foot_slip_mode = "g1_sole"
    quality.differentiate_support = differentiate_support
    quality.biped = True
    quality.sole_centers = torch.zeros(2, 1, 3, dtype=torch.float64)
    quality.sole_radii = torch.zeros(2, 1, dtype=torch.float64)
    quality.foot_slide_weight = .008
    quality.feet_close_weight = .008
    quality.default_q = quality.kp = quality.kd = torch.zeros(6, dtype=torch.float64)
    quality.action_scale = torch.ones(6, dtype=torch.float64)
    quality.qlo = -torch.ones(6, dtype=torch.float64)
    quality.qhi = quality.effort = quality.velocity_limit = torch.ones(6, dtype=torch.float64)

    def foot_kinematics(state):
        p, rotation, velocity, angular, _, _ = state
        position = p.reshape(1, 2, 3)
        return (position, rotation[:, None].expand(1, 2, 3, 3),
                velocity.reshape(1, 2, 3), angular[:, None].expand(1, 2, 3))

    quality._foot_kinematics = foot_kinematics
    return quality


def point_state(height=.01, *, grounded=0., speeds=(0., 2.)):
    return (
        torch.tensor([[0., .1, grounded, 0., -.1, height]],
                     dtype=torch.float64, requires_grad=True),
        torch.eye(3, dtype=torch.float64)[None],
        torch.tensor([[speeds[0], 0., 0., speeds[1], 0., 0.]],
                     dtype=torch.float64, requires_grad=True),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 6, dtype=torch.float64),
        torch.zeros(1, 6, dtype=torch.float64),
    )


def reward(quality, state):
    return quality.reward(state, state, torch.zeros_like(state[4]),
                          state[4].new_zeros(1, 10)).sum()


@pytest.mark.parametrize("enabled", [False, True])
def test_height_finite_difference_and_autograd(enabled):
    quality = point_quality(enabled)
    state = point_state()
    actual, = torch.autograd.grad(reward(quality, state), state[0])
    eps = 1e-7
    finite_difference = (reward(quality, point_state(.01 + eps))
                         - reward(quality, point_state(.01 - eps))) / (2 * eps)
    torch.testing.assert_close(finite_difference, finite_difference.new_tensor(8 / 15),
                               atol=1e-9, rtol=0)
    torch.testing.assert_close(actual[0, 5], actual.new_tensor(8 / 15 if enabled else 0.),
                               atol=1e-12, rtol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("height", [-.01, 0., .01, .02, .03])
def test_forward_values_and_velocity_gradients_are_identical(height):
    state = point_state(height)
    detached, smooth = point_quality(), point_quality(True)
    for name, expected in detached.measurements(state).items():
        torch.testing.assert_close(smooth.measurements(state)[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(reward(smooth, state), reward(detached, state), rtol=0, atol=0)
    gradients = [torch.autograd.grad(reward(q, state), state[2])[0]
                 for q in (detached, smooth)]
    torch.testing.assert_close(*gradients, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("height", [0., .01, .0199])
def test_equal_speed_plateau_has_zero_height_gradient(enabled, height):
    quality, state = point_quality(enabled), point_state(height, speeds=(.4, .4))
    slip = quality.measurements(state)["support_foot_slip_scale_m_s"]
    torch.testing.assert_close(slip, slip.new_tensor([.4]), rtol=0, atol=1e-16)
    gradient, = torch.autograd.grad(reward(quality, state), state[0])
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=1e-14)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("height", [.02, .03, 1.])
def test_air_has_zero_reward_and_gradients(enabled, height):
    quality = point_quality(enabled)
    state = point_state(height, grounded=height, speeds=(2., 9.))
    metrics = quality.measurements(state)
    assert torch.count_nonzero(metrics["sole_support_heuristic"]) == 0
    assert metrics["support_foot_slip_scale_m_s"].item() == 0.
    value = reward(quality, state)
    assert value.item() == 0.
    for gradient in torch.autograd.grad(value, (state[0], state[2])):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("height", [-1e-8, 0., 1e-8, .02 - 1e-8, .02, .02 + 1e-8])
def test_support_boundary_gradients_are_finite(height):
    quality, state = point_quality(True), point_state(height)
    for gradient in torch.autograd.grad(reward(quality, state), (state[0], state[2])):
        assert torch.isfinite(gradient).all()
    if height in (0., .02):
        gradient, = torch.autograd.grad(reward(quality, state), state[0])
        assert gradient[0, 5].item() == 0.


def test_new_style_probes_missing_flag_retain_detached_support():
    quality, state = point_quality(), point_state()
    del quality.differentiate_support
    assert not quality.measurements(state)["sole_support_heuristic"].requires_grad
    gradient, = torch.autograd.grad(reward(quality, state), state[0])
    assert gradient[0, 5].item() == 0.


def constructor_kin(robot="g1"):
    bodies = FOOT_BODIES[robot]
    return SimpleNamespace(
        robot=SimpleNamespace(name=robot, default_q=(0.,) * 6,
                              action_scale=(.25,) * 6, mjcf_path=None),
        device="cpu", body_names=bodies, tracked_idx=tuple(range(len(bodies))),
        q_lo=torch.full((6,), -10.), q_hi=torch.full((6,), 10.))


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], torch.tensor(True)])
def test_constructor_requires_boolean(value):
    with pytest.raises(ValueError, match="differentiate_support must be a bool"):
        LocomotionQuality(constructor_kin(), Plane(), [], differentiate_support=value)


def test_constructor_opt_in_requires_g1_sole():
    with pytest.raises(ValueError, match="requires foot_slip_mode='g1_sole'"):
        LocomotionQuality(constructor_kin(), Plane(), [], differentiate_support=True)
    with pytest.raises(ValueError, match="only for G1"):
        LocomotionQuality(constructor_kin("go2"), Plane(), [],
                          foot_slip_mode="g1_sole", differentiate_support=True)
    geometry = (torch.zeros(2, 1, 3), torch.zeros(2, 1))
    with patch("trackerLab.algorithms.supermimic.locomotion_quality._g1_sole_geometry",
               return_value=geometry):
        for enabled in (False, True):
            quality = LocomotionQuality(constructor_kin(), Plane(), [(20., 12., 0., 0.)] * 6,
                                        foot_slip_mode="g1_sole", differentiate_support=enabled)
            assert quality.differentiate_support is enabled


@pytest.mark.parametrize("robot", list(FOOT_BODIES))
def test_explicit_false_preserves_legacy_for_all_robots(robot):
    kin, state = constructor_kin(robot), point_state()
    count = len(kin.body_names)
    position = torch.zeros(1, count, 3, dtype=torch.float64)
    position[..., 1] = torch.linspace(.1, -.1, count)
    rotation = torch.eye(3, dtype=torch.float64).expand(1, count, 3, 3)
    velocity, angular = torch.ones_like(position) * .4, torch.zeros_like(position)
    kin.forward = lambda *args: (position, rotation, velocity, angular)
    default = LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * 6)
    explicit = LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * 6,
                                differentiate_support=False)
    for name, expected in default.measurements(state).items():
        torch.testing.assert_close(explicit.measurements(state)[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(reward(explicit, state), reward(default, state), rtol=0, atol=0)


def test_optional_actual_g1_support_gradient():
    from trackerLab.kinematics.supermimic.robot import RobotKinematics
    from trackerTask.supermimic.robots import get_robot

    spec = get_robot("g1")
    if not Path(spec.mjcf_path).is_file():
        pytest.skip("Configured G1 assets are optional; no downloads")
    spec = replace(spec, mjcf_path=os.path.relpath(spec.mjcf_path))
    kin = RobotKinematics("cpu", spec)
    qualities = [LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * spec.num_joints,
                                   foot_slip_mode="g1_sole", differentiate_support=enabled)
                 for enabled in (False, True)]
    state = [torch.tensor([[0., 0., .8]]), torch.eye(3)[None],
             torch.zeros(1, 3), torch.zeros(1, 3),
             torch.tensor(spec.default_q)[None], torch.zeros(1, spec.num_joints)]
    state[5][0, spec.joint_names.index("right_hip_pitch_joint")] = 1.
    points, _ = qualities[0].sole_state(state)
    state[0][0, 2] += .01 - points[..., 2].amin()
    state[0].requires_grad_()
    state[4].requires_grad_()
    action = torch.zeros_like(state[4])
    xi = torch.zeros(1, spec.num_joints + 4)
    values = [q.reward(state, state, action, xi).sum() for q in qualities]
    torch.testing.assert_close(*values, rtol=0, atol=0)
    derivatives = [torch.autograd.grad(value, (state[0], state[4])) for value in values]
    for root_gradient, joint_gradient in derivatives:
        assert torch.isfinite(root_gradient).all()
        assert torch.isfinite(joint_gradient).all()
    assert derivatives[0][0][0, 2].item() == 0.
    assert abs(derivatives[1][0][0, 2].item()) > 1e-6
    eps = 1e-4
    displaced = []
    for delta in (eps, -eps):
        shifted = list(state)
        shifted[0] = state[0].detach().clone()
        shifted[0][0, 2] += delta
        displaced.append(qualities[1].reward(shifted, shifted, action, xi).sum())
    finite_difference = (displaced[0] - displaced[1]) / (2 * eps)
    torch.testing.assert_close(derivatives[1][0][0, 2], finite_difference,
                               rtol=.02, atol=2e-5)
    arms = [i for i, name in enumerate(spec.joint_names)
            if any(part in name for part in ("shoulder", "elbow", "wrist"))]
    torch.testing.assert_close(derivatives[0][1][:, arms], derivatives[1][1][:, arms],
                               rtol=0, atol=0)

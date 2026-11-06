"""CPU sole-contact checks using a tiny MJCF and optional local G1 assets."""

from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.locomotion_quality import LocomotionQuality
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.kinematics.supermimic.rotations import so3_exp
from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS
from trackerTask.supermimic.robots import get_robot


CENTERS = ((-.05, .025, -.03), (-.05, -.025, -.03),
           (.12, .03, -.03), (.12, -.03, -.03))


def sole_xml(geom_type="sphere"):
    # These positions/radii come from this repository's configured G1 MJCF.
    geoms = "".join(f'<geom class="foot" pos="{x} {y} {z}"/>' for x, y, z in CENTERS)
    bodies = "".join(
        f'<body name="{name}"><freejoint/>{geoms}'
        '<geom type="box" size=".2 .1 .1" contype="0" conaffinity="0"/>'
        '</body>' for name in FOOT_BODIES["g1"])
    return (
        '<mujoco><default><default class="foot">'
        f'<geom type="{geom_type}" size=".005 .005 .005" contype="1" conaffinity="1"/>'
        f'</default></default><worldbody>{bodies}</worldbody></mujoco>')


class RigidFeetKinematics:
    """Two translating feet with root rotation and world angular velocity."""

    device = "cpu"
    body_names = FOOT_BODIES["g1"]
    tracked_idx = (0, 1)
    q_lo, q_hi = torch.full((6,), -10.), torch.full((6,), 10.)

    def __init__(self, path):
        self.robot = SimpleNamespace(name="g1", default_q=(0.,) * 6,
                                     action_scale=(.25,) * 6, mjcf_path=path)

    def forward(self, p, R, v, w, q, qd):
        local = p.new_tensor(((0., .1, 0.), (0., -.1, 0.))) + q.reshape(*q.shape[:-1], 2, 3)
        arm = (R[..., None, :, :] @ local[..., None]).squeeze(-1)
        position = p[..., None, :] + arm
        angular = w[..., None, :].expand_as(position)
        joint_v = (R[..., None, :, :] @ qd.reshape(*qd.shape[:-1], 2, 3, 1)).squeeze(-1)
        velocity = v[..., None, :] + torch.cross(angular, arm, dim=-1) + joint_v
        rotation = R[..., None, :, :].expand(*R.shape[:-2], 2, 3, 3)
        return position, rotation, velocity, angular


class Plane:
    scale = 1.

    def __init__(self, height=0., slope=0.):
        self.height, self.slope = height, slope

    def sample(self, xy):
        return self.height + self.slope * xy[..., 0]


class SoleQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.path = Path(cls.directory.name) / "sole.xml"
        cls.path.write_text(sole_xml(), encoding="ascii")

    def quality(self, *, mode="g1_sole", field=None, foot_slide_weight=.008,
                feet_close_weight=.008):
        # Zero PD gains isolate geometric rewards from actuator regularization.
        return LocomotionQuality(RigidFeetKinematics(self.path), field or Plane(),
                                 [(20., 12., 0., 0.)] * 6,
                                 foot_slip_mode=mode, foot_slide_weight=foot_slide_weight,
                                 feet_close_weight=feet_close_weight)

    def state(self, *, dtype=torch.float64):
        return (torch.tensor([[0., 0., .035]], dtype=dtype), torch.eye(3, dtype=dtype)[None],
                torch.zeros(1, 3, dtype=dtype), torch.zeros(1, 3, dtype=dtype),
                torch.zeros(1, 6, dtype=dtype), torch.zeros(1, 6, dtype=dtype))

    def reward(self, quality, state):
        return quality.reward(state, state, torch.zeros_like(state[4]),
                              state[0].new_zeros((*state[0].shape[:-1], 10)))

    def test_compiled_geometry_resolves_defaults_and_ignores_visuals(self):
        quality = self.quality()
        expected = torch.tensor(CENTERS).expand(2, 4, 3)
        torch.testing.assert_close(quality.sole_centers, expected)
        torch.testing.assert_close(quality.sole_radii, torch.full((2, 4), .005))
        points, velocity = quality.sole_state(self.state())
        torch.testing.assert_close(points[..., 2], torch.zeros_like(points[..., 2]), atol=2e-9, rtol=0)
        torch.testing.assert_close(velocity, torch.zeros_like(velocity))

    def test_planted_sole_has_zero_slip_while_ankles_rotate_and_translate(self):
        quality = self.quality()
        state = list(self.state())
        state[3][..., 1] = 2.
        state[2][..., 0] = .07  # Cancels omega x (center - radius * world_up) in XY.
        _, velocity = quality.sole_state(state)
        torch.testing.assert_close(velocity[..., :2], torch.zeros_like(velocity[..., :2]),
                                   atol=3e-9, rtol=0)
        self.assertLess(quality.measurements(state)["support_foot_slip_scale_m_s"].item(), 3e-9)
        self.assertGreater(self.quality(mode="legacy").measurements(state)[
            "support_foot_slip_scale_m_s"].item(), .02)

    def test_material_velocity_includes_sphere_radius_and_angular_motion(self):
        quality = self.quality()
        state = list(self.state())
        state[1] = so3_exp(torch.tensor([[.24, -.17, .5]], dtype=torch.float64))
        state[3][...] = torch.tensor([[.8, -.3, .7]], dtype=torch.float64)
        body_p, _, body_v, body_w = quality.kin.forward(*state)
        points, velocity = quality.sole_state(state)
        expected = body_v[..., None, :] + torch.cross(
            body_w[..., None, :].expand_as(points), points - body_p[..., None, :], dim=-1)
        torch.testing.assert_close(velocity, expected)
        # Freeze the sampled material point for a finite-difference check.
        arm = points - body_p[..., None, :]
        eps = 1e-6
        step_R = so3_exp(state[3] * eps)
        plus = (body_p[..., None, :] + body_v[..., None, :] * eps
                + (step_R[..., None, None, :, :] @ arm[..., None]).squeeze(-1))
        minus = (body_p[..., None, :] - body_v[..., None, :] * eps
                 + (step_R.transpose(-1, -2)[..., None, None, :, :] @ arm[..., None]).squeeze(-1))
        torch.testing.assert_close(velocity, (plus - minus) / (2 * eps), atol=2e-10, rtol=1e-8)

    def test_stationary_ankles_with_yaw_rotation_have_sole_slip(self):
        quality = self.quality()
        state = list(self.state())
        state[3][..., 2] = 1.
        # Counteract the root rotation's body-origin velocity using foot joints.
        local = torch.tensor([[[0., .1, 0.], [0., -.1, 0.]]], dtype=torch.float64)
        state[5] = -torch.cross(state[3][..., None, :].expand_as(local), local, dim=-1).reshape(1, 6)
        _, _, ankle_v = quality.foot_state(state)
        torch.testing.assert_close(ankle_v, torch.zeros_like(ankle_v))
        self.assertGreater(quality.measurements(state)["support_foot_slip_scale_m_s"].item(), .05)

    def test_material_velocity_pose_and_rate_gradients_match_finite_differences(self):
        quality = self.quality()
        p = self.state()[0]
        rotation_vector = torch.tensor([[.2, -.3, .5]], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([[.3, -.2, .1]], dtype=torch.float64, requires_grad=True)
        w = torch.tensor([[.4, .6, -.7]], dtype=torch.float64, requires_grad=True)
        q = torch.linspace(-.1, .1, 6, dtype=torch.float64).reshape(1, 6).requires_grad_()
        qd = torch.linspace(-.4, .5, 6, dtype=torch.float64).reshape(1, 6).requires_grad_()

        def material_velocity(r, v, w, q, qd):
            return quality.sole_state((p, so3_exp(r), v, w, q, qd))[1]

        self.assertTrue(torch.autograd.gradcheck(material_velocity, (rotation_vector, v, w, q, qd)))

    def test_horizontal_support_speed_has_finite_nonzero_reward_gradient(self):
        quality = self.quality()
        state = list(self.state())
        state[2] = torch.tensor([[.3, .4, 0.]], dtype=torch.float64, requires_grad=True)
        slip = quality.measurements(state)["support_foot_slip_scale_m_s"]
        torch.testing.assert_close(slip, torch.tensor([.5], dtype=torch.float64))
        gradient, = torch.autograd.grad(self.reward(quality, state).sum(), state[2])
        torch.testing.assert_close(gradient, torch.tensor([[-.0048, -.0064, 0.]], dtype=torch.float64))
        self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_stationary_and_airborne_states_have_exact_zero_slip_and_finite_gradients(self):
        quality = self.quality()
        for height, speed in ((.035, 0.), (.06, .4), (.5, 1.)):
            with self.subTest(height=height, speed=speed):
                state = list(self.state())
                state[0][..., 2] = height
                state[2] = torch.tensor([[speed, 0., 0.]], dtype=torch.float64, requires_grad=True)
                slip = quality.measurements(state)["support_foot_slip_scale_m_s"]
                self.assertEqual(slip.item(), 0.)
                gradient, = torch.autograd.grad(slip.sum(), state[2])
                torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_one_weakly_supported_point_has_no_support_mass_discount(self):
        quality = self.quality()
        state = list(self.state())
        state[1] = so3_exp(torch.tensor([[.4, .3, 0.]], dtype=torch.float64))
        # Lift the right foot, leaving a single corner on the left near cutoff.
        state[4][..., 5] = .5
        points, _ = quality.sole_state(state)
        state[0][..., 2] += .019 - points[..., 0, :, 2].amin(-1)
        state[2][..., 0] = .4
        metrics = quality.measurements(state)
        support = metrics["sole_support_heuristic"]
        self.assertEqual(torch.count_nonzero(support).item(), 1)
        self.assertGreater(support.sum().item(), 0)
        self.assertLess(support.sum().item(), .01)
        self.assertAlmostEqual(metrics["support_foot_slip_scale_m_s"].item(), .4)
        state[2].requires_grad_()
        gradient, = torch.autograd.grad(
            quality.measurements(state)["support_foot_slip_scale_m_s"].sum(), state[2])
        torch.testing.assert_close(gradient, torch.tensor([[1., 0., 0.]], dtype=torch.float64))

    def test_fast_airborne_foot_does_not_dilute_or_increase_supported_slip(self):
        quality = self.quality()
        state = list(self.state())
        state[4][..., 5] = .3
        state[5][..., 0], state[5][..., 3] = .4, 9.
        metrics = quality.measurements(state)
        self.assertAlmostEqual(metrics["support_foot_slip_scale_m_s"].item(), .4)
        self.assertEqual(metrics["sole_support_heuristic"][..., 1, :].sum().item(), 0.)

    def test_reward_has_no_height_or_slope_gradient_through_support_gates(self):
        quality = self.quality(field=Plane(slope=.01), feet_close_weight=0.)
        state = list(self.state())
        state[0] = torch.tensor([[0., 0., .045]], dtype=torch.float64, requires_grad=True)
        state[4] = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
        state[5] = torch.tensor([[.2, 0., 0., .7, 0., 0.]], dtype=torch.float64)
        metrics = quality.measurements(state)
        self.assertFalse(metrics["sole_support_heuristic"].requires_grad)
        self.assertFalse(metrics["support_heuristic"].requires_grad)
        gradients = torch.autograd.grad(self.reward(quality, state).sum(), (state[0], state[4]))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_crossing_feet_never_improves_spacing_reward(self):
        quality = self.quality()
        rewards = []
        for spacing in (.2, .05, 0., -.05, -.2):
            state = list(self.state())
            state[4][..., 1] = (spacing - .2) / 2
            state[4][..., 4] = -state[4][..., 1]
            state[4].requires_grad_()
            metrics = quality.measurements(state)
            self.assertAlmostEqual(metrics["foot_separation_scale_m"].item(), abs(spacing))
            self.assertAlmostEqual(metrics["foot_lateral_separation_scale_m"].item(), spacing)
            reward = self.reward(quality, state)
            rewards.append(reward.item())
            if spacing <= 0:
                gradient, = torch.autograd.grad(reward.sum(), state[4])
                self.assertGreater(gradient[0, 1].item(), 0.)
                self.assertLess(gradient[0, 4].item(), 0.)
        self.assertTrue(all(a > b for a, b in zip(rewards, rewards[1:])))

    def test_signed_spacing_uses_heading_independent_of_yaw_and_pitch(self):
        quality = self.quality()
        state = list(self.state())
        state[1] = (so3_exp(torch.tensor([[0., 0., 1.3]], dtype=torch.float64))
                    @ so3_exp(torch.tensor([[0., .6, 0.]], dtype=torch.float64)))
        self.assertAlmostEqual(quality.measurements(state)["foot_lateral_separation_scale_m"].item(), .2)

    def test_geometry_mode_ignores_reference_and_saved_ankle_calibration(self):
        quality = self.quality()
        state = list(self.state())
        state[2][..., 0] = .4
        expected = quality.measurements(state)
        quality.sole_height.fill_(.5)
        actual = quality.measurements(state)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])
        quality.calibrate(SimpleNamespace())
        torch.testing.assert_close(quality.sole_height, torch.zeros(2))

    def test_batched_shapes_scaling_and_existing_weight_controls(self):
        field = Plane(height=.2)
        field.scale = 2.
        quality = self.quality(field=field)
        state = list(self.state(dtype=torch.float32))
        state[0][..., 2], state[2][..., 0] = .235, .4
        batch = tuple(value.expand(3, 2, *value.shape[1:]).clone() for value in state)
        metrics = quality.measurements(batch)
        torch.testing.assert_close(metrics["support_foot_slip_scale_m_s"], torch.full((3, 2), .2))
        self.assertEqual(metrics["sole_support_heuristic"].shape, (3, 2, 2, 4))
        self.assertEqual(quality.sole_state(batch)[0].shape, (3, 2, 2, 4, 3))
        weighted = self.quality(field=field, foot_slide_weight=.016)
        torch.testing.assert_close(self.reward(quality, batch) - self.reward(weighted, batch),
                                   .008 * metrics["support_foot_slip_scale_m_s"])

    def test_geometry_errors_are_explicit_and_legacy_never_reads_assets(self):
        kin = RigidFeetKinematics(self.path)
        for mode in ("sole", "", None):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "foot_slip_mode"):
                LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * 6, foot_slip_mode=mode)
        kin.robot.name = "go2"
        with self.assertRaisesRegex(ValueError, "only for G1"):
            LocomotionQuality(kin, Plane(), [], foot_slip_mode="g1_sole")
        kin.robot.name, kin.robot.mjcf_path = "g1", None
        with self.assertRaisesRegex(ValueError, "mjcf_path"):
            LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * 6, foot_slip_mode="g1_sole")
        unsupported = Path(self.directory.name) / "box.xml"
        unsupported.write_text(sole_xml("box"), encoding="ascii")
        kin.robot.mjcf_path = unsupported
        with self.assertRaisesRegex(ValueError, "spherical collision"):
            LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * 6, foot_slip_mode="g1_sole")
        with patch("mujoco.MjModel.from_xml_path", side_effect=AssertionError("asset read")):
            default = LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * 6)
            self.assertEqual(default.foot_slip_mode, "legacy")
            with self.assertRaisesRegex(ValueError, "sole_state requires"):
                default.sole_state(self.state())

    def test_legacy_formula_and_offsets_are_preserved_for_every_robot(self):
        state = self.state(dtype=torch.float32)
        for name, bodies in FOOT_BODIES.items():
            with self.subTest(robot=name):
                count = len(bodies)
                p = torch.zeros(1, count, 3)
                p[..., 2] = .01
                p[..., 1] = torch.linspace(.1, -.1, count)
                R = torch.eye(3).expand(1, count, 3, 3)
                v, w = torch.full_like(p, .4), torch.full_like(p, .2)
                kin = SimpleNamespace(
                    robot=SimpleNamespace(name=name, default_q=(0.,) * 6, action_scale=(.25,) * 6),
                    device="cpu", body_names=bodies, tracked_idx=tuple(range(count)),
                    q_lo=torch.full((6,), -10.), q_hi=torch.full((6,), 10.),
                    forward=lambda *args: (p, R, v, w))
                quality = LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * 6)
                offset = p.new_tensor(FOOT_OFFSETS.get(name, (0., 0., 0.))).expand_as(p)
                point = p + offset
                velocity = v + torch.cross(w, offset, dim=-1)
                support = torch.sigmoid((.02 - point[..., 2]) / .01)
                speed = torch.sqrt(velocity[..., :2].square().sum(-1) + 1e-8)
                slip = (speed * support).sum(-1) / support.sum(-1).clamp_min(1)
                measured = quality.measurements(state)
                torch.testing.assert_close(measured["support_foot_slip_scale_m_s"], slip, rtol=0, atol=0)
                torch.testing.assert_close(measured["support_heuristic"], support, rtol=0, atol=0)
                self.assertNotIn("sole_support_heuristic", measured)
                self.assertNotIn("foot_lateral_separation_scale_m", measured)
                if count == 2:
                    spacing = (point[..., 0, 1] - point[..., 1, 1]).abs()
                    torch.testing.assert_close(measured["foot_separation_scale_m"], spacing, rtol=0, atol=0)
                expected_reward = -.008 * slip - .001 * state[3][..., :2].square().sum(-1)
                torch.testing.assert_close(self.reward(quality, state), expected_reward, rtol=0, atol=0)

    def test_optional_actual_g1_geometry_and_mujoco_point_jacobians(self):
        spec = get_robot("g1")
        if not Path(spec.mjcf_path).is_file():
            self.skipTest("Configured G1 assets are optional; no downloads")
        # Relative paths also work with Windows MuJoCo builds using ANSI fopen.
        spec = replace(spec, mjcf_path=os.path.relpath(spec.mjcf_path))
        kin = RobotKinematics("cpu", spec)
        quality = LocomotionQuality(kin, Plane(), [(20., 12., 0., 0.)] * spec.num_joints,
                                     foot_slip_mode="g1_sole")
        torch.testing.assert_close(quality.sole_centers, torch.tensor(CENTERS).expand(2, 4, 3))
        torch.testing.assert_close(quality.sole_radii, torch.full((2, 4), .005))
        model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
        data = mujoco.MjData(model)
        data.qpos[:3] = (.1, -.2, .76)
        data.qpos[3:7] = np.array((.98, .1, -.05, .14)) / np.linalg.norm((.98, .1, -.05, .14))
        for name, value in zip(spec.joint_names, spec.default_q):
            data.qpos[model.joint(name).qposadr[0]] = value
        data.qvel[:] = np.linspace(-.4, .5, model.nv)
        mujoco.mj_forward(model, data)
        root = model.body(spec.root_body).id
        root_R = data.xmat[root].reshape(3, 3)
        joint_q = [data.qpos[model.joint(name).qposadr[0]] for name in spec.joint_names]
        joint_v = [data.qvel[model.joint(name).dofadr[0]] for name in spec.joint_names]
        state = tuple(torch.tensor(np.asarray(value).copy(), dtype=torch.float32)[None] for value in (
            data.xpos[root], root_R, data.qvel[:3], root_R @ data.qvel[3:6], joint_q, joint_v))
        points, velocity = quality.sole_state(state)
        jacobian = np.zeros((3, model.nv))
        for foot, name in enumerate(FOOT_BODIES["g1"]):
            body = model.body(name).id
            for point in range(4):
                actual_point = points[0, foot, point].detach().numpy().astype(np.float64)
                expected_point = (data.xpos[body] + data.xmat[body].reshape(3, 3) @
                                  np.asarray(CENTERS[point]) - np.array((0., 0., .005)))
                np.testing.assert_allclose(actual_point, expected_point, atol=2e-7, rtol=1e-6)
                mujoco.mj_jac(model, data, jacobian, None, actual_point, body)
                np.testing.assert_allclose(velocity[0, foot, point].detach().numpy(),
                                           jacobian @ data.qvel, atol=3e-7, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()

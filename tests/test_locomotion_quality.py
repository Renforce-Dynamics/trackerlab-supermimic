"""Foot geometry, regularization gradients and data-derived velocity coverage."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from command_envelope import fixed_command_envelope, reference_envelope, validate_envelope
from trackerLab.algorithms.supermimic.locomotion_quality import LocomotionQuality
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField
from trackerTask.supermimic.terrain import generate


class TestKinematics:
    device = "cpu"
    robot = SimpleNamespace(name="g1", default_q=(0., 0.), action_scale=(.25, .25))
    body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    tracked_idx = (0, 1)
    q_lo, q_hi = torch.full((2,), -2.), torch.full((2,), 2.)

    def forward(self, p, R, v, w, q, qd):
        local = torch.stack((
            torch.stack((q[..., 0], torch.full_like(q[..., 0], .1), q[..., 0] * 0 - .74), -1),
            torch.stack((q[..., 1], torch.full_like(q[..., 1], -.1), q[..., 1] * 0 - .74), -1),
        ), -2)
        position = p[..., None, :] + (R[..., None, :, :] @ local[..., None]).squeeze(-1)
        joint_v = torch.stack((qd, torch.zeros_like(qd), torch.zeros_like(qd)), -1)
        velocity = v[..., None, :] + joint_v
        return position, R[..., None, :, :].expand(*R.shape[:-2], 2, 3, 3), velocity, w[..., None, :].expand_as(velocity)


class LocomotionQualityTests(unittest.TestCase):
    def state(self):
        return (torch.tensor([[0., 0., .76]]), torch.eye(3)[None], torch.zeros(1, 3),
                torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, 2))

    def test_foot_slip_and_actuator_penalties_share_finite_gradients(self):
        field = HeightField(generate("velocity", .76, 1), .76, "cpu")
        quality = LocomotionQuality(TestKinematics(), field, [(20., 12., 25., .5, .01)] * 2)
        previous = self.state()
        qd = torch.full((1, 2), .4, requires_grad=True)
        state = (*previous[:5], qd)
        action = torch.full((1, 2), .2, requires_grad=True)
        xi = torch.zeros(1, 6)
        measured = quality.measurements(state, previous)
        self.assertGreater(measured["support_foot_slip_scale_m_s"].item(), .1)
        self.assertAlmostEqual(measured["foot_separation_scale_m"].item(), .2)
        reward = quality.reward(previous, state, action, xi)
        self.assertLess(reward.item(), 0)
        reward.sum().backward()
        for gradient in (qd.grad, action.grad):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_stealth_foot_slide_weight_scales_measured_slip(self):
        field = HeightField(generate("velocity", .76, 1), .76, "cpu")
        motors = [(20., 12., 25., .5, .01)] * 2
        default = LocomotionQuality(TestKinematics(), field, motors)
        weighted = LocomotionQuality(TestKinematics(), field, motors, foot_slide_weight=.04)
        previous = self.state()
        current = (*previous[:2], torch.tensor([[.4, 0., 0.]]), *previous[3:])
        action, xi = torch.zeros(1, 2), torch.zeros(1, 6)
        slip = default.measurements(current, previous)["support_foot_slip_scale_m_s"]
        difference = default.reward(previous, current, action, xi) - weighted.reward(
            previous, current, action, xi)
        torch.testing.assert_close(difference, .032 * slip)
        with self.assertRaises(ValueError):
            LocomotionQuality(TestKinematics(), field, motors, foot_slide_weight=float("nan"))

    def test_stealth_feet_spacing_weight_scales_close_foot_penalty(self):
        field = HeightField(generate("velocity", .76, 1), .76, "cpu")
        motors = [(20., 12., 25., .5, .01)] * 2
        class CloseFeetKinematics(TestKinematics):
            def forward(self, p, R, v, w, q, qd):
                position, rotation, velocity, angular = super().forward(p, R, v, w, q, qd)
                position[..., 1] = p[..., None, 1]
                return position, rotation, velocity, angular

        default = LocomotionQuality(CloseFeetKinematics(), field, motors)
        weighted = LocomotionQuality(CloseFeetKinematics(), field, motors, feet_close_weight=.04)
        previous = self.state()
        close = (torch.tensor([[0., 0., .76]]), torch.eye(3)[None], torch.zeros(1, 3),
                 torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, 2))
        action, xi = torch.zeros(1, 2), torch.zeros(1, 6)
        difference = default.reward(previous, close, action, xi) - weighted.reward(
            previous, close, action, xi)
        self.assertGreater(difference.item(), 0.)
        self.assertEqual(weighted.feet_close_weight, .04)

    def test_reference_envelope_retains_si_units_and_rejects_invalid_ranges(self):
        t = torch.cat((torch.zeros(30), torch.linspace(.2, .6, 50)))
        rotations = torch.eye(3).repeat(80, 1, 1)
        motion = SimpleNamespace(root_R=rotations,
                                 root_v=torch.stack((t, t * 0, t * 0), -1),
                                 root_w=torch.zeros(80, 3),
                                 clip_id=torch.zeros(80, dtype=torch.long))
        envelope = reference_envelope(motion)
        self.assertAlmostEqual(envelope["ranges"][0][0], .24, places=5)
        self.assertAlmostEqual(envelope["ranges"][0][1], .56, places=5)
        self.assertEqual(envelope["moving_frames"], 50)
        self.assertEqual(envelope["ranges"][1:], [[0., 0.], [0., 0.]])
        terrain = reference_envelope(motion, terrain_navigation=True)
        self.assertGreater(terrain["ranges"][0][0], .1)
        self.assertEqual(terrain["ranges"][1:], [[0., 0.], [0., 0.]])
        self.assertEqual(envelope["history_steps"], 10)
        self.assertEqual(envelope["minimum_moving_frames_per_window"], 8)
        with self.assertRaises(ValueError):
            validate_envelope(dict(version=1, ranges=[[1., 0.], [0., 0.], [0., 0.]]))

    def test_fixed_command_envelope_keeps_the_requested_velocity_constant(self):
        envelope = fixed_command_envelope((.3, 0., 0.))
        self.assertEqual(envelope["ranges"], [[.3, .3], [0., 0.], [0., 0.]])
        self.assertEqual(envelope["fixed_command"], {
            "body_vx_m_s": .3, "body_vy_m_s": 0., "body_yaw_rate_rad_s": 0.,
        })
        self.assertFalse(envelope["standing_commands"])
        self.assertTrue(envelope["forward_only"])
        for command in ((.3, 0.), (.3, 0., 0., 0.), (.3, float("nan"), 0.), (True, 0., 0.)):
            with self.subTest(command=command), self.assertRaises(ValueError):
                fixed_command_envelope(command)

    def test_keyboard_stealth_envelope_describes_forward_pivot_and_arc_modes(self):
        from command_envelope import keyboard_stealth_envelope

        envelope = keyboard_stealth_envelope()
        modes = {row["name"]: row for row in envelope["command_modes"]}
        self.assertEqual(modes["forward"]["command"], [.25, 0., 0.])
        self.assertEqual(modes["turn_left"]["command"], [0., 0., .55])
        self.assertEqual(modes["turn_right"]["command"], [0., 0., -.55])
        self.assertEqual(modes["forward_left"]["command"], [.25, 0., .45])
        self.assertEqual(modes["forward_right"]["command"], [.25, 0., -.45])
        self.assertAlmostEqual(sum(row["probability"] for row in modes.values()), 1.)
        self.assertEqual(envelope["ranges"], [[0., .25], [0., 0.], [-.55, .55]])

    def test_reference_windows_never_cross_clip_boundaries(self):
        velocity = torch.zeros(10, 3)
        velocity[:, 0] = .3
        motion = SimpleNamespace(root_R=torch.eye(3).repeat(10, 1, 1),
                                 root_v=velocity,
                                 root_w=torch.zeros(10, 3),
                                 clip_id=torch.tensor([0] * 5 + [1] * 5))
        with self.assertRaisesRegex(ValueError, "no sustained"):
            reference_envelope(motion, history_steps=10)
        motion.clip_id = torch.zeros(10, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "no complete"):
            reference_envelope(motion, history_steps=11)

    def test_stealth_envelope_retains_sideways_and_reverse_motion(self):
        velocity = torch.zeros(40, 3)
        velocity[:20, 0] = -.2
        velocity[20:, 1] = .3
        motion = SimpleNamespace(root_R=torch.eye(3).repeat(40, 1, 1),
                                 root_v=velocity, root_w=torch.zeros(40, 3),
                                 clip_id=torch.tensor([0] * 20 + [1] * 20))
        envelope = reference_envelope(motion, minimum_forward_speed=.08, forward_only=False)
        self.assertEqual(envelope["moving_frames"], 40)
        self.assertFalse(envelope["forward_only"])
        self.assertLess(envelope["ranges"][0][0], -.1)
        self.assertGreater(envelope["ranges"][1][1], .2)

    def test_crouched_forward_motion_uses_world_vertical_speed_guard(self):
        from trackerLab.kinematics.supermimic.rotations import so3_exp

        rotation = so3_exp(torch.tensor([[0., .5, 0.]])).expand(20, 3, 3)
        motion = SimpleNamespace(root_R=rotation, root_v=torch.tensor([[1., 0., 0.]]).expand(20, 3),
                                 root_w=torch.zeros(20, 3),
                                 clip_id=torch.zeros(20, dtype=torch.long))
        envelope = reference_envelope(motion, forward_only=False)
        self.assertEqual(envelope["moving_frames"], 20)
        self.assertEqual(envelope["vertical_speed_frame"], "world")


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for fixed-pose stealth upper-body constraints."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.stealth_upper_body import (
    UpperBodyConstraint,
    blend_upper_body_reference,
    calibrate_upper_body_reference,
    extract_static_upper_body_pose,
)


class UpperBodyConstraintTests(unittest.TestCase):
    joint_names = (
        "left_hip_pitch_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_roll_joint",
    )
    default_q = (0.1, -0.2, 0.0, 0.1, -0.1, 0.0)
    action_scale = (0.2, 0.3, 0.2, 0.1, 0.05, 0.25)
    reference_q = {
        "left_shoulder_pitch_joint": 0.5,
        "left_elbow_joint": 0.6,
        "left_wrist_roll_joint": -0.3,
        "right_shoulder_roll_joint": 0.4,
    }

    def make_constraint(self, **kwargs):
        config = dict(
            joint_names=self.joint_names,
            default_q=self.default_q,
            action_scale=self.action_scale,
            reference_q=self.reference_q,
        )
        config.update(kwargs)
        return UpperBodyConstraint(**config)

    def test_first_action_is_rate_limited_in_joint_radians(self):
        constraint = self.make_constraint()
        raw = torch.full((2, len(self.joint_names)), 5.0)
        previous = torch.zeros_like(raw)

        applied = constraint(raw, previous)
        upper = constraint.upper_indices
        default = torch.tensor(self.default_q)[upper]
        scale = torch.tensor(self.action_scale)[upper]
        joint_delta = (applied[:, upper] - previous[:, upper]) * scale

        self.assertTrue(bool((joint_delta > 0).all()))
        self.assertTrue(bool((joint_delta.abs() <= 1.2 * 0.02 + 1e-6).all()))
        self.assertTrue(bool((joint_delta.abs() < 0.1).all()))
        torch.testing.assert_close(
            applied[:, :2], raw[:, :2], atol=0, rtol=0,
        )
        self.assertTrue(bool(torch.isfinite(applied).all()))
        self.assertEqual(default.shape, scale.shape)

    def test_constraint_respects_policy_action_limit_without_exceeding_rate(self):
        constraint = self.make_constraint(action_limit=4.)
        raw = torch.full((1, len(self.joint_names)), 20.)
        previous = torch.zeros_like(raw)
        action = constraint(raw, previous)
        upper = constraint.upper_indices
        self.assertLessEqual(action[:, upper].abs().max().item(), 4.)
        delta = (action - previous) * constraint.action_scale
        self.assertLessEqual(delta[:, upper].abs().max().item(), .024 + 1e-6)

    def test_recurrent_outputs_obey_rate_and_pose_bounds(self):
        constraint = self.make_constraint()
        action = torch.zeros(1, len(self.joint_names))
        reference = torch.tensor([
            self.reference_q[name] for name in constraint.upper_joint_names
        ])
        default = constraint.default_q[constraint.upper_indices]
        scale = constraint.action_scale[constraint.upper_indices]
        previous_q = default.clone()

        for _ in range(800):
            raw = torch.full_like(action, 20.0)
            applied = constraint(raw, action)
            current_q = default + applied[0, constraint.upper_indices] * scale
            delta = current_q - previous_q
            self.assertTrue(bool((delta.abs() <= 1.2 * 0.02 + 1e-6).all()))
            previous_q = current_q
            action = applied

        self.assertTrue(bool(((previous_q - reference).abs() <= 0.25 + 1e-6).all()))
        self.assertTrue(bool(((previous_q - reference).abs() > 0.24).all()))

    def test_reset_and_aiming_pose_can_be_blended_before_rate_limited_approach(self):
        blended = blend_upper_body_reference(
            self.joint_names, self.default_q, self.reference_q,
        )
        constraint = self.make_constraint(reference_q=blended)
        upper = constraint.upper_indices
        default = constraint.default_q[upper]
        scale = constraint.action_scale[upper]
        previous = torch.zeros((1, len(self.joint_names)))
        raw = torch.zeros_like(previous)
        raw[:, upper] = (constraint.reference_q - default) / scale
        applied = constraint(raw, previous)
        first_target = default + applied[0, upper] * scale
        reference = constraint.reference_q
        displacement = first_target - default

        torch.testing.assert_close(
            constraint.reference_q,
            (default + torch.tensor([
                self.reference_q[name] for name in constraint.upper_joint_names
            ])) / 2,
        )
        self.assertTrue(bool((displacement.abs() <= 1.2 * 0.02 + 1e-6).all()))
        self.assertTrue(bool((displacement * (reference - default) > 0).all()))

    def test_legs_and_waist_are_unchanged(self):
        constraint = self.make_constraint()
        raw = torch.randn(3, len(self.joint_names))
        previous = torch.zeros_like(raw)
        applied = constraint(raw, previous)
        torch.testing.assert_close(applied[:, :2], raw[:, :2], atol=0, rtol=0)

    def test_constraint_is_differentiable_for_policy_and_history_actions(self):
        constraint = self.make_constraint()
        raw = torch.full((2, len(self.joint_names)), 0.2, requires_grad=True)
        previous = torch.full((2, len(self.joint_names)), -0.1, requires_grad=True)

        loss = constraint(raw, previous).square().sum()
        loss.backward()

        for value in (raw.grad, previous.grad):
            self.assertIsNotNone(value)
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertGreater(value.abs().sum().item(), 0)

    def test_config_roundtrip_and_scripted_execution_match(self):
        constraint = self.make_constraint().eval()
        restored = UpperBodyConstraint.from_dict(constraint.to_dict()).eval()
        torch.testing.assert_close(
            restored.reference_q, constraint.reference_q, atol=0, rtol=0,
        )
        self.assertEqual(restored.to_dict(), constraint.to_dict())
        scripted = torch.jit.script(constraint)
        raw = torch.randn(4, len(self.joint_names))
        previous = torch.randn_like(raw)
        torch.testing.assert_close(
            scripted(raw, previous), restored(raw, previous), atol=1e-6, rtol=1e-6,
        )

    def test_calibration_uses_only_arm_medians_from_one_clip(self):
        q = torch.tensor([
            [0.0, 9.0, 0.2, 0.4, -0.1, 0.3],
            [1.0, 8.0, 0.6, 0.8, -0.5, 0.7],
            [2.0, 7.0, 1.0, 1.2, -0.9, 1.1],
        ])
        motion = SimpleNamespace(q=q, clip_id=torch.zeros(3, dtype=torch.long))

        reference = calibrate_upper_body_reference(
            motion, self.joint_names, method="median",
        )

        self.assertEqual(set(reference), set(self.reference_q))
        for name, expected in zip(self.reference_q, (0.6, 0.8, -0.5, 0.7)):
            self.assertAlmostEqual(reference[name], expected)

    def test_static_pose_extraction_uses_a_contiguous_clip_range(self):
        q = torch.tensor([
            [0., 9., 0.2, 0.4, -0.1, 0.3],
            [1., 8., 0.6, 0.8, -0.5, 0.7],
            [2., 7., 1.0, 1.2, -0.9, 1.1],
            [3., 6., 5.0, 5.0, 5.0, 5.0],
        ])
        motion = SimpleNamespace(
            q=q, clip_id=torch.tensor([0, 0, 0, 1]),
        )
        pose = extract_static_upper_body_pose(
            motion, self.joint_names, frame_start=0, frame_end=2, method="medoid",
        )
        self.assertEqual(pose, {
            "left_shoulder_pitch_joint": 0.6000000238418579,
            "left_elbow_joint": 0.800000011920929,
            "left_wrist_roll_joint": -0.5,
            "right_shoulder_roll_joint": 0.699999988079071,
        })

    def test_external_reference_is_accepted_without_motion_data(self):
        reference = calibrate_upper_body_reference(
            None, self.joint_names, reference_q=self.reference_q,
        )
        for name, expected in self.reference_q.items():
            self.assertAlmostEqual(reference[name], expected)

    def test_posture_reward_favors_static_aiming_and_has_finite_gradients(self):
        constraint = self.make_constraint()
        joints = constraint.default_q.repeat(2, 1)
        joints[0, constraint.upper_indices] = constraint.reference_q
        joints[1, constraint.upper_indices] = constraint.reference_q + .5
        joints.requires_grad_()
        joint_velocities = torch.zeros_like(joints)
        joint_velocities[1, constraint.upper_indices] = 2.
        reward = constraint.posture_reward((None, None, None, None, joints, joint_velocities))
        self.assertAlmostEqual(reward[0].item(), 0.)
        self.assertLess(reward[1].item(), reward[0].item())
        reward.sum().backward()
        self.assertTrue(bool(torch.isfinite(joints.grad).all()))
        self.assertGreater(joints.grad[1].abs().sum().item(), 0.)

    def test_calibration_rejects_multiple_clips_and_nonfinite_motion(self):
        q = torch.zeros(4, len(self.joint_names))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            calibrate_upper_body_reference(
                SimpleNamespace(q=q, clip_id=torch.tensor([0, 0, 1, 1])),
                self.joint_names,
            )
        q[0, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            calibrate_upper_body_reference(
                SimpleNamespace(q=q, clip_id=torch.zeros(4, dtype=torch.long)),
                self.joint_names,
            )

    def test_invalid_joint_metadata_and_config_values_are_rejected(self):
        for overrides in (
            {"action_scale": (0.2, 0.3, 0.2, 0.1, 0.0, 0.25)},
            {"residual_radius_rad": float("nan")},
            {"max_joint_rate_rad_s": float("inf")},
            {"control_dt": -0.02},
            {"reference_q": {"left_elbow_joint": float("nan")}},
            {"joint_names": ("hip", "waist")},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.make_constraint(**overrides)


if __name__ == "__main__":
    unittest.main()

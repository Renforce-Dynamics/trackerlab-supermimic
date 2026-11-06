"""CPU contracts for the pose-independent stealth arm rate limiter."""

import io
import json
from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.stealth_action_filter import UpperBodyRateLimiter
from trackerTask.supermimic.robots import get_robot


class UpperBodyRateLimiterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.spec = get_robot("g1")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def limiter(self, **overrides):
        config = dict(joint_names=self.spec.joint_names, action_scale=self.spec.action_scale)
        config.update(overrides)
        return UpperBodyRateLimiter(**config)

    def test_g1_selects_exactly_fourteen_arm_joints_and_preserves_other_outputs(self):
        limiter = self.limiter()
        self.assertEqual(limiter.num_joints, 29)
        self.assertEqual(len(limiter.upper_joint_names), 14)
        expected = [index for index, name in enumerate(self.spec.joint_names)
                    if any(part in name for part in ("shoulder", "elbow", "wrist"))]
        self.assertEqual(limiter.upper_indices.tolist(), expected)
        self.assertFalse(hasattr(limiter, "reference_q"))
        self.assertFalse(hasattr(limiter, "posture_reward"))
        raw = torch.linspace(-30., 30., 29).expand(3, -1).clone()
        previous = torch.zeros_like(raw)
        result = limiter(raw, previous, previous)
        mask = torch.ones(29, dtype=torch.bool)
        mask[limiter.upper_indices] = False
        torch.testing.assert_close(result[:, mask], raw[:, mask], rtol=0, atol=0)
        torch.testing.assert_close(raw, torch.linspace(-30., 30., 29).expand(3, -1), rtol=0, atol=0)
        torch.testing.assert_close(previous, torch.zeros_like(previous), rtol=0, atol=0)

    def test_large_entry_jump_obeys_first_step_acceleration_in_joint_radians(self):
        limiter = self.limiter()
        raw = torch.stack((torch.full((29,), 20.), torch.full((29,), -20.)))
        previous = torch.zeros_like(raw)
        result = limiter(raw, previous, previous, entry_time=torch.zeros(2))
        delta = (result - previous) * limiter.action_scale
        self.assertLessEqual(delta[:, limiter.upper_indices].abs().max().item(), .024)
        self.assertGreater(delta[:, limiter.upper_indices].abs().min().item(), 0.)
        self.assertLessEqual(delta[:, limiter.upper_indices].abs().max().item(), 3. * .02**2)

    def test_moderate_arm_targets_retain_policy_gradient_over_full_imagined_horizon(self):
        limiter = self.limiter()
        raw = torch.zeros(1, 29)
        raw[:, limiter.upper_indices] = .25
        raw.requires_grad_()
        previous = torch.zeros_like(raw)
        older = torch.zeros_like(raw)
        total = torch.zeros(())
        for _ in range(32):
            current = limiter(raw, previous, older)
            total = total + current[:, limiter.upper_indices].sum()
            older, previous = previous, current
        gradient = torch.autograd.grad(total, raw)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient[:, limiter.upper_indices].abs().sum()), 1e-3)

    def test_repeated_reversals_obey_rate_and_acceleration_with_executed_history(self):
        limiter = self.limiter().double()
        previous = torch.zeros(2, 29, dtype=torch.float64)
        older, velocity = previous.clone(), previous.clone()
        peak_rate = 0.
        for index in range(360):
            raw = torch.full_like(previous, 20. if (index // 30) % 2 == 0 else -20.)
            raw[1] = -raw[1]
            result = limiter(raw, previous, older)
            next_velocity = (result - previous) * limiter.action_scale / limiter.control_dt
            acceleration = (next_velocity - velocity) / limiter.control_dt
            upper = limiter.upper_indices
            peak_rate = max(peak_rate, next_velocity[:, upper].abs().max().item())
            self.assertLessEqual(next_velocity[:, upper].abs().max().item(), 1.2 + 1e-10)
            self.assertLessEqual(acceleration[:, upper].abs().max().item(), 3. + 1e-9)
            self.assertTrue(bool(torch.isfinite(result).all()))
            older, previous, velocity = previous, result, next_velocity
        self.assertGreater(peak_rate, 1.)

    def test_filter_has_no_pose_attractor_or_entry_clock_dependency(self):
        limiter = self.limiter()
        raw = torch.linspace(-1., 1., 29).expand(3, -1).clone()
        # Holding any actor-selected pose holds it exactly, including idle.
        for time in (None, torch.zeros(3), torch.tensor([.5, 4.84, 100.])):
            torch.testing.assert_close(limiter(raw, raw, raw, entry_time=time), raw, rtol=0, atol=0)
        previous = torch.zeros_like(raw)
        expected = limiter(raw, previous, previous)
        for time in (torch.zeros(3), torch.tensor([.5, 4.84, 100.])):
            torch.testing.assert_close(limiter(raw, previous, previous, time), expected, rtol=0, atol=0)
        # A zero raw arm action moves toward action zero, never toward a pose reference.
        result = limiter(previous, raw, raw)
        upper = limiter.upper_indices
        self.assertTrue(bool((result[:, upper].abs() < raw[:, upper].abs()).all()))

    def test_action_cap_applies_to_arms_and_takes_precedence_at_boundary(self):
        limiter = self.limiter(action_limit=.5)
        raw = torch.full((1, 29), 20.)
        previous = torch.full_like(raw, .4999)
        older = previous - .024 / limiter.action_scale
        result = limiter(raw, previous, older)
        torch.testing.assert_close(result[:, limiter.upper_indices], torch.full((1, 14), .5))
        self.assertLessEqual(result[:, limiter.upper_indices].abs().max().item(), .5)
        # The incoming history is moving outwards at maximum rate next to a hard
        # action cap. The cap cannot also permit a full acceleration-limited stop.
        acceleration = ((result - previous) * limiter.action_scale - .024) / .02**2
        self.assertGreater(acceleration[:, limiter.upper_indices].abs().max().item(), 3.)
        torch.testing.assert_close(result[:, :15], raw[:, :15], rtol=0, atol=0)

    def test_braking_at_both_action_boundaries_preserves_acceleration(self):
        limiter = self.limiter(action_limit=.5).double()
        previous = torch.zeros(1, 29, dtype=torch.float64)
        older, velocity = previous.clone(), previous.clone()
        for index in range(180):
            raw = torch.full_like(previous, 20. if index < 90 else -20.)
            result = limiter(raw, previous, older)
            next_velocity = (result - previous) * limiter.action_scale / .02
            acceleration = (next_velocity - velocity) / .02
            upper = limiter.upper_indices
            self.assertLessEqual(result[:, upper].abs().max().item(), .5)
            self.assertLessEqual(next_velocity[:, upper].abs().max().item(), 1.2 + 1e-10)
            self.assertLessEqual(acceleration[:, upper].abs().max().item(), 3. + 1e-9)
            if index in (89, 179):
                expected = .5 if index == 89 else -.5
                torch.testing.assert_close(result[:, upper], torch.full((1, 14), expected, dtype=torch.float64))
                torch.testing.assert_close(next_velocity[:, upper], torch.zeros(1, 14, dtype=torch.float64))
            older, previous, velocity = previous, result, next_velocity

    def test_actor_and_history_gradients_are_finite_and_nonarm_gradient_is_identity(self):
        limiter = self.limiter().double()
        raw = torch.full((2, 29), .0001, dtype=torch.float64, requires_grad=True)
        previous = torch.zeros_like(raw, requires_grad=True)
        older = torch.zeros_like(raw, requires_grad=True)
        time = torch.ones(2, dtype=torch.float64, requires_grad=True)
        result = limiter(raw, previous, older, entry_time=time)
        grads = torch.autograd.grad(result.sum(), (raw, previous, older, time), allow_unused=True)
        for gradient in grads[:3]:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertIsNone(grads[3])
        torch.testing.assert_close(grads[0][:, :15], torch.ones(2, 15, dtype=torch.float64))
        torch.testing.assert_close(grads[1][:, :15], torch.zeros(2, 15, dtype=torch.float64))
        torch.testing.assert_close(grads[2][:, :15], torch.zeros(2, 15, dtype=torch.float64))
        self.assertGreater(grads[0][:, limiter.upper_indices].abs().sum().item(), 0.)
        self.assertTrue(torch.autograd.gradcheck(limiter, (raw, previous, older)))
        # In the acceleration-limited regime both history inputs affect the step.
        limited = limiter(torch.full_like(raw, 10.), previous, older)
        history_grads = torch.autograd.grad(limited[:, limiter.upper_indices].sum(), (previous, older))
        for gradient in history_grads:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(gradient[:, limiter.upper_indices].abs().sum().item(), 0.)

    def test_config_roundtrip_owns_metadata_and_script_export_matches(self):
        scale = torch.tensor(self.spec.action_scale, requires_grad=True)
        limiter = self.limiter(action_scale=scale).eval()
        saved_scale = limiter.action_scale.clone()
        with torch.no_grad():
            scale.fill_(99.)
        torch.testing.assert_close(limiter.action_scale, saved_scale)
        self.assertFalse(limiter.action_scale.requires_grad)
        config = json.loads(json.dumps(limiter.to_dict()))
        self.assertEqual(set(config), {
            "joint_names", "action_scale", "max_joint_rate_rad_s",
            "max_joint_acceleration_rad_s2", "control_dt", "action_limit",
            "rate_error_scale_rad",
        })
        restored = UpperBodyRateLimiter.from_dict(config).eval()
        self.assertEqual(restored.to_dict(), config)
        scripted = torch.jit.script(restored)
        archive = io.BytesIO()
        torch.jit.save(scripted, archive)
        archive.seek(0)
        loaded = torch.jit.load(archive)
        for shape in ((29,), (7, 29), (2, 3, 29)):
            raw = torch.randn(shape)
            previous = torch.zeros_like(raw)
            older = torch.zeros_like(raw)
            time = torch.zeros(shape[:-1])
            torch.testing.assert_close(loaded(raw, previous, older),
                                       restored(raw, previous, older), rtol=0, atol=0)
            torch.testing.assert_close(loaded(raw, previous, older, time),
                                       restored(raw, previous, older, time), rtol=0, atol=0)

    def test_aot_eager_fullgraph_matches_outputs_and_all_action_gradients(self):
        limiter = self.limiter()
        compiled = torch.compile(limiter, backend="aot_eager", fullgraph=True)
        for magnitude in (.0001, 10.):
            raw = torch.full((3, 29), magnitude, requires_grad=True)
            previous = torch.zeros_like(raw, requires_grad=True)
            older = torch.zeros_like(raw, requires_grad=True)
            inputs = (raw, previous, older)
            compiled_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
            expected = limiter(*inputs, entry_time=torch.zeros(3))
            actual = compiled(*compiled_inputs, entry_time=torch.ones(3))
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(), compiled_inputs),
                                       torch.autograd.grad(expected.sum(), inputs))

    def test_invalid_metadata_and_action_shapes_are_rejected(self):
        for invalid in (
            {"joint_names": []}, {"joint_names": ["hip", "waist"]},
            {"joint_names": ["left_elbow_joint", "left_elbow_joint"], "action_scale": [1., 1.]},
            {"action_scale": [1.]}, {"action_scale": [0.] * 29},
            {"action_scale": [float("nan")] * 29},
            {"action_scale": [-.1] * 29},
            *({name: value} for name in ("max_joint_rate_rad_s", "max_joint_acceleration_rad_s2",
                                         "control_dt", "action_limit")
              for value in (0., -1., True, "1", float("nan"), float("inf"))),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.limiter(**invalid)
        for config in (None, [], "metadata"):
            with self.subTest(config=config), self.assertRaises(ValueError):
                UpperBodyRateLimiter.from_dict(config)
        limiter = self.limiter()
        raw = torch.zeros(2, 29)
        for inputs in ((raw, raw[:1], raw), (raw, raw, raw[:, :-1]),
                       (torch.zeros(()), raw, raw), (raw.double(), raw.double(), raw.double()),
                       (raw, raw.double(), raw), (raw, raw, raw.double())):
            with self.subTest(shapes=[value.shape for value in inputs]), self.assertRaises(ValueError):
                limiter(*inputs)


if __name__ == "__main__":
    unittest.main()

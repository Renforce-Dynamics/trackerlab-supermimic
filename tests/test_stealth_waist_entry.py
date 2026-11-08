"""CPU tests for opt-in, persistent waist targets without policy integration."""

import io
import json
import math
from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.stealth_upper_body import UpperBodyConstraint
from trackerLab.algorithms.supermimic.stealth_waist_entry import (
    CombinedEntryConstraint, WaistEntryConstraint, compose_entry_waist_constraint,
)


NAMES = tuple(
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
) + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")
)


class WaistEntryTests(unittest.TestCase):
    def make_constraint(self, **overrides):
        config = {
            "joint_names": NAMES,
            "default_q": [0.] * 29,
            "action_scale": [.25] * 29,
            "reference_q": dict(zip(NAMES[12:15], (.06, -.03, .5))),
            "ramp_duration_s": 1.02,
        }
        config.update(overrides)
        return WaistEntryConstraint(**config)

    def make_primary(self):
        return UpperBodyConstraint(
            NAMES, [0.] * 29, [.25] * 29,
            {name: .1 for name in NAMES[15:]},
            residual_radius_rad=0., posture_tolerance_rad=.25,
        )

    def test_defaults_midpoint_final_and_persistent_hold(self):
        constraint = self.make_constraint()
        times = torch.tensor([-5., 0., .51, 1.02, 4.84, 1000.])
        reference = constraint.reference_at(times)
        torch.testing.assert_close(reference[:2], torch.zeros(2, 3), rtol=0, atol=0)
        torch.testing.assert_close(reference[2], constraint.reference_q / 2)
        torch.testing.assert_close(reference[3:], constraint.reference_q.expand(3, 3), rtol=0, atol=0)
        raw = torch.randn(6, 29)
        raw[:, 12] = math.radians(116.) / .25
        result = constraint(raw, -raw, raw * 9, entry_time=times)
        torch.testing.assert_close(result[3:, 12:15], (constraint.reference_q / .25).expand(3, 3))
        self.assertAlmostEqual(constraint.max_joint_rate_rad_s, 1.2)
        self.assertAlmostEqual(constraint.max_joint_acceleration_rad_s2, 3.)

    def test_nonzero_default_and_shuffled_joint_names(self):
        names = tuple(reversed(NAMES))
        default = [.1] * 29
        constraint = self.make_constraint(joint_names=names, default_q=default)
        raw = torch.arange(29, dtype=torch.float32)
        result = constraint(raw, raw, entry_time=torch.tensor(1.02))
        indices = [names.index(name) for name in NAMES[12:15]]
        torch.testing.assert_close(result[indices], (constraint.reference_q - .1) / .25)
        others = [i for i in range(29) if i not in indices]
        self.assertTrue(torch.equal(result[others], raw[others]))

    def test_scalar_and_multidimensional_batches(self):
        constraint = self.make_constraint()
        for shape in ((29,), (4, 29), (2, 3, 29)):
            raw = torch.randn(shape)
            result = constraint(raw, raw, entry_time=torch.tensor(.51))
            torch.testing.assert_close(result[..., 12:15], torch.tensor([.12, -.06, 1.]).expand(*shape[:-1], 3))
            clock = torch.full(shape[:-1], 1.02)
            result = constraint(raw, raw, entry_time=clock)
            torch.testing.assert_close(result[..., 12:15], (constraint.reference_q / .25).expand(*shape[:-1], 3))

    def test_exact_nonwaist_values_gradients_and_clock_gradient(self):
        constraint = self.make_constraint().double()
        raw = torch.randn(3, 29, dtype=torch.float64, requires_grad=True)
        previous = torch.randn_like(raw, requires_grad=True)
        older = torch.randn_like(raw, requires_grad=True)
        time = torch.tensor([.2, .4, .6], dtype=torch.float64, requires_grad=True)
        result = constraint(raw, previous, older, entry_time=time)
        mask = [i for i in range(29) if i not in (12, 13, 14)]
        self.assertTrue(torch.equal(raw[:, mask], result[:, mask]))
        grad, clock_grad, prev_grad, older_grad = torch.autograd.grad(
            result.sum(), (raw, time, previous, older), allow_unused=True)
        self.assertTrue(torch.equal(grad[:, mask], torch.ones_like(grad[:, mask])))
        self.assertTrue(torch.equal(grad[:, 12:15], torch.zeros_like(grad[:, 12:15])))
        self.assertTrue(bool((clock_grad > 0).all()))
        self.assertIsNone(prev_grad)
        self.assertIsNone(older_grad)
        self.assertEqual(list(constraint.parameters()), [])

    def test_analytic_peak_rate_and_acceleration_validation(self):
        constraint = self.make_constraint().double()
        times = torch.linspace(0., 1.02, 4001, dtype=torch.float64, requires_grad=True)
        q = constraint.reference_at(times)[:, 2]
        rate = torch.autograd.grad(q.sum(), times, create_graph=True)[0]
        acceleration = torch.autograd.grad(rate.sum(), times)[0]
        distance = float(constraint.reference_q[2])
        self.assertAlmostEqual(float(rate.detach().max()), 1.875 * distance / 1.02, places=7)
        self.assertAlmostEqual(float(acceleration.detach().abs().max()),
                               (10 / math.sqrt(3)) * distance / 1.02**2, places=6)
        for options, message in (
            ({"max_joint_rate_rad_s": .9}, "max_joint_rate"),
            ({"max_joint_acceleration_rad_s2": 2.7}, "max_joint_acceleration"),
            ({"ramp_duration_s": .2}, "max_joint_rate"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                self.make_constraint(**options)

    def test_discrete_endpoints_have_no_target_jump_or_rate_acceleration_jump(self):
        constraint = self.make_constraint().double()
        t = torch.arange(-.04, 1.08, .02, dtype=torch.float64)
        target = constraint.reference_at(t)
        speed = torch.diff(target, dim=0) / .02
        acceleration = torch.diff(speed, dim=0) / .02
        self.assertLessEqual(float(speed.abs().max()), constraint.max_joint_rate_rad_s)
        self.assertLessEqual(float(acceleration.abs().max()), constraint.max_joint_acceleration_rad_s2)
        endpoints = torch.tensor([-.000001, 0., .000001, 1.019999, 1.02, 1.020001],
                                 dtype=torch.float64, requires_grad=True)
        q = constraint.reference_at(endpoints)
        v = torch.autograd.grad(q.sum(), endpoints, create_graph=True)[0]
        a = torch.autograd.grad(v.sum(), endpoints)[0]
        self.assertLess(float(v.detach().abs().max()), 1e-9)
        self.assertLess(float(a.detach().abs().max()), 1e-4)
        torch.testing.assert_close(q[-2], q[-1], rtol=0, atol=0)

    def test_action_range_checks_all_scheduled_targets(self):
        constraint = self.make_constraint(action_limit=2.)
        times = torch.linspace(-1., 5., 501)
        raw = torch.zeros(501, 29)
        self.assertLessEqual(float(constraint(raw, raw, entry_time=times)[:, 12:15].abs().max()), 2.)
        with self.assertRaisesRegex(ValueError, "action_limit"):
            self.make_constraint(action_limit=1.99)
        with self.assertRaisesRegex(ValueError, "action_limit"):
            self.make_constraint(default_q=[5.] * 29, ramp_duration_s=30.)

    def test_constructor_rejects_invalid_scalar_options(self):
        for name in ("ramp_duration_s", "control_dt", "max_joint_rate_rad_s",
                     "max_joint_acceleration_rad_s2", "action_limit"):
            for bad in (True, False, 0., -1., float("nan"), float("inf"), "1", 1e300, 1e-300):
                with self.subTest(name=name, value=bad), self.assertRaises(ValueError):
                    self.make_constraint(**{name: bad})

    def test_constructor_rejects_bad_names_and_vectors(self):
        for names in (NAMES[:-1], (*NAMES[:-1], NAMES[0]),
                      (*NAMES[:12], "unknown", *NAMES[13:]), (*NAMES[:-1], ""),
                      (*NAMES[:-1], 1), None):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.make_constraint(joint_names=names)
        for name in ("default_q", "action_scale"):
            for bad in ([0.] * 28, [float("nan")] * 29, [float("inf")] * 29,
                        [1e300] * 29, [True] * 29, [0.] * 28 + [True],
                        torch.ones(29, dtype=torch.bool), torch.ones(29, dtype=torch.complex64),
                        [[.25] * 29], "bad", None):
                with self.subTest(name=name, value=str(bad)), self.assertRaises(ValueError):
                    self.make_constraint(**{name: bad})
        for bad in ([0.] * 29, [-.25] * 29, [1e-300] * 29):
            with self.assertRaises(ValueError):
                self.make_constraint(action_scale=bad)

    def test_reference_requires_exact_named_finite_numeric_fields(self):
        good = dict(zip(NAMES[12:15], (.06, -.03, .5)))
        bads = [None, [.06, -.03, .5], {**good, "extra": 0.}, {NAMES[12]: 0.}]
        for value in (True, "0.5", float("nan"), float("inf"), 1e300):
            bads.append({**good, NAMES[14]: value})
        for bad in bads:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.make_constraint(reference_q=bad)
        with self.assertRaises(ValueError):
            WaistEntryConstraint.from_dict([])

    def test_missing_clock_shape_dtype_and_nonfinite_time(self):
        constraint = self.make_constraint()
        raw = torch.zeros(2, 29)
        with self.assertRaisesRegex(ValueError, "entry_time is required"):
            constraint(raw, raw)
        for clock in (torch.zeros(1), torch.zeros(2, 1), torch.tensor(float("nan")),
                      torch.tensor(float("inf")), torch.zeros(2, dtype=torch.float64)):
            with self.subTest(clock=clock), self.assertRaises(ValueError):
                constraint(raw, raw, entry_time=clock)
        for action, previous, older in (
            (torch.zeros(2, 28), torch.zeros(2, 28), None),
            (raw, torch.zeros(1, 29), None),
            (raw, raw, torch.zeros(2, 28)),
            (raw, raw.double(), None),
            (raw, raw, raw.double()),
            (torch.tensor(0.), torch.tensor(0.), None),
        ):
            with self.subTest(shape=action.shape), self.assertRaises(ValueError):
                constraint(action, previous, older, entry_time=torch.tensor(.2))

    def test_commands_and_previous_actions_cannot_reset_entry_clock(self):
        constraint = self.make_constraint()
        raw = torch.randn(9, 29)
        time = torch.full((9,), 8.)
        first = constraint(raw, raw * 9, -raw, entry_time=time)
        for raw_command_response in (torch.zeros_like(raw), -raw * 4, torch.ones_like(raw) * 20):
            changed = constraint(raw_command_response, raw * -5, entry_time=time)
            self.assertTrue(torch.equal(first[:, 12:15], changed[:, 12:15]))
        torch.testing.assert_close(constraint(raw, raw, entry_time=torch.tensor(0.))[:, 12:15],
                                   torch.zeros(9, 3))

    def test_metadata_state_dict_and_script_roundtrips(self):
        constraint = self.make_constraint()
        metadata = json.loads(json.dumps(constraint.to_dict(), allow_nan=False))
        restored = WaistEntryConstraint.from_dict(metadata)
        self.assertEqual(metadata, restored.to_dict())
        self.assertEqual(restored.ramp_duration_s, 1.02)
        restored.load_state_dict(constraint.state_dict(), strict=True)
        raw, previous = torch.randn(3, 29), torch.randn(3, 29)
        clock = torch.tensor([0., .51, 20.])
        expected = constraint(raw, previous, entry_time=clock)
        scripted = torch.jit.script(constraint)
        torch.testing.assert_close(scripted(raw, previous, None, clock), expected)
        torch.testing.assert_close(scripted.reference_at(clock), constraint.reference_at(clock))
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        torch.testing.assert_close(loaded(raw, previous, None, clock), expected)
        torch.testing.assert_close(loaded.reference_at(clock), constraint.reference_at(clock))

    def test_trace_dynamic_clock_batch_and_serialization(self):
        constraint = self.make_constraint()
        raw = torch.randn(2, 29)
        example = (raw, raw, raw, torch.tensor([.2, .4]))
        traced = torch.jit.trace(constraint, example)
        for batch, time in ((1, -.2), (4, .7), (3, 100.)):
            action = torch.randn(batch, 29)
            clock = torch.full((batch,), time)
            actual = (action, -action, action * 4, clock)
            torch.testing.assert_close(traced(*actual), constraint(*actual))
        buffer = io.BytesIO()
        torch.jit.save(traced, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        torch.testing.assert_close(loaded(*example), constraint(*example))
        torch.testing.assert_close(loaded.reference_at(example[-1]), constraint.reference_at(example[-1]))

    def test_composition_none_is_identical_and_config_needs_primary(self):
        primary = self.make_primary()
        self.assertIs(compose_entry_waist_constraint(primary, None), primary)
        self.assertIsNone(compose_entry_waist_constraint(None, None))
        with self.assertRaisesRegex(ValueError, "existing primary"):
            compose_entry_waist_constraint(None, self.make_constraint().to_dict())
        with self.assertRaises(ValueError):
            compose_entry_waist_constraint(object(), self.make_constraint().to_dict())
        raw, previous, older = torch.randn(2, 29), torch.zeros(2, 29), torch.zeros(2, 29)
        clock = torch.tensor([.1, 6.])
        self.assertTrue(torch.equal(
            compose_entry_waist_constraint(primary, None)(raw, previous, older, clock),
            primary(raw, previous, older, clock)))

    def test_composition_preserves_primary_arms_legs_and_history_arguments(self):
        primary = self.make_primary()
        waist = self.make_constraint()
        combined = CombinedEntryConstraint(primary, waist)
        raw = torch.randn(3, 29, requires_grad=True)
        previous, older = torch.randn_like(raw), torch.randn_like(raw)
        clock = torch.tensor([0., .6, 10.])
        primary_result = primary(raw, previous, older, clock)
        expected = waist(primary_result, previous, older, clock)
        result = combined(raw, previous, older, clock)
        self.assertTrue(torch.equal(result, expected))
        self.assertTrue(torch.equal(result[:, :12], raw[:, :12]))
        self.assertTrue(torch.equal(result[:, 15:], primary_result[:, 15:]))
        self.assertIs(combined.primary, primary)
        self.assertIs(combined.waist, waist)
        gradient = torch.autograd.grad(result.sum(), raw)[0]
        self.assertTrue(torch.equal(gradient[:, :12], torch.ones(3, 12)))
        self.assertTrue(torch.equal(gradient[:, 12:15], torch.zeros(3, 3)))
        with self.assertRaises(ValueError):
            CombinedEntryConstraint(None, waist)

    def test_composition_infers_primary_dtype_device_and_traces(self):
        primary = self.make_primary().double()
        config = self.make_constraint().to_dict()
        combined = compose_entry_waist_constraint(primary, config)
        self.assertIs(combined.primary, primary)
        self.assertEqual(combined.waist.default_q.dtype, torch.float64)
        self.assertEqual(combined.waist.default_q.device, primary.default_q.device)
        explicit = compose_entry_waist_constraint(primary, config, device="cpu")
        self.assertEqual(explicit.waist.default_q.device.type, "cpu")
        raw = torch.randn(2, 29, dtype=torch.float64)
        example = (raw, raw * 0, raw * 0, torch.tensor([.2, 5.], dtype=torch.float64))
        traced = torch.jit.trace(combined, example)
        torch.testing.assert_close(traced(*example), combined(*example))
        scripted = torch.jit.script(combined)
        torch.testing.assert_close(scripted(*example), combined(*example))
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        torch.testing.assert_close(loaded(*example), combined(*example))

    def test_cpu_fullgraph_compile_keeps_clock_gradients(self):
        constraint = self.make_constraint()
        compiled = torch.compile(constraint, backend="eager", fullgraph=True)
        raw = torch.randn(2, 29, requires_grad=True)
        time = torch.tensor([.3, .6], requires_grad=True)
        actual = compiled(raw, raw, None, time)
        torch.testing.assert_close(actual, constraint(raw, raw, None, time))
        gradient, clock_gradient = torch.autograd.grad(actual.sum(), (raw, time))
        self.assertTrue(torch.equal(gradient[:, 12:15], torch.zeros(2, 3)))
        self.assertTrue(bool((clock_gradient > 0).all()))


if __name__ == "__main__":
    unittest.main()

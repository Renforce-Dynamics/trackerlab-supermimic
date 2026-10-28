"""CPU regression checks for storage-preserving running-normalizer updates."""

from io import BytesIO
from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths

configure_paths()

from trackerLab.algorithms.supermimic.networks import RunningNorm


class LegacyRunningNorm(nn.Module):
    """Independent pre-change formula, including its buffer reassignment."""

    def __init__(self, dim, max_count=2e8, eps=1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (
            self.var * self.count
            + v * n
            + delta**2 * self.count * n / tot
        ) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


class RunningNormStorageTests(unittest.TestCase):
    DTYPES = (torch.float32, torch.float64, torch.float16, torch.bfloat16)

    def setUp(self):
        self.generator = torch.Generator().manual_seed(20261028)

    def sample(self, shape, dtype=torch.float32):
        return torch.randn(shape, generator=self.generator).to(dtype)

    def assert_exact(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def assert_same_state(self, actual, expected):
        self.assertEqual(set(actual.state_dict()), {"mean", "var", "count"})
        self.assertEqual(set(actual.state_dict()), set(expected.state_dict()))
        for name in actual.state_dict():
            with self.subTest(buffer=name):
                self.assert_exact(actual.state_dict()[name], expected.state_dict()[name])
                self.assertFalse(getattr(actual, name).requires_grad)
                self.assertIsNone(getattr(actual, name).grad_fn)
        self.assert_exact(actual.std(), expected.std())

    @staticmethod
    def storage_identity(norm):
        return {
            name: (id(buffer), buffer.untyped_storage().data_ptr(), buffer.data_ptr())
            for name, buffer in norm.named_buffers()
        }

    def assert_same_forward_and_input_gradient(self, actual, expected, dtype):
        probe = self.sample((2, 3, 7), dtype)
        actual_input = probe.clone().requires_grad_()
        expected_input = probe.clone().requires_grad_()
        actual_output = actual(actual_input)
        expected_output = expected(expected_input)
        self.assert_exact(actual_output, expected_output)
        weights = torch.linspace(0.2, 1.1, 7, dtype=actual_output.dtype)
        (actual_output.square() * weights).sum().backward()
        (expected_output.square() * weights).sum().backward()
        self.assert_exact(actual_input.grad, expected_input.grad)

    def test_exact_legacy_values_outputs_and_gradients_across_dtypes(self):
        for norm_dtype in self.DTYPES:
            for input_dtype in self.DTYPES:
                with self.subTest(norm_dtype=norm_dtype, input_dtype=input_dtype):
                    actual = RunningNorm(7).to(dtype=norm_dtype)
                    expected = LegacyRunningNorm(7).to(dtype=norm_dtype)
                    for index, shape in enumerate(((4, 7), (3, 2, 7), (1, 7))):
                        batch = (self.sample(shape, input_dtype) + index * 0.25)
                        batch.requires_grad_()
                        actual.update(batch)
                        expected.update(batch)
                        self.assert_same_state(actual, expected)
                        self.assertIsNone(batch.grad)
                        self.assert_same_forward_and_input_gradient(
                            actual, expected, input_dtype,
                        )

    def test_fp32_and_fp64_active_updates_keep_all_buffer_storage(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                norm = RunningNorm(7).to(dtype=dtype)
                original = self.storage_identity(norm)
                for shape in ((2, 7), (4, 3, 7), (1, 7), (5, 7)):
                    norm.update(self.sample(shape, dtype))
                    self.assertEqual(self.storage_identity(norm), original)
                self.assertEqual(norm.count.item(), 20)

    def test_low_precision_variance_keeps_original_fp32_promotion(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                actual = RunningNorm(7).to(dtype=dtype)
                expected = LegacyRunningNorm(7).to(dtype=dtype)
                original = self.storage_identity(actual)
                batch = self.sample((5, 7), dtype)
                actual.update(batch)
                expected.update(batch)
                self.assert_same_state(actual, expected)
                self.assertEqual(actual.var.dtype, torch.float32)
                self.assertEqual(actual.mean.dtype, dtype)
                self.assertEqual(actual.count.dtype, dtype)
                promoted = self.storage_identity(actual)
                self.assertNotEqual(promoted["var"][0], original["var"][0])
                self.assertEqual(promoted["mean"], original["mean"])
                self.assertEqual(promoted["count"], original["count"])
                actual.update(batch + 0.5)
                expected.update(batch + 0.5)
                self.assert_same_state(actual, expected)
                self.assertEqual(self.storage_identity(actual), promoted)

    def test_saturation_is_a_no_op_and_preserves_original_threshold_behavior(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                actual = RunningNorm(7, max_count=8).to(dtype=dtype)
                expected = LegacyRunningNorm(7, max_count=8).to(dtype=dtype)
                for _ in range(2):
                    batch = self.sample((6, 7), dtype)
                    actual.update(batch)
                    expected.update(batch)
                # The original check occurs before the batch, so counts may
                # exceed max_count rather than truncating the crossing batch.
                self.assertEqual(actual.count.item(), 12)
                before = {key: value.clone() for key, value in actual.state_dict().items()}
                identity = self.storage_identity(actual)
                for batch in (
                    torch.full((2, 7), float("nan"), dtype=dtype),
                    self.sample((3, 7), dtype),
                ):
                    actual.update(batch)
                    expected.update(batch)
                    self.assert_same_state(actual, expected)
                    self.assertEqual(self.storage_identity(actual), identity)
                    for name, saved in before.items():
                        self.assert_exact(actual.state_dict()[name], saved)

    def test_legacy_and_current_state_dict_checkpoints_are_compatible(self):
        for dtype in self.DTYPES:
            for source_class in (LegacyRunningNorm, RunningNorm):
                with self.subTest(dtype=dtype, source_class=source_class.__name__):
                    source = source_class(7).to(dtype=dtype)
                    source.update(self.sample((6, 7), dtype))
                    payload = BytesIO()
                    torch.save(source.state_dict(), payload)
                    payload.seek(0)
                    saved = torch.load(payload, map_location="cpu", weights_only=True)
                    actual = RunningNorm(7).to(dtype=dtype)
                    expected = LegacyRunningNorm(7).to(dtype=dtype)
                    # Loading uses the same destination-buffer casting rules
                    # as existing checkpoints, including low-precision ones.
                    for norm in (actual, expected):
                        result = norm.load_state_dict(saved, strict=True)
                        self.assertEqual(result.missing_keys, [])
                        self.assertEqual(result.unexpected_keys, [])
                    self.assert_same_state(actual, expected)
                    self.assert_same_forward_and_input_gradient(actual, expected, dtype)
                    continuation = self.sample((3, 2, 7), dtype)
                    actual.update(continuation)
                    expected.update(continuation)
                    self.assert_same_state(actual, expected)
                    self.assert_same_forward_and_input_gradient(actual, expected, dtype)


if __name__ == "__main__":
    unittest.main()

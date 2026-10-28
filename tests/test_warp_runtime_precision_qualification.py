"""Qualification failure-path tests using CPU-only substitutes for CUDA.

These tests exercise admission logic, comparison checks, and RNG restoration.
They do not qualify a GPU, BF16 arithmetic, CUDA Graphs, or training throughput.
"""

from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "scripts/rls/supermimic/warp_runtime"
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
sys.path.insert(0, str(RUNTIME))
from bootstrap import configure_paths

configure_paths()

from trackerLab.algorithms.supermimic import networks
from legacy_normalizer_reference import LegacyRunningNorm
import canonical_source_contract
import native_bf16_precision
import norm_actual_width_fixture


@contextmanager
def cpu_fixture_harness():
    """Replace device operations solely for admission-control unit testing."""
    arange = torch.arange
    fake_cuda_rng = [torch.tensor([1, 2, 3], dtype=torch.uint8)]

    def cpu_arange(*args, **kwargs):
        kwargs["device"] = "cpu"
        return arange(*args, **kwargs)

    def restore_fake_cuda_rng(states):
        fake_cuda_rng[:] = [value.clone() for value in states]

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
        stack.enter_context(patch.object(
            torch.cuda, "get_rng_state_all",
            side_effect=lambda: [value.clone() for value in fake_cuda_rng],
        ))
        stack.enter_context(patch.object(
            torch.cuda, "set_rng_state_all", side_effect=restore_fake_cuda_rng,
        ))
        stack.enter_context(patch.object(torch.nn.Module, "cuda", lambda module: module))
        stack.enter_context(patch.object(torch, "arange", side_effect=cpu_arange))
        yield fake_cuda_rng


class WarpPrecisionQualificationTests(unittest.TestCase):
    @contextmanager
    def precision_context(self):
        """Set and restore host-side flags; never create CUDA tensors."""
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "NVIDIA_TF32_OVERRIDE": "0",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                "SM_FROZEN_NORM_GUARD": "0",
            }))
            stack.enter_context(patch.object(networks, "AMP_DTYPE", torch.bfloat16))
            stack.enter_context(patch.object(
                torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=False),
            ))
            stack.enter_context(patch.object(
                torch.backends, "cudnn", SimpleNamespace(allow_tf32=False),
            ))
            stack.enter_context(patch.object(torch, "get_float32_matmul_precision", return_value="highest"))
            stack.enter_context(patch.object(torch, "is_autocast_enabled", return_value=False))
            yield

    def contract(self, modules, normalizer_source=None):
        if normalizer_source is None:
            normalizer_source = canonical_source_contract.canonical_contracts(networks)["normalizer"]
        return native_bf16_precision.contract(
            networks, SimpleNamespace(amp="bf16"), modules,
            normalizer_source=normalizer_source,
        )

    def fixture(self, norm_class=networks.RunningNorm):
        # Only the widths are read from wm; the fixture must not change it.
        wm = SimpleNamespace(
            in_norm=SimpleNamespace(mean=torch.zeros(133)),
            out_norm=SimpleNamespace(mean=torch.zeros(70)),
        )
        source = SimpleNamespace(RunningNorm=norm_class)
        with cpu_fixture_harness():
            result = norm_actual_width_fixture.verify_actual_widths(
                source, wm, LegacyRunningNorm.update,
            )
        self.assertTrue(torch.equal(wm.in_norm.mean, torch.zeros(133)))
        self.assertTrue(torch.equal(wm.out_norm.mean, torch.zeros(70)))
        return result

    def test_cpu_substitute_exercises_all_fixture_comparisons(self):
        result = self.fixture()
        self.assertTrue(result["passed"])
        self.assertEqual(result["comparisons"], 90)
        self.assertEqual(len(result["rows"]), 15)
        self.assertEqual(result["widths"], {
            "pi.obs_norm": 99, "pi.goal_norm": 730,
            "wm.in_norm": 133, "wm.out_norm": 70, "critic.norm": 914,
        })
        self.assertTrue(result["cpu_rng_unchanged"])
        self.assertTrue(result["cuda_rng_unchanged"])
        self.assertTrue(result["rng_restored"])
        self.assertFalse(result["optimizer_created"])
        self.assertFalse(result["actual_training_model_mutated"])

    def test_fixture_rejects_reassignment_even_when_values_match(self):
        # Reintroduce the old allocation behavior, with unchanged arithmetic.
        result = self.fixture(LegacyRunningNorm)
        self.assertFalse(result["passed"])
        self.assertTrue(all(
            comparison["exact"]
            for row in result["rows"]
            for comparison in row["comparisons"].values()
        ))
        self.assertTrue(any(
            not row["candidate_storage_preserved"] for row in result["rows"]
        ))

    def test_fixture_rejects_finite_but_wrong_statistics(self):
        class CorruptedNorm(networks.RunningNorm):
            @torch.no_grad()
            def update(self, values):
                super().update(values)
                self.var.add_(0.01)

        result = self.fixture(CorruptedNorm)
        self.assertFalse(result["passed"])
        self.assertTrue(all(
            comparison["both_finite"]
            for row in result["rows"]
            for comparison in row["comparisons"].values()
        ))
        self.assertTrue(any(
            not row["comparisons"]["var"]["exact"] for row in result["rows"]
        ))

    def test_fixture_rejects_rng_mutation_and_restores_cpu_state(self):
        class RandomConsumingNorm(networks.RunningNorm):
            @torch.no_grad()
            def update(self, values):
                super().update(values)
                torch.rand(1)

        before = torch.get_rng_state().clone()
        result = self.fixture(RandomConsumingNorm)
        self.assertFalse(result["passed"])
        self.assertFalse(result["cpu_rng_unchanged"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_fixture_restores_rng_when_a_comparison_raises(self):
        class RaisingNorm(networks.RunningNorm):
            @torch.no_grad()
            def update(self, values):
                torch.rand(1)
                raise RuntimeError("intentional fixture failure")

        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "intentional fixture failure"):
            self.fixture(RaisingNorm)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_source_receipt_uses_shared_methods_without_installing_replacements(self):
        update = networks.RunningNorm.update
        methods = {
            name: getattr(networks.RunningNorm, name)
            for name in ("__init__", "forward", "std")
        }
        result = canonical_source_contract.canonical_contracts(networks)
        self.assertIs(networks.RunningNorm.update, update)
        for name, method in methods.items():
            self.assertIs(getattr(networks.RunningNorm, name), method)
        self.assertFalse(result["normalizer"]["method_replaced_at_runtime"])
        self.assertFalse(result["goal"]["method_replaced_at_runtime"])
        self.assertTrue(result["normalizer"]["fresh_cuda_fixture_required"])
        self.assertEqual(
            result["normalizer"]["network"]["source"],
            result["normalizer"]["update"]["source"],
        )

    def test_source_receipt_rejects_replaced_update_and_mismatched_digest(self):
        with patch.object(networks.RunningNorm, "update", LegacyRunningNorm.update):
            with self.assertRaisesRegex(RuntimeError, "canonical source differs"):
                canonical_source_contract.canonical_contracts(networks)
        with patch.object(canonical_source_contract, "NETWORKS_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "canonical source differs"):
                canonical_source_contract.canonical_contracts(networks)

    def test_source_digest_accepts_only_line_ending_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            raw = b"one\ntwo\n"
            path.write_bytes(raw)
            expected = hashlib.sha256(raw).hexdigest()
            self.assertEqual(canonical_source_contract.source_sha256(path), expected)
            path.write_bytes(raw.replace(b"\n", b"\r\n"))
            self.assertEqual(canonical_source_contract.source_sha256(path), expected)
            path.write_bytes(b"one\r\ntwo \r\n")
            self.assertNotEqual(canonical_source_contract.source_sha256(path), expected)

    def test_contract_rejects_stale_receipt_before_cuda_fixture(self):
        with self.precision_context():
            with patch.object(norm_actual_width_fixture, "verify_actual_widths") as fixture:
                with self.assertRaisesRegex(RuntimeError, "source receipt differs"):
                    self.contract({"wm": None}, normalizer_source={"passed": True})
        fixture.assert_not_called()

    def test_contract_rejects_failed_fresh_fixture(self):
        with self.precision_context():
            with patch.object(
                norm_actual_width_fixture, "verify_actual_widths",
                return_value={"passed": False},
            ) as fixture:
                with self.assertRaisesRegex(RuntimeError, "Fresh CUDA normalizer"):
                    self.contract({"wm": None})
        fixture.assert_called_once_with(networks, None, LegacyRunningNorm.update)

    def test_contract_calls_fresh_fixture_each_time_and_retains_scope_limits(self):
        # A mocked fixture here tests admission orchestration and report scope;
        # the returned report is not a hardware qualification artifact.
        with torch.random.fork_rng(devices=[]):
            module = torch.nn.Linear(4, 2)
        parameters = {name: value.clone() for name, value in module.state_dict().items()}
        update = networks.RunningNorm.update
        with self.precision_context():
            with patch.object(
                norm_actual_width_fixture, "verify_actual_widths",
                return_value={"passed": True},
            ) as fixture:
                for _ in range(2):
                    result = self.contract({"wm": module})
                    self.assertFalse(result["runtime_rng_bitwise_parity_verified"])
                    self.assertFalse(result["model_or_optimizer_mutation"])
                    self.assertFalse(result["normalizer_update_method_replaced_at_runtime"])
                    self.assertFalse(result["custom_tf32_kernel_used"])
                    self.assertIn("actual WM batches", result["qualification"])
        self.assertEqual(fixture.call_count, 2)
        self.assertIs(networks.RunningNorm.update, update)
        for name, original in parameters.items():
            torch.testing.assert_close(module.state_dict()[name], original, rtol=0, atol=0)

    def test_contract_rejects_non_native_linear_and_non_fp32_parameters(self):
        class ModifiedLinear(torch.nn.Linear):
            pass

        with torch.random.fork_rng(devices=[]):
            candidates = (
                (ModifiedLinear(4, 2), "original nn.Linear"),
                (torch.nn.Linear(4, 2).to(dtype=torch.bfloat16), "must remain FP32"),
            )
        with self.precision_context():
            with patch.object(
                norm_actual_width_fixture, "verify_actual_widths", return_value={"passed": True},
            ):
                for module, message in candidates:
                    with self.subTest(message=message):
                        with self.assertRaisesRegex(RuntimeError, message):
                            self.contract({"wm": module})

    def test_contract_rejects_precision_and_frozen_statistics_overrides(self):
        cases = (
            ("NVIDIA_TF32_OVERRIDE", "1", "NVIDIA_TF32_OVERRIDE=0"),
            ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1", "cuBLAS TF32 override"),
            ("SM_FROZEN_NORM_GUARD", "1", "updates must remain active"),
        )
        with self.precision_context():
            with patch.object(norm_actual_width_fixture, "verify_actual_widths") as fixture:
                for variable, value, message in cases:
                    with self.subTest(variable=variable):
                        with patch.dict(os.environ, {variable: value}):
                            with self.assertRaisesRegex(RuntimeError, message):
                                self.contract({"wm": None})
                with patch.object(networks, "AMP_DTYPE", None):
                    with self.assertRaisesRegex(RuntimeError, "requires BF16 autocast"):
                        self.contract({"wm": None})
                with patch.object(torch, "get_float32_matmul_precision", return_value="high"):
                    with self.assertRaisesRegex(RuntimeError, "require IEEE FP32"):
                        self.contract({"wm": None})
                with patch.object(torch, "is_autocast_enabled", return_value=True):
                    with self.assertRaisesRegex(RuntimeError, "require IEEE FP32"):
                        self.contract({"wm": None})
        fixture.assert_not_called()

    def test_admission_requires_cuda_before_creating_any_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "admission.json"
            with patch.object(sys, "argv", ["native_bf16_precision", "--output", str(output)]):
                with patch.object(torch.cuda, "is_available", return_value=False):
                    with patch.object(torch.cuda, "is_bf16_supported") as support:
                        with self.assertRaisesRegex(RuntimeError, "CUDA BF16 support"):
                            native_bf16_precision.main()
            support.assert_not_called()
            self.assertFalse(output.exists())

    def test_admission_requires_bf16_support_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "admission.json"
            with ExitStack() as stack:
                stack.enter_context(patch.object(
                    sys, "argv", ["native_bf16_precision", "--output", str(output)],
                ))
                stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
                stack.enter_context(patch.object(torch.cuda, "is_bf16_supported", return_value=False))
                model = stack.enter_context(patch.object(networks, "WorldModel"))
                with self.assertRaisesRegex(RuntimeError, "CUDA BF16 support"):
                    native_bf16_precision.main()
            model.assert_not_called()
            self.assertFalse(output.exists())

    def test_admission_preserves_existing_receipt_before_device_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "admission.json"
            original = b'{"passed": false, "evidence": "retained"}\n'
            output.write_bytes(original)
            with patch.object(sys, "argv", ["native_bf16_precision", "--output", str(output)]):
                with patch.object(torch.cuda, "is_available") as available:
                    with self.assertRaises(FileExistsError):
                        native_bf16_precision.main()
            available.assert_not_called()
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

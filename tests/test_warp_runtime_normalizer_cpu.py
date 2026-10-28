"""CPU compatibility checks, not CUDA Graph or BF16 numerical qualification."""

from contextlib import nullcontext
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths

configure_paths()

from trackerLab.algorithms.supermimic import networks
from trackerTask.supermimic.robots import get_robot


@torch.no_grad()
def legacy_update(norm, values):
    """Original formula with buffer reassignment, independent of the candidate."""
    values = values.reshape(-1, values.shape[-1]).float()
    if norm.count >= norm.max_count:
        return
    n = values.shape[0]
    mean = values.mean(0)
    variance = values.var(0, unbiased=False)
    total = norm.count + n
    delta = mean - norm.mean
    norm.mean += delta * n / total
    norm.var = (
        norm.var * norm.count
        + variance * n
        + delta**2 * norm.count * n / total
    ) / total
    norm.count = total


class WarpNormalizerCpuCompatibilityTests(unittest.TestCase):
    """Exercise production dimensions without initializing a GPU or simulator."""

    @classmethod
    def setUpClass(cls):
        spec = get_robot("g1")
        goal_dim = 10 * spec.goal_frame_dim
        cls.widths = {
            "pi.obs_norm": 99,
            "pi.goal_norm": goal_dim,
            "wm.in_norm": spec.state_dim + spec.num_joints + spec.context_dim,
            "wm.out_norm": 12 + 2 * spec.num_joints,
            "critic.norm": spec.state_dim + spec.critic_extra + goal_dim,
        }

    def test_dimensions_match_relative_ori99_g1_training_interface(self):
        self.assertEqual(self.widths, {
            "pi.obs_norm": 99,
            "pi.goal_norm": 730,
            "wm.in_norm": 133,
            "wm.out_norm": 70,
            "critic.norm": 914,
        })
        # A small hidden layer has identical state interfaces without allocating
        # the full production network or changing the global RNG.
        with torch.random.fork_rng(devices=[]):
            wm = networks.WorldModel(hidden=(8,))
        self.assertEqual(wm.in_norm.mean.numel(), self.widths["wm.in_norm"])
        self.assertEqual(wm.out_norm.mean.numel(), self.widths["wm.out_norm"])

    def test_actual_width_values_gradients_storage_and_rng_are_exact_on_cpu(self):
        rng_before = torch.get_rng_state().clone()
        update_before = networks.RunningNorm.update
        amp_before = networks.AMP_DTYPE
        for dtype in (torch.float32, torch.float64):
            for name, width in self.widths.items():
                with self.subTest(dtype=dtype, normalizer=name, width=width):
                    epsilon = 1e-4 if name == "wm.out_norm" else 1e-2
                    reference = networks.RunningNorm(width, eps=epsilon).to(dtype=dtype)
                    candidate = networks.RunningNorm(width, eps=epsilon).to(dtype=dtype)
                    buffers = {
                        key: (id(value), value.data_ptr())
                        for key, value in candidate.named_buffers()
                    }
                    base = torch.arange(64 * width, dtype=torch.float32).reshape(64, width)
                    weights = (base.remainder(31) - 15) / 31
                    for step in range(3):
                        values = (base.remainder(127) - 63) / 64 + step / 8
                        legacy_update(reference, values)
                        candidate.update(values)
                        left = (values / 2).clone().requires_grad_()
                        right = left.detach().clone().requires_grad_()
                        left_output, right_output = reference(left), candidate(right)
                        left_grad, = torch.autograd.grad(
                            (left_output * weights).sum(), left,
                        )
                        right_grad, = torch.autograd.grad(
                            (right_output * weights).sum(), right,
                        )
                        pairs = {
                            **{
                                key: (getattr(reference, key), getattr(candidate, key))
                                for key in ("mean", "var", "count")
                            },
                            "std": (reference.std(), candidate.std()),
                            "forward": (left_output, right_output),
                            "input_gradient": (left_grad, right_grad),
                        }
                        for field, (expected, actual) in pairs.items():
                            with self.subTest(field=field, update=step + 1):
                                self.assertEqual(actual.dtype, expected.dtype)
                                self.assertTrue(bool(torch.isfinite(actual).all()))
                                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        for key, value in candidate.named_buffers():
                            self.assertEqual((id(value), value.data_ptr()), buffers[key])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertIs(networks.RunningNorm.update, update_before)
        self.assertIs(networks.AMP_DTYPE, amp_before)

    def test_mlp_precision_wrapper_dispatches_only_its_inner_call(self):
        # This mock verifies context selection and the FP32 return contract.
        # It deliberately makes no CUDA/BF16 arithmetic or throughput claim.
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        result = values.to(dtype=torch.bfloat16)
        net = Mock(return_value=result)
        with patch.object(networks, "AMP_DTYPE", torch.bfloat16):
            with patch.object(torch, "autocast", return_value=nullcontext()) as autocast:
                output = networks.run_mlp(net, values)
        autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)
        net.assert_called_once_with(values)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, result.float(), rtol=0, atol=0)

    def test_mlp_fp32_path_does_not_open_an_autocast_context(self):
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        result = values + 1
        net = Mock(return_value=result)
        with patch.object(networks, "AMP_DTYPE", None):
            with patch.object(torch, "autocast") as autocast:
                output = networks.run_mlp(net, values)
        autocast.assert_not_called()
        self.assertIs(output, result)
        net.assert_called_once_with(values)


if __name__ == "__main__":
    unittest.main()

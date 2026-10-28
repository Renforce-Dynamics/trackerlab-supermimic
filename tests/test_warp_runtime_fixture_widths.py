"""CPU value/storage checks at the five preset-selected G1 Warp normalizer widths."""

import copy
import json
from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.networks import Critic, Policy, WorldModel
from trackerLab.tracker_env.mdp.supermimic import observation_dim
from trackerTask.supermimic.robots import get_robot


@torch.no_grad()
def legacy_update(norm, values):
    """Reference formula before the storage-preservation change."""
    values = values.reshape(-1, values.shape[-1]).float()
    if norm.count >= norm.max_count:
        return
    count = values.shape[0]
    mean, variance = values.mean(0), values.var(0, unbiased=False)
    total = norm.count + count
    delta = mean - norm.mean
    norm.mean += delta * count / total
    norm.var = (
        norm.var * norm.count + variance * count
        + delta**2 * norm.count * count / total
    ) / total
    norm.count = total


class WarpFixtureWidthTests(unittest.TestCase):
    def normalizers(self):
        spec = get_robot("g1")
        preset = json.loads((
            REPO / "scripts/rls/supermimic/configs/g1_warp_bf16_8096.json"
        ).read_text(encoding="utf-8"))
        goal_dim = spec.goal_frame_dim * preset["goal_frames"]
        obs_dim = observation_dim(preset["obs_mode"], spec.num_joints)
        # Hidden layer sizes do not affect WM normalization dimensions.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261028)
            world_model = WorldModel(hidden=(16, 16), num_joints=spec.num_joints)
            policy = Policy(obs_dim, goal_dim, hist=preset["hist"], num_joints=spec.num_joints)
            critic = Critic(spec.state_dim + spec.critic_extra + goal_dim)
        return {
            "pi.obs_norm": policy.obs_norm, "pi.goal_norm": policy.goal_norm,
            "wm.in_norm": world_model.in_norm, "wm.out_norm": world_model.out_norm,
            "critic.norm": critic.norm,
        }

    def test_preset_matches_qualified_actual_normalizer_widths(self):
        norms = self.normalizers()
        self.assertEqual(
            {name: norm.mean.numel() for name, norm in norms.items()},
            {"pi.obs_norm": 99, "pi.goal_norm": 730, "wm.in_norm": 133,
             "wm.out_norm": 70, "critic.norm": 914},
        )
        for name, norm in norms.items():
            with self.subTest(normalizer=name):
                self.assertEqual(norm.max_count, 2e8)
                self.assertEqual(norm.eps, 1e-4 if name == "wm.out_norm" else 1e-2)
                self.assertEqual(norm.update.__func__.__module__, "trackerLab.algorithms.supermimic.networks")

    def test_actual_width_values_gradients_and_storage_match_legacy_formula(self):
        norms = self.normalizers()
        rng_before = torch.get_rng_state().clone()
        comparisons = 0
        for name, candidate in norms.items():
            with self.subTest(normalizer=name):
                reference = copy.deepcopy(candidate)
                pointers = {
                    field: getattr(candidate, field).data_ptr()
                    for field in ("mean", "var", "count")
                }
                width = candidate.mean.numel()
                base = torch.arange(64 * width, dtype=torch.float32).reshape(64, width)
                weights = (base.remainder(31) - 15) / 31
                for step in range(3):
                    values = (base.remainder(127) - 63) / 64 + step / 8
                    legacy_update(reference, values)
                    candidate.update(values)
                    left = (values / 2).clone().requires_grad_(True)
                    right = left.detach().clone().requires_grad_(True)
                    output_left, output_right = reference(left), candidate(right)
                    gradient_left, = torch.autograd.grad((output_left * weights).sum(), left)
                    gradient_right, = torch.autograd.grad((output_right * weights).sum(), right)
                    pairs = {
                        **{
                            field: (getattr(reference, field), getattr(candidate, field))
                            for field in ("mean", "var", "count")
                        },
                        "std": (reference.std(), candidate.std()),
                        "forward": (output_left, output_right),
                        "input_gradient": (gradient_left, gradient_right),
                    }
                    for field, (expected, actual) in pairs.items():
                        with self.subTest(update=step + 1, field=field):
                            self.assertEqual(actual.dtype, torch.float32)
                            self.assertTrue(bool(torch.isfinite(actual).all()))
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            comparisons += 1
                    self.assertEqual(
                        {field: getattr(candidate, field).data_ptr() for field in pointers},
                        pointers,
                    )
                    self.assertEqual(candidate.count.item(), (step + 1) * 64)
        self.assertEqual(comparisons, 90)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))


if __name__ == "__main__":
    unittest.main()

"""CPU checks for fixed-bank world-model validation."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from validate_command_world import evaluate_bank, main
from trackerLab.algorithms.supermimic.networks import RunningNorm, WorldModel


class ToyWorld(torch.nn.Module):
    num_joints = 29
    state_error = staticmethod(WorldModel.state_error)

    def __init__(self, bias=0., variance=1.):
        super().__init__()
        self.bias = bias
        self.out_norm = RunningNorm(70, eps=0.)
        self.out_norm.var.fill_(variance)

    def step(self, state, action, xi):
        p, R, v, w, q, qd = state
        return p + action[:, :3] + xi[:, :1] + self.bias, R, v, w, q, qd


def bank_fixture(steps=80, envs=2):
    actions = torch.arange(steps * envs * 29).reshape(steps, envs, 29).float() % 3
    positions = torch.cat((torch.zeros(1, envs, 3), actions[:, :, :3].cumsum(0)), 0)
    s = torch.zeros(steps, envs, 71)
    s[:, :, 3] = 1.
    s[:, :, :3] = positions[:-1]
    s1 = s.clone()
    s1[:, :, :3] = positions[1:]
    xi = torch.zeros(steps, envs, 33)
    xi[:, :, 0] = .75
    return {"s": s, "s1": s1, "a": actions, "xi": xi,
            "done": torch.zeros(steps, envs, dtype=torch.bool),
            "stealth_mode_id": torch.zeros(steps, envs, dtype=torch.long),
            "c": torch.zeros(steps, envs, 7)}


class CommandWorldValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_model_has_zero_error_and_preserves_buffers(self):
        reference, candidate = ToyWorld(), ToyWorld()
        snapshots = [{key: value.clone() for key, value in world.named_buffers()}
                     for world in (reference, candidate)]
        result = evaluate_bank(reference, candidate, bank_fixture())
        self.assertEqual(result["sampling"]["samples"], 4)
        self.assertEqual(len(result["results"]), 6)
        for row in result["results"]:
            self.assertEqual(row["nonfinite_samples"], 0)
            self.assertEqual(row["common_normalized_cumulative_mse"], 0.)
            self.assertEqual(row["common_normalized_endpoint_mse"], 0.)
            self.assertTrue(all(value == 0. for key, value in row.items() if key.endswith("_rmse")))
        for world, snapshot in zip((reference, candidate), snapshots):
            for key, value in world.named_buffers():
                torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
        large = evaluate_bank(reference, candidate, bank_fixture(800, 90))
        self.assertEqual(large["sampling"]["samples"], 4230)

    def test_bias_grows_with_horizon_using_reference_std_floor(self):
        result = evaluate_bank(ToyWorld(variance=0.), ToyWorld(bias=.25, variance=100.),
                               bank_fixture())
        scale = result["common_output_std"][0]
        self.assertAlmostEqual(scale, .01)
        for row in result["results"][3:]:
            horizon = row["horizon_steps"]
            factor = 3 * (.25 / scale) ** 2 / 70
            self.assertAlmostEqual(row["root_position_m_rmse"], horizon * .25)
            self.assertAlmostEqual(row["common_normalized_endpoint_mse"],
                                   factor * horizon**2, places=5)
            self.assertAlmostEqual(row["common_normalized_cumulative_mse"],
                                   factor * sum(k*k for k in range(1, horizon + 1)), places=5)

    def test_reset_windows_use_same_mask_for_every_horizon(self):
        bank = bank_fixture()
        bank["done"][31, 0] = True
        bank["done"][63, 1] = True
        result = evaluate_bank(ToyWorld(), ToyWorld(), bank)
        self.assertEqual(result["sampling"]["excluded_reset_windows"], 2)
        self.assertTrue(all(row["samples"] == 2 for row in result["results"]))
        bank["done"].fill_(True)
        with self.assertRaisesRegex(ValueError, "no reset-free"):
            evaluate_bank(ToyWorld(), ToyWorld(), bank)

    def test_malformed_nonfinite_and_existing_output_fail(self):
        for field, replacement in (("s1", torch.zeros(80, 2, 70)),
                                   ("xi", torch.zeros(80, 2, 32)),
                                   ("done", torch.full((80, 2), 2))):
            bank = bank_fixture()
            bank[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_bank(ToyWorld(), ToyWorld(), bank)
        bank = bank_fixture()
        del bank["a"]
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate_bank(ToyWorld(), ToyWorld(), bank)
        bank = bank_fixture()
        bank["s"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            evaluate_bank(ToyWorld(), ToyWorld(), bank)
        with self.assertRaisesRegex(FloatingPointError, "nonfinite_samples=4"):
            evaluate_bank(ToyWorld(), ToyWorld(float("inf")), bank_fixture())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                main(["--checkpoint", "unused", "--reference-checkpoint", "unused",
                      "--bank", "unused", "--bank-protocol", "unused", "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()

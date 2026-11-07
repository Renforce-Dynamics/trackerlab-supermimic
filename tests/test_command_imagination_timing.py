"""Check the strict command-horizon boundary and the legacy disabled path."""

from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from train_terrain import command_timing_start_mask


class ImaginationCommandTimingTests(unittest.TestCase):
    def test_strict_boundary_excludes_resampling_inside_horizon_and_margin(self):
        data = {
            "ep_step": torch.tensor([0, 100, 500, 900, 100]),
            "command_time_left_steps": torch.tensor([1, 32, 34, 35, 100], dtype=torch.int32),
        }
        self.assertEqual(command_timing_start_mask(data, 32, 2).tolist(),
                         [False, False, False, True, True])

    def test_legacy_path_does_not_require_new_replay_field(self):
        data = {"ep_step": torch.tensor([1, 2])}
        self.assertEqual(command_timing_start_mask(data, 32, None).tolist(), [True, True])

    def test_enabled_path_rejects_missing_receipts_and_invalid_margin(self):
        data = {"ep_step": torch.tensor([1])}
        with self.assertRaises(KeyError):
            command_timing_start_mask(data, 32, 2)
        for margin in (-1, True, 1.5):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                command_timing_start_mask(data, 32, margin)


if __name__ == "__main__":
    unittest.main()

"""Continuation must wait for the exact completed run and saved weights."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from continue_g1_stealth import source_checkpoint


class StealthContinuationTests(unittest.TestCase):
    def test_waits_until_completion_and_immutable_weights_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertIsNone(source_checkpoint(run, 20000))
            summary = {
                "task": "locomotion/g1/stealth",
                "phase": "completed", "completed_iterations": 20000,
            }
            (run / "run_summary.json").write_text(json.dumps(summary))
            self.assertIsNone(source_checkpoint(run, 20000))
            checkpoint = run / "iteration_020000.pt"
            checkpoint.write_bytes(b"saved-model")
            self.assertEqual(source_checkpoint(run, 20000), checkpoint)

    def test_rejects_wrong_task_incomplete_runs_and_wrong_rounds(self):
        for change in (
            {"task": "locomotion/g1/velocity"},
            {"phase": "stopped"},
            {"completed_iterations": 12000},
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                summary = {
                    "task": "locomotion/g1/stealth",
                    "phase": "completed", "completed_iterations": 20000,
                } | change
                (run / "run_summary.json").write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    source_checkpoint(run, 20000)


if __name__ == "__main__":
    unittest.main()

"""CPU checks for configuration, resource integrity and evaluation denominators."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts/rls/supermimic")]
from bootstrap import configure_paths
configure_paths()

from configuration import apply_config
import prepare_data
from trackerLab.metrics.supermimic import summarize_motion_returns


class ConfigurationTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config")
        parser.add_argument("--num_envs", type=int, default=4096)
        parser.add_argument("--obs_mode", choices=("paper_obs93", "relative_ori99"))
        parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=True)
        return parser

    def test_cli_overrides_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            path.write_text(json.dumps({"num_envs": 8096, "reset_noise": True}))
            argv = ["--config", str(path), "--num_envs", "4", "--no-reset_noise"]
            parser = self.parser()
            apply_config(parser, argv)
            args = parser.parse_args(argv)
            self.assertEqual(args.num_envs, 4)
            self.assertFalse(args.reset_noise)

    def test_bad_presets_fail_before_launch(self):
        cases = [
            {"num_env": 4096}, {"num_envs": 2.5}, {"num_envs": 4.0}, {"num_envs": [4]},
            {"reset_noise": "false"}, {"obs_mode": "typo"}, {"resume": "old.pt"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            for config in cases:
                with self.subTest(config=config):
                    path.write_text(json.dumps(config))
                    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                        apply_config(self.parser(), ["--config", str(path)])


class ResourceTests(unittest.TestCase):
    def test_csv_rejects_invalid_shapes_values_and_quaternions(self):
        good = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0] + [0.0] * 29
        cases = [good[:-1], [float("nan"), *good[1:]], [*good[:6], 0.0, *good[7:]]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "clip.csv")
            for row in cases:
                path.write_text((",".join(map(str, row)) + "\n") * 2)
                with self.assertRaises(ValueError):
                    prepare_data.inspect_csv(path)

    def test_offline_verification_detects_changed_data_and_extra_clips(self):
        row = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0] + [0.0] * 29
        content = ((",".join(map(str, row)) + "\n") * 2).encode()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            prepare_data.LOCK, {"motion_count": 1, "frames": 2}
        ):
            root = Path(directory)
            path = root / "walk.csv"
            path.write_bytes(content)
            manifest = {
                "dataset": prepare_data.REPO_ID, "revision": prepare_data.REVISION,
                "files": [{"file": path.name, "frames": 2,
                           "sha256": hashlib.sha256(content).hexdigest(),
                           "git_blob_sha1": prepare_data.blob_hash(content)}],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            for name in ("SOURCE_README.md", "SOURCE_LICENSE", "source_robot.urdf"):
                (root / name).write_text("source notice")
            prepare_data.verify_data(root)
            path.write_bytes(content + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare_data.verify_data(root)
            path.write_bytes(content)
            (root / "extra.csv").write_bytes(content)
            with self.assertRaisesRegex(ValueError, "exactly"):
                prepare_data.verify_data(root)


class MetricTests(unittest.TestCase):
    def test_early_failure_keeps_full_clip_denominator(self):
        result = summarize_motion_returns(
            ["short", "long"], [1.0, 5.0], [10, 100], [False, True], 0.1
        )
        self.assertAlmostEqual(result["clip_weighted_return"], 0.75)
        self.assertAlmostEqual(result["frame_weighted_return"], 6.0 / 11.0)
        self.assertEqual(result["failed_motions"], ["long"])
        self.assertEqual(result["coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()

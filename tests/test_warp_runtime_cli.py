"""CPU checks for portable Warp CLI, numerical settings and source integrity."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO / "scripts/rls/supermimic/train_warp.py"
RUNTIME = ENTRYPOINT.with_name("warp_runtime")
PRESET = REPO / "scripts/rls/supermimic/configs/g1_warp_bf16_8096.json"
sys.path.insert(0, str(RUNTIME))
from runtime_settings import FIXED_ENVIRONMENT, configure_environment


# Import interception proves that help, validation and dry runs do not initialize
# Torch or either physics implementation, even on machines with them installed.
IMPORT_GUARD = """
import importlib.abc
import runpy
import sys

class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mujoco', 'mujoco_warp', 'warp', 'isaaclab', 'omni'}:
            raise AssertionError('Training import during CLI inspection: ' + fullname)
        return None

sys.meta_path.insert(0, RejectTrainingImports())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


class WarpRuntimeCliTests(unittest.TestCase):
    def command(self, *arguments):
        environment = dict(os.environ)
        for key in (*FIXED_ENVIRONMENT, "TRACKERLAB_BASE_REPO"):
            environment.pop(key, None)
        return subprocess.run(
            [sys.executable, "-B", "-c", IMPORT_GUARD, str(ENTRYPOINT), *map(str, arguments)],
            cwd=REPO, env=environment, text=True, capture_output=True, timeout=30,
        )

    def test_help_without_torch_or_physics_imports(self):
        result = self.command("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--task {g1}", result.stdout)

    def test_preset_dry_run_is_portable_without_training_imports_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never_created"
            result = self.command(
                "--config", PRESET, "--motion_config", "data/motions.json",
                "--motion_root_cache", "data/root83", "--log_dir", output, "--dry-run",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertFalse(value["training_started"])
            self.assertEqual(Path(value["repository"]), REPO)
            self.assertEqual(value["arguments"]["num_envs"], 8096)
            self.assertEqual(value["arguments"]["amp"], "bf16")
            self.assertEqual(value["environment"]["SM_GRAPH_SCOPE"], "wm")
            self.assertEqual(value["environment"]["SM_FROZEN_NORM_GUARD"], "0")
            self.assertEqual(value["environment"]["NVIDIA_TF32_OVERRIDE"], "0")
            self.assertTrue(Path(value["arguments"]["motion_root_cache"]).is_absolute())
            self.assertFalse(output.exists())

    def test_cli_overrides_preset(self):
        result = self.command(
            "--config", PRESET, "--num_envs", "64", "--motion_config", "motions.json",
            "--motion_root_cache", "cache", "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["arguments"]["num_envs"], 64)

    def test_resume_rejected_before_checkpoint_or_physics_loading(self):
        result = self.command(
            "--resume", "unread.pt", "--motion_config", "motions.json",
            "--motion_root_cache", "cache", "--dry-run",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("random initialization", result.stderr)

    def test_other_robots_are_rejected(self):
        result = self.command("--task", "go2", "--motion_root_cache", "cache", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_explicit_motion_configuration_is_required(self):
        result = self.command("--motion_root_cache", "cache", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--motion_config is required", result.stderr)

    def test_unqualified_observation_and_goal_widths_are_rejected(self):
        for flags in (("--obs_mode", "paper_obs93"), ("--goal_frames", "20")):
            with self.subTest(flags=flags):
                result = self.command(
                    *flags, "--motion_config", "motions.json",
                    "--motion_root_cache", "cache", "--dry-run",
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("qualified normalizer fixture", result.stderr)

    def test_mmap_flag_rejected_for_device_resident_runtime(self):
        result = self.command(
            "--motion_storage", "mmap", "--motion_config", "motions.json",
            "--motion_root_cache", "cache", "--dry-run",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("argument --motion_storage: invalid choice", result.stderr)
        # Python 3.11 quotes choices; later argparse versions omit the quotes.
        self.assertRegex(result.stderr, r"""choose from ['"]?device['"]?""")

    def test_invalid_workload_rejected_without_training_imports(self):
        for flags, message in (
            (("--num_envs", "0"), "must be positive"),
            (("--wm_window", "8", "--buffer_steps", "7"), "must cover"),
            (("--utd", "0"), "invalid update ratio"),
        ):
            with self.subTest(flags=flags):
                result = self.command(
                    *flags, "--motion_config", "motions.json",
                    "--motion_root_cache", "cache", "--dry-run",
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Training import during CLI inspection", result.stderr)

    def test_existing_training_artifacts_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            previous = output / "train_log.csv"
            previous.write_text("iteration,loss\n50,0.1\n", encoding="utf-8")
            original = previous.read_bytes()
            result = self.command(
                "--motion_config", "motions.json", "--motion_root_cache", "cache",
                "--log_dir", output, "--dry-run",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("preserve previous artifacts", result.stderr)
            self.assertEqual(previous.read_bytes(), original)

    def test_environment_conflicts_fail_without_partial_changes(self):
        for key in FIXED_ENVIRONMENT:
            with self.subTest(key=key):
                environment = {key: "different"}
                with self.assertRaisesRegex(ValueError, key):
                    configure_environment(environment)
                self.assertEqual(environment, {key: "different"})
        environment = {"OMP_NUM_THREADS": "3"}
        accepted = configure_environment(environment)
        self.assertEqual(accepted["OMP_NUM_THREADS"], "3")
        for key, value in FIXED_ENVIRONMENT.items():
            self.assertEqual(accepted[key], value)

    def test_sources_compile_and_match_integration_manifest(self):
        manifest = json.loads((RUNTIME / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["source_files"])
        self.assertEqual(
            set(manifest["source_files"]), {path.name for path in RUNTIME.glob("*.py")},
        )
        for name, entry in manifest["source_files"].items():
            raw = (RUNTIME / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["integrated_sha256"], name)
            ast.parse(raw, filename=name)
        for relative, entry in manifest["entrypoint_preset_requirements"].items():
            raw = (REPO / relative).read_bytes()
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(), entry["integrated_sha256"], relative,
            )
        for relative, expected in manifest["canonical_shared_source_sha256_lf"].items():
            raw = (REPO / relative).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, relative)
        for path in RUNTIME.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            for private_marker in ("/root/autodl-tmp/", "connect.westd.", "28665"):
                self.assertNotIn(private_marker, source, path.name)
        self.assertNotIn(
            "NORM_PROBE_TERMINAL.json",
            (RUNTIME / "native_bf16_precision.py").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()

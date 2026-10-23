"""Host overlay preserves old exports, detects conflicts and can be repeated."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from install_trackerlab import install, LAZY


class InstallationTests(unittest.TestCase):
    def host(self, directory):
        root = Path(directory)
        for package in ("trackerLab", "trackerTask"):
            folder = root / f"source/{package}/{package}"
            folder.mkdir(parents=True)
            (folder / "__init__.py").write_text("HOST_SENTINEL = 42\n")
        return root

    def test_plan_conflict_and_idempotent_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.host(directory)
            original = root / "source/trackerLab/trackerLab/__init__.py"
            plan = install(root)
            self.assertTrue(plan["files"])
            self.assertFalse((root / ".supermimic-install.json").exists())
            target = root / "source/trackerLab/trackerLab/managers/supermimic.py"
            target.parent.mkdir(parents=True)
            target.write_text("# user customization\n")
            with self.assertRaisesRegex(ValueError, "Existing files differ"):
                install(root, apply=True)
            self.assertFalse((root / "scripts/rls/supermimic/train.py").exists())
            result = install(root, apply=True, replace_extensions=True)
            self.assertEqual(original.read_text(), "HOST_SENTINEL = 42\n")
            self.assertEqual(
                (Path(result["backup"]) / target.relative_to(root)).read_text(),
                "# user customization\n",
            )
            self.assertEqual(install(root, apply=True)["files"], [])
            target.write_text("# changed after installation\n")
            with self.assertRaises(ValueError):
                install(root, apply=True)

    def test_lazy_parent_preserves_exports_and_relative_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.host(directory)
            parent = root / "source/trackerLab/trackerLab/motion_buffer"
            parent.mkdir()
            (parent / "__init__.py").write_text("from .legacy import Sentinel\n")
            (parent / "legacy.py").write_text("Sentinel = 73\n")
            install(root, apply=True)
            code = (
                "import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(root / 'source/trackerLab')!r}); "
                "import trackerLab.motion_buffer as parent; "
                "assert 'trackerLab.motion_buffer.legacy' not in sys.modules; "
                "from trackerLab.motion_buffer import Sentinel; assert Sentinel == 73; "
                "from trackerLab.motion_buffer import *; assert Sentinel == 73; "
                "import trackerLab; assert trackerLab.HOST_SENTINEL == 42"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            self.assertEqual((parent / "_supermimic_original_init.py").read_text(),
                             "from .legacy import Sentinel\n")
            self.assertEqual(install(root, apply=True)["files"], [])

    def test_symlink_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            root = self.host(directory)
            target = root / "scripts"
            try:
                target.symlink_to(other, target_is_directory=True)
            except OSError:
                self.skipTest("Symlink creation unavailable")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                install(root, apply=True)
            self.assertEqual(list(Path(other).iterdir()), [])

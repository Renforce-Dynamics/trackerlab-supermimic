"""Install the same SuperMimic sources in an existing TrackerLab checkout.

The default is a read-only plan. --apply saves backups and writes the plan.
Existing parent package exports are preserved through a lazy initializer:
importing the SuperMimic submodule does not import unrelated tasks or assets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

REPO = Path(__file__).resolve().parents[1]
RECORD = ".supermimic-install.json"
MARKER = "# TrackerLab SuperMimic lazy parent v1"
PARENTS = (
    "source/trackerLab/trackerLab/algorithms",
    "source/trackerLab/trackerLab/kinematics",
    "source/trackerLab/trackerLab/managers",
    "source/trackerLab/trackerLab/metrics",
    "source/trackerLab/trackerLab/motion_buffer",
    "source/trackerLab/trackerLab/motion_buffer/motion_lib",
    "source/trackerLab/trackerLab/tracker_env",
    "source/trackerLab/trackerLab/tracker_env/mdp",
)
EXTENSIONS = (
    "source/trackerLab/trackerLab/algorithms/supermimic",
    "source/trackerLab/trackerLab/kinematics/supermimic",
    "source/trackerLab/trackerLab/managers/supermimic.py",
    "source/trackerLab/trackerLab/metrics/supermimic.py",
    "source/trackerLab/trackerLab/motion_buffer/supermimic.py",
    "source/trackerLab/trackerLab/motion_buffer/motion_lib/supermimic.py",
    "source/trackerLab/trackerLab/tracker_env/supermimic",
    "source/trackerLab/trackerLab/tracker_env/mdp/supermimic.py",
    "source/trackerLab/trackerLab/tracker_env/mdp/supermimic_chase_dodge.py",
    "source/trackerLab/trackerLab/tracker_env/mdp/supermimic_velocity_dodge.py",
    "source/trackerTask/trackerTask/supermimic",
    "scripts/rls/supermimic",
)

LAZY = MARKER + '''
from pathlib import Path as _SMPath
from importlib import import_module as _sm_import
from importlib.util import find_spec as _sm_spec

_sm_loaded = False


def _sm_load():
    global _sm_loaded
    if _sm_loaded:
        return
    _sm_loaded = True
    _source = _SMPath(__file__).with_name("_supermimic_original_init.py")
    try:
        exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
    except BaseException:
        _sm_loaded = False
        raise


def __getattr__(name):
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    if name != "__all__" and _sm_spec(f"{__name__}.{name}") is not None:
        return _sm_import(f"{__name__}.{name}")
    _sm_load()
    if name == "__all__":
        return globals().get("__all__", [key for key in globals() if not key.startswith("_")])
    if name in globals():
        return globals()[name]
    raise AttributeError(f"{__name__} has no attribute {name!r}")


def __dir__():
    _sm_load()
    return sorted(globals())
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def destination(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid destination: {relative}")
    target = root / relative
    # A symlink inside the host could otherwise redirect writes into another checkout.
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Symlink in installation destination: {current}")
        current = current.parent
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Destination escapes the host: {relative}")
    return target


def make_plan(root, source=REPO, replace_extensions=False):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root == source:
        raise ValueError("Standalone sources already have this layout")
    for package in ("trackerLab", "trackerTask"):
        if not (root / f"source/{package}/{package}/__init__.py").is_file():
            raise ValueError(f"Expected an existing TrackerLab checkout: missing {package}")
    record_path = destination(root, RECORD)
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    installed = previous.get("installed_sha256", {})
    desired = {}
    for relative in EXTENSIONS:
        item = source / relative
        paths = item.rglob("*") if item.is_dir() else [item]
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                desired[path.relative_to(source).as_posix()] = path.read_bytes()
    parent_changes = set()
    for relative in PARENTS:
        name = relative + "/__init__.py"
        target = destination(root, name)
        if not target.exists():
            desired[name] = (source / name).read_bytes()
            continue
        content = target.read_bytes()
        if content.decode("utf-8").startswith(MARKER):
            preserved = relative + "/_supermimic_original_init.py"
            original = destination(root, preserved)
            if not original.is_file():
                raise ValueError(f"Missing preserved parent initializer: {relative}")
            if content != LAZY.encode() or installed.get(preserved) != digest(original.read_bytes()):
                raise ValueError(f"Parent initializer was modified after installation: {relative}")
            continue
        if installed.get(name) == digest(content) and content == (source / name).read_bytes():
            continue
        preserved = relative + "/_supermimic_original_init.py"
        if destination(root, preserved).exists():
            raise ValueError(f"Preserved initializer already exists: {preserved}")
        desired[preserved] = content
        desired[name] = LAZY.encode()
        parent_changes.add(name)
    changes = {}
    conflicts = []
    for relative, content in desired.items():
        target = destination(root, relative)
        current = target.read_bytes() if target.is_file() else None
        if current == content:
            continue
        managed = current is not None and installed.get(relative) == digest(current)
        if current is not None and relative not in parent_changes and not managed and not replace_extensions:
            conflicts.append(relative)
        changes[relative] = content
    if conflicts:
        raise ValueError("Existing files differ; inspect them or use --replace_extensions with backups:\n"
                         + "\n".join(conflicts))
    hashes = dict(installed)
    hashes.update({name: digest(content) for name, content in desired.items()})
    return changes, hashes


def install(root, apply=False, replace_extensions=False, source=REPO):
    root = Path(root).resolve()
    changes, hashes = make_plan(root, source, replace_extensions)
    result = {"host": str(root), "mode": "apply" if apply else "plan", "files": sorted(changes)}
    if not apply or not changes:
        return result
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    backup = destination(root, ".supermimic-backups/" + stamp)
    originals = {}
    # Back up all originals before the first replacement.
    for name in [*changes, RECORD]:
        target = destination(root, name)
        originals[name] = target.read_bytes() if target.exists() else None
        if target.exists():
            copy = backup / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, copy)
    try:
        for name, content in changes.items():
            target = destination(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        record = {
            "version": 1, "installed_at_utc": stamp,
            "installed_sha256": hashes, "backup": str(backup.relative_to(root)),
        }
        destination(root, RECORD).write_text(json.dumps(record, indent=2) + "\n")
    except BaseException:
        for name, content in originals.items():
            target = destination(root, name)
            if content is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(content)
        raise
    result["backup"] = str(backup)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trackerlab", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--replace_extensions", action="store_true",
                        help="Back up and replace conflicting SuperMimic extension files")
    args = parser.parse_args()
    print(json.dumps(install(args.trackerlab, args.apply, args.replace_extensions), indent=2))


if __name__ == "__main__":
    main()

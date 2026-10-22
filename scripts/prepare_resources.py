"""Fetch pinned G1 assets and original motions, or verify an existing offline installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_data import fetch, verify_data

REPO = Path(__file__).resolve().parents[1]
LOCK = json.loads((REPO / "resources.lock.json").read_text(encoding="utf-8"))


def git(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def prepare_assets(directory: Path, verify_only: bool) -> dict:
    config = LOCK["assets"]
    if not directory.exists():
        if verify_only:
            raise FileNotFoundError(f"Asset checkout missing: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir()
        git(directory, "init", "--quiet")
        git(directory, "remote", "add", "origin", config["repository"])
        git(directory, "sparse-checkout", "init", "--cone")
        git(directory, "sparse-checkout", "set", config["subdirectory"])
        git(directory, "fetch", "--depth", "1", "origin", config["revision"])
        git(directory, "checkout", "--quiet", "--detach", config["revision"])
    if git(directory, "rev-parse", "HEAD") != config["revision"]:
        raise ValueError(f"{directory} must be at {config['revision']}; use a separate resource directory")
    dirty = git(directory, "status", "--porcelain", "--untracked-files=no", "--", config["subdirectory"])
    if dirty:
        raise ValueError(f"Tracked G1 assets have local modifications in {directory}")
    robot = directory / config["subdirectory"]
    for relative in config["required"]:
        path = robot / relative
        with path.open("rb") as stream:
            header = stream.read(128)
        if not header or header.startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"Asset is empty or an unresolved Git LFS pointer: {path}")
    hashes = {}
    for path in sorted(robot.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                hashes[path.relative_to(directory).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"root": str(directory), "revision": config["revision"], "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_root", type=Path, default=Path(
        os.environ.get("SUPERMIMIC_ASSETLIB_ROOT", REPO / "data/assetslib")))
    parser.add_argument("--motion_dir", type=Path, default=REPO / "data/lafan1_g1")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--with_reference", action="store_true", help="Fetch pinned vibed MDP for optional parity checks")
    args = parser.parse_args()
    args.asset_root = args.asset_root.expanduser().resolve()
    args.motion_dir = args.motion_dir.expanduser().resolve()
    assets = prepare_assets(args.asset_root, args.verify_only)
    command = [sys.executable, str(REPO / "scripts/prepare_data.py"), "--output", str(args.motion_dir)]
    if args.verify_only:
        command.append("--verify_only")
    subprocess.run(command, check=True)
    motions = verify_data(args.motion_dir)
    if args.with_reference:
        config = LOCK["adaptation"]
        target = REPO / "data/supermimic_reference/mdp.py"
        if args.verify_only:
            content = target.read_bytes()
        else:
            content = fetch(
                "https://raw.githubusercontent.com/zita-ch/supermimic_vibed/"
                + config["revision"] + "/supermimic/mdp.py"
            )
        if hashlib.sha256(content).hexdigest() != config["mdp_sha256"]:
            raise ValueError("Pinned reference MDP hash mismatch")
        if not args.verify_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    manifest = {
        "assets": assets,
        "motions": {"root": str(args.motion_dir), "revision": motions["revision"],
                    "motion_count": motions["motion_count"], "total_frames": motions["total_frames"]},
        "reference_mdp_checked": args.with_reference,
    }
    output = REPO / "data/supermimic_resources.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Resources verified. Asset root: {args.asset_root}\nMotions: {args.motion_dir}")
    print("For a custom asset directory, set SUPERMIMIC_ASSETLIB_ROOT to the asset root above.")


if __name__ == "__main__":
    main()

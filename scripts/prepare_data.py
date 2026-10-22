"""Download and validate the pinned, public G1 LAFAN1 retargets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
LOCK = json.loads((REPO / "resources.lock.json").read_text(encoding="utf-8"))["motions"]
REPO_ID = LOCK["repository"]
REVISION = LOCK["revision"]
BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}"


def fetch(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "TrackerLab-SuperMimic"}), timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("download failed")


def blob_hash(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def inspect_csv(path: Path) -> dict:
    rows = 0
    max_quat_error = 0.0
    with path.open(newline="", encoding="utf-8") as stream:
        for rows, row in enumerate(csv.reader(stream), 1):
            values = [float(value) for value in row]
            if len(values) != 36 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{rows}: expected 36 finite values")
            norm = math.sqrt(sum(value * value for value in values[3:7]))
            if norm < 0.99 or norm > 1.01:
                raise ValueError(f"{path}:{rows}: invalid quaternion norm {norm}")
            max_quat_error = max(max_quat_error, abs(norm - 1))
    if rows < 2:
        raise ValueError(f"{path}: expected at least two frames")
    return {
        "frames": rows,
        "duration_seconds": (rows - 1) / 30.0,
        "max_quaternion_norm_error": max_quat_error,
    }


def verify_data(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION or manifest["dataset"] != REPO_ID:
        raise ValueError("Motion manifest does not match resources.lock.json")
    names = {record["file"] for record in manifest["files"]}
    if len(names) != LOCK["motion_count"] or names != {p.name for p in directory.glob("*.csv")}:
        raise ValueError("Expected exactly the pinned 40 G1 CSVs; additional files change the experiment")
    for record in manifest["files"]:
        if Path(record["file"]).name != record["file"]:
            raise ValueError("Manifest filenames must be plain CSV basenames")
        path = directory / record["file"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"] or blob_hash(content) != record["git_blob_sha1"]:
            raise ValueError(f"Motion hash mismatch: {path}")
        actual = inspect_csv(path)
        if actual["frames"] != record["frames"]:
            raise ValueError(f"Motion frame count mismatch: {path}")
    if sum(record["frames"] for record in manifest["files"]) != LOCK["frames"]:
        raise ValueError("Total source frame count differs from the pinned dataset")
    for name in ("SOURCE_README.md", "SOURCE_LICENSE", "source_robot.urdf"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "data" / "lafan1_g1",
    )
    parser.add_argument("--verify_only", action="store_true", help="Offline integrity check using manifest.json")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_data(args.output)
        print(f"Verified {manifest['motion_count']} clips in {args.output}", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    entries = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{REVISION}/g1?limit=100"))
    files = sorted(
        (entry for entry in entries if entry["type"] == "file" and entry["path"].endswith(".csv")),
        key=lambda entry: entry["path"],
    )
    if len(files) != LOCK["motion_count"]:
        raise RuntimeError(f"expected {LOCK['motion_count']} G1 clips at pinned revision, got {len(files)}")
    extra = {p.name for p in args.output.glob("*.csv")} - {Path(entry["path"]).name for entry in files}
    if extra:
        raise ValueError(f"Unexpected CSVs in the pinned data directory: {sorted(extra)}")
    records = []
    for index, entry in enumerate(files, 1):
        path = args.output / Path(entry["path"]).name
        content = path.read_bytes() if path.exists() else b""
        expected = entry["oid"]
        if blob_hash(content) != expected:
            content = fetch(f"{BASE}/{entry['path']}")
            if blob_hash(content) != expected:
                raise ValueError(f"source hash mismatch: {entry['path']}")
            temp = path.with_suffix(".csv.part")
            temp.write_bytes(content)
            temp.replace(path)
        record = inspect_csv(path)
        record.update(
            file=path.name, bytes=len(content), git_blob_sha1=expected,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        records.append(record)
        print(f"[{index:02d}/40] {path.name}: {record['frames']} frames", flush=True)
    for remote, local in [
        ("README.md", "SOURCE_README.md"),
        ("LICENSE", "SOURCE_LICENSE"),
        ("robot_description/g1/g1_29dof_rev_1_0.urdf", "source_robot.urdf"),
    ]:
        (args.output / local).write_bytes(fetch(f"{BASE}/{remote}"))
    manifest = {
        "dataset": REPO_ID,
        "revision": REVISION,
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "motion_license": "CC BY-NC-ND 4.0 (source README); preserve original CSVs",
        "fps": 30,
        "quaternion_order": "xyzw",
        "columns": 36,
        "motion_count": len(records),
        "total_frames": sum(record["frames"] for record in records),
        "duration_hours": sum(record["duration_seconds"] for record in records) / 3600,
        "files": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_data(args.output)
    print(f"Validated {len(records)} clips, {manifest['duration_hours']:.6f} hours in {args.output}", flush=True)


if __name__ == "__main__":
    main()

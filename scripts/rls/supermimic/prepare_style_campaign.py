"""Build bounded, immutable expert pools and atomically update style-map.json."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from bootstrap import configure_paths

REPO = configure_paths()
from prepare_gait_references import digest, motion_config, verify_audit
from prepare_style_references import (
    OFFICIAL_ARCHIVE_SHA256, OFFICIAL_REVISION, ORIGINAL_AIMING_SHA256,
    align_constant_floor, physical_floor_distances, resample_g1, source_group,
)
from trackerTask.supermimic.robots import get_robot

ROBOTS = ("g1", "a3", "t1", "bumi", "pi_plus", "go2")
FPS = 50
DEFAULT_HOLDOUT = ("walk2", "walk4")
TRAIN_FAMILIES = ("jumps", "obstacles", "run", "walk")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_bytes(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=convert) + "\n").encode()


def _identity(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _basename(value):
    if (not isinstance(value, str) or not value or Path(value).name != value
            or "/" in value or "\\" in value or value in (".", "..")):
        raise ValueError("Manifest filenames must be plain basenames")
    return value


def _local_file(root, name):
    path = (root / _basename(name)).resolve()
    if path.parent != root.resolve():
        raise ValueError("Source file escapes its declared directory")
    return path


def _validate_values(values, robot):
    if (values.ndim != 2 or values.shape[1] != 7 + get_robot(robot).num_joints
            or len(values) < 7 or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, atol=1e-3)):
        raise ValueError(f"Invalid {robot} motion: expected finite robot joints and unit XYZW quaternions")


def _read_motion(path, robot, expected=None, frames=None):
    if expected is not None and digest(path) != expected:
        raise ValueError(f"Motion hash mismatch: {path.name}")
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    _validate_values(values, robot)
    if frames is not None and len(values) != frames:
        raise ValueError(f"Motion frame count mismatch: {path.name}")
    return values


def read_config(path, robot, *, require_50=True):
    path = Path(path).resolve()
    cfg = _read_json(path)
    allowed = {"version", "root", "source_fps", "quaternion_order", "joint_names", "files"}
    if (not isinstance(cfg, dict) or set(cfg) != allowed or cfg["version"] != 1
            or cfg["quaternion_order"] != "xyzw"
            or cfg["joint_names"] != list(get_robot(robot).joint_names)
            or not isinstance(cfg["root"], str) or not cfg["root"]
            or not isinstance(cfg["files"], list) or not cfg["files"]):
        raise ValueError(f"Invalid {robot} motion config: {path}")
    fps = cfg["source_fps"]
    if (isinstance(fps, bool) or not isinstance(fps, (float, int))
            or not math.isfinite(fps) or fps <= 0 or (require_50 and fps != FPS)):
        raise ValueError("Campaign source configuration must declare the required frame rate")
    if len(cfg["files"]) != len(set(cfg["files"])):
        raise ValueError("Duplicate motion names in config")
    root = (path.parent / cfg["root"]).resolve()
    for name in cfg["files"]:
        _local_file(root, name)
    return cfg, root


def _family(group):
    name = re.sub(r"\d+$", "", group).lower()
    return "run" if name == "sprint" else name


@dataclass
class Clip:
    path: Path
    values: np.ndarray
    record: dict
    split: str
    group: str
    family: str
    begin: int
    end: int
    metadata: dict

    @property
    def selected(self):
        return self.values[self.begin:self.end]

    @property
    def original_id(self):
        return self.record["source_sha256"]

    def descriptor(self):
        return {
            "prepared_file": self.path.name, "prepared_sha256": digest(self.path),
            "prepared_frame_range": [self.begin, self.end], "prepared_fps": FPS,
            "source_file": self.record["source_file"],
            "source_sha256": self.record["source_sha256"], "source_group": self.group,
            "family": self.family, "split": self.split,
            "upstream_split": self.record.get("split"),
            "frames": self.end - self.begin,
            "vertical_range_m": float(np.ptp(self.selected[:, 2])),
        }


def crop_window(values, limit, family):
    """Choose one contiguous window without changing sampling times."""
    count = min(len(values), limit)
    if count == len(values):
        return 0, count
    if family in ("jumps", "obstacles"):
        windows = np.lib.stride_tricks.sliding_window_view(values[:, 2], count)
        scores = np.ptp(windows, axis=1)
    else:
        distances = np.r_[0., np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)].cumsum()
        scores = distances[count - 1:] - distances[:len(values) - count + 1]
    begin = int(np.argmax(scores))
    return begin, begin + count


def load_style(directory, style, holdout=DEFAULT_HOLDOUT, clip_frames=400, *,
               require_all_holdouts=True):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    if (manifest.get("status") != "ready" or manifest.get("robot") != "g1"
            or manifest.get("style") != style or manifest.get("output_fps") != FPS
            or manifest.get("quaternion_order") != "xyzw"
            or manifest.get("joint_names") != list(get_robot("g1").joint_names)):
        raise ValueError(f"{style} needs a ready G1 prepared manifest at 50 Hz")
    membership = {}
    for split in ("train", "test"):
        config, root = read_config(directory / f"{split}.json", "g1")
        if root != directory:
            raise ValueError("Prepared G1 configuration must refer to its manifest directory")
        for name in config["files"]:
            if name in membership:
                raise ValueError("Prepared source configs overlap")
            membership[name] = split
    if style == "stealth":
        official = manifest.get("reviewed_official_source") or {}
        if (official.get("revision") != OFFICIAL_REVISION
                or official.get("archive_sha256") != OFFICIAL_ARCHIVE_SHA256):
            raise ValueError("Stealth source lacks verified original Aiming archive provenance")
    if style == "parkour" and "jumps1" in holdout:
        raise ValueError("jumps1 must be training data in this campaign")
    candidates, seen, identity_splits = [], set(), {}
    metadata = {
        "source_manifest_sha256": digest(manifest_path),
        "source_manifest": str(manifest_path),
        "dataset": manifest.get("dataset"), "revision": manifest.get("revision"),
        "license": manifest.get("license"), "style": style,
    }
    for row in manifest.get("clips", []):
        name = _basename(row["file"])
        _basename(row["source_file"])
        if (not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_sha256", "")))
                or not isinstance(row.get("source_window_frames"), list)
                or len(row["source_window_frames"]) != 2
                or any(type(value) is not int for value in row["source_window_frames"])
                or not 0 <= row["source_window_frames"][0] < row["source_window_frames"][1]
                or isinstance(row.get("source_fps"), bool)
                or not isinstance(row.get("source_fps"), (int, float))
                or not math.isfinite(row["source_fps"]) or row["source_fps"] <= 0):
            raise ValueError("Prepared clip needs a valid original hash, frame interval and FPS")
        if name in seen or membership.get(name) != row.get("split") or row.get("fps") != FPS:
            raise ValueError("Prepared clip membership, uniqueness or FPS is inconsistent")
        seen.add(name)
        path = _local_file(directory, name)
        values = _read_motion(path, "g1", row["sha256"], row["frames"])
        group = row["source_group"]
        if not isinstance(group, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", group):
            raise ValueError("Each source needs its original take group")
        if style == "stealth":
            origin = row.get("origin") or {}
            original_name = origin.get("source_file")
            expected_split = "test" if group == "aiming1" else "train" if group == "aiming2" else None
            if (original_name not in ORIGINAL_AIMING_SHA256
                    or origin.get("source_sha256") != ORIGINAL_AIMING_SHA256[original_name]
                    or group != source_group(original_name)
                    or expected_split is None or row.get("split") != expected_split
                    or (origin.get("split") is not None and origin["split"] != expected_split)
                    or not (row.get("upper_body_handling") or {}).get("source_posture_retained")):
                raise ValueError("Stealth clips must retain verified Aiming provenance, take split and arm posture")
        elif group != source_group(row["source_file"]):
            raise ValueError("Parkour take group differs from original source identity")
        family = _family(group)
        if style == "parkour" and family not in TRAIN_FAMILIES:
            continue
        split = ("test" if group in holdout else "train") if style == "parkour" else row["split"]
        for identity in (("source", row["source_sha256"]), ("prepared", row["sha256"]), ("group", group)):
            prior = identity_splits.setdefault(identity, split)
            if prior != split:
                raise ValueError("Source identity crosses campaign train/test splits")
        begin, end = crop_window(values, clip_frames, family)
        candidates.append(Clip(path, values, row, split, group, family, begin, end, metadata))
    if not candidates:
        raise ValueError(f"No valid {style} clips in prepared manifest")
    if (style == "parkour" and require_all_holdouts
            and set(holdout) - {clip.group for clip in candidates}):
        raise ValueError("Requested parkour holdout groups are absent from prepared inputs")
    return candidates, metadata


def load_parkour_sources(primary, additional=(), holdout=DEFAULT_HOLDOUT, clip_frames=400):
    """Merge source pools; the first pool owns all windows for a source/group."""
    candidates, metadata = [], []
    owners, identity_splits, source_groups, visited = {}, {}, {}, set()
    for directory in (primary, *additional):
        directory = Path(directory).resolve()
        if directory in visited:
            continue
        visited.add(directory)
        clips, receipt = load_style(
            directory, "parkour", holdout, clip_frames, require_all_holdouts=False)
        retained, duplicates = 0, 0
        for clip in clips:
            source_hash = clip.record["source_sha256"]
            group = source_groups.setdefault(source_hash, clip.group)
            if group != clip.group:
                raise ValueError("Original source hash names conflicting groups across parkour pools")
            for identity in (("source", source_hash), ("prepared", clip.record["sha256"]),
                             ("group", clip.group)):
                split = identity_splits.setdefault(identity, clip.split)
                if split != clip.split:
                    raise ValueError("Source identity crosses combined parkour train/test splits")
            owner = owners.setdefault((source_hash, clip.group), directory)
            if owner != directory:
                duplicates += 1
                continue
            candidates.append(clip)
            retained += 1
        metadata.append(dict(receipt, retained_clips=retained, duplicate_source_clips=duplicates))
    if set(holdout) - {clip.group for clip in candidates}:
        raise ValueError("Requested parkour holdout groups are absent from combined prepared inputs")
    return candidates, {
        "source_manifests": metadata,
        "deduplication": "Original SHA256/group pair; first pool wins, retaining its distinct windows.",
        "holdout_groups": sorted(set(holdout)),
    }


def select_bounded(candidates, *, max_per_source=2, frame_budget=4000):
    """Reserve coverage for each train family and held-out take, then diversify."""
    def rank(clip):
        return (clip.family == "jumps" and clip.group != "jumps1",
                -float(np.ptp(clip.selected[:, 2])) if clip.family in ("jumps", "obstacles") else 0,
                -(clip.end - clip.begin), clip.record["source_file"], clip.path.name, clip.begin)

    buckets = defaultdict(list)
    for clip in candidates:
        buckets[(clip.split, clip.family if clip.split == "train" else clip.group)].append(clip)
    if not any(key[0] == "train" for key in buckets) or not any(key[0] == "test" for key in buckets):
        raise ValueError("Bounded styles require disjoint train and test recordings")
    for pool in buckets.values():
        pool.sort(key=rank)
    selected, counts, used, total = [], Counter(), set(), 0
    source_ranges = defaultdict(list)

    def usable(clip):
        key = (str(clip.path), clip.begin, clip.end)
        if key in used or counts[clip.original_id] >= max_per_source:
            return False
        raw_start, _ = clip.record["source_window_frames"]
        raw_fps = clip.record["source_fps"]
        a = raw_start / raw_fps + clip.begin / FPS
        b = raw_start / raw_fps + clip.end / FPS
        return not any(a < end and b > start for start, end in source_ranges[clip.original_id])

    def append(clip):
        nonlocal total
        selected.append(clip)
        counts[clip.original_id] += 1
        used.add((str(clip.path), clip.begin, clip.end))
        total += clip.end - clip.begin
        raw_start = clip.record["source_window_frames"][0] / clip.record["source_fps"]
        source_ranges[clip.original_id].append((raw_start + clip.begin / FPS, raw_start + clip.end / FPS))

    keys = sorted(buckets, key=lambda key: (key[0] == "test", key[1]))
    for key in keys:
        clip = next((clip for clip in buckets[key] if usable(clip)), None)
        if clip is None or total + clip.end - clip.begin > frame_budget:
            raise ValueError("Frame budget cannot cover every available training family and held-out group")
        append(clip)
    while True:
        added = False
        for key in keys:
            choices = sorted(buckets[key], key=lambda clip: (counts[clip.original_id], *rank(clip)))
            clip = next((clip for clip in choices if usable(clip)
                         and total + clip.end - clip.begin <= frame_budget), None)
            if clip is not None:
                append(clip)
                added = True
        if not added:
            break
    selected.sort(key=lambda clip: (clip.split, clip.group, clip.path.name, clip.begin))
    aiming_only = all(clip.metadata["style"] == "stealth" for clip in candidates)
    walk_only = all(clip.family == "walk" for clip in candidates)
    expected_families = {"aiming"} if aiming_only else {"walk"} if walk_only else set(TRAIN_FAMILIES)
    report = {
        "selected_frames": total, "frame_budget": frame_budget,
        "max_clips_per_original_source": max_per_source,
        "training_families": sorted({clip.family for clip in selected if clip.split == "train"}),
        "held_out_groups": sorted({clip.group for clip in selected if clip.split == "test"}),
        "missing_training_families": sorted(expected_families - {
            clip.family for clip in selected if clip.split == "train"}),
        "scope": "Contiguous crops at 50 Hz; timing preserved; source labels describe provenance.",
        "split_revision": ("Verified Aiming source splits preserved." if aiming_only else
                           "Parkour uses campaign take holdouts; upstream splits retained in each record."),
    }
    if total < min(2000, frame_budget):
        report["limitation"] = "Available disjoint sources did not fill the bounded target frame budget."
    return selected, report


def _verify_split(records, *, require_both=True):
    sides = defaultdict(dict)
    for row in records:
        if row["split"] not in ("train", "test"):
            raise ValueError("Every output needs an explicit train/test split")
        for field in ("sha256", "original_sha256", "source_group", "raw_sha256", "source_pose_sha256"):
            if row.get(field):
                previous = sides[field].setdefault(row[field], row["split"])
                if previous != row["split"]:
                    raise ValueError(f"Identical {field} crosses train/test splits")
        for field in ("original_sha256", "raw_sha256", "source_pose_sha256"):
            if row.get(field):
                previous = sides["source_content"].setdefault(row[field], row["split"])
                if previous != row["split"]:
                    raise ValueError("Identical source content crosses train/test splits")
    if require_both and {row["split"] for row in records} != {"train", "test"}:
        raise ValueError("Both expert splits must contain recorded motion")


def gait_inputs(directory, robot):
    directory = Path(directory).resolve()
    audit = verify_audit(directory / "train.json", robot)
    inputs = []
    for split, field in (("train", "training_files"), ("test", "evaluation_files")):
        cfg, root = read_config(directory / f"{split}.json", robot)
        if root != directory or cfg["files"] != audit[field]:
            raise ValueError("Gait configs differ from audited split membership")
        for name in cfg["files"]:
            path = _local_file(directory, name)
            receipt = audit["records"][name]
            _read_motion(path, robot, receipt["sha256"], receipt["frames"])
            inputs.append({
                "path": path, "split": split, "sha256": receipt["sha256"],
                "frames": receipt["frames"], "source_group": f"gait:{Path(name).stem}",
                "provenance": receipt, "audit_sha256": digest(directory / "gait_audit.json"),
            })
    _verify_split([dict(row, original_sha256=row["sha256"]) for row in inputs])
    return inputs


def retarget(values, robot):
    from retarget_gait_references import retarget as mink_retarget
    return mink_retarget(values, robot)


def _atomic_json(path, value):
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


@contextmanager
def _campaign_lock(output):
    path = output / ".prepare.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Another campaign preparation owns .prepare.lock; use sequential --robots calls") from error
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def _require_ignored_output(output):
    if not output.is_relative_to((REPO / "data").resolve()):
        raise ValueError("Campaign derivatives must remain in the ignored repository data/ tree")
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--quiet", "--no-index", str(output)],
        capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError("Campaign output must be ignored by Git")


def verify_pool(directory, robot, style):
    directory = Path(directory)
    manifest = _read_json(directory / "manifest.json")
    if (manifest.get("schema") != "style-campaign-pool-v1" or manifest.get("robot") != robot
            or manifest.get("style") != style or manifest.get("status") != "ready"
            or manifest.get("model_sha256") != digest(get_robot(robot).mjcf_path)
            or _identity(manifest.get("recipe")) != manifest.get("recipe_sha256")
            or manifest.get("fps") != FPS or manifest.get("quaternion_order") != "xyzw"
            or manifest.get("joint_names") != list(get_robot(robot).joint_names)):
        raise ValueError("Existing campaign pool identity differs")
    records = manifest["files"]
    _verify_split(records)
    if len({row["file"] for row in records}) != len(records):
        raise ValueError("Duplicate output filenames")
    for row in records:
        _read_motion(_local_file(directory, row["file"]), robot, row["sha256"], row["frames"])
    for split in ("train", "test"):
        cfg_path = directory / f"{split}.json"
        cfg, root = read_config(cfg_path, robot)
        if (root != directory.resolve()
                or cfg["files"] != [row["file"] for row in records if row["split"] == split]
                or digest(cfg_path) != manifest["configs"][split]["sha256"]):
            raise ValueError("Campaign config hash or membership differs")
    return manifest


def _build_pool(output, robot, style, recipe, writer):
    identity = _identity(recipe)
    parent = output / "pools" / robot
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{style}-{identity[:20]}"
    if destination.exists():
        manifest = verify_pool(destination, robot, style)
        if manifest["recipe_sha256"] != identity:
            raise ValueError("Existing pool recipe differs")
        return destination, manifest
    staging = Path(tempfile.mkdtemp(prefix=f".{style}-building-", dir=parent))
    records, details = writer(staging)
    _verify_split(records)
    configs = {}
    for split in ("train", "test"):
        path = staging / f"{split}.json"
        path.write_bytes(_json_bytes(motion_config(robot, [
            row["file"] for row in records if row["split"] == split])))
        configs[split] = {"file": path.name, "sha256": digest(path)}
    manifest = {
        "schema": "style-campaign-pool-v1", "status": "ready", "robot": robot, "style": style,
        "fps": FPS, "quaternion_order": "xyzw", "joint_names": list(get_robot(robot).joint_names),
        "model_sha256": digest(get_robot(robot).mjcf_path), "recipe_sha256": identity,
        "recipe": recipe, "files": records, "configs": configs, **details,
        "distribution": "Local research only; upstream licenses continue to apply.",
        "scope": "Expert motion preparation; physical tracking and actuator feasibility remain untested.",
    }
    (staging / "manifest.json").write_bytes(_json_bytes(manifest))
    verify_pool(staging, robot, style)
    staging.rename(destination)
    return destination, manifest


def _copy_gaits(directory, inputs, prefix="gait"):
    records = []
    for index, row in enumerate(inputs):
        path = row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Gait source changed after validation")
        name = f"{prefix}_{index:03d}__{path.name}"
        destination = directory / name
        shutil.copyfile(path, destination)
        records.append({
            "file": name, "sha256": digest(destination), "frames": row["frames"], "fps": FPS,
            "duration_seconds": (row["frames"] - 1) / FPS, "split": row["split"],
            "source_file": str(path), "source_group": row["source_group"],
            "original_sha256": row["sha256"], "source_frame_range": [0, row["frames"]],
            "source_fps": FPS, "gait_audit_sha256": row["audit_sha256"],
            "upstream_record": row["provenance"], "transform": "byte-preserving copy",
        })
    return records


def _write_styles(directory, clips, robot, report):
    records = []
    for index, clip in enumerate(clips):
        source = _read_motion(clip.path, "g1", clip.record["sha256"], len(clip.values))
        values = source[clip.begin:clip.end].copy()
        retarget_report, floor = None, None
        if robot != "g1":
            values, retarget_report = retarget(values, robot)
            _validate_values(values, robot)
            if len(values) != clip.end - clip.begin:
                raise ValueError("Retargeter changed the number of timestamps")
            retarget_report = dict(retarget_report, source="G1 style crop identified in this manifest")
            values, floor = align_constant_floor(values, physical_floor_distances(values, robot))
        name = f"clip_{index:03d}__{clip.group}__{clip.begin:05d}-{clip.end:05d}.csv"
        path = directory / name
        np.savetxt(path, values, delimiter=",", fmt="%.17g")
        descriptor = clip.descriptor()
        original_start = clip.record["source_window_frames"][0]
        original_fps = clip.record["source_fps"]
        records.append({
            "file": name, "sha256": digest(path), "frames": len(values), "fps": FPS,
            "duration_seconds": (len(values) - 1) / FPS, "split": clip.split,
            "source_group": clip.group, "original_sha256": clip.record["source_sha256"],
            "source": descriptor, "upstream_record": clip.record,
            "upstream_manifest": clip.metadata,
            "original_sample_frame_positions": [
                original_start + clip.begin * original_fps / FPS,
                original_start + (clip.end - 1) * original_fps / FPS],
            "range_convention": "Crops are zero-based [begin,end); mapped original samples are first/last, possibly fractional.",
            "transform": "G1 contiguous crop" if robot == "g1" else "Morphology-scaled Mink IK",
            "retarget": retarget_report, "floor_alignment": floor,
            "inherited_floor_alignment": clip.record.get("floor_alignment") if robot == "g1" else None,
            "upper_body": "Source posture retained; no firearm object or control.",
        })
    return records, {"selection": report}


def _write_walk(directory, gaits, clips=(), report=None, robot="g1"):
    records = _copy_gaits(directory, gaits)
    details = {"scope_source": "Audited native per-robot gait pool", "expanded_walk_style": bool(clips)}
    if clips:
        added, _ = _write_styles(directory, clips, robot, report)
        records.extend(added)
        details.update(scope_source="Audited gait plus bounded G1 walk-family posture crops",
                       walk_selection=report)
    return records, details


def _go2_manifest(root):
    path = root / "manifest.json"
    if not path.exists():
        return None, {}
    manifest = _read_json(path)
    spec = get_robot("go2")
    if (manifest.get("schema") != "go2-style-expanded-v1" or manifest.get("version") != 1
            or manifest.get("status") != "ready" or manifest.get("robot") != "go2"
            or manifest.get("output_fps") != FPS or manifest.get("quaternion_order") != "xyzw"
            or manifest.get("joint_names") != list(spec.joint_names)
            or manifest.get("mjcf_sha256") != digest(spec.mjcf_path)):
        raise ValueError("Expanded Go2 manifest model, schema or FPS differs from the campaign")
    source_fps = manifest.get("source_fps")
    if (isinstance(source_fps, bool) or not isinstance(source_fps, (int, float))
            or not math.isfinite(source_fps) or source_fps <= 0):
        raise ValueError("Expanded Go2 manifest has invalid source FPS")
    rows, identities = {}, []
    for row in manifest.get("files", []):
        name = _basename(row["file"])
        if name in rows or row.get("fps") != FPS or row.get("source_fps") != source_fps:
            raise ValueError("Expanded Go2 clip identity or FPS differs")
        for field in ("sha256", "raw_sha256", "source_pose_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get(field, ""))):
                raise ValueError(f"Expanded Go2 clip requires {field}")
        if (not isinstance(row.get("clip_type"), str) or not row["clip_type"]
                or not isinstance(row.get("clip"), str) or not row["clip"]):
            raise ValueError("Expanded Go2 clip needs publisher type and source identity")
        _read_motion(_local_file(root, name), "go2", row["sha256"], row["frames"])
        fk, frames = row.get("raw_fk") or {}, row.get("source_frames")
        error = fk.get("foot_link_max_error_m")
        if (type(frames) is not int or frames < 3 or type(fk.get("frames_checked")) is not int
                or fk["frames_checked"] != frames or isinstance(error, bool)
                or not isinstance(error, (int, float)) or not math.isfinite(error)
                or not 0 <= error <= 1e-4):
            raise ValueError("Expanded Go2 clip lacks a passing every-frame FK receipt")
        group = row.get("source_group", row["clip"])
        if not isinstance(group, str) or not group:
            raise ValueError("Expanded Go2 clip needs an indivisible source group")
        identities.append({"split": row["split"], "sha256": row["sha256"],
                           "raw_sha256": row["raw_sha256"], "source_pose_sha256": row["source_pose_sha256"],
                           "source_group": group})
        rows[name] = row
    _verify_split(identities)
    for split, field in (("train", "training_files"), ("test", "evaluation_files")):
        receipt = manifest.get("configs", {}).get(split, {})
        cfg_path = _local_file(root, receipt.get("file"))
        cfg, cfg_root = read_config(cfg_path, "go2")
        names = [row["file"] for row in rows.values() if row["split"] == split]
        if (cfg_root != root or names != manifest.get(field) or cfg["files"] != names
                or digest(cfg_path) != receipt.get("sha256")):
            raise ValueError("Expanded Go2 config hash or split membership differs")
    return manifest, rows


def _go2_extra(config_path, limit, budget, split="train"):
    config_path = Path(config_path).resolve()
    if split not in ("train", "test"):
        raise ValueError("Go2 extras need an explicit train/test split")
    cfg, root = read_config(config_path, "go2", require_50=False)
    manifest, rows = _go2_manifest(root)
    if manifest is not None:
        receipt = manifest["configs"][split]
        if (config_path != _local_file(root, receipt["file"])
                or cfg["source_fps"] != FPS or digest(config_path) != receipt["sha256"]):
            raise ValueError("Expanded Go2 config differs from its requested split")
    names = sorted(cfg["files"], key=lambda name: (
        rows.get(name, {}).get("clip_type") != "jump", name))
    result, total = [], 0
    for name in names:
        path = _local_file(root, name)
        row = rows.get(name)
        source = _read_motion(path, "go2", row["sha256"] if row else None)
        values = resample_g1(source, cfg["source_fps"]) if cfg["source_fps"] != FPS else source
        length = min(len(values), limit)
        if total + length > budget:
            continue
        begin, end = crop_window(values, limit, "jumps")
        result.append({
            "path": path, "values": values[begin:end].copy(), "sha256": digest(path),
            "range": [begin, end], "source_fps": cfg["source_fps"],
            "config_sha256": digest(config_path),
            "split": split, "clip_type": row["clip_type"] if row else "unverified",
            "source_group": row.get("source_group", row["clip"]) if row else path.stem,
            "raw_sha256": row["raw_sha256"] if row else None,
            "source_pose_sha256": row["source_pose_sha256"] if row else None,
            "source_verification": {
                "verified": row is not None, "every_frame_fk": row is not None,
                "manifest_sha256": digest(root / "manifest.json") if row else None,
                "model_sha256": manifest["mjcf_sha256"] if row else None,
                "scope": "CSV/model/config hashes checked; raw/pose hashes and every-frame FK retained from importer receipt."
                         if row else "No verified source manifest; filename is not an action label.",
            },
            "source_record": row,
        })
        total += length
    if not result:
        raise ValueError("Optional Go2 source config has no usable bounded clips")
    return result


def _write_go2(directory, gaits, extras):
    records = _copy_gaits(directory, gaits)
    for index, extra in enumerate(extras):
        if digest(extra["path"]) != extra["sha256"]:
            raise ValueError("Go2 source changed after planning")
        name = f"go2_extra_{extra['split']}_{index:03d}.csv"
        values = extra["values"]
        np.savetxt(directory / name, values, delimiter=",", fmt="%.17g")
        records.append({
            "file": name, "sha256": digest(directory / name), "frames": len(values), "fps": FPS,
            "duration_seconds": (len(values) - 1) / FPS, "split": extra["split"],
            "original_sha256": extra["raw_sha256"] or extra["sha256"],
            "source_group": f"go2-extra:{extra['source_group']}",
            "raw_sha256": extra["raw_sha256"], "source_pose_sha256": extra["source_pose_sha256"],
            "clip_type": extra["clip_type"], "source_verification": extra["source_verification"],
            "upstream_record": extra["source_record"],
            "source_file": str(extra["path"]), "source_config_sha256": extra["config_sha256"],
            "source_fps": extra["source_fps"], "resampled_frame_range": extra["range"],
            "source_sample_frame_positions": [
                extra["range"][0] * extra["source_fps"] / FPS,
                (extra["range"][1] - 1) * extra["source_fps"] / FPS],
            "transform": "Native Go2 data; optional time-preserving resampling and contiguous crop",
        })
    verified_jump_splits = {extra["split"] for extra in extras if extra["clip_type"] == "jump"
                            and extra["source_verification"]["verified"]
                            and extra["source_verification"]["every_frame_fk"]}
    verified_jump = verified_jump_splits == {"train", "test"}
    return records, {
        "limitations": [
            "Go2 uses native audited walking/trotting; no human-to-quadruped mapping.",
            "Jump references in both splits have verified importer FK receipts; publisher types describe provenance, not physical feasibility."
            if verified_jump else "Jump reference coverage is unverified; optional data require separate semantic/physical review."
            if extras else "No Go2 jump reference supplied; parkour expert pool contains gait dynamics only.",
        ],
        "additional_native_go2_clips": len(extras), "verified_jump_reference": verified_jump,
    }


def prepare_campaign(gait_root, g1_parkour, output, *, g1_stealth=None, parkour_additional=(),
                     go2_parkour_config=None, robots=ROBOTS, holdout=DEFAULT_HOLDOUT,
                     clip_frames=400, max_per_source=2, frame_budget=4000,
                     go2_parkour_test_config=None, expand_walk_style=False):
    if go2_parkour_test_config is not None and go2_parkour_config is None:
        raise ValueError("Go2 test config requires the paired train config")
    if type(expand_walk_style) is not bool:
        raise ValueError("expand_walk_style must be boolean")
    if not robots or len(set(robots)) != len(robots) or set(robots) - set(ROBOTS):
        raise ValueError("Select distinct supported robots")
    if (type(clip_frames) is not int or not 7 <= clip_frames <= 400
            or type(max_per_source) is not int or not 1 <= max_per_source <= 2
            or type(frame_budget) is not int or not 7 <= frame_budget <= 4000):
        raise ValueError("Bounds: 7..400 frames/clip, 1..2 clips/source, 7..4000 frames/style")
    output, gait_root = Path(output).resolve(), Path(gait_root).resolve()
    _require_ignored_output(output)
    for source in (gait_root, g1_parkour, g1_stealth, *parkour_additional,
                   *(Path(cfg).resolve().parent for cfg in (go2_parkour_config, go2_parkour_test_config)
                     if cfg is not None)):
        if source is not None:
            source = Path(source).resolve()
            if output == source or source in output.parents or output in source.parents:
                raise ValueError("Campaign output and source directories must be disjoint")
    output.mkdir(parents=True, exist_ok=True)
    with _campaign_lock(output):
        map_path = output / "style-map.json"
        mapping = _read_json(map_path) if map_path.exists() else {
            "version": 1, "robots": {}, "evaluations": {}, "pools": {}, "limitations": {}}
        if mapping.get("version") != 1 or not isinstance(mapping.get("robots"), dict):
            raise ValueError("Existing style map has an incompatible schema")
        for field in ("evaluations", "pools", "limitations"):
            mapping.setdefault(field, {})
        plans, walk_plan = {}, None
        if any(robot != "go2" for robot in robots):
            candidates, source_union = load_parkour_sources(
                g1_parkour, parkour_additional, tuple(holdout), clip_frames)
            plans["parkour"] = select_bounded(
                candidates, max_per_source=max_per_source, frame_budget=frame_budget)
            plans["parkour"][1]["source_union"] = source_union
            if expand_walk_style:
                walk_plan = select_bounded(
                    [clip for clip in candidates if clip.family == "walk"],
                    max_per_source=max_per_source, frame_budget=min(1600, frame_budget))
                walk_plan[1]["source_union"] = source_union
                walk_plan[1]["scope"] = "Walk-family contiguous posture crops only; command coverage needs separate evaluation."
            if g1_stealth is not None:
                candidates, _ = load_style(g1_stealth, "stealth", (), clip_frames)
                plans["stealth"] = select_bounded(
                    candidates, max_per_source=max_per_source, frame_budget=frame_budget)
        implementation = digest(__file__)
        for robot in robots:
            gaits = gait_inputs(gait_root / robot, robot)
            base_recipe = {
                "implementation_sha256": implementation, "robot": robot,
                "model_sha256": digest(get_robot(robot).mjcf_path), "fps": FPS,
                "style_preparation_utilities_sha256": digest(
                    Path(__file__).with_name("prepare_style_references.py")),
                "numpy_version": np.__version__,
            }
            gait_recipe = [{key: value for key, value in row.items()
                            if key not in ("path", "provenance")} for row in gaits]
            prepared = {}
            recipe = dict(base_recipe, style="walk", audited_gait_inputs=gait_recipe)
            walk_clips, walk_report = walk_plan if walk_plan is not None and robot != "go2" else ((), None)
            if walk_clips:
                recipe.update(
                    expanded_walk_style=True, walk_selection=walk_report,
                    sources=[dict(clip.descriptor(), metadata=clip.metadata) for clip in walk_clips],
                    source_model_sha256=digest(get_robot("g1").mjcf_path),
                    retarget_sha256=digest(Path(__file__).with_name("retarget_gait_references.py"))
                    if robot != "g1" else None)
            prepared["walk"] = _build_pool(
                output, robot, "walk", recipe, lambda directory: _write_walk(
                    directory, gaits, walk_clips, walk_report, robot))
            if robot == "go2":
                test_extras = (_go2_extra(go2_parkour_test_config, clip_frames, frame_budget // 2, "test")
                               if go2_parkour_test_config else [])
                train_budget = frame_budget - sum(len(extra["values"]) for extra in test_extras)
                extras = (_go2_extra(go2_parkour_config, clip_frames, train_budget)
                          if go2_parkour_config else [])
                extras += test_extras
                _verify_split([{key: value for key, value in row.items() if key != "values"}
                               for row in extras], require_both=False)
                extra_recipe = [{key: str(value) if isinstance(value, Path) else value
                                 for key, value in row.items() if key != "values"} for row in extras]
                recipe = dict(base_recipe, style="parkour", audited_gait_inputs=gait_recipe,
                              additional_go2_inputs=extra_recipe)
                prepared["parkour"] = _build_pool(
                    output, robot, "parkour", recipe, lambda directory: _write_go2(directory, gaits, extras))
            else:
                for style, (clips, report) in plans.items():
                    recipe = dict(
                        base_recipe, style=style, selection=report,
                        source_model_sha256=digest(get_robot("g1").mjcf_path),
                        sources=[dict(clip.descriptor(), metadata=clip.metadata) for clip in clips],
                        retarget_sha256=digest(Path(__file__).with_name("retarget_gait_references.py"))
                        if robot != "g1" else None,
                    )
                    prepared[style] = _build_pool(
                        output, robot, style, recipe,
                        lambda directory, clips=clips, report=report: _write_styles(directory, clips, robot, report))
            robot_map, evaluation_map, pool_map = {}, {}, {}
            for style, (directory, manifest) in prepared.items():
                relative = directory.relative_to(output).as_posix()
                robot_map[style] = f"{relative}/train.json"
                evaluation_map[style] = f"{relative}/test.json"
                pool_map[style] = {
                    "manifest": f"{relative}/manifest.json",
                    "sha256": digest(directory / "manifest.json"),
                    "config_sha256": manifest["configs"],
                }
            if "stealth" not in prepared:
                old = mapping["robots"].get(robot, {})
                if robot != "go2" and old.get("stealth"):
                    robot_map["stealth"] = old["stealth"]
                    evaluation_map["stealth"] = mapping["evaluations"][robot]["stealth"]
                    pool_map["stealth"] = mapping["pools"][robot]["stealth"]
                else:
                    robot_map["stealth"] = None
            mapping["robots"][robot] = robot_map
            mapping["evaluations"][robot] = evaluation_map
            mapping["pools"][robot] = pool_map
            mapping["limitations"][robot] = (
                prepared["parkour"][1].get("limitations", []) if robot == "go2" else
                ([] if robot_map.get("stealth") else ["Stealth waits for verified prepared Aiming references."]))
        # Validate all retained and new entries before publishing a single map.
        for robot, styles in mapping["robots"].items():
            for style, config in styles.items():
                if config is None:
                    continue
                path = (output / config).resolve()
                if not path.is_relative_to(output):
                    raise ValueError("Style map points outside the campaign")
                manifest = verify_pool(path.parent, robot, style)
                receipt = mapping["pools"][robot][style]
                if (digest(path.parent / "manifest.json") != receipt["sha256"]
                        or path.name != "train.json"
                        or mapping["evaluations"][robot][style] !=
                        (path.parent / "test.json").relative_to(output).as_posix()):
                    raise ValueError("Style map receipt differs from its pool")
        _atomic_json(map_path, mapping)
    return mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait-root", type=Path, default=REPO / "data/gait-foundation-v5")
    parser.add_argument("--g1-parkour", type=Path, required=True)
    parser.add_argument("--g1-parkour-additional", type=Path, nargs="*", default=[],
                        help="Additional prepared 50 Hz G1 parkour directories; holdouts apply to the union.")
    parser.add_argument("--g1-stealth", type=Path)
    parser.add_argument("--go2-parkour-config", type=Path)
    parser.add_argument("--go2-parkour-test-config", type=Path,
                        help="Optional held-out native Go2 config paired with --go2-parkour-config.")
    parser.add_argument("--expand-walk-style", action="store_true",
                        help="Add at most 1600 contiguous 50 Hz G1 walk-family frames to humanoid gait pools.")
    parser.add_argument("--output", type=Path, default=REPO / "data/style-references-v1")
    parser.add_argument("--robots", nargs="+", choices=ROBOTS, default=list(ROBOTS))
    parser.add_argument("--holdout-groups", nargs="+", default=list(DEFAULT_HOLDOUT))
    parser.add_argument("--clip-frames", type=int, default=400)
    parser.add_argument("--max-per-source", type=int, default=2)
    parser.add_argument("--frame-budget", type=int, default=4000)
    args = parser.parse_args(argv)
    result = prepare_campaign(
        args.gait_root, args.g1_parkour, args.output, g1_stealth=args.g1_stealth,
        parkour_additional=tuple(args.g1_parkour_additional),
        go2_parkour_config=args.go2_parkour_config, robots=tuple(args.robots),
        go2_parkour_test_config=args.go2_parkour_test_config, expand_walk_style=args.expand_walk_style,
        holdout=tuple(args.holdout_groups), clip_frames=args.clip_frames,
        max_per_source=args.max_per_source, frame_budget=args.frame_budget)
    print(json.dumps({"style_map": str(args.output.resolve() / "style-map.json"),
                      "robots": result["robots"], "limitations": result["limitations"]}, indent=2))


if __name__ == "__main__":
    main()

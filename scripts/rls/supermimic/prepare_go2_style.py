"""Prepare a local-only, FK-audited Go2 locomotion/jump style pool from Kine2Go."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from bootstrap import configure_paths
from prepare_data import blob_hash, fetch
from prepare_kine2go import API, BASE, DATASET, LOCK, REVISION, convert_array, digest, validate_fk
from prepare_style_references import align_constant_floor, physical_floor_distances

REPO = configure_paths()
from trackerTask.supermimic.robots import get_robot

FPS = 50
PLAN = (
    ("ai4_dog_walk_00", "walk", "train"),
    ("ai4_dog_trot_00", "trot", "train"),
    ("ai4_dog_run_00", "run", "train"),
    ("ai4_dog_run_04", "run", "train"),
    ("ai4_dog_canter", "canter", "train"),
    ("ai4_dog_left_turn", "turn", "train"),
    ("ai4_dog_right_turn", "turn", "train"),
    ("ai4_dog_synth_tight_strafe", "strafe", "train"),
    ("solo8_jump_forward_a", "jump", "train"),
    ("ai4_dog_run_02", "run", "test"),
    ("solo8_jump_forward_b", "jump", "test"),
)
DOCUMENTS = ("README.md", "LICENSE", "metadata.json",
             "LICENSES/AI4Animation_CC-BY-NC-4.0.txt", "LICENSES/cassi_BSD-3-Clause.txt")
LICENSES = {"CC BY-NC 4.0": DOCUMENTS[3], "BSD-3-Clause": DOCUMENTS[4]}


def go2_spec():
    spec = get_robot("go2")
    path = Path(spec.mjcf_path).resolve()
    if path.is_relative_to(Path.cwd()):
        spec = replace(spec, mjcf_path=str(path.relative_to(Path.cwd())))
    return spec


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_ignored_data(path):
    path = path.expanduser().resolve()
    if not path.is_relative_to((REPO / "data").resolve()):
        raise ValueError("Motion sources and derivatives must remain under local data/")
    result = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--quiet", "--no-index", str(path)],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Motion sources and derivatives must be ignored by Git")
    return path


def expected_files():
    return set(DOCUMENTS) | {f"data/{clip}/{name}" for clip, _, _ in PLAN
                             for name in ("motion.npy", "clip.json", "config.json")}


def download_source(source):
    """Fetch only pinned numeric NPY, JSON metadata, README and license texts."""
    source.mkdir(parents=True, exist_ok=True)
    entries = {}
    for folder in ("", "LICENSES", *(f"data/{clip}" for clip, _, _ in PLAN)):
        for entry in json.loads(fetch(f"{API}/{folder}?limit=100")):
            if entry["type"] == "file":
                entries[entry["path"]] = entry
    files = []
    local_cache = REPO / "data/raw/kine2go-gait-foundation"
    for name in sorted(expected_files()):
        entry = entries[name]
        algorithm = "sha256" if "lfs" in entry else "git_blob_sha1"
        expected = entry["lfs"]["oid"] if "lfs" in entry else entry["oid"]
        target = source / name
        content = target.read_bytes() if target.is_file() else b""
        hash_bytes = (lambda value: hashlib.sha256(value).hexdigest()) if algorithm == "sha256" else blob_hash
        if not target.exists() and (local_cache / name).is_file():
            cached = (local_cache / name).read_bytes()
            if hash_bytes(cached) == expected:
                content = cached
        if target.exists() and hash_bytes(content) != expected:
            raise ValueError(f"Existing source hash mismatch: {name}")
        if hash_bytes(content) != expected:
            content = fetch(f"{BASE}/{name}")
        if hash_bytes(content) != expected or len(content) != entry["size"]:
            raise ValueError(f"Pinned source hash mismatch: {name}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(target)
        files.append({"file": name, "sha256": digest(target), "bytes": len(content),
                      "source_hash_algorithm": algorithm, "source_hash": expected})
    manifest = {"version": 1, "dataset": DATASET, "revision": REVISION,
                "clips": [row[0] for row in PLAN], "files": files,
                "download_policy": "Numeric motion.npy, clip/config JSON, README, schema and licenses only.",
                "pickle_loaded": False}
    write_json(source / "source_manifest.json", manifest)
    return verify_source(source)


def verify_source(source):
    manifest = json.loads((source / "source_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("dataset") != DATASET or manifest.get("revision") != REVISION
            or manifest.get("clips") != [row[0] for row in PLAN]):
        raise ValueError("Source manifest differs from the pinned Go2 style selection")
    rows = manifest.get("files", [])
    if len(rows) != len(expected_files()) or {row["file"] for row in rows} != expected_files():
        raise ValueError("Source manifest requires exactly the allowed numeric/JSON/license files")
    for row in rows:
        content = (source / row["file"]).read_bytes()
        algorithm = row["source_hash_algorithm"]
        if algorithm not in ("sha256", "git_blob_sha1"):
            raise ValueError("Unknown source hash algorithm")
        original = hashlib.sha256(content).hexdigest() if algorithm == "sha256" else blob_hash(content)
        if (hashlib.sha256(content).hexdigest() != row["sha256"]
                or original != row["source_hash"] or len(content) != row["bytes"]):
            raise ValueError(f"Source hash mismatch: {row['file']}")
    return manifest


def validate_metadata(clip, meta, config, spec):
    env = config.get("env_cfg", {})
    urdf = str(env.get("urdf_path", "")).replace("\\", "/")
    expected_license = "BSD-3-Clause" if clip.startswith("solo8_") else "CC BY-NC 4.0"
    if (meta.get("name") != clip or meta.get("fps") != LOCK["source_fps"]
            or meta.get("license") != expected_license or not meta.get("attribution")):
        raise ValueError(f"Invalid source identity, license or FPS: {clip}")
    if (env.get("num_actions") != 12 or env.get("num_dofs") != 12
            or env.get("control_freq") != LOCK["source_fps"] or "/go2/" not in "/" + urdf
            or set(env.get("dof_names", [])) != set(spec.joint_names)):
        raise ValueError(f"Config does not describe the 12-joint Go2 robot: {clip}")


def resample_50hz(values, source_fps=60):
    if (values.ndim != 2 or values.shape[1] != 19 or len(values) < 3
            or not np.isfinite(values).all() or not math.isfinite(source_fps) or source_fps <= 0
            or not np.allclose(np.linalg.norm(values[:, 3:7], axis=-1), 1, atol=1e-4)):
        raise ValueError("Expected finite Go2 CSV poses and a positive source FPS")
    source_t = np.arange(len(values), dtype=np.float64) / source_fps
    target_t = np.arange(math.floor(source_t[-1] * FPS + 1e-9) + 1) / FPS
    output = np.column_stack([np.interp(target_t, source_t, values[:, index])
                              for index in range(19)])
    output[:, 3:7] = Slerp(source_t, Rotation.from_quat(values[:, 3:7]))(target_t).as_quat()
    return output


def group_sources(records):
    """Deduplicate aliases, rejecting shared raw or pose content across splits."""
    seen, selected, aliases = {}, [], []
    for row in records:
        keys = (("raw", row["raw_sha256"]), ("poses", row["source_pose_sha256"]))
        matches = [seen[key] for key in keys if key in seen]
        if any(old["split"] != row["split"] for old in matches):
            raise ValueError("Identical source content crosses train/test splits")
        if matches:
            aliases.append({"clip": row["clip"], "alias_of": matches[0]["clip"],
                            "split": row["split"], "raw_sha256": row["raw_sha256"],
                            "publisher_clip_type": row["clip_type"]})
        else:
            selected.append(row)
        for key in keys:
            seen[key] = matches[0] if matches else row
    return selected, aliases


def motion_config(spec, names):
    return {"version": 1, "root": ".", "source_fps": FPS, "quaternion_order": "xyzw",
            "joint_names": list(spec.joint_names), "files": names}


def prepare(source, output, *, walking_per_frame=False):
    source, output = require_ignored_data(source), require_ignored_data(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use separate source and derivative directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty derivative directory")
    source_manifest = verify_source(source)
    spec = go2_spec()
    records, arrays = [], {}
    for clip, kind, split in PLAN:
        folder = source / "data" / clip
        meta = json.loads((folder / "clip.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        validate_metadata(clip, meta, config, spec)
        raw = np.load(folder / "motion.npy", allow_pickle=False)
        if not isinstance(raw, np.ndarray) or not np.issubdtype(raw.dtype, np.number):
            raise ValueError(f"Expected numeric NPY: {clip}")
        values = convert_array(raw, spec.joint_names)
        validation = validate_fk(raw, values, spec)
        record = {
            "clip": clip, "file": clip + ".csv", "clip_type": kind, "split": split,
            "raw_sha256": digest(folder / "motion.npy"),
            "source_pose_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
            "source_frames": len(raw), "source_fps": meta["fps"],
            "publisher_metadata_frames": meta["n_frames"], "publisher_tags": meta.get("tags", []),
            "source": meta["source"], "source_url": meta["source_url"],
            "license": meta["license"], "license_file": LICENSES[meta["license"]],
            "attribution": meta["attribution"], "raw_fk": validation,
            "mapping": "Existing Kine2Go column-group mapping, verified against all four stored feet on every frame.",
            "publisher_config_dof_order_used_for_mapping": False,
            "publisher_config_robot": config["env_cfg"]["urdf_path"],
        }
        records.append(record)
        arrays[clip] = values
        print(f"FK passed {clip}: {len(raw)} frames, max {validation['foot_link_max_error_m']:.3g} m",
              flush=True)
    selected, aliases = group_sources(records)
    prepared = {}
    for row in selected:
        values = resample_50hz(arrays[row["clip"]], row["source_fps"])
        distances = physical_floor_distances(values, "go2")
        clearance = .003 * .30 / .76
        if walking_per_frame and row["clip_type"] == "walk":
            values[:, 2] += clearance - distances
            alignment = {"method": "per_frame_walking_support_alignment",
                         "per_frame_clamp": True, "airborne_trajectory_preserved": False}
        else:
            values, alignment = align_constant_floor(values, distances, clearance_m=clearance)
        after = physical_floor_distances(values, "go2")
        if not np.isfinite(after).all() or after.min() < clearance - 2e-5:
            raise ValueError(f"Physical foot floor validation failed: {row['clip']}")
        row.update(
            frames=len(values), fps=FPS, duration_seconds=(len(values) - 1) / FPS,
            floor_alignment=alignment,
            physical_floor={"minimum_m": float(after.min()), "maximum_m": float(after.max()),
                            "frames_all_feet_above_3cm": int((after > .03).sum()),
                            "scope": "All-frame physical foot-collider distance to a flat MuJoCo plane."},
            root_z_range_m=[float(values[:, 2].min()), float(values[:, 2].max())],
            mean_horizontal_speed_m_s=float(np.linalg.norm(np.gradient(values[:, :2], 1/FPS, axis=0), axis=-1).mean()),
        )
        prepared[row["file"]] = values
    output.mkdir(parents=True, exist_ok=True)
    for row in selected:
        target = output / row["file"]
        np.savetxt(target, prepared[row["file"]], delimiter=",", fmt="%.10f")
        row["sha256"] = digest(target)
    split_files = {split: [row["file"] for row in selected if row["split"] == split]
                   for split in ("train", "test")}
    hashes = {split: {row["sha256"] for row in selected if row["split"] == split}
              for split in ("train", "test")}
    if not all(split_files.values()) or hashes["train"] & hashes["test"]:
        raise ValueError("Output train/test must be nonempty and content-disjoint")
    for split, names in split_files.items():
        write_json(output / f"{split}.json", motion_config(spec, names))
    manifest = {
        "schema": "go2-style-expanded-v1", "version": 1, "status": "ready",
        "dataset": DATASET, "revision": REVISION, "robot": "go2", "task": "go2",
        "source": "Pinned Kine2Go Go2-retargeted quadruped references",
        "source_pipeline": LOCK["pipeline_repository"], "pipeline_revision": LOCK["pipeline_revision"],
        "source_fps": LOCK["source_fps"], "output_fps": FPS, "quaternion_order": "xyzw",
        "joint_names": list(spec.joint_names), "raw_joint_order": LOCK["raw_joint_order"],
        "raw_foot_order": LOCK["raw_foot_order"], "mjcf_sha256": digest(spec.mjcf_path),
        "importer_sha256": digest(__file__),
        "fk_importer_sha256": digest(Path(__file__).with_name("prepare_kine2go.py")),
        "source_directory": str(source), "source_manifest_sha256": digest(source / "source_manifest.json"),
        "source_files": source_manifest["files"], "files": selected,
        "aliases": aliases, "candidates": records,
        "training_files": split_files["train"], "evaluation_files": split_files["test"],
        "configs": {split: {"file": f"{split}.json", "sha256": digest(output / f"{split}.json")}
                    for split in split_files},
        "license": "Per-clip CC BY-NC 4.0 or BSD-3-Clause; texts and attribution retained.",
        "distribution": "Local research only; this utility publishes no data.",
        "split_policy": "Indivisible source clips; raw and canonical pose hashes disjoint across splits; same-split aliases deduplicated.",
        "label_scope": "Clip types are publisher metadata categories; turn aliases are not independently observed left/right recordings.",
        "conversion": "Named Go2 mapping; 60-to-50 Hz position/joint interpolation and quaternion SLERP; no human mapping or IK.",
        "physical_scope": "Full-frame source FK and physical foot-floor geometry; actuator feasibility remains untested.",
        "pickle_loaded": False, "walking_per_frame": walking_per_frame,
    }
    write_json(output / "manifest.json", manifest)
    verify_output(output)
    return manifest


def verify_output(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("dataset") != DATASET or manifest.get("revision") != REVISION
            or manifest.get("robot") != "go2" or manifest.get("output_fps") != FPS):
        raise ValueError("Output manifest differs from pinned Go2 style contract")
    files = manifest["files"]
    if {path.name for path in output.glob("*.csv")} != {row["file"] for row in files}:
        raise ValueError("Output CSV inventory differs from the manifest")
    source_hashes = {split: set() for split in ("train", "test")}
    output_hashes = {split: set() for split in ("train", "test")}
    for row in files:
        if Path(row["file"]).name != row["file"] or "/" in row["file"] or "\\" in row["file"]:
            raise ValueError("Output filenames must be plain CSV basenames")
        path = output / row["file"]
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        if (digest(path) != row["sha256"] or values.shape != (row["frames"], 19)
                or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, atol=1e-4)):
            raise ValueError(f"Output CSV validation failed: {row['file']}")
        source_hashes[row["split"]].update((row["raw_sha256"], row["source_pose_sha256"]))
        output_hashes[row["split"]].add(row["sha256"])
    if source_hashes["train"] & source_hashes["test"] or output_hashes["train"] & output_hashes["test"]:
        raise ValueError("Identical source/output content crosses train/test")
    spec = get_robot("go2")
    for split, key in (("train", "training_files"), ("test", "evaluation_files")):
        path = output / f"{split}.json"
        names = [row["file"] for row in files if row["split"] == split]
        if (not names or manifest[key] != names or digest(path) != manifest["configs"][split]["sha256"]
                or json.loads(path.read_text(encoding="utf-8")) != motion_config(spec, names)):
            raise ValueError(f"Invalid motion configuration: {split}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "data/raw/kine2go-style-expanded")
    parser.add_argument("--output", type=Path, default=REPO / "data/go2-style-expanded")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--walking-per-frame", action="store_true")
    args = parser.parse_args()
    source, output = require_ignored_data(args.source), require_ignored_data(args.output)
    if args.verify_only:
        verify_source(source)
        manifest = verify_output(output)
    else:
        if not args.offline:
            download_source(source)
        verify_source(source)
        if args.download_only:
            print(f"Verified pinned sources: {source}", flush=True)
            return
        manifest = prepare(source, output, walking_per_frame=args.walking_per_frame)
    print(json.dumps({"status": manifest["status"], "train_config": str(output / "train.json"),
                      "test_config": str(output / "test.json"), "manifest": str(output / "manifest.json"),
                      "train_clips": len(manifest["training_files"]),
                      "test_clips": len(manifest["evaluation_files"]),
                      "aliases": manifest["aliases"]}, indent=2))


if __name__ == "__main__":
    main()

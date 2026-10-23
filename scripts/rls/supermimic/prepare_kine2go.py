"""Import pinned, Go2-retargeted Kine2Go references into TrackerLab CSVs.

Only numeric NPY references and source documentation are downloaded. Training
rollout pickles are not needed. FK validates the actual raw DOF order against
independently stored foot link positions before any CSV is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from prepare_data import blob_hash, fetch

REPO = Path(__file__).resolve().parents[3]
LOCK = json.loads(Path(__file__).with_name("resources.lock.json").read_text())["go2_motions"]
DATASET, REVISION = LOCK["repository"], LOCK["revision"]
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(source: Path, clips: list[str]):
    source.mkdir(parents=True, exist_ok=True)
    entries = {}
    for folder in ("", "LICENSES", *(f"data/{clip}" for clip in clips)):
        for entry in json.loads(fetch(f"{API}/{folder}?limit=100")):
            if entry["type"] == "file":
                entries[entry["path"]] = entry
    names = ["README.md", "LICENSE", "LICENSES/AI4Animation_CC-BY-NC-4.0.txt"]
    names += [f"data/{clip}/{name}" for clip in clips for name in ("motion.npy", "clip.json", "config.json")]
    records = []
    for name in names:
        entry = entries[name]
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        algorithm = "sha256" if "lfs" in entry else "git_blob_sha1"
        expected = entry["lfs"]["oid"] if "lfs" in entry else entry["oid"]

        def matches(content):
            actual = hashlib.sha256(content).hexdigest() if algorithm == "sha256" else blob_hash(content)
            return actual == expected

        content = target.read_bytes() if target.exists() else b""
        if not matches(content):
            content = fetch(f"{BASE}/{name}")
            if not matches(content):
                raise ValueError(f"Pinned source hash mismatch: {name}")
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(content)
            temp.replace(target)
        records.append({"file": name, "sha256": digest(target), "bytes": len(content),
                        "source_hash_algorithm": algorithm, "source_hash": expected})
    manifest = {"dataset": DATASET, "revision": REVISION, "clips": clips, "files": records}
    (source / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_source(source: Path, clips: list[str]):
    manifest = json.loads((source / "source_manifest.json").read_text())
    if manifest["dataset"] != DATASET or manifest["revision"] != REVISION:
        raise ValueError("Source data differs from the pinned Kine2Go revision")
    if manifest["clips"] != clips:
        raise ValueError("Source clip selection differs; use a separate source directory")
    expected = {"README.md", "LICENSE", "LICENSES/AI4Animation_CC-BY-NC-4.0.txt"}
    expected.update(f"data/{clip}/{name}" for clip in clips for name in ("motion.npy", "clip.json", "config.json"))
    if {record["file"] for record in manifest["files"]} != expected:
        raise ValueError("Source manifest must include references, metadata and licenses")
    for record in manifest["files"]:
        path = source / record["file"]
        content = path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest() if record["source_hash_algorithm"] == "sha256" else blob_hash(content)
        if digest(path) != record["sha256"] or source_hash != record["source_hash"]:
            raise ValueError(f"Source data hash mismatch: {path}")
    return manifest


def convert_array(raw, joint_names):
    if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] != 61 or not np.isfinite(raw).all():
        raise ValueError("Expected finite Kine2Go (T, 61) array with at least three frames")
    if not np.allclose(np.linalg.norm(raw[:, 51:55], axis=1), 1, atol=1e-4):
        raise ValueError("Kine2Go root WXYZ quaternions must have unit norm")
    indices = [6 + LOCK["raw_joint_order"].index(name) for name in joint_names]
    return np.concatenate((raw[:, 48:51], raw[:, [52, 53, 54, 51]], raw[:, indices]), axis=1)


def validate_fk(raw, values, spec):
    import mujoco

    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    joint_ids = np.array([model.joint(name).id for name in spec.joint_names])
    qids = model.jnt_qposadr[joint_ids]
    legs = ("FL", "FR", "RL", "RR")
    calves = [model.body(leg + "_calf").id for leg in legs]
    feet = raw[:, 36:48].reshape(-1, 4, 3)[:, [LOCK["raw_foot_order"].index(leg) for leg in legs]]
    errors = []
    for frame, row in enumerate(values):
        data.qpos[:3], data.qpos[3:7], data.qpos[qids] = row[:3], row[[6, 3, 4, 5]], row[7:]
        mujoco.mj_forward(model, data)
        predicted = np.stack([
            data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.array([0., 0., -.213])
            for body in calves
        ])
        errors.append(np.linalg.norm(predicted - feet[frame], axis=1))
    errors = np.asarray(errors)
    if errors.max() > 1e-4:
        raise ValueError(f"Kine2Go joint/root mapping failed full-frame FK: {errors.max():.6g} m")
    q = values[:, 7:]
    lo, hi = model.jnt_range[joint_ids].T
    if ((q < lo - 1e-4) | (q > hi + 1e-4)).any():
        raise ValueError("Retargeted joints exceed Go2 model limits")
    return {
        "frames_checked": len(raw), "foot_link_max_error_m": float(errors.max()),
        "foot_link_mean_error_m": float(errors.mean()),
        "joint_limit_min_margin_rad": float(np.minimum(q - lo, hi - q).min()),
        "root_height_range_m": [float(values[:, 2].min()), float(values[:, 2].max())],
        "joint_max_velocity_rad_s": float(np.abs(np.gradient(q, 1 / LOCK["source_fps"], axis=0)).max()),
        "dynamic_feasibility": "FK validates geometry; physical execution is evaluated in PhysX",
    }


def verify_output(output: Path, clips: list[str]):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["dataset"] != DATASET or manifest["revision"] != REVISION or manifest["clips"] != clips:
        raise ValueError("Output manifest differs from selected pinned data")
    if {p.name for p in output.glob("*.csv")} != {clip + ".csv" for clip in clips}:
        raise ValueError("Unexpected motion CSVs in the output directory")
    for record in manifest["files"]:
        path = output / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Converted CSV hash mismatch: {path}")
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        if values.shape != (record["frames"], 19) or not np.isfinite(values).all():
            raise ValueError(f"Malformed converted CSV: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=REPO / "data/raw/kine2go")
    parser.add_argument("--output", type=Path, default=REPO / "data/retargeted/go2_kine2go")
    parser.add_argument("--motion_config", type=Path, default=REPO / "data/configs/go2_kine2go.json")
    parser.add_argument("--clips", nargs="+", default=LOCK["default_clips"])
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    if len(set(args.clips)) != len(args.clips) or any(not re.fullmatch(r"ai4_[a-z0-9_]+", c) for c in args.clips):
        parser.error("Select unique AI4Animation clip identifiers; other source licenses need an importer extension")
    source, output, config_path = (p.expanduser().resolve() for p in (args.source_dir, args.output, args.motion_config))
    if not args.offline and not args.verify_only:
        download(source, args.clips)
    source_manifest = verify_source(source, args.clips)
    if args.download_only:
        print(json.dumps(source_manifest, indent=2))
        return
    from bootstrap import configure_paths
    configure_paths()
    from trackerTask.supermimic.robots import get_robot
    spec = get_robot("go2")
    if (output / "manifest.json").exists() or args.verify_only:
        manifest = verify_output(output, args.clips)
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("Use an empty output directory to preserve existing data")
        records = []
        for clip in args.clips:
            path = source / "data" / clip / "motion.npy"
            raw = np.load(path, allow_pickle=False)
            meta = json.loads(path.with_name("clip.json").read_text())
            if meta["fps"] != LOCK["source_fps"]:
                raise ValueError("Clip frame rate differs from pinned schema")
            values = convert_array(raw, spec.joint_names)
            validation = validate_fk(raw, values, spec)
            target = output / (clip + ".csv")
            np.savetxt(target, values, delimiter=",", fmt="%.10f")
            records.append({
                "file": target.name, "frames": len(values), "duration_seconds": (len(values) - 1) / meta["fps"],
                "sha256": digest(target), "raw_sha256": digest(path),
                "publisher_metadata_frames": meta["n_frames"],
                "source": meta["source"], "source_url": meta["source_url"], "license": meta["license"],
                "attribution": meta["attribution"], "validation": validation,
            })
        manifest = {
            "version": 1, "task": "go2", "dataset": DATASET, "revision": REVISION,
            "source": "Kine2Go Go2-retargeted AI4Animation reference motion",
            "source_pipeline": LOCK["pipeline_repository"], "pipeline_revision": LOCK["pipeline_revision"],
            "retargeted_from_mocap": True, "clips": args.clips, "source_fps": LOCK["source_fps"],
            "quaternion_order": "xyzw", "joint_names": list(spec.joint_names),
            "raw_joint_order": LOCK["raw_joint_order"], "raw_foot_order": LOCK["raw_foot_order"],
            "mjcf_sha256": digest(spec.mjcf_path), "importer_sha256": digest(__file__),
            "files": records, "source_files": source_manifest["files"],
            "conversion": "Full NPY clip; reorder joints and WXYZ->XYZW. No smoothing, cropping or IK rerun.",
            "metadata_note": "Actual NPY row count is authoritative; publisher clip.json frame counts can differ.",
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_output(output, args.clips)
    config = {
        "version": 1, "root": os.path.relpath(output, config_path.parent).replace("\\", "/"),
        "source_fps": LOCK["source_fps"], "quaternion_order": "xyzw", "joint_names": list(spec.joint_names),
        "files": [record["file"] for record in manifest["files"]],
    }
    if args.verify_only:
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Motion configuration differs from verified data")
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"motion_config": str(config_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()

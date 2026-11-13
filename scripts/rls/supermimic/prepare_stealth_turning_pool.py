"""Add audited left/right turning references to an existing G1 full-body pool."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
from scipy.spatial.transform import Rotation

from bootstrap import configure_paths

REPO = configure_paths()
from full_body_reference import assess_reference, solve_reference, validate_motion
from prepare_stealth_full_body import (
    ROLES, SCHEMA, TURNING_SCHEMA, load_selected, local_file, verify_full_body_pool,
)
from prepare_style_campaign import (
    _identity, _json_bytes, _require_ignored_output, digest, motion_config, verify_pool,
)
from prepare_style_references import ORIGINAL_AIMING_SHA256, OFFICIAL_REVISION
from stealth_turning import mirror_g1_motion, mirror_receipt
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import generate, robot_parameters


def _origin_record(path: Path, source_name: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("schema") != "lafan1-g1-origin-v1"
            or document.get("revision") != OFFICIAL_REVISION
            or document.get("task_group") != "aiming"):
        raise ValueError("Turning source needs the pinned original Aiming provenance")
    rows = [row for row in document.get("files", []) if row.get("file") == source_name]
    if len(rows) != 1:
        raise ValueError(f"Origin manifest needs exactly one record for {source_name}")
    return rows[0]


def _load_origin(path: Path, source: Path, bvh: Path) -> dict:
    record = _origin_record(path, source.name)
    if record.get("sha256") != digest(source):
        raise ValueError("Turning source differs from its origin manifest")
    if (bvh.name not in ORIGINAL_AIMING_SHA256 or not bvh.name.startswith("aiming2_")
            or digest(bvh) != ORIGINAL_AIMING_SHA256[bvh.name]
            or record.get("source_file") != bvh.name
            or record.get("source_sha256") != ORIGINAL_AIMING_SHA256[bvh.name]
            or record.get("source_group") != "aiming2" or record.get("split") != "train"
            or record.get("source_fps") != 30 or record.get("output_fps") != 50
            or record.get("target_model_sha256") != digest(get_robot("g1").mjcf_path)
            or record.get("joint_names") != list(get_robot("g1").joint_names)):
        raise ValueError("Turning provenance must bind original Aiming2 and the current G1 model")
    frames = record.get("source_frame_range")
    if (not isinstance(frames, list) or len(frames) != 2
            or any(type(frame) is not int for frame in frames)
            or not 0 <= frames[0] < frames[1] <= record.get("source_total_frames", 0)):
        raise ValueError("Turning source needs a valid original frame interval")
    values = validate_motion(np.loadtxt(source, delimiter=",", ndmin=2))
    expected_frames = (frames[1] - frames[0] - 1) * 50 // 30 + 1
    if len(values) != record.get("output_frames") or len(values) != expected_frames:
        raise ValueError("Turning frame count must preserve the declared 30-to-50 Hz timing")
    yaw = np.unwrap(Rotation.from_quat(values[:, 3:7]).as_euler("xyz")[:, 2])
    if np.degrees(yaw[-1] - yaw[0]) < 15:
        raise ValueError("The chronological donor must contain a measured left turn of at least15deg")
    return record


def verify_turning_extension(pool, manifest, style):
    """Recompute mirror derivation and original-source identity from bundled bytes."""
    pool = Path(pool).resolve()
    declaration = manifest.get("turning_extension", {})
    if (declaration.get("audit_file") != "turning-reference-audit.json"
            or declaration.get("roles") != ["turn_left", "turn_right"]):
        raise ValueError("Turning extension needs both roles and its audit")
    audit_path = local_file(pool, declaration["audit_file"])
    if digest(audit_path) != declaration.get("audit_sha256"):
        raise ValueError("Turning audit bytes differ")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit != manifest["recipe"].get("turning_extension")
            or audit.get("schema") != "g1-stealth-turning-reference-v2"
            or audit.get("mirror") != mirror_receipt(tuple(get_robot("g1").joint_names)).to_dict()):
        raise ValueError("Turning audit or mirror convention differs from the recipe")
    origin = audit["source"]
    inputs = {}
    for name in ("csv", "origin_manifest", "bvh", "base_manifest"):
        binding = origin[name]
        path = local_file(pool, binding["file"])
        if digest(path) != binding["sha256"]:
            raise ValueError(f"Turning {name} source bytes changed")
        inputs[name] = path
    record = _load_origin(inputs["origin_manifest"], inputs["csv"], inputs["bvh"])
    if record != audit["origin_record"]:
        raise ValueError("Turning origin record differs from the bundled provenance")
    base = json.loads(inputs["base_manifest"].read_text(encoding="utf-8"))
    if (base.get("recipe", {}).get("purpose") != SCHEMA
            or _identity(base["recipe"]) != base.get("recipe_sha256")
            or origin["base_manifest"]["sha256"] != manifest["recipe"].get("base_pool_manifest_sha256")
            or {row.get("clip_role") for row in base["files"] if row["split"] == "train"} != set(ROLES)):
        raise ValueError("Turning extension must preserve the three-role base pool")
    by_name = {row["file"]: row for row in manifest["files"]}
    for row in base["files"]:
        if by_name.get(row["file"]) != row or digest(local_file(pool, row["file"])) != row["sha256"]:
            raise ValueError("Turning extension changed a base motion or heldout record")
    if (manifest["recipe"]["guidance"] != base["recipe"]["guidance"]
            or manifest["recipe"]["source"] != base["recipe"]["source"]
            or audit["selected_pose_sha256"] != style["selection"]["pose_sha256"]):
        raise ValueError("Turning extension changed selected aim guidance")
    left = np.loadtxt(local_file(pool, style["sources"]["turn_left"]["file"]), delimiter=",", ndmin=2)
    right = np.loadtxt(local_file(pool, style["sources"]["turn_right"]["file"]), delimiter=",", ndmin=2)
    source = np.loadtxt(inputs["csv"], delimiter=",", ndmin=2)
    expected = mirror_g1_motion(source, tuple(get_robot("g1").joint_names))
    if (left.shape != source.shape or right.shape != source.shape
            or not np.array_equal(left, source)
            or not np.allclose(right, expected, atol=1e-12, rtol=0)):
        raise ValueError("Turning donor bytes do not reproduce the declared sagittal mirror")
    for role in ("turn_left", "turn_right"):
        rows = [row for row in manifest["files"] if row.get("clip_role") == role]
        if len(rows) != 1:
            raise ValueError("Turning role must occur exactly once")
        row = rows[0]
        if any(row.get(key) != record[key] for key in
               ("source_file", "source_fps", "source_frame_range", "source_group")):
            raise ValueError("Turning role source identity changed")
        if (row.get("original_sha256") != record["source_sha256"]
                or row.get("frames") != len(source) or row.get("fps") != 50):
            raise ValueError("Turning role timing or original hash changed")
    return audit


def _row(role: str, name: str, values: np.ndarray, source: Path, origin: dict,
         report: dict, mirrored: bool) -> dict:
    source_hash = digest(source)
    source_record = {
        "family": "aiming",
        "prepared_file": source.name,
        "prepared_sha256": source_hash,
        "prepared_fps": 50,
        "prepared_frame_range": [0, len(values)],
        "source_file": origin["source_file"],
        "source_sha256": origin["source_sha256"],
        "source_group": origin["source_group"],
        "split": "train",
        "frames": len(values),
        "vertical_range_m": float(np.ptp(values[:, 2])),
        "turning_transform": "sagittal G1 reflection" if mirrored else None,
    }
    return {
        "clip_role": role,
        "duration_seconds": (len(values) - 1) / 50,
        "file": name,
        "fps": 50,
        "frames": len(values),
        "full_body_geometry": report,
        "original_sha256": origin["source_sha256"],
        "parent_csv_sha256": source_hash,
        "source_file": origin["source_file"],
        "source_fps": origin["source_fps"],
        "source_frame_range": origin["source_frame_range"],
        "source_group": origin["source_group"],
        "split": "train",
        "source": source_record,
        "transform": (
            "Sagittal G1 reflection followed by full-body IK: selected torso/height, "
            "bounded joints, chronological foot targets."
            if mirrored else
            "Full-body IK: selected torso/height, bounded joints, chronological foot targets."
        ),
        "upper_body": "Selected arm pose; waist remains an IK variable. Policy outputs stay learned.",
    }


def prepare(base_pool: Path, turning_source: Path, origin_manifest: Path, output: Path,
            source_bvh: Path) -> dict:
    base_pool = Path(base_pool).resolve()
    turning_source = Path(turning_source).resolve()
    origin_manifest = Path(origin_manifest).resolve()
    source_bvh = Path(source_bvh).resolve()
    output = Path(output).resolve()
    _require_ignored_output(output)
    for parent in (base_pool, turning_source.parent, origin_manifest.parent, source_bvh.parent):
        if output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ValueError("Turning output and source trees must be disjoint")
    if output.exists():
        raise FileExistsError(output)
    if not turning_source.is_file() or not origin_manifest.is_file():
        raise FileNotFoundError("Turning source and origin manifest are required")
    base_result = verify_full_body_pool(base_pool)
    base_manifest = json.loads((base_pool / "manifest.json").read_text(encoding="utf-8"))
    base_style = json.loads((base_pool / "full-body-style.json").read_text(encoding="utf-8"))
    if base_style["schema"] != SCHEMA:
        raise ValueError("Use the original three-role full-body pool")
    selected = (base_pool / base_style["selection"]["pose"]).resolve()
    selected_report = (base_pool / base_style["selection"]["report"]).resolve()
    reference, _ = load_selected(selected, selected_report)
    source = validate_motion(np.loadtxt(turning_source, delimiter=",", ndmin=2))
    origin = _load_origin(origin_manifest, turning_source, source_bvh)
    inputs = {path: digest(path) for path in (
        base_pool / "manifest.json", turning_source, origin_manifest, source_bvh,
        selected, selected_report)}
    model, spec, *_ = build_model("g1", generate("stealth", .76, 101), self_collisions=True)
    motors = robot_parameters("g1")[2]
    mirrored_source = mirror_g1_motion(source, tuple(spec.joint_names))
    left_values = solve_reference(source, reference, model, spec, motors)
    right_values = solve_reference(mirrored_source, reference, model, spec, motors)
    reports = {
        "turn_left": assess_reference(
            left_values, source, reference, model, spec, motors,
        ),
        "turn_right": assess_reference(
            right_values, mirrored_source, reference, model, spec, motors,
        ),
    }
    if any(not report["passed"] for report in reports.values()):
        raise ValueError(f"Turning geometry failed: {reports}")
    if any(min(report["candidate_lifts"]) < 1 for report in reports.values()):
        raise ValueError("Turning references need sustained lifting by both feet")

    implementation = digest(Path(__file__))
    mirror = mirror_receipt(tuple(spec.joint_names)).to_dict()
    audit = {
        "schema": "g1-stealth-turning-reference-v2",
        "source": {
            "csv": {"file": f"sources/turning/{turning_source.name}",
                    "sha256": inputs[turning_source]},
            "origin_manifest": {"file": "sources/turning/origin.json",
                                "sha256": inputs[origin_manifest]},
            "bvh": {"file": f"sources/turning/{source_bvh.name}",
                    "sha256": inputs[source_bvh]},
            "base_manifest": {"file": "sources/base-manifest.json",
                              "sha256": inputs[base_pool / "manifest.json"]},
        },
        "origin_record": origin,
        "selected_pose_sha256": digest(selected),
        "mirror": mirror,
        "reports": reports,
        "implementation_sha256": {
            path.name: digest(path) for path in (
                Path(__file__), Path(__file__).with_name("stealth_turning.py"),
                Path(__file__).with_name("full_body_reference.py"))},
        "scope": "Kinematic reference preparation; policy dynamics and visual quality remain unproved.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stealth-turning-", dir=output.parent))
    try:
        shutil.copytree(base_pool, staging, dirs_exist_ok=True)
        (staging / "sources").mkdir(exist_ok=True)
        (staging / "sources/turning").mkdir()
        for name, path in (
            ("csv", turning_source), ("origin_manifest", origin_manifest),
            ("bvh", source_bvh), ("base_manifest", base_pool / "manifest.json")):
            shutil.copyfile(path, staging / audit["source"][name]["file"])
        left_source_name = "sources/turn_left.csv"
        right_source_name = "sources/turn_right.csv"
        np.savetxt(staging / left_source_name, source, delimiter=",", fmt="%.17g")
        np.savetxt(staging / right_source_name, mirrored_source, delimiter=",", fmt="%.17g")
        begin, end = origin["source_frame_range"]
        suffix = f"{source_bvh.stem}__f{begin:05d}-{end:05d}.csv"
        left_output, right_output = "turn_left__" + suffix, "turn_right__" + suffix
        np.savetxt(staging / left_output, left_values, delimiter=",", fmt="%.17g")
        np.savetxt(staging / right_output, right_values, delimiter=",", fmt="%.17g")
        left_binding = {
            "file": left_source_name, "input_path": str(turning_source),
            "kind": "chronological", "output_file": left_output,
            "sha256": digest(staging / left_source_name),
        }
        right_binding = {
            "file": right_source_name, "input_path": "generated from turn_left by sagittal reflection",
            "kind": "mirrored_chronological", "output_file": right_output,
            "sha256": digest(staging / right_source_name), "mirror": mirror,
        }
        style = copy.deepcopy(base_style)
        style["schema"] = TURNING_SCHEMA
        style["sources"].update(turn_left=left_binding, turn_right=right_binding)
        style["checks"].update(reports)
        style["scope"] = (
            "Kinematic reference quality with audited left/right turning extensions; "
            "trained physical and visual quality remain unproved."
        )
        (staging / "full-body-style.json").write_bytes(_json_bytes(style))
        records = copy.deepcopy(base_manifest["files"])
        records.extend([
            _row("turn_left", left_output, left_values, staging / left_source_name,
                 origin, reports["turn_left"], False),
            _row("turn_right", right_output, right_values, staging / right_source_name,
                 origin, reports["turn_right"], True),
        ])
        records[-2]["sha256"] = digest(staging / left_output)
        records[-1]["sha256"] = digest(staging / right_output)
        recipe = copy.deepcopy(base_manifest["recipe"])
        recipe["purpose"] = TURNING_SCHEMA
        recipe["turning_extension"] = audit
        recipe["base_pool_manifest_sha256"] = digest(base_pool / "manifest.json")
        recipe["turning_preparation_sha256"] = implementation
        for split in ("train", "test"):
            names = [row["file"] for row in records if row["split"] == split]
            (staging / f"{split}.json").write_bytes(_json_bytes(motion_config("g1", names)))
        manifest = copy.deepcopy(base_manifest)
        manifest["full_body_style"] = {"schema": TURNING_SCHEMA, "passed": True}
        manifest["recipe"] = recipe
        manifest["recipe_sha256"] = _identity(recipe)
        manifest["files"] = records
        manifest["configs"] = {
            split: {
                "file": f"{split}.json",
                "sha256": digest(staging / f"{split}.json"),
            }
            for split in ("train", "test")
        }
        manifest["auxiliary_files"]["full_body_style"]["sha256"] = digest(
            staging / "full-body-style.json"
        )
        (staging / "turning-reference-audit.json").write_bytes(_json_bytes(audit))
        manifest["turning_extension"] = {
            "audit_file": "turning-reference-audit.json",
            "audit_sha256": digest(staging / "turning-reference-audit.json"),
            "roles": ["turn_left", "turn_right"],
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        if any(digest(path) != value for path, value in inputs.items()):
            raise RuntimeError("Turning source changed during preparation")
        verify_pool(staging, "g1", "stealth")
        verification = verify_full_body_pool(staging)
        staging.rename(output)
    except Exception as error:
        raise RuntimeError(f"Turning preparation failed; evidence retained at {staging}") from error
    return {
        "pool": str(output),
        "manifest_sha256": digest(output / "manifest.json"),
        "base_pool_manifest_sha256": digest(base_pool / "manifest.json"),
        "turning_audit_sha256": digest(output / "turning-reference-audit.json"),
        "records": [left_output, right_output],
        "base_verification": base_result,
        "verification": verification,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--turning-source", type=Path, required=True)
    parser.add_argument("--origin-manifest", type=Path, required=True)
    parser.add_argument("--source-bvh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        args.base_pool, args.turning_source, args.origin_manifest, args.output, args.source_bvh
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

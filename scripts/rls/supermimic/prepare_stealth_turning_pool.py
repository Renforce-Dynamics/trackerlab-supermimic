"""Add audited left/right turning references to an existing G1 full-body pool."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from bootstrap import configure_paths

REPO = configure_paths()
from full_body_reference import assess_reference, load_selected, solve_reference, validate_motion
from prepare_stealth_full_body import verify_full_body_pool
from prepare_style_campaign import _identity, _json_bytes, digest, motion_config, verify_pool
from stealth_turning import mirror_g1_motion, mirror_receipt
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import robot_parameters


def _origin_record(path: Path, source_name: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    for row in document.get("files", []):
        if row.get("file") == source_name:
            return row
    raise ValueError(f"Origin manifest has no record for {source_name}")


def _load_origin(path: Path, source: Path) -> tuple[dict, dict]:
    record = _origin_record(path, source.name)
    expected = record.get("sha256")
    if expected != digest(source):
        raise ValueError("Turning source differs from its origin manifest")
    return record, {
        "manifest": str(path.resolve()),
        "manifest_sha256": digest(path),
        "source_csv": source.name,
        "source_csv_sha256": expected,
    }


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


def prepare(base_pool: Path, turning_source: Path, origin_manifest: Path, output: Path) -> dict:
    base_pool = Path(base_pool).resolve()
    turning_source = Path(turning_source).resolve()
    origin_manifest = Path(origin_manifest).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not turning_source.is_file() or not origin_manifest.is_file():
        raise FileNotFoundError("Turning source and origin manifest are required")
    base_result = verify_full_body_pool(base_pool)
    base_manifest = json.loads((base_pool / "manifest.json").read_text(encoding="utf-8"))
    base_style = json.loads((base_pool / "full-body-style.json").read_text(encoding="utf-8"))
    selected = (base_pool / base_style["selection"]["pose"]).resolve()
    selected_report = (base_pool / base_style["selection"]["report"]).resolve()
    reference, _ = load_selected(selected, selected_report)
    source = validate_motion(np.loadtxt(turning_source, delimiter=",", ndmin=2))
    origin, origin_receipt = _load_origin(origin_manifest, turning_source)

    spec = get_robot("g1")
    import mujoco

    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
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

    implementation = digest(Path(__file__))
    mirror = mirror_receipt(tuple(spec.joint_names)).to_dict()
    audit = {
        "schema": "g1-stealth-turning-reference-v1",
        "source": origin_receipt | {
            "origin_source_file": origin["source_file"],
            "origin_source_sha256": origin["source_sha256"],
            "source_frame_range": origin["source_frame_range"],
            "source_group": origin["source_group"],
        },
        "selected_pose_sha256": digest(selected),
        "mirror": mirror,
        "reports": reports,
        "implementation_sha256": implementation,
        "scope": "Kinematic reference preparation; policy dynamics and visual quality remain unproved.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stealth-turning-", dir=output.parent))
    try:
        shutil.copytree(base_pool, staging, dirs_exist_ok=True)
        (staging / "sources").mkdir(exist_ok=True)
        left_source_name = "sources/turn_left.csv"
        right_source_name = "sources/turn_right.csv"
        np.savetxt(staging / left_source_name, source, delimiter=",", fmt="%.17g")
        np.savetxt(staging / right_source_name, mirrored_source, delimiter=",", fmt="%.17g")
        left_output = "turn_left__aiming2_subject2__f00200-00261.csv"
        right_output = "turn_right__aiming2_subject2__f00200-00261.csv"
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
        recipe = copy.deepcopy(base_manifest["recipe"])
        recipe["turning_extension"] = audit
        recipe["base_pool_manifest_sha256"] = digest(base_pool / "manifest.json")
        recipe["implementation_sha256"] = implementation
        for split in ("train", "test"):
            names = [row["file"] for row in records if row["split"] == split]
            (staging / f"{split}.json").write_bytes(_json_bytes(motion_config("g1", names)))
        manifest = copy.deepcopy(base_manifest)
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
        verify_pool(staging, "g1", "stealth")
        verify_full_body_pool(staging)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "pool": str(output),
        "manifest_sha256": digest(output / "manifest.json"),
        "base_pool_manifest_sha256": digest(base_pool / "manifest.json"),
        "turning_audit_sha256": digest(output / "turning-reference-audit.json"),
        "records": [left_output, right_output],
        "base_verification": base_result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--turning-source", type=Path, required=True)
    parser.add_argument("--origin-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        args.base_pool, args.turning_source, args.origin_manifest, args.output
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

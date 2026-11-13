"""Publish source-bound G1 full-body stealth references and verify their bytes."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np

from bootstrap import configure_paths

REPO = configure_paths()
from full_body_reference import (
    CRITERIA, STAGGERED_IDLE, assess_reference, assess_staggered_idle,
    solve_reference, solve_staggered_idle, validate_motion,
)
from prepare_style_campaign import _build_pool, _json_bytes, _require_ignored_output, verify_pool
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import generate, robot_parameters


SCHEMA = "g1-stealth-full-body-style-v1"
ROLES = ("forward", "backward", "idle")
REQUIRED_ROLES = frozenset(ROLES)
OPTIONAL_ROLES = frozenset(("turn_left", "turn_right"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_file(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("Expected a relative artifact path without traversal")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("Reference artifact must be a file inside its pool")
    return path


def load_selected(path, report_path):
    spec = get_robot("g1")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (report.get("schema") != "selected-canonical-aim-g1-v2"
            or report.get("geometry_acceptance", {}).get("passed") is not True
            or report.get("sha256") != digest(path)
            or report.get("joint_names") != list(spec.joint_names)
            or report.get("model_sha256") != digest(spec.mjcf_path)
            or report.get("source", {}).get("canonical_sha256")
            != "6095c299915e19c8afd04034eb3c4d3b54d8bf60854648e6a2e53e7019980766"
            or report["source"].get("source_frame") != 68):
        raise ValueError("Selected pose differs from the reviewed source, robot or geometry")
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if values.shape != (1, 36) or not np.isfinite(values).all():
        raise ValueError("This recipe requires the single reviewed selected pose")
    return values[0], report


def aim_guidance(reference, selected_report, source_pool, forward_pool, selected_path, report_path):
    import mujoco
    import torch
    from trackerLab.algorithms.supermimic.stealth_aim_style import AimStyleGuidance, calibrate_aim_config
    from trackerLab.algorithms.supermimic.stealth_palm_separation import calibrate_palm_separation_config
    from trackerLab.kinematics.supermimic.robot import RobotKinematics
    from trackerLab.kinematics.supermimic.rotations import quat_to_mat

    spec = get_robot("g1")
    if os.name == "nt":
        spec = replace(spec, mjcf_path=os.path.relpath(spec.mjcf_path))
    kin = RobotKinematics("cpu", spec)
    state = (
        torch.tensor(reference[:3], dtype=torch.float32)[None],
        quat_to_mat(torch.tensor(reference[[6, 3, 4, 5]], dtype=torch.float32)[None]),
        torch.zeros(1, 3), torch.zeros(1, 3),
        torch.tensor(reference[7:], dtype=torch.float32)[None], torch.zeros(1, 29),
    )
    source = {
        **selected_report["source"], "selected_output_frame": 0,
        "selected_csv_sha256": digest(selected_path),
        "retarget_report_sha256": digest(report_path),
        "parent_manifest_sha256": digest(source_pool / "manifest.json"),
        "forward_manifest_sha256": digest(forward_pool / "manifest.json"),
        "scope": "Selected whole-body posture for chronological moving clips and a planted static idle hold.",
    }
    config = calibrate_aim_config(kin, state, source, weight=.01, duration=.6)
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    qids = [int(model.joint(name).qposadr[0]) for name in spec.joint_names]
    data.qpos[:3], data.qpos[3:7], data.qpos[qids] = reference[:3], reference[[6, 3, 4, 5]], reference[7:]
    config["palm_separation"] = calibrate_palm_separation_config(
        model, data, source, weight=.006, margin=.010, scale=.010)
    return AimStyleGuidance(kin, config).to_dict()


def prepare(source_pool, forward_pool, selected_path, report_path, output):
    source_pool, forward_pool, selected_path, report_path, output = [
        Path(path).resolve() for path in (source_pool, forward_pool, selected_path, report_path, output)]
    if output.exists():
        raise FileExistsError(output)
    _require_ignored_output(output)
    for parent in (source_pool, forward_pool):
        if output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ValueError("Use a separate immutable output tree")
    parent = verify_pool(source_pool, "g1", "stealth")
    original = verify_pool(forward_pool, "g1", "stealth")
    role_rows = {}
    for row in parent["files"]:
        if row["split"] == "train":
            role = row.get("clip_role")
            if role not in ROLES or role in role_rows:
                raise ValueError("Parent pool needs one chronological training clip per role")
            role_rows[role] = (source_pool, row)
    if set(role_rows) != set(ROLES):
        raise ValueError("Parent pool requires forward, backward and idle roles")
    forwards = [row for row in original["files"] if row["split"] == "train"]
    if len(forwards) != 1 or forwards[0].get("source_group") != "aiming2":
        raise ValueError("Original forward pool must have one Aiming2 training clip")
    role_rows["forward"] = (forward_pool, forwards[0])
    reference, selected_report = load_selected(selected_path, report_path)
    guidance = aim_guidance(
        reference, selected_report, source_pool, forward_pool, selected_path, report_path)
    model, spec, *_ = build_model("g1", generate("stealth", .76, 101), self_collisions=True)
    _, _, motors = robot_parameters("g1")
    inputs = {path: digest(path) for path in (
        source_pool / "manifest.json", forward_pool / "manifest.json", selected_path, report_path)}
    transformed, source_bindings, source_payloads, records, checks = {}, {}, {}, [], {}
    for role in ROLES:
        directory, row = role_rows[role]
        if role == "idle":
            candidate, values = solve_staggered_idle(reference, row["frames"], model, spec, motors)
            check = assess_staggered_idle(candidate, values, reference, model, spec, motors)
            name = "idle__selected_staggered.csv"
            stream = BytesIO()
            np.savetxt(stream, values, delimiter=",", fmt="%.17g")
            payload = stream.getvalue()
            new_row = {
                "file": name, "fps": 50, "frames": len(values), "split": "train",
                "duration_seconds": (len(values) - 1) / 50,
                "original_sha256": digest(selected_path),
                "source_pose_sha256": digest(selected_path),
                "source_group": "owner_selected_pose",
                "source_file": selected_path.name, "source_frame_range": [0, 1],
                "source_kind": "selected-pose static hold",
                "selected_source": copy.deepcopy(selected_report["source"]),
            }
            binding = {"kind": "selected_staggered_hold",
                       "selected_pose_sha256": digest(selected_path)}
        else:
            source = directory / row["file"]
            inputs[source] = digest(source)
            payload = source.read_bytes()
            values = validate_motion(np.loadtxt(source, delimiter=",", ndmin=2))
            candidate = solve_reference(values, reference, model, spec, motors)
            check = assess_reference(candidate, values, reference, model, spec, motors)
            name = "forward__" + row["file"] if role == "forward" else row["file"]
            new_row = copy.deepcopy(row)
            binding = {"kind": "chronological", "input_path": str(source)}
        if not check["passed"]:
            raise ValueError(f"Full-body reference failed for {role}: {check['failures']}")
        if Path(name).name != name:
            raise ValueError("Output clip must have a plain filename")
        payload_hash = hashlib.sha256(payload).hexdigest()
        if "shared_aim_source" in new_row:
            new_row["parent_shared_aim_source"] = new_row.pop("shared_aim_source")
        new_row.update(
            file=name, clip_role=role, parent_csv_sha256=payload_hash,
            transform=("Selected staggered static hold, flat planted soles, bounded whole-body IK."
                       if role == "idle" else
                       "Full-body IK: selected torso/height, bounded joints, chronological foot targets."),
            upper_body="Selected arm pose; waist remains an IK variable. Policy outputs stay learned.",
        )
        records.append(new_row)
        transformed[name] = candidate
        source_payloads[role] = payload
        source_bindings[role] = {
            **binding, "file": f"sources/{role}.csv", "sha256": payload_hash, "output_file": name}
        checks[role] = check
    heldout = [row for row in parent["files"] if row["split"] == "test"]
    if not heldout:
        raise ValueError("Preserve the independent Aiming1 holdout")
    for row in heldout:
        inputs[source_pool / row["file"]] = digest(source_pool / row["file"])
    recipe = {
        "version": 1, "purpose": SCHEMA, "source": guidance["source"], "guidance": guidance,
        "parent_recipe_sha256": parent["recipe_sha256"],
        "forward_recipe_sha256": original["recipe_sha256"],
        "target_clearance_m": float(reference[2]), "criteria": CRITERIA,
        "idle_stance": STAGGERED_IDLE,
        "solver": {"mink": importlib.metadata.version("mink"),
                   "daqp": importlib.metadata.version("daqp"), "mujoco": __import__("mujoco").__version__,
                   "dt": 1 / 50, "substeps": 20, "first_pose_iterations": 100},
        "implementation_sha256": {
            path.name: digest(path) for path in (Path(__file__), Path(__file__).with_name("full_body_reference.py"))},
    }

    def unchanged():
        if any(digest(path) != value for path, value in inputs.items()):
            raise RuntimeError("A reference input changed during preparation")

    def writer(directory):
        unchanged()
        (directory / "sources").mkdir()
        for role, binding in source_bindings.items():
            (directory / binding["file"]).write_bytes(source_payloads[role])
        (directory / "sources/selected-pose.csv").write_bytes(selected_path.read_bytes())
        (directory / "sources/selected-pose.json").write_bytes(report_path.read_bytes())
        output_records = copy.deepcopy(records)
        for row in output_records:
            path = directory / row["file"]
            np.savetxt(path, transformed[row["file"]], delimiter=",", fmt="%.17g")
            row["sha256"] = digest(path)
            row["full_body_geometry"] = checks[row["clip_role"]]
        for row in heldout:
            (directory / row["file"]).write_bytes((source_pool / row["file"]).read_bytes())
            output_records.append(copy.deepcopy(row))
        style = {
            "schema": SCHEMA, "passed": True, "target_clearance_m": float(reference[2]),
            "criteria": CRITERIA, "sources": source_bindings, "idle_stance": STAGGERED_IDLE,
            "selection": {
                "pose": "sources/selected-pose.csv", "pose_sha256": digest(selected_path),
                "report": "sources/selected-pose.json", "report_sha256": digest(report_path)},
            "heldout": {row["file"]: row["sha256"] for row in heldout},
            "checks": checks,
            "scope": "Kinematic reference quality; trained physical and visual quality remain unproved.",
        }
        auxiliary = {}
        for key, name, value in (
            ("aim_style", "aim-style-reference.json", guidance),
            ("full_body_style", "full-body-style.json", style),
        ):
            path = directory / name
            path.write_bytes(_json_bytes(value))
            auxiliary[key] = {"file": name, "sha256": digest(path)}
        unchanged()
        return output_records, {
            "auxiliary_files": auxiliary, "full_body_style": {"schema": SCHEMA, "passed": True},
        }

    output.mkdir(parents=True)
    pool, manifest = _build_pool(output, "g1", "stealth", recipe, writer)
    result = verify_full_body_pool(pool)
    result.update(pool=str(pool), recipe_sha256=manifest["recipe_sha256"])
    (output / "preparation.json").write_bytes(_json_bytes(result))
    return result


def verify_full_body_pool(pool):
    """Recheck source bindings, holdout and geometry on the current CPU model."""
    pool = Path(pool).resolve()
    manifest = verify_pool(pool, "g1", "stealth")
    auxiliary = manifest.get("auxiliary_files", {}).get("full_body_style", {})
    if auxiliary.get("file") != "full-body-style.json":
        raise ValueError("Pool does not declare a full-body reference audit")
    path = local_file(pool, auxiliary["file"])
    if digest(path) != auxiliary.get("sha256"):
        raise ValueError("Full-body audit bytes differ from the manifest")
    style = json.loads(path.read_text(encoding="utf-8"))
    if (style.get("schema") != SCHEMA or style.get("passed") is not True
            or style.get("criteria") != CRITERIA
            or style.get("idle_stance") != STAGGERED_IDLE
            or manifest.get("recipe", {}).get("purpose") != SCHEMA
            or manifest["recipe"].get("criteria") != CRITERIA
            or manifest["recipe"].get("idle_stance") != STAGGERED_IDLE):
        raise ValueError("Full-body reference schema or criteria differ")
    selection = style["selection"]
    selected = local_file(pool, selection["pose"])
    selected_report = local_file(pool, selection["report"])
    if digest(selected) != selection["pose_sha256"] or digest(selected_report) != selection["report_sha256"]:
        raise ValueError("Selected source bytes changed")
    reference, _ = load_selected(selected, selected_report)
    if (style.get("target_clearance_m") != float(reference[2])
            or manifest["recipe"].get("target_clearance_m") != float(reference[2])):
        raise ValueError("Clearance target differs from the selected pose")
    from train_terrain import load_shared_aim_reference
    load_shared_aim_reference(pool / "aim-style-reference.json", pool / "train.json")
    model, spec, *_ = build_model("g1", generate("stealth", .76, 101), self_collisions=True)
    _, _, motors = robot_parameters("g1")
    train = [row for row in manifest["files"] if row["split"] == "train"]
    train_roles = [row.get("clip_role") for row in train]
    if (len(train_roles) != len(set(train_roles))
            or not REQUIRED_ROLES.issubset(train_roles)
            or set(train_roles) - REQUIRED_ROLES - OPTIONAL_ROLES):
        raise ValueError("Require the forward, backward and idle roles, with optional turn roles")
    results = {}
    for row in train:
        role = row["clip_role"]
        binding = style["sources"][role]
        source = local_file(pool, binding["file"])
        if (digest(source) != binding["sha256"] or binding["output_file"] != row["file"]
                or binding["sha256"] != row.get("parent_csv_sha256")):
            raise ValueError("Chronological source bytes or role binding changed")
        if role == "idle":
            if (binding.get("kind") != "selected_staggered_hold"
                    or binding.get("selected_pose_sha256") != selection["pose_sha256"]):
                raise ValueError("Idle stance must bind the selected-pose hold")
            assess = assess_staggered_idle
        else:
            if binding.get("kind") not in ("chronological", "mirrored_chronological"):
                raise ValueError("Moving references must retain a chronological donor")
            assess = assess_reference
        results[role] = assess(
            np.loadtxt(pool / row["file"], delimiter=",", ndmin=2),
            np.loadtxt(source, delimiter=",", ndmin=2),
            reference, model, spec, motors)
        if not results[role]["passed"]:
            raise ValueError(f"Full-body geometry failed for {role}: {results[role]['failures']}")
    holdout = {row["file"]: row["sha256"] for row in manifest["files"] if row["split"] == "test"}
    if not holdout or holdout != style.get("heldout"):
        raise ValueError("Independent heldout data changed")
    return {
        "schema": SCHEMA, "passed": True, "manifest_sha256": digest(pool / "manifest.json"),
        "target_clearance_m": float(reference[2]), "criteria": CRITERIA, "checks": results,
        "runtime_joint_override": False, "visual_policy_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-pool", type=Path)
    parser.add_argument("--source-pool", type=Path)
    parser.add_argument("--forward-pool", type=Path)
    parser.add_argument("--selected-csv", type=Path)
    parser.add_argument("--selected-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inputs = (args.source_pool, args.forward_pool, args.selected_csv, args.selected_report, args.output)
    if args.verify_pool is not None:
        if any(value is not None for value in inputs):
            parser.error("--verify-pool cannot be combined with preparation inputs")
        result = verify_full_body_pool(args.verify_pool)
    else:
        if any(value is None for value in inputs):
            parser.error("Preparation requires both pools, selected CSV/report and output")
        result = prepare(*inputs)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

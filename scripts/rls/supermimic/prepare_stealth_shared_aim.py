"""Publish one selected aiming upper-body pose across all stealth expert roles."""

import argparse
import copy
import csv
from dataclasses import replace
import hashlib
from io import StringIO
import json
import os
from pathlib import Path

import numpy as np
import torch

from bootstrap import configure_paths
configure_paths()
from prepare_style_campaign import _build_pool, _json_bytes, _require_ignored_output, verify_pool
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.kinematics.supermimic.rotations import quat_to_mat
from trackerTask.supermimic.robots import get_robot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_upper(payload, reference, joint_names, frames):
    columns = {7+joint_names.index(name): format(value, ".17g")
               for name, value in reference.items()}
    rows = list(csv.reader(StringIO(payload.decode("ascii")), strict=True))
    if len(rows) != frames or any(len(row) != 7+len(joint_names) for row in rows):
        raise ValueError("Expected the declared plain numeric motion rows")
    before = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(before).all():
        raise ValueError("Motion fields must be finite")
    for row in rows:
        for index, value in columns.items():
            row[index] = value
    stream = StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    after = np.asarray(rows, dtype=np.float64)
    preserved = [i for i in range(before.shape[1]) if i not in columns]
    if not np.array_equal(before[:, preserved], after[:, preserved]):
        raise ValueError("Shared aiming changed a root or leg coordinate")
    return stream.getvalue().encode("ascii")


def prepare(source_pool, selected_report, selected_csv, output, *, frame=0,
            aim_weight=.01, entry_duration=.60):
    source_pool, selected_report, selected_csv, output = map(
        lambda p: Path(p).resolve(), (source_pool, selected_report, selected_csv, output))
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source_pool) or source_pool.is_relative_to(output):
        raise ValueError("Use a separate immutable output tree")
    _require_ignored_output(output)
    source_manifest = verify_pool(source_pool, "g1", "stealth")
    report = json.loads(selected_report.read_text(encoding="utf-8"))
    if report.get("schema") != "selected-canonical-aim-g1-v1":
        raise ValueError("Selection needs the selected current canonical-human retarget")
    row = report
    spec = get_robot("g1")
    if (row["sha256"] != digest(selected_csv) or row["file"] != selected_csv.name
            or row["source"]["canonical_sha256"] != "6095c299915e19c8afd04034eb3c4d3b54d8bf60854648e6a2e53e7019980766"
            or row["source"]["source_frame"] != 68
            or row["joint_names"] != list(spec.joint_names)
            or row["model_sha256"] != digest(spec.mjcf_path)):
        raise ValueError("Selection differs from the owner's subject5 six-second asset")
    values = np.loadtxt(selected_csv, delimiter=",", ndmin=2)
    if (values.shape != (row["output_frames"], 36) or not np.isfinite(values).all()
            or type(frame) is not int or not 0 <= frame < len(values)):
        raise ValueError("Invalid selected pose index or retargeted coordinates")
    reference = values[frame]
    indices = [i for i, name in enumerate(spec.joint_names)
               if any(part in name.split("_") for part in ("waist", "shoulder", "elbow", "wrist"))]
    if len(indices) != 17:
        raise ValueError("Expected three waist and fourteen arm joints")
    reference_q = {spec.joint_names[i]: float(reference[7+i]) for i in indices}
    source = {
        **row["source"],
        "asset_label": "LAFAN1 Aiming subject5 six-second current library display",
        "selected_output_frame": frame,
        "owner_screenshot_time_s": 2.27,
        "selected_csv_sha256": digest(selected_csv),
        "retarget_report_sha256": digest(selected_report),
        "parent_manifest_sha256": digest(source_pool / "manifest.json"),
        "scope": "Morphology-scaled robot pose near the owner-selected screenshot; no finger or weapon model.",
    }
    from trackerLab.algorithms.supermimic.stealth_aim_style import calibrate_aim_config
    state = (torch.tensor(reference[:3], dtype=torch.float32)[None],
             quat_to_mat(torch.tensor(reference[[6, 3, 4, 5]], dtype=torch.float32)[None]),
             torch.zeros(1, 3), torch.zeros(1, 3),
             torch.tensor(reference[7:], dtype=torch.float32)[None], torch.zeros(1, 29))
    kin_spec = replace(spec, mjcf_path=os.path.relpath(spec.mjcf_path)) if os.name == "nt" else spec
    kin = RobotKinematics("cpu", kin_spec)
    guidance = calibrate_aim_config(kin, state, source, weight=aim_weight, duration=entry_duration)
    transformed = {}
    inputs = {selected_report: digest(selected_report), selected_csv: digest(selected_csv),
              source_pool / "manifest.json": digest(source_pool / "manifest.json")}
    for record in source_manifest["files"]:
        path = source_pool / record["file"]
        inputs[path] = digest(path)
        transformed[record["file"]] = (replace_upper(
            path.read_bytes(), reference_q, list(spec.joint_names), record["frames"])
            if record["split"] == "train" else path.read_bytes())
    recipe = {
        "version": 1, "purpose": "Owner-selected upper-body aiming pose shared by every command",
        "source": source, "reference_q": reference_q,
        "parent_pool_recipe_sha256": source_manifest["recipe_sha256"],
        "guidance": guidance,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
    }
    def unchanged():
        if any(digest(path) != expected for path, expected in inputs.items()):
            raise RuntimeError("An input changed during shared aim preparation")
    def writer(directory):
        unchanged()
        records = copy.deepcopy(source_manifest["files"])
        for record in records:
            payload = transformed[record["file"]]
            (directory / record["file"]).write_bytes(payload)
            if record["split"] == "train":
                record["parent_csv_sha256"] = record["sha256"]
                record["sha256"] = hashlib.sha256(payload).hexdigest()
                record["transform"] = "Shared source-bound waist and arm pose; root, leg coordinates and timing retained."
                record["shared_aim_source"] = source
        payload = _json_bytes(guidance)
        (directory / "aim-style-reference.json").write_bytes(payload)
        return records, {
            "shared_aim": {"reference_q": reference_q, "source": source},
            "auxiliary_files": {"aim_style": {"file": "aim-style-reference.json",
                "sha256": hashlib.sha256(payload).hexdigest()}},
            "scope": "Upper-body reference is shared; leg exemplars remain chronological source motions. Policy actions stay learned.",
        }
    output.mkdir(parents=True)
    pool, manifest = _build_pool(output, "g1", "stealth", recipe, writer)
    unchanged()
    result = {"pool": str(pool), "recipe_sha256": manifest["recipe_sha256"],
              "manifest_sha256": digest(pool/"manifest.json"), "source_inputs_unchanged": True,
              "nonupper_coordinates_preserved": True,
              "selected_frame": frame, "source": source}
    (output / "preparation.json").write_bytes(_json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-pool", "selected-report", "selected-csv", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    result = prepare(args.source_pool, args.selected_report, args.selected_csv,
                     args.output, frame=args.frame)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

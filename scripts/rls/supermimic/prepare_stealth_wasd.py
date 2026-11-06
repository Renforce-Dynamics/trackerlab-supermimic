"""Publish verified forward/backward/crouched-idle G1 clips with equal role mass.

Inputs are explicit prepared CSVs plus original retarget receipts and BVHs.
All three clips must have equal lengths, so uniform complete-history sampling
assigns one third to each role without duplicating or reversing frames.
Optional static arms replace only the 14 named arm CSV fields. Aiming1 is
copied byte-for-byte from an existing verified pool.
"""

from __future__ import annotations

import argparse
import copy
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys

if sys.platform == "win32":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
from scipy.spatial.transform import Rotation

from bootstrap import configure_paths
configure_paths()
from prepare_stealth_entry import _read_reference, _replace_arm_fields, _sha, _write_new
from prepare_style_campaign import (
    _build_pool, _json_bytes, _local_file, _require_ignored_output, _validate_values, verify_pool,
)
from prepare_style_references import ORIGINAL_AIMING_SHA256
from retarget_lafan_aiming import BODY_BASIS, parse_bvh, forward_kinematics, human_to_robot
from trackerTask.supermimic.robots import get_robot


ROLES = ("forward", "backward", "idle")


def _quantiles(values):
    return np.quantile(values, [.05, .5, .95], axis=0).tolist()


def _source_metrics(payload, begin, end, scale):
    motion = parse_bvh(payload.decode("utf-8-sig"), begin, end - begin)
    if len(motion.positions) != end - begin:
        raise ValueError("Source range exceeds the original BVH")
    p, R = human_to_robot(*forward_kinematics(motion.positions, motion.rotations, motion.parents))
    body_R = R[:, 0] @ BODY_BASIS
    velocity = np.gradient(p[:, 0], motion.frame_time, axis=0) * scale
    local_v = np.einsum("nji,nj->ni", body_R, velocity)
    heading = np.unwrap(np.arctan2(body_R[:, 1, 0], body_R[:, 0, 0]))
    yaw = np.gradient(heading, motion.frame_time)
    speed = np.linalg.norm(local_v[:, :2], axis=-1)
    feet = [motion.names.index(side + part)
            for side in ("Left", "Right") for part in ("Foot", "Toe")]
    toe_ids = [motion.names.index(side + "Toe") for side in ("Left", "Right")]
    foot_v = np.gradient(p[:, feet], motion.frame_time, axis=0) * scale
    clearance = (p[:, 0, 2] - p[:, toe_ids, 2].min(-1)) * scale
    hand_ids = [motion.names.index(side + "Hand") for side in ("Left", "Right")]
    hands_local = np.einsum("nji,nkj->nki", body_R, p[:, hand_ids] - p[:, :1]) * scale
    upright = body_R[:, 2, 2]
    return {
        "method": "Original BVH local channels -> FK -> pelvis BODY_BASIS; unchanged frame order.",
        "source_frame_time_s": motion.frame_time,
        "root_scale_g1": scale,
        "body_vx_m_s_p05_p50_p95": _quantiles(local_v[:, 0]),
        "body_vy_m_s_p05_p50_p95": _quantiles(local_v[:, 1]),
        "planar_speed_m_s_p05_p50_p95": _quantiles(speed),
        "yaw_rate_rad_s_p05_p50_p95": _quantiles(yaw),
        "forward_fraction_above_0_08": float((local_v[:, 0] > .08).mean()),
        "backward_fraction_below_minus_0_08": float((local_v[:, 0] < -.08).mean()),
        "low_speed_fraction_below_0_15": float((speed < .15).mean()),
        "low_yaw_fraction_below_0_25": float((abs(yaw) < .25).mean()),
        "clearance_proxy_m_p05_p50_p95": _quantiles(clearance),
        "upright_cos_p05_p50_p95": _quantiles(upright),
        "hands_forward_g1_proxy_m_p05_p50_p95": _quantiles(hands_local[..., 0].min(-1)),
        "foot_xy_speed_proxy_mean_m_s": float(np.linalg.norm(foot_v[..., :2], axis=-1).mean()),
        "foot_xy_range_proxy_max_m": float(
            np.linalg.norm(np.ptp(p[:, feet, :2], axis=0), axis=-1).max() * scale),
        "qualification": "Kinematic source classification; no measured contact forces or physical tracking guarantee.",
    }


def _validate_role(role, metrics):
    if role == "forward":
        valid = metrics["forward_fraction_above_0_08"] >= .85
    elif role == "backward":
        valid = metrics["backward_fraction_below_minus_0_08"] >= .85
    else:
        valid = (metrics["low_speed_fraction_below_0_15"] >= .9
                 and metrics["low_yaw_fraction_below_0_25"] >= .9
                 and metrics["planar_speed_m_s_p05_p50_p95"][1] < .08
                and .3 <= metrics["clearance_proxy_m_p05_p50_p95"][1] <= .55
                and metrics["clearance_proxy_m_p05_p50_p95"][2] <= .65
                and metrics["upright_cos_p05_p50_p95"][0] >= .7
                and metrics["hands_forward_g1_proxy_m_p05_p50_p95"][0] >= .08
                 and metrics["foot_xy_speed_proxy_mean_m_s"] < .04)
    if not valid:
        raise ValueError(f"Original source FK does not support the declared {role} role")


def _prepared_metrics(values):
    R = Rotation.from_quat(values[:, 3:7]).as_matrix()
    v = np.gradient(values[:, :3], .02, axis=0)
    local = np.einsum("nji,nj->ni", R, v)
    return {
        "body_vx_m_s_p05_p50_p95": _quantiles(local[:, 0]),
        "planar_speed_m_s_p05_p50_p95": _quantiles(np.linalg.norm(local[:, :2], axis=-1)),
        "root_z_m_p05_p50_p95": _quantiles(values[:, 2]),
        "joint_velocity_abs_p95_rad_s": float(np.quantile(abs(np.gradient(values[:, 7:], .02, axis=0)), .95)),
    }


def prepare(prepared_manifest, source_root, heldout_pool, output, provenance, *, arm_reference=None):
    prepared_manifest, source_root, heldout_pool, output = (
        Path(value).expanduser().resolve()
        for value in (prepared_manifest, source_root, heldout_pool, output)
    )
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    for source in (prepared_manifest.parent, source_root, heldout_pool):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Input and output trees must be separate")
    _require_ignored_output(output)
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must describe this research derivative")
    spec = get_robot("g1")
    manifest_bytes = prepared_manifest.read_bytes()
    document = json.loads(manifest_bytes)
    if (document.get("schema") != "stealth-wasd-prepared-v1" or document.get("robot") != "g1"
            or document.get("fps") != 50 or document.get("quaternion_order") != "xyzw"
            or document.get("joint_names") != list(spec.joint_names)):
        raise ValueError("Prepared manifest schema, FPS or G1 joint order differs")
    clips = document.get("clips")
    if (not isinstance(clips, list) or len(clips) != 3 or
            sorted(row.get("role", "") for row in clips) != sorted(ROLES)):
        raise ValueError("Exactly one forward, backward and idle clip is required")
    source_manifest_bytes = (heldout_pool / "manifest.json").read_bytes()
    parent = verify_pool(heldout_pool, "g1", "stealth")
    heldout = [row for row in parent["files"] if row["split"] == "test"]
    if len(heldout) != 1 or heldout[0].get("source_group") != "aiming1":
        raise ValueError("Held-out pool must contain one Aiming1 test clip")
    test = heldout[0]
    inputs = {prepared_manifest: manifest_bytes,
              heldout_pool / "manifest.json": source_manifest_bytes,
              heldout_pool / test["file"]: (heldout_pool / test["file"]).read_bytes()}
    if _sha(inputs[heldout_pool / test["file"]]) != test["sha256"]:
        raise ValueError("Held-out CSV changed")
    reference_bytes, pose = None, None
    if arm_reference is not None:
        arm_reference = Path(arm_reference).expanduser().resolve()
        reference_bytes = arm_reference.read_bytes()
        pose = _read_reference(reference_bytes, list(spec.joint_names))
        inputs[arm_reference] = reference_bytes
    selected = []
    for role in ROLES:
        row = next(item for item in clips if item["role"] == role)
        name = row.get("source_file")
        if name not in ORIGINAL_AIMING_SHA256 or not name.startswith("aiming2_"):
            raise ValueError("Training clips must come from pinned original Aiming2 sources")
        source = _local_file(source_root, name)
        csv_path = _local_file(prepared_manifest.parent, row["file"])
        report_path = _local_file(prepared_manifest.parent, row["retarget_report"])
        for path in (source, csv_path, report_path):
            inputs[path] = path.read_bytes()
        if (_sha(inputs[source]) != ORIGINAL_AIMING_SHA256[name]
                or row["source_sha256"] != ORIGINAL_AIMING_SHA256[name]
                or _sha(inputs[csv_path]) != row["sha256"]
                or _sha(inputs[report_path]) != row["retarget_report_sha256"]):
            raise ValueError("Prepared clip, report or original source hash differs")
        receipt = json.loads(inputs[report_path])
        if receipt.get("schema") != "lafan1-g1-origin-v1" or len(receipt.get("files", [])) != 1:
            raise ValueError("Expected one original retarget receipt per prepared clip")
        original = receipt["files"][0]
        span = row["source_frame_range"]
        if (not isinstance(span, list) or len(span) != 2
                or any(type(value) is not int for value in span)
                or not 0 <= span[0] < span[1] or not 2 <= span[1] - span[0] <= 500):
            raise ValueError("Invalid original source frame range")
        expected_frames = math.floor((span[1] - span[0] - 1) * 50 / 30 + 1e-9) + 1
        fields = {
            "source_file": name, "source_sha256": row["source_sha256"],
            "source_frame_range": span, "file": row["file"], "sha256": row["sha256"],
            "source_group": "aiming2", "split": "train", "output_frames": expected_frames,
            "output_fps": 50., "source_fps": 30., "joint_names": list(spec.joint_names),
        }
        mismatches = {key: {"receipt": original.get(key), "prepared": value}
                      for key, value in fields.items() if original.get(key) != value}
        if row.get("frames") != expected_frames:
            mismatches["frame count"] = {"prepared_manifest": row.get("frames"),
                                         "expected": expected_frames}
        if expected_frames < 100:
            mismatches["minimum_frames"] = {"receipt": expected_frames, "required": 100}
        if mismatches:
            raise ValueError(f"Prepared clip metadata differs from its original retarget receipt: {mismatches}")
        if original.get("target_model_sha256") != _sha(Path(spec.mjcf_path).read_bytes()):
            raise ValueError("Retarget model identity differs")
        scale = original.get("root_scale")
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Missing finite source-to-robot scale")
        metrics = _source_metrics(inputs[source], *span, scale)
        _validate_role(role, metrics)
        values = np.loadtxt(BytesIO(inputs[csv_path]), delimiter=",", ndmin=2)
        _validate_values(values, "g1")
        if len(values) != expected_frames:
            raise ValueError("Prepared CSV frame count differs")
        final = inputs[csv_path]
        if pose is not None:
            final, _ = _replace_arm_fields(final, pose, list(spec.joint_names), len(values))
        selected.append((row, original, metrics, final, _prepared_metrics(values)))
    if len({row["frames"] for row, *_ in selected}) != 1:
        raise ValueError("Equal clip lengths are required for equal role probability without resampling")
    if len({row["file"] for row, *_ in selected} | {test["file"]}) != 4:
        raise ValueError("Training and held-out output filenames must be unique")
    sampling = {
        "method": "Uniform complete-history starts over equal-length clips.",
        "role_probabilities": {role: 1 / 3 for role in ROLES},
        "clip_frames": selected[0][0]["frames"],
        "history_steps_example": 10,
        "valid_windows_per_role_at_10": selected[0][0]["frames"] - 9,
        "sequence_transform": "None; no reversal, repetition, time warp or clip concatenation.",
        "idle_command": "Zero vx/vy/yaw holding the crouched pose; standing belongs to an external FSM.",
    }
    recipe = {
        "version": 1, "purpose": "real-source forward/backward/crouched-idle G1 expert coverage",
        "provenance": provenance.strip(), "prepared_manifest_sha256": _sha(manifest_bytes),
        "heldout_source_recipe_sha256": parent["recipe_sha256"],
        "heldout_manifest_sha256": _sha(source_manifest_bytes), "heldout_csv_sha256": test["sha256"],
        "reference_document_sha256": _sha(reference_bytes) if reference_bytes is not None else None,
        "sampling": sampling,
        "clips": [{**row, "published_sha256": _sha(final), "source_fk": metrics}
                  for row, _, metrics, final, _ in selected],
        "implementation_sha256": _sha(Path(__file__).read_text(encoding="utf-8").encode()),
    }

    def unchanged():
        if any(path.read_bytes() != payload for path, payload in inputs.items()):
            raise ValueError("An input changed during publication")

    def writer(directory):
        unchanged()
        records = []
        for row, original, metrics, final, prepared_metrics in selected:
            _write_new(directory / row["file"], final)
            report_name = f"{row['role']}-retarget.json"
            _write_new(directory / report_name, inputs[prepared_manifest.parent / row["retarget_report"]])
            records.append({
                "file": row["file"], "sha256": _sha(final), "frames": row["frames"], "fps": 50,
                "duration_seconds": (row["frames"] - 1) / 50., "split": "train",
                "source_group": "aiming2", "original_sha256": row["source_sha256"],
                "source_file": row["source_file"], "source_frame_range": row["source_frame_range"],
                "source_fps": original["source_fps"], "clip_role": row["role"],
                "source_fk": metrics, "prepared_kinematics": prepared_metrics,
                "upstream_record": original, "retarget_report": report_name,
                "retarget_report_sha256": row["retarget_report_sha256"],
                "transform": ("Original chronological retarget with explicit local static arm replacement."
                              if pose is not None else "Original chronological source-driven G1 retarget."),
                "local_static_arms": ({"reference_q": pose, "source_upper_body_posture_retained": False,
                                      "reference_sha256": _sha(reference_bytes)} if pose is not None else None),
            })
        _write_new(directory / test["file"], inputs[heldout_pool / test["file"]])
        records.append(copy.deepcopy(test))
        if reference_bytes is not None:
            _write_new(directory / "upper-body-reference.json", reference_bytes)
        return records, {
            "source": "Local G1 research derivatives of verified Aiming2 originals; Aiming1 held out unchanged.",
            "sampling": sampling, "hand_status": "Explicit static correction" if pose is not None else "Source-driven retarget; hand review pending",
            "reference_sha256": _sha(reference_bytes) if reference_bytes is not None else None,
            "qualification": "Source coverage and content identity only; physical training quality untested.",
        }

    unchanged()
    output.mkdir(parents=True, exist_ok=False)
    pool, _ = _build_pool(output, "g1", "stealth", recipe, writer)
    verified = verify_pool(pool, "g1", "stealth")
    unchanged()
    if (pool / test["file"]).read_bytes() != inputs[heldout_pool / test["file"]]:
        raise ValueError("Held-out bytes differ")
    result = {
        "pool": pool.relative_to(output).as_posix(), "recipe_sha256": verified["recipe_sha256"],
        "manifest_sha256": _sha((pool / "manifest.json").read_bytes()),
        "source_inputs_unchanged": True, "heldout_bytes_identical": True,
        "sampling": sampling, "hand_status": verified["hand_status"],
    }
    _write_new(output / "preparation.json", _json_bytes(result))
    return pool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-manifest", "source-root", "heldout-pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--arm-reference", type=Path,
                        help="Optional named 14-joint static pose or wrapper with reference_q and entry metadata")
    args = parser.parse_args(argv)
    pool = prepare(args.prepared_manifest, args.source_root, args.heldout_pool,
                   args.output, args.provenance, arm_reference=args.arm_reference)
    print(json.dumps({"pool": str(pool), "train_config": str(pool / "train.json"),
                      "test_config": str(pool / "test.json"),
                      "manifest": str(pool / "manifest.json"),
                      "reference": str(pool / "upper-body-reference.json") if args.arm_reference else None,
                      "recipe_sha256": verify_pool(pool, "g1", "stealth")["recipe_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

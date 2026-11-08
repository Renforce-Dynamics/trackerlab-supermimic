"""Bind a forward-expert waist pose to an opt-in G1 crouched-entry constraint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bootstrap import configure_paths
configure_paths()
from prepare_style_campaign import verify_pool
from trackerTask.supermimic.robots import get_robot


WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(pool):
    pool = Path(pool).resolve()
    manifest = verify_pool(pool, "g1", "stealth")
    candidates = [row for row in manifest["files"]
                  if row.get("clip_role") == "forward" and row.get("split") == "train"]
    if len(candidates) != 1:
        raise ValueError("Require one explicitly classified forward training expert")
    spec = get_robot("g1")
    row = candidates[0]
    clip = pool / row["file"]
    if digest(clip) != row["sha256"]:
        raise ValueError("Forward expert changed after pool verification")
    values = np.loadtxt(clip, delimiter=",", ndmin=2)
    if values.shape != (row["frames"], 7 + spec.num_joints) or not np.isfinite(values).all():
        raise ValueError("Invalid forward expert shape or values")
    indices = [spec.joint_names.index(name) for name in WAIST_NAMES]
    selected = values[:, 7 + np.asarray(indices)]
    # Use one observed pose, rather than independently combining joint medians.
    frame = int(np.abs(selected - np.median(selected, axis=0)).sum(-1).argmin())
    target = selected[frame]
    initial = np.asarray(spec.default_q)[indices]
    rate, acceleration, dt = 1.2, 3., .02
    distance = float(np.abs(target - initial).max())
    required = max(1.875 * distance / rate,
                   math.sqrt((10 / math.sqrt(3)) * distance / acceleration))
    duration = max(dt, math.ceil(required / dt) * dt)
    config = {
        "joint_names": list(spec.joint_names), "default_q": list(spec.default_q),
        "action_scale": list(spec.action_scale),
        "reference_q": dict(zip(WAIST_NAMES, target.tolist())),
        "ramp_duration_s": duration, "control_dt": dt,
        "max_joint_rate_rad_s": rate, "max_joint_acceleration_rad_s2": acceleration,
        "action_limit": 20.,
    }
    from trackerLab.algorithms.supermimic.stealth_waist_entry import WaistEntryConstraint
    WaistEntryConstraint.from_dict(config)
    return {
        "schema": "g1-stealth-waist-entry-v1",
        "constraint": config,
        "source": {
            "pool_manifest_sha256": digest(pool / "manifest.json"),
            "forward_csv": row["file"], "forward_csv_sha256": digest(clip),
            "converted_frame": frame, "source_file": row.get("source_file"),
            "source_frame_range": row.get("source_frame_range"),
            "selection": "Actual three-waist-joint sample nearest their componentwise median in L1.",
            "timing": "Minimum common quintic duration from existing 1.2rad/s and 3rad/s2 bounds, rounded up to 20ms.",
            "expert_pose_radians": target.tolist(),
        },
        "scope": "Explicit waist target controller, not learned waist control or recorded entry motion. "
                 "Holds the expert waist pose after entry; legs remain learned. "
                 "Kinematic target bounds do not establish physical or visual acceptance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output file")
    result = prepare(args.pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "reference_q": result["constraint"]["reference_q"],
                      "ramp_duration_s": result["constraint"]["ramp_duration_s"]}))


if __name__ == "__main__":
    main()

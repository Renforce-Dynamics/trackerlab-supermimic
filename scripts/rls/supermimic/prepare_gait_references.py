"""Prepare and audit real locomotion references with recording-disjoint splits."""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS, ROBOTS
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import ROBOT_HEIGHTS
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.kinematics.supermimic.rotations import mat_to_quat


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def motion_config(robot, names):
    return {"version": 1, "root": ".", "source_fps": 50, "quaternion_order": "xyzw",
            "joint_names": list(get_robot(robot).joint_names), "files": names}


def physical_floor_distances(values, robot):
    import mujoco
    from trackerTask.supermimic.mujoco_terrain import build_model
    from trackerTask.supermimic.terrain import generate

    terrain = generate("stairs", ROBOT_HEIGHTS[robot], 0)
    terrain = replace(terrain, heights=np.zeros_like(terrain.heights))
    model, _, _, qids, *_ = build_model(robot, terrain, ground_plane=True)
    feet = {model.body(name).id for name in FOOT_BODIES[robot]}
    geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in feet
             and (model.geom_contype[i] or model.geom_conaffinity[i])]
    if not geoms:
        raise ValueError("The configured feet need physical collision geometry")
    plane = model.geom("supermimic_ground").id
    data = mujoco.MjData(model)
    distances = []
    for row in values:
        data.qpos[:3], data.qpos[3:7], data.qpos[qids] = row[:3], row[[6, 3, 4, 5]], row[7:]
        mujoco.mj_forward(model, data)
        distances.append(min(mujoco.mj_geomDistance(model, data, plane, geom, 10., None)
                             for geom in geoms))
    return np.asarray(distances)


def align_reference_floor(values, robot):
    """Walking references keep a physical support foot on the same plane as training."""
    distance = physical_floor_distances(values, robot)
    clearance = .003 * ROBOT_HEIGHTS[robot] / .76
    required_shift = clearance - distance
    window = min(21, len(values) if len(values) % 2 else len(values) - 1)
    smooth_shift = savgol_filter(required_shift, window, 2) if window >= 5 else required_shift
    shift = np.maximum(smooth_shift, required_shift)
    values = values.copy()
    values[:, 2] += shift
    return values, {
        "method": "Per-frame root-height alignment to the lowest physical foot collider",
        "assumption": "Walking has a supporting foot; this pass is not for flight-phase motions",
        "clearance_m": clearance, "root_height_shift_range_m": [float(shift.min()), float(shift.max())],
        "original_floor_distance_range_m": [float(distance.min()), float(distance.max())],
    }


def reference_audit(root, robot, train_files, test_files, sources):
    """Measure actual FK and file hashes; missing or invalid evidence fails."""
    spec, scale = get_robot(robot), ROBOT_HEIGHTS[robot] / .76
    kin = RobotKinematics("cpu", spec)
    tracked = [kin.body_names[index] for index in kin.tracked_idx]
    feet = [tracked.index(name) for name in FOOT_BODIES[robot]]
    offset = torch.tensor(FOOT_OFFSETS.get(robot, (0., 0., 0.)))
    records, failures = {}, []
    training_hashes = {digest(root / name) for name in train_files}
    test_hashes = {digest(root / name) for name in test_files}
    if set(train_files) & set(test_files) or training_hashes & test_hashes:
        failures.append("Training and evaluation must have distinct recordings and bytes")
    for name in [*train_files, *test_files]:
        values = np.loadtxt(root / name, delimiter=",", ndmin=2)
        motion = MotionLib([str(root / name)], kin, "cpu", src_fps=50)
        arm = (motion.body_R[:, feet] @ offset[..., None]).squeeze(-1)
        foot = motion.body_p[:, feet] + arm
        velocity = motion.body_v[:, feet] + torch.cross(motion.body_w[:, feet], arm, dim=-1)
        zfloor = foot[..., 2].quantile(.10, dim=0)
        stance = (foot[..., 2] <= zfloor + .035 * scale) & (velocity[..., :2].norm(dim=-1) < .25 * scale)
        speed = motion.root_v[:, :2].norm(dim=-1) / scale
        tilt = torch.acos(motion.root_R[:, 2, 2].clamp(-1, 1))
        violations = torch.maximum(kin.q_lo - motion.q, motion.q - kin.q_hi).clamp_min(0)
        stance_speed = velocity[..., :2].norm(dim=-1)[stance] / scale
        row = {
            "sha256": digest(root / name), "frames": len(values),
            "duration_seconds": (len(values) - 1) / 50,
            "mean_speed_scale_m_s": float(speed.mean()),
            "moving_fraction": float((speed > .12).float().mean()),
            "root_tilt_p95_rad": float(tilt.quantile(.95)),
            "joint_limit_violation_max_rad": float(violations.max()),
            "joint_velocity_p99_rad_s": float(motion.qd.abs().quantile(.99)),
            "root_vertical_speed_p99_scale_m_s": float(
                (motion.root_v[:, 2].abs() / scale).quantile(.99)),
            "stance_fraction": float(stance.float().mean()),
            "stance_foot_frame_speed_scale_m_s": float(stance_speed.mean()) if len(stance_speed) else None,
            "foot_swing_range_scale_m": ((foot[..., 2].quantile(.95, dim=0) - zfloor) / scale).tolist(),
            "source": sources.get(name),
        }
        floor_distance = physical_floor_distances(values, robot) / scale
        row["physical_foot_floor_distance_scale_m"] = {
            "min": float(floor_distance.min()), "max": float(floor_distance.max()),
            "p95": float(np.quantile(floor_distance, .95)),
        }
        reasons = []
        if len(values) < 128:
            reasons.append("Fewer than 128 reference frames")
        if not .12 <= row["mean_speed_scale_m_s"] <= 2.0 or row["moving_fraction"] < .5:
            reasons.append("Insufficient sustained locomotion or excessive speed")
        if row["root_tilt_p95_rad"] > .4:
            reasons.append("Excessive reference torso tilt")
        if row["joint_velocity_p99_rad_s"] > 20:
            reasons.append("Reference joint velocity has a retargeting discontinuity")
        if row["root_vertical_speed_p99_scale_m_s"] > .8:
            reasons.append("Reference root height has an abrupt contact correction")
        if row["joint_limit_violation_max_rad"] > .02:
            reasons.append("Reference exceeds robot joint limits")
        if not .08 <= row["stance_fraction"] <= .9:
            reasons.append("Reference does not alternate stance and swing")
        if max(row["foot_swing_range_scale_m"]) > .6:
            reasons.append("Excessive swing height")
        if floor_distance.min() < -.005 or floor_distance.max() > .03:
            reasons.append("Reference support foot penetrates or floats above the physical ground")
        row["failures"] = reasons
        records[name] = row
        failures.extend(f"{name}: {reason}" for reason in reasons)
    return {
        "schema": "supermimic-gait-reference-audit-v1", "robot": robot,
        "status": "passed" if not failures else "failed", "failures": failures,
        "training_files": train_files, "evaluation_files": test_files,
        "model_sha256": digest(spec.mjcf_path), "records": records,
        "split": "Different source recordings; source derivatives stay on their original side",
        "scope": "Reference kinematics and locomotion content; physical tracking is evaluated separately",
    }


def verify_audit(config_path, robot):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / config["root"]).resolve()
    audit = json.loads((root / "gait_audit.json").read_text(encoding="utf-8"))
    if (audit["robot"] != robot or audit["status"] != "passed"
            or audit["training_files"] != config["files"]
            or audit["model_sha256"] != digest(get_robot(robot).mjcf_path)):
        raise ValueError("Gait reference audit, training membership or robot model differs")
    for name, row in audit["records"].items():
        if digest(root / name) != row["sha256"]:
            raise ValueError(f"Gait reference changed after audit: {name}")
        floor = row.get("physical_foot_floor_distance_scale_m")
        if not floor or floor["min"] < -.005 or floor["max"] > .03:
            raise ValueError(f"Physical support-foot alignment needs validation: {name}")
    return audit


def canonicalize(values):
    values = values.copy()
    rotation = Rotation.from_quat(values[:, 3:7])
    yaw = rotation[0].as_euler("xyz")[2]
    align = Rotation.from_euler("z", -yaw)
    origin = values[0, :3].copy()
    origin[2] = 0
    values[:, :3] = align.apply(values[:, :3] - origin)
    values[:, 3:7] = (align * rotation).as_quat()
    return values


def prepare_bones(source, output):
    """Use one morphology per take; retain take 004 for evaluation."""
    spec = get_robot("g1")
    train, test, sources = [], [], {}
    for take in range(1, 5):
        paths = sorted(Path(source).rglob(f"neutral_walk_180_R_{take:03d}__*.csv"))
        paths = [p for p in paths if not p.stem.endswith("_M")]
        if not paths:
            raise FileNotFoundError(f"Missing BONES-SEED neutral walk take {take}")
        path = next((p for p in paths if "__A083" in p.stem), paths[0])
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = ["root_translate" + axis for axis in "XYZ"] + [
                "root_rotate" + axis for axis in "XYZ"] + [name + "_dof" for name in spec.joint_names]
            raw = np.array([[float(row[key]) for key in fields] for row in reader])
        values = np.column_stack((
            raw[:, :3] * .01, Rotation.from_euler("xyz", raw[:, 3:6], degrees=True).as_quat(),
            np.deg2rad(raw[:, 6:]),
        ))
        values = canonicalize(values)
        values, alignment = align_reference_floor(values, "g1")
        name = path.stem + ".csv"
        np.savetxt(output / name, values, delimiter=",", fmt="%.9g")
        (test if take == 4 else train).append(name)
        sources[name] = {"dataset": "bones-studio/seed",
                         "revision": "2f59b2077b9da34dd4e43618e705c7cb962c9a66",
                         "path": str(path.resolve()), "sha256": digest(path), "source_fps": 50,
                         "conversion": "cm/degrees to m/radians; XYZ Euler to XYZW; initial SE(2) alignment",
                         "ground_alignment": alignment}
    return train, test, sources


def prepare_import(config_path, output, robot, holdout):
    """Resample the pinned importer output and deduplicate original trajectories."""
    spec = get_robot(robot)
    kin = RobotKinematics("cpu", spec)
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    source_root = (config_path.parent / config["root"]).resolve()
    train, test, sources, seen = [], [], {}, {}
    for name in config["files"]:
        original = source_root / name
        content_hash = digest(original)
        split = "test" if Path(name).stem in holdout else "train"
        if content_hash in seen:
            if seen[content_hash] != split:
                raise ValueError("Duplicate source motion crosses the training/evaluation split")
            continue
        seen[content_hash] = split
        motion = MotionLib([str(original)], kin, "cpu", src_fps=config["source_fps"])
        quaternion = mat_to_quat(motion.root_R)[:, [1, 2, 3, 0]]
        values = torch.cat((motion.root_p, quaternion, motion.q), -1).numpy()
        # Kine2Go includes bows and idle intervals inside otherwise locomotion-labelled clips.
        speed = motion.root_v[:, :2].norm(dim=-1).numpy() / (ROBOT_HEIGHTS[robot] / .76)
        tilt = torch.acos(motion.root_R[:, 2, 2].clamp(-1, 1)).numpy()
        valid = (speed > .15) & (speed < 3.5) & (tilt < .32)
        edges = np.diff(np.r_[False, valid, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if not len(starts):
            continue
        selected = int(np.argmax(ends - starts))
        begin, end = int(starts[selected]), int(ends[selected])
        if end - begin < 100:
            continue
        values = values[begin:end]
        original_time = np.arange(len(values), dtype=float) / 50
        # A uniform 0.75x reference pace retains the recorded gait and reduces initial demands.
        sample_time = np.arange(0, original_time[-1] + 1e-9, .02 * .75)
        position = np.stack([np.interp(sample_time, original_time, values[:, k]) for k in range(3)], -1)
        joints = np.stack([np.interp(sample_time, original_time, values[:, k])
                           for k in range(7, values.shape[1])], -1)
        orientation = Slerp(original_time, Rotation.from_quat(values[:, 3:7]))(sample_time).as_quat()
        values = np.column_stack((position, orientation, joints))
        values = canonicalize(values)
        values, alignment = align_reference_floor(values, robot)
        np.savetxt(output / name, values, delimiter=",", fmt="%.9g")
        (test if split == "test" else train).append(name)
        sources[name] = {"path": str(original), "sha256": content_hash,
                         "source_fps": config["source_fps"], "source_50hz_window": [begin, end],
                         "reference_pace": .75,
                         "ground_alignment": alignment,
                         "conversion": "50 Hz resampling; longest upright locomotion interval; 0.75x pace; initial SE(2) alignment"}
    return train, test, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=ROBOTS, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--bones-source", type=Path)
    sources.add_argument("--source-config", type=Path)
    parser.add_argument("--holdout", nargs="+", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output.resolve()
    if (output / "gait_audit.json").is_file():
        print(json.dumps(verify_audit(output / "train.json", args.robot), indent=2))
        return
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new reference directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.bones_source:
        if args.robot != "g1":
            parser.error("BONES-SEED robot-space source is G1")
        train, test, records = prepare_bones(args.bones_source, output)
    else:
        train, test, records = prepare_import(args.source_config, output, args.robot, set(args.holdout))
    if len(train) < 2 or not test:
        raise ValueError("Require at least two training recordings and one held-out recording")
    write_json(output / "train.json", motion_config(args.robot, train))
    write_json(output / "test.json", motion_config(args.robot, test))
    audit = reference_audit(output, args.robot, train, test, records)
    write_json(output / "gait_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit("Gait reference audit failed")


if __name__ == "__main__":
    main()

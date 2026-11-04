"""Prepare local, provenance-tracked style references from explicit G1 CSVs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from bootstrap import configure_paths

REPO = configure_paths()
from prepare_gait_references import digest, motion_config, write_json
from trackerTask.supermimic.robot import G1_JOINT_NAMES
from trackerTask.supermimic.robots import get_robot

LOCK = json.loads(Path(__file__).with_name("resources.lock.json").read_text(encoding="utf-8"))
DATASET = LOCK["motions"]
TARGET_FPS = 50
MIN_SEGMENT_FRAMES = 100
ORIGINAL_AIMING_FILES = (
    "aiming1_subject1.bvh", "aiming1_subject4.bvh",
    "aiming2_subject2.bvh", "aiming2_subject3.bvh", "aiming2_subject5.bvh",
)
ORIGINAL_AIMING_SHA256 = {
    "aiming1_subject1.bvh": "53c90d858221d02a623776b6bca9d4cfed17ea832fc7bd641b35549055af912c",
    "aiming1_subject4.bvh": "651f3c00bfe50fd19f6757faa31df2407ad70de11c30ae1f8525c6e49ab4b07b",
    "aiming2_subject2.bvh": "76ddbd67bd40710f2196f732e7686823744c3fdfcd5c3410838f4c8508508d98",
    "aiming2_subject3.bvh": "8f0bf3935998e1118dda8b01ded62d70bd3c679bdc5b5780a602924236a51cc9",
    "aiming2_subject5.bvh": "788d43f6ecd38a937e2423bf8139cd4c562e45502924034c9c23b7a0e83a7061",
}
OFFICIAL_ARCHIVE_SHA256 = "ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977"
OFFICIAL_REVISION = "94084601bacdf9cc3764b5c73daaeccae6035fac"
STYLE_POLICIES = {
    "walk": {
        "description": "Kinematically selected walking intervals from caller-listed recordings.",
        "minimum_speed_m_s": 0.10,
        "maximum_speed_m_s": 2.5,
        "minimum_clearance_m": 0.30,
        "maximum_clearance_m": 0.78,
        "require_forward_hands": False,
        "minimum_hand_forward_m": None,
        "maximum_hand_separation_m": None,
        "minimum_root_height_m": None,
        "maximum_root_vertical_speed_m_s": None,
    },
    "stealth": {
        "description": "Low-profile locomotion candidate selected by root clearance, speed and hand position.",
        "minimum_speed_m_s": 0.08,
        "maximum_speed_m_s": 2.0,
        "minimum_clearance_m": 0.30,
        "maximum_clearance_m": 0.70,
        "require_forward_hands": True,
        "minimum_hand_forward_m": 0.12,
        "maximum_hand_separation_m": 0.72,
        "minimum_root_height_m": None,
        "maximum_root_vertical_speed_m_s": None,
    },
    "parkour": {
        "description": "Moving intervals with vertical motion retained; flight and crouch phases are allowed.",
        "minimum_speed_m_s": 0.08,
        "maximum_speed_m_s": 3.5,
        "minimum_clearance_m": None,
        "maximum_clearance_m": None,
        "require_forward_hands": False,
        "minimum_hand_forward_m": None,
        "maximum_hand_separation_m": None,
        "minimum_root_height_m": 0.20,
        "maximum_root_vertical_speed_m_s": 3.0,
    },
}


def source_group(filename: str) -> str:
    """Return take identity shared by contemporaneous subject recordings."""
    stem = Path(filename).stem
    match = re.fullmatch(r"(.+)_subject\d+", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def resample_g1(values: np.ndarray, source_fps: float, target_fps: int = TARGET_FPS) -> np.ndarray:
    if not math.isfinite(source_fps) or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame rates must be positive and finite")
    source_time = np.arange(len(values), dtype=np.float64) / source_fps
    target_time = np.arange(math.floor(source_time[-1] * target_fps + 1e-9) + 1) / target_fps
    position = np.column_stack([
        np.interp(target_time, source_time, values[:, column]) for column in range(3)
    ])
    orientation = Slerp(source_time, Rotation.from_quat(values[:, 3:7]))(target_time).as_quat()
    joints = np.column_stack([
        np.interp(target_time, source_time, values[:, column])
        for column in range(7, values.shape[1])
    ])
    return np.column_stack((position, orientation, joints))


def _segments(mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    begins, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(begin), int(end)) for begin, end in zip(begins, ends)
            if end - begin >= minimum_frames]


@lru_cache(maxsize=5)
def _robot_kinematics(robot: str):
    from trackerLab.kinematics.supermimic.robot import RobotKinematics

    spec = get_robot(robot)
    model_path = Path(spec.mjcf_path)
    if model_path.is_relative_to(Path.cwd()):
        spec = replace(spec, mjcf_path=str(model_path.relative_to(Path.cwd())))
    return RobotKinematics("cpu", spec)


def inspect_kinematics(values: np.ndarray, robot: str = "g1",
                       fps: float = 30.0) -> dict[str, np.ndarray]:
    """Measure root, ankle and wrist FK using the selected robot model."""
    kin = _robot_kinematics(robot)
    spec = kin.robot
    body_lookup = {name: kin.tracked_idx.index(kin.body_names.index(name))
                   for name in (spec.root_body,)}
    if robot == "g1":
        wrist_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    else:
        from trackerTask.supermimic.gait import FOOT_BODIES
        wrist_names = tuple(name for name in spec.tracked_bodies
                            if ("wrist" in name.lower() or "hand" in name.lower()))
        if len(wrist_names) < 2:
            raise ValueError(f"{robot} has no two tracked wrist/hand frames for style screening")
        wrist_names = wrist_names[:2]
        foot_names = FOOT_BODIES[robot]
    for name in (*wrist_names, *foot_names):
        body_lookup[name] = kin.tracked_idx.index(kin.body_names.index(name))

    root_rotation = Rotation.from_quat(values[:, 3:7]).as_matrix()
    q = torch.as_tensor(values[:, 7:], dtype=torch.float32)
    with torch.no_grad():
        body_p = kin.forward(
            torch.as_tensor(values[:, :3], dtype=torch.float32),
            torch.as_tensor(root_rotation, dtype=torch.float32),
            torch.zeros((len(values), 3)),
            torch.zeros((len(values), 3)),
            q,
            torch.zeros_like(q),
        )[0].cpu().numpy()
    root = values[:, :3]
    hand_world = body_p[:, [body_lookup[name] for name in wrist_names]]
    foot_world = body_p[:, [body_lookup[name] for name in foot_names]]
    hand_root = np.einsum("nji,nkj->nki", root_rotation, hand_world - root[:, None, :])
    return {
        "speed": np.linalg.norm(np.gradient(root[:, :2], axis=0) * fps, axis=1),
        "clearance": root[:, 2] - np.min(foot_world[:, :, 2], axis=1),
        "root_height": root[:, 2],
        "root_vertical_speed": np.gradient(root[:, 2]) * fps,
        "hands_root": hand_root,
        "hands_forward_min": np.min(hand_root[:, :, 0], axis=1),
        "hands_separation": np.linalg.norm(hand_root[:, 0] - hand_root[:, 1], axis=1),
    }


def select_intervals(values: np.ndarray, fps: float, style: str, robot: str = "g1",
                     minimum_frames: int = MIN_SEGMENT_FRAMES) -> tuple[list[tuple[int, int]], dict]:
    if style not in STYLE_POLICIES:
        raise ValueError(f"Unknown style {style!r}")
    measured = inspect_kinematics(values, robot, fps)
    policy = STYLE_POLICIES[style]
    mask = ((measured["speed"] >= policy["minimum_speed_m_s"])
            & (measured["speed"] <= policy["maximum_speed_m_s"]))
    if policy["minimum_clearance_m"] is not None:
        mask &= measured["clearance"] >= policy["minimum_clearance_m"]
        mask &= measured["clearance"] <= policy["maximum_clearance_m"]
    if policy["minimum_root_height_m"] is not None:
        mask &= measured["root_height"] >= policy["minimum_root_height_m"]
    if policy["maximum_root_vertical_speed_m_s"] is not None:
        mask &= np.abs(measured["root_vertical_speed"]) <= policy["maximum_root_vertical_speed_m_s"]
    if policy["require_forward_hands"]:
        mask &= measured["hands_forward_min"] >= policy["minimum_hand_forward_m"]
        mask &= measured["hands_separation"] <= policy["maximum_hand_separation_m"]
    intervals = _segments(mask, minimum_frames)
    metrics = {
        "source_fps": fps,
        "qualifying_frames": int(mask.sum()),
        "candidate_intervals": [[start, end] for start, end in intervals],
        "mean_horizontal_speed_m_s": float(np.mean(measured["speed"])),
        "root_clearance_m_quantiles": np.quantile(measured["clearance"], [.05, .5, .95]).tolist(),
        "minimum_forward_hand_x_m_quantiles": np.quantile(measured["hands_forward_min"], [.05, .5, .95]).tolist(),
        "hand_separation_m_quantiles": np.quantile(measured["hands_separation"], [.05, .5, .95]).tolist(),
        "root_clearance_definition": "World root Z minus the lower ankle-link origin Z; a kinematic support-height proxy, not a collider-ground distance.",
        "policy": policy,
        "selection_semantics": "Kinematic candidate filter only; source filenames and poses do not certify an action class.",
    }
    return intervals, metrics


def _read_source(path: Path, minimum_frames: int = MIN_SEGMENT_FRAMES) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if values.shape[1] != 7 + len(G1_JOINT_NAMES) or values.shape[0] < minimum_frames:
        raise ValueError(f"{path.name} must have at least {minimum_frames} rows and 36 columns")
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, atol=1e-3):
        raise ValueError(f"{path.name} contains invalid values or non-unit XYZW quaternions")
    return values


def _resolve_inputs(source_dir: Path, patterns: tuple[str, ...]) -> list[Path]:
    files = []
    for pattern in patterns:
        if not pattern or Path(pattern).name != pattern or "/" in pattern or "\\" in pattern:
            raise ValueError("--files patterns must be plain CSV basenames")
        matches = sorted(source_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Explicit input pattern matched no files: {pattern}")
        files.extend(path.resolve() for path in matches if path.is_file())
    result = sorted(set(files))
    if any(path.parent != source_dir for path in result):
        raise ValueError("Input CSVs must be direct children of --source-g1-dir")
    if any(path.suffix.lower() != ".csv" for path in result):
        raise ValueError("--files accepts CSV files only")
    return result


def _load_gait_refs(directory: Path, robot: str, output: Path) -> tuple[list[str], list[str], list[dict]]:
    audit = json.loads((directory / "gait_audit.json").read_text(encoding="utf-8"))
    if audit.get("robot") != robot or audit.get("status") != "passed":
        raise ValueError("Existing gait references need a passed audit for the selected robot")
    if audit.get("model_sha256") and audit["model_sha256"] != digest(get_robot(robot).mjcf_path):
        raise ValueError("Existing gait audit names a different robot model")
    records = audit.get("records", {})
    train, test, provenance = [], [], []
    hashes = {}
    for split, names in (("train", audit["training_files"]), ("test", audit["evaluation_files"])):
        for name in names:
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Gait references must use plain CSV basenames")
            source = directory / name
            expected = records.get(name, {}).get("sha256")
            if not source.is_file() or not expected or digest(source) != expected:
                raise ValueError(f"Existing gait reference hash does not match its audit: {name}")
            if expected in hashes and hashes[expected] != split:
                raise ValueError("Identical gait source bytes cross train/test splits")
            hashes[expected] = split
            values = np.loadtxt(source, delimiter=",", ndmin=2)
            if (values.shape[1] != 7 + get_robot(robot).num_joints or len(values) < 2
                    or not np.isfinite(values).all()
                    or not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, atol=1e-3)):
                raise ValueError(f"Malformed audited gait reference: {name}")
            if records[name].get("frames", len(values)) != len(values):
                raise ValueError(f"Existing gait frame count differs from its audit: {name}")
            target_name = f"gait_{split}__{Path(name).name}"
            shutil.copy2(source, output / target_name)
            provenance.append({
                "file": target_name, "sha256": digest(output / target_name),
                "source_file": str(source), "source_sha256": expected,
                "split": split, "kind": "audited_gait_reference",
                "fps": 50, "frames": len(values),
                "duration_seconds": (len(values) - 1) / 50,
                "source_audit_sha256": digest(directory / "gait_audit.json"),
            })
            (train if split == "train" else test).append(target_name)
    return train, test, provenance


def _read_origins(path: Path | None) -> tuple[dict, dict]:
    if path is None:
        return {}, {}
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("files")
    if not isinstance(rows, list):
        raise ValueError("Origin manifest needs a files array")
    origins = {}
    for row in rows:
        name = row.get("file")
        if (not isinstance(name, str) or Path(name).name != name or name in origins
                or "/" in name or "\\" in name or not name.endswith(".csv")):
            raise ValueError("Origin file entries need unique CSV basenames")
        retained = {key: row[key] for key in (
            "file", "sha256", "source_file", "source_sha256", "source_group",
            "source_fps", "retargeter", "retargeter_revision", "retarget_config_sha256",
        ) if key in row}
        for key in ("sha256", "source_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(retained.get(key, ""))):
                raise ValueError(f"Origin {name} requires a lowercase SHA256 for {key}")
        origins[name] = retained
    provenance = {key: document[key] for key in (
        "dataset", "revision", "license", "archive_url", "archive_sha256",
    ) if key in document}
    provenance["origin_manifest_sha256"] = digest(path)
    return origins, provenance


def align_constant_floor(values: np.ndarray, distances: np.ndarray,
                         clearance_m: float = .003) -> tuple[np.ndarray, dict]:
    """Translate one clip vertically as a whole; retain airborne height changes."""
    distances = np.asarray(distances, dtype=np.float64)
    if (distances.shape != (len(values),) or not np.isfinite(distances).all()
            or not math.isfinite(clearance_m) or clearance_m < 0):
        raise ValueError("Floor distances must be finite and match the clip")
    shift = clearance_m - float(distances.min())
    result = values.copy()
    result[:, 2] += shift
    return result, {
        "method": "single_constant_root_z_translation",
        "target_minimum_physical_foot_clearance_m": clearance_m,
        "source_physical_foot_clearance_range_m": [
            float(distances.min()), float(distances.max())],
        "root_z_shift_m": shift,
        "airborne_trajectory_preserved": True,
        "per_frame_clamp": False,
    }


@lru_cache(maxsize=5)
def _physical_floor_model(robot: str):
    import mujoco
    from dataclasses import replace
    from trackerTask.supermimic.mujoco_terrain import build_model
    from trackerTask.supermimic.gait import FOOT_BODIES
    from trackerTask.supermimic.terrain import ROBOT_HEIGHTS, generate

    field = generate("stairs", ROBOT_HEIGHTS[robot], 0)
    field = replace(field, heights=np.zeros_like(field.heights))
    model, spec, _, qpos_ids, *_ = build_model(robot, field, ground_plane=True)
    feet = {model.body(name).id for name in FOOT_BODIES[robot]}
    foot_geoms = tuple(index for index in range(model.ngeom)
                       if model.geom_bodyid[index] in feet
                       and (model.geom_contype[index] or model.geom_conaffinity[index]))
    if not foot_geoms:
        raise ValueError(f"{robot} foot bodies have no physical collision geometry")
    return model, qpos_ids, foot_geoms, model.geom("supermimic_ground").id


def physical_floor_distances(values: np.ndarray, robot: str) -> np.ndarray:
    import mujoco

    model, qpos_ids, foot_geoms, plane = _physical_floor_model(robot)
    data = mujoco.MjData(model)
    output = np.empty(len(values), dtype=np.float64)
    for index, row in enumerate(values):
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]
        data.qpos[qpos_ids] = row[7:]
        mujoco.mj_forward(model, data)
        output[index] = min(mujoco.mj_geomDistance(model, data, plane, geom, 10., None)
                            for geom in foot_geoms)
    return output


def prepare_style_references(source_dir: Path, output_dir: Path, files: tuple[str, ...],
                             style: str, robot: str = "g1", source_fps: float = 30.0,
                             holdout_groups: tuple[str, ...] = (),
                             gait_reference_dir: Path | None = None,
                             minimum_frames: int = MIN_SEGMENT_FRAMES,
                             retarget_humanoid: bool = False,
                             source_config: Path | None = None,
                             origin_manifest: Path | None = None) -> dict:
    if source_config is not None:
        source_config = source_config.expanduser().resolve()
        cfg = json.loads(source_config.read_text(encoding="utf-8"))
        if (cfg.get("version") != 1 or cfg.get("quaternion_order") != "xyzw"
                or cfg.get("joint_names") != list(G1_JOINT_NAMES)):
            raise ValueError("Source config must describe G1 joints in XYZW order")
        source_dir = source_config.parent / cfg["root"]
        source_fps = cfg["source_fps"]
        if not files:
            files = tuple(cfg["files"])
    if isinstance(source_fps, bool) or not isinstance(source_fps, (int, float)):
        raise ValueError("Source FPS must be a positive number")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Source FPS must be a positive number")
    if not isinstance(minimum_frames, int) or isinstance(minimum_frames, bool) or minimum_frames < 2:
        raise ValueError("minimum_frames must be an integer of at least two")
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if style not in STYLE_POLICIES:
        raise ValueError(f"Unknown style {style!r}")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if robot == "go2":
        raise ValueError("No human-to-quadruped LAFAN1 retarget is supported")
    if robot not in ("g1", "a3", "t1", "bumi", "pi_plus"):
        raise ValueError(f"Unsupported robot {robot!r}")
    if robot != "g1" and not retarget_humanoid:
        raise ValueError("Humanoid Mink retargeting requires explicit --retarget-humanoid")
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must remain outside the source motion directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new or empty local research output: {output_dir}")
    if not output_dir.is_relative_to(REPO / "data"):
        raise ValueError("Local style derivatives must be stored under the ignored repository data/ tree")
    ignored = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--quiet", "--no-index", str(output_dir)],
        check=False, capture_output=True,
    )
    if ignored.returncode != 0:
        raise ValueError("Research output must be ignored by Git before creating motion derivatives")
    output_dir.mkdir(parents=True, exist_ok=True)

    origins, origin_provenance = _read_origins(origin_manifest)
    if style == "stealth" and origin_manifest is not None:
        expected_dataset = "ubisoft/ubisoft-laforge-animation-dataset"
        if (origin_provenance.get("dataset") != expected_dataset
                or origin_provenance.get("revision") != OFFICIAL_REVISION
                or origin_provenance.get("archive_sha256") != OFFICIAL_ARCHIVE_SHA256):
            raise ValueError("Aiming origin manifest must identify the reviewed Ubisoft archive and revision")
    source_train, source_test, gait_records = [], [], []
    if gait_reference_dir is not None:
        gait_root = gait_reference_dir.expanduser().resolve()
        if gait_root == output_dir or gait_root in output_dir.parents:
            raise ValueError("Output must remain outside the gait source directory")
        source_train, source_test, gait_records = _load_gait_refs(gait_root, robot, output_dir)

    sources = _resolve_inputs(source_dir, files)
    available_groups = {source_group(origins[path.name]["source_file"] if path.name in origins
                                     else path.name) for path in sources}
    if sources and set(holdout_groups) - available_groups:
        raise ValueError("Every explicit holdout group must name an available source take")
    input_rows, rejected_rows = [], []
    for source in sources:
        source_name = source.name
        source_hash = digest(source)
        origin = origins.get(source_name)
        if origin is not None and origin["sha256"] != source_hash:
            raise ValueError(f"Origin CSV hash differs from {source_name}")
        if style == "stealth" and (origin is None
                                  or origin.get("source_file") not in ORIGINAL_AIMING_FILES):
            rejected_rows.append({
                "source_file": source_name, "source_sha256": source_hash,
                "status": "waiting_for_correct_data",
                "reason": "Stealth requires an origin-manifest entry for a verified original Aiming BVH; walk files are not substituted.",
            })
            continue
        if style == "stealth" and origin["source_sha256"] != ORIGINAL_AIMING_SHA256[origin["source_file"]]:
            raise ValueError("Original Aiming BVH hash differs from the reviewed official archive member")
        try:
            original = _read_source(source, minimum_frames)
            intervals, metrics = select_intervals(original, source_fps, style, "g1", minimum_frames)
        except (ValueError, OSError) as error:
            rejected_rows.append({
                "source_file": source_name, "source_sha256": source_hash,
                "status": "rejected", "reason": str(error),
            })
            continue
        if not intervals:
            rejected_rows.append({
                "source_file": source_name, "source_sha256": source_hash,
                "status": "rejected", "reason": "No interval met the declared kinematic policy.",
                "kinematics": metrics,
            })
            continue
        group = source_group(origin["source_file"] if origin else source_name)
        if origin and origin.get("source_group", group) != group:
            raise ValueError("Origin source_group differs from original take identity")
        split = "test" if group in holdout_groups else "train"
        if not holdout_groups:
            split = "pending_deterministic_split"
        for start, end in intervals:
            values = resample_g1(original[start:end], source_fps)
            source_frame_hash = hashlib.sha256(
                original[start:end].astype("<f8").tobytes(order="C")).hexdigest()
            source_clip_name = f"{source.stem}__{style}__{start:05d}-{end:05d}__g1.csv"
            g1_derivative = output_dir / source_clip_name
            np.savetxt(g1_derivative, values, delimiter=",", fmt="%.9g")
            derivative = g1_derivative
            retarget_report = None
            result = values
            floor_alignment = None
            if robot != "g1":
                from retarget_gait_references import retarget

                retargeted, retarget_report = retarget(values, robot)
                if (retargeted.shape != (len(values), 7 + get_robot(robot).num_joints)
                        or not np.isfinite(retargeted).all()
                        or not np.allclose(np.linalg.norm(retargeted[:, 3:7], axis=1), 1, atol=1e-3)):
                    raise ValueError("Retargeter output violates the robot CSV contract")
                retarget_report = dict(
                    retarget_report,
                    source="Caller-supplied G1 LAFAN1 style window, identified in this manifest.",
                    implementation_sha256=digest(Path(__file__).with_name("retarget_gait_references.py")),
                )
                result = retargeted
                derivative = output_dir / source_clip_name.replace(
                    "__g1.csv", f"__{robot}.csv")
            if style == "parkour":
                distances = physical_floor_distances(result, robot)
                result, floor_alignment = align_constant_floor(result, distances)
            np.savetxt(derivative, result, delimiter=",", fmt="%.9g")
            row = {
                "source_file": source_name,
                "source_sha256": source_hash,
                "source_dataset": origin_provenance.get("dataset", "local_G1_CSVs"),
                "source_revision": origin_provenance.get("revision"),
                "source_identity_verified": origin is not None,
                "source_identity_scope": "CSV hash checked against supplied origin receipt; BVH hash is declared by the caller.",
                "origin": origin,
                "source_license": "CC BY-NC-ND 4.0; local research derivative only.",
                "source_fps": source_fps,
                "source_frames": int(len(original)),
                "source_window_frames": [start, end],
                "source_window_seconds": [start / source_fps, (end - 1) / source_fps],
                "source_window_float64_sha256": source_frame_hash,
                "window_convention": "Zero-based frame indices [begin,end); seconds identify first and last included samples.",
                "source_group": group,
                "file": derivative.name,
                "sha256": digest(derivative),
                "frames": int(len(result)),
                "fps": TARGET_FPS,
                "duration_seconds": (len(result) - 1) / TARGET_FPS,
                "robot": robot,
                "style": style,
                "split": split,
                "selection_kinematics": metrics,
                "upper_body_handling": {
                    "source_posture_retained": True,
                    "posture_policy": "Source upper-body posture retained through resampling and optional retargeting.",
                    "weapon_operation": "No firearm object, control, or weapon operation is modeled.",
                } if style == "stealth" else None,
                "floor_alignment": floor_alignment,
                "retarget": json.loads(json.dumps(retarget_report, default=lambda value:
                                                 value.item() if isinstance(value, np.generic) else value.tolist())),
            }
            input_rows.append(row)
    if not holdout_groups and input_rows:
        groups = sorted({row["source_group"] for row in input_rows})
        held_out = groups[-1]
        for row in input_rows:
            row["split"] = "test" if row["source_group"] == held_out else "train"
    train_names = source_train + [row["file"] for row in input_rows if row["split"] == "train"]
    test_names = source_test + [row["file"] for row in input_rows if row["split"] == "test"]
    training_hashes = {row["source_sha256"] for row in input_rows + gait_records if row["split"] == "train"}
    test_hashes = {row["source_sha256"] for row in input_rows + gait_records if row["split"] == "test"}
    if training_hashes & test_hashes:
        raise ValueError("Identical source bytes cross the training/test split")
    train_output_hashes = {row["sha256"] for row in input_rows + gait_records if row["split"] == "train"}
    test_output_hashes = {row["sha256"] for row in input_rows + gait_records if row["split"] == "test"}
    if train_output_hashes & test_output_hashes:
        raise ValueError("Identical derivative bytes cross the training/test split")
    style_train = [row["file"] for row in input_rows if row["split"] == "train"]
    style_test = [row["file"] for row in input_rows if row["split"] == "test"]
    if style == "walk":
        style_train = source_train + style_train
        style_test = source_test + style_test
    status = ("ready" if style_train and style_test else
              "waiting_for_correct_data" if not input_rows else "waiting_for_disjoint_style_split")
    if status == "ready":
        write_json(output_dir / "train.json", motion_config(robot, train_names))
        write_json(output_dir / "test.json", motion_config(robot, test_names))
    manifest = {
        "schema": "lafan1-style-reference-manifest-v1",
        "status": status,
        "dataset": origin_provenance.get("dataset", "local_G1_CSVs"),
        "revision": origin_provenance.get("revision"),
        "expected_public_G1_upstream": {"dataset": DATASET["repository"], "revision": DATASET["revision"]},
        "license": "CC BY-NC-ND 4.0 as declared by upstream dataset README.",
        "distribution": "No motion data or derivatives are to be published or redistributed by this utility.",
        "style": style,
        "style_description": STYLE_POLICIES[style]["description"],
        "robot": robot,
        "source_directory": str(source_dir),
        "source_config_sha256": digest(source_config) if source_config is not None else None,
        "origin_provenance": origin_provenance,
        "required_original_aiming_files": list(ORIGINAL_AIMING_FILES) if style == "stealth" else [],
        "reviewed_official_source": {
            "repository": "ubisoft/ubisoft-laforge-animation-dataset",
            "revision": OFFICIAL_REVISION,
            "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "aiming_member_sha256": ORIGINAL_AIMING_SHA256,
        } if style == "stealth" else None,
        "source_fps": source_fps,
        "output_fps": TARGET_FPS,
        "quaternion_order": "xyzw",
        "joint_names": list(get_robot(robot).joint_names),
        "source_model_sha256": digest(get_robot("g1").mjcf_path),
        "target_model_sha256": digest(get_robot(robot).mjcf_path),
        "preparation_sha256": digest(__file__),
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__},
        "minimum_source_frames": minimum_frames,
        "minimum_source_duration_seconds": (minimum_frames - 1) / source_fps,
        "selection_policy": STYLE_POLICIES[style],
        "selection_disclaimer": "Kinematic screening supports candidate discovery and does not establish the semantic action represented by a source take.",
        "split_policy": "Source takes are indivisible groups; all subject recordings sharing a take stay in the same split.",
        "held_out_source_groups": sorted({row["source_group"] for row in input_rows if row["split"] == "test"}),
        "training_files": train_names,
        "evaluation_files": test_names,
        "style_training_files": style_train,
        "style_evaluation_files": style_test,
        "clips": input_rows,
        "rejected_clips": rejected_rows,
        "existing_gait_references": gait_records,
        "quadruped_constraint": "Go2 is excluded; no humanoid-to-quadruped retargeting is inferred.",
        "audit_scope": "Kinematic window selection and file provenance only; no actuator-feasibility, gait-qualification, or physical-tracking claim.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-g1-dir", type=Path)
    inputs.add_argument("--source-config", type=Path)
    parser.add_argument("--files", nargs="+", default=[],
                        help="Explicit filenames or glob patterns, e.g. walk3_subject*.csv.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style", choices=tuple(STYLE_POLICIES), required=True)
    parser.add_argument("--robot", choices=("g1", "a3", "t1", "bumi", "pi_plus", "go2"), default="g1")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--holdout-groups", nargs="*", default=[])
    parser.add_argument("--gait-reference-dir", type=Path)
    parser.add_argument("--origin-manifest", type=Path,
                        help="CSV and original BVH SHA256s; required to establish Aiming provenance.")
    parser.add_argument("--minimum-frames", type=int, default=MIN_SEGMENT_FRAMES,
                        help="Minimum contiguous source-frame count; interpreted at --source-fps.")
    parser.add_argument("--retarget-humanoid", action="store_true")
    parser.add_argument("--confirm-local-research-only", action="store_true")
    args = parser.parse_args()
    if not args.confirm_local_research_only:
        parser.error("Confirm local-only use of CC BY-NC-ND motion derivatives.")
    if args.source_g1_dir is not None and not args.files:
        parser.error("--source-g1-dir requires explicit --files")
    manifest = prepare_style_references(
        args.source_g1_dir or Path("."), args.output, tuple(args.files), args.style,
        args.robot, args.source_fps, tuple(args.holdout_groups),
        args.gait_reference_dir, args.minimum_frames, args.retarget_humanoid,
        args.source_config, args.origin_manifest,
    )
    print(json.dumps({
        "manifest": str(args.output.resolve() / "manifest.json"),
        "status": manifest["status"],
        "style_clips": len(manifest["clips"]),
        "training": len(manifest["training_files"]),
        "held_out": len(manifest["evaluation_files"]),
        "rejected": len(manifest["rejected_clips"]),
        "held_out_groups": manifest["held_out_source_groups"],
    }, sort_keys=True))
    if manifest["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

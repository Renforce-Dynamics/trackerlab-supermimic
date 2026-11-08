"""Offline flat-G1 trajectory measurements; CPU FK only, no policy or physics steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


LIFT_M, BOUT_S, ENTRY_S = .02, .08, 4.84


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(np.float64)


def quat_matrix(wxyz):
    """Active WXYZ rotation; positive Y pitch tilts local up toward forward +X."""
    q = finite(wxyz, "WXYZ quaternion")
    if q.shape[-1:] != (4,) or not np.allclose(np.linalg.norm(q, axis=-1), 1., atol=1e-4, rtol=0):
        raise ValueError("Expected unit WXYZ quaternions (..., 4)")
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def lifted_bouts(clearance, dt):
    """Observed first-to-last sample spans; boundary bouts retain censor flags."""
    mask = np.asarray(clearance) > LIFT_M
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = []
    for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        duration = (end - begin - 1) * dt
        if duration + 1e-12 >= BOUT_S:
            result.append({"first_sample": int(begin), "last_sample": int(end-1),
                           "duration_s": float(duration), "left_censored": bool(begin == 0),
                           "right_censored": bool(end == len(mask))})
    return result


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {name: float(value) for name, value in zip(
        ("min", "max", "mean", "p95"),
        (values.min(), values.max(), values.mean(), np.percentile(values, 95)))}


def pitch_and_lean(rotation, heading):
    """ZYX pitch and backward-positive up-axis lean in the root-heading plane."""
    pitch = np.arctan2(-rotation[..., 2, 0], np.hypot(rotation[..., 0, 0], rotation[..., 1, 0]))
    lean = np.arctan2(-np.sum(rotation[..., :2, 2] * heading, axis=-1), rotation[..., 2, 2])
    return np.rad2deg(pitch), np.rad2deg(lean)


def sole_metrics(position, rotation, centers, radii, dt):
    """Material speed v + omega x arm at sphere poles, never pole differentiation."""
    arm = np.einsum("tfij,fsj->tfsi", rotation, centers)
    arm[..., 2] -= radii
    height = (position[:, :, None, :] + arm)[..., 2]
    velocity = np.gradient(position, dt, axis=0)
    spin = np.gradient(rotation, dt, axis=0) @ np.swapaxes(rotation, -1, -2)
    omega = .5 * np.stack((spin[..., 2, 1]-spin[..., 1, 2],
                           spin[..., 0, 2]-spin[..., 2, 0],
                           spin[..., 1, 0]-spin[..., 0, 1]), axis=-1)
    speed = np.linalg.norm((velocity[:, :, None, :] + np.cross(omega[:, :, None, :], arm))[..., :2], axis=-1)
    clearance = height.min(axis=-1)
    both_down = (clearance <= LIFT_M).all(axis=1)
    down_duration = float(np.count_nonzero(both_down[:-1] & both_down[1:]) * dt)
    feet = {}
    for foot, name in enumerate(("left", "right")):
        supported = height[:, foot] <= LIFT_M
        bouts = lifted_bouts(clearance[:, foot], dt)
        feet[name] = {"whole_foot_clearance_m": stats(clearance[:, foot]),
                      "sustained_lift_count": len(bouts), "sustained_lift_bouts": bouts,
                      "stance_pole_material_speed_m_s": stats(speed[:, foot][supported]),
                      "supported_pole_samples": int(supported.sum())}
    return {"feet": feet, "both_feet_down_duration_s": down_duration,
            "both_feet_down_proportion": down_duration / ((len(position)-1)*dt)}


def load_record(path, lane=0, start_second=0., end_second=None):
    start = finite(start_second, "start_second")
    end = None if end_second is None else finite(end_second, "end_second")
    if start.ndim or start < 0 or (end is not None and (end.ndim or end <= start)):
        raise ValueError("Require 0 <= start_second < end_second")
    with np.load(path, allow_pickle=False) as archive:
        required = ("qpos", "control_dt", "terminal_steps", "task", "iteration", "seed", "recorded_env_ids")
        if any(key not in archive for key in required):
            raise ValueError(f"Trajectory requires {required}")
        q = archive["qpos"]
        if q.ndim != 3 or q.shape[2] != 36 or q.shape[0] < 2 or q.shape[1] < 1:
            raise ValueError("Expected qpos (T>=2, N>=1, 36)")
        if isinstance(lane, bool) or not isinstance(lane, (int, np.integer)) or not 0 <= lane < q.shape[1]:
            raise ValueError("lane must index a recorded lane")
        dt = finite(archive["control_dt"], "control_dt")
        if dt.ndim or dt <= 0:
            raise ValueError("control_dt must be a positive scalar")
        dt = float(dt)
        terminal, ids = archive["terminal_steps"], archive["recorded_env_ids"]
        for name, values in (("terminal_steps", terminal), ("recorded_env_ids", ids)):
            if values.shape != (q.shape[1],) or values.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer vector of lane count")
        if np.any(terminal < -1) or np.any(terminal >= len(q)):
            raise ValueError("terminal_steps must be -1 or an in-range inclusive frame")
        if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("recorded_env_ids must be unique and nonnegative")
        meta = {}
        for key in ("task", "iteration", "seed"):
            value = archive[key]
            if value.ndim or (key == "task" and value.dtype.kind != "U") or (
                    key != "task" and (value.dtype.kind not in "iu" or value < 0)):
                raise ValueError(f"Invalid scalar {key}")
            meta[key] = value.item()
        if meta["task"] != "locomotion/g1/stealth":
            raise ValueError("Only flat locomotion/g1/stealth recordings are supported")
        last = int(terminal[lane]) if terminal[lane] >= 0 else len(q)-1
        times = np.arange(last+1) * dt
        indices = np.flatnonzero((times >= float(start)-1e-12) &
                                 (times <= (float(end) if end is not None else times[-1])+1e-12))
        if len(indices) < 2:
            raise ValueError("Selected pre-terminal window needs at least two samples")
        poses = finite(q[indices, lane], "selected qpos")
        quat_matrix(poses[:, 3:7])
        return poses, dt, {**meta, "lane": int(lane), "recorded_env_ids": ids.tolist(),
                           "environment_id": int(ids[lane]), "qpos_dtype": str(q.dtype),
                           "qpos_shape": list(q.shape),
                           "first_terminal_step": int(terminal[lane]), "last_episode_frame": last,
                           "frame_range_inclusive": [int(indices[0]), int(indices[-1])],
                           "observed_interval_s": [float(times[indices[0]]), float(times[indices[-1]])]}


def g1_geometry():
    import mujoco
    from bootstrap import configure_paths
    configure_paths()
    from trackerTask.supermimic.robots import get_robot
    from trackerTask.supermimic.gait import FOOT_BODIES

    spec = get_robot("g1")
    xml = Path(spec.mjcf_path).resolve()
    # Relative asset paths avoid ANSI fopen issues in Windows MuJoCo builds.
    path = os.path.relpath(xml) if xml.anchor.lower() == Path.cwd().anchor.lower() else str(xml)
    model = mujoco.MjModel.from_xml_path(path)
    root, torso = model.body(spec.root_body).id, model.body(spec.com_body).id
    addresses = [int(model.joint(name).qposadr[0]) for name in spec.joint_names]
    if (model.nq != 36 or model.njnt != 30 or model.jnt_qposadr[0] != 0
            or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[0] != root
            or sorted(addresses) != list(range(7, 36))
            or np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError("Configured G1 must have a free root plus the named 29 hinges")
    order = ["root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz"] + [""]*29
    for address, name in zip(addresses, spec.joint_names):
        order[address] = name
    feet = [model.body(name).id for name in FOOT_BODIES["g1"]]
    centers, radii, geom_ids = [], [], []
    for body in feet:
        geoms = np.flatnonzero((model.geom_bodyid == body) &
                              ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        if not len(geoms) or np.any(model.geom_type[geoms] != mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError("Configured G1 feet require spherical collision geoms")
        centers.append(model.geom_pos[geoms].copy())
        radii.append(model.geom_size[geoms, 0].copy())
        geom_ids.append(geoms.tolist())
    if len(centers[0]) != len(centers[1]):
        raise ValueError("Configured feet must have equal sphere counts")
    receipt = {"xml": str(xml), "xml_sha256": sha256(xml), "mujoco_version": mujoco.__version__,
               "qpos_order": order, "root_body": spec.root_body, "torso_body": spec.com_body,
               "foot_bodies": list(FOOT_BODIES["g1"]), "sphere_geom_ids": geom_ids,
               "sphere_centers_body_m": np.asarray(centers).tolist(), "sphere_radii_m": np.asarray(radii).tolist()}
    return model, [*feet, root, torso], np.asarray(centers), np.asarray(radii), receipt


def measure_poses(poses, dt, first_frame, model, bodies, centers, radii):
    import mujoco

    data = mujoco.MjData(model)
    position, rotation = [], []
    for pose in poses:
        data.qpos[:] = pose  # Original compiled MuJoCo order, including WXYZ; no remapping.
        mujoco.mj_kinematics(model, data)
        position.append(data.xpos[bodies].copy())
        rotation.append(data.xmat[bodies].reshape(4, 3, 3).copy())
    position, rotation = np.asarray(position), np.asarray(rotation)
    metrics = sole_metrics(position[:, :2], rotation[:, :2], centers, radii, dt)
    time = (np.arange(len(poses)) + first_frame) * dt
    entry = time <= ENTRY_S + 1e-12
    heading = rotation[:, 2, :2, 0]
    norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    if np.any(norm[entry] < 1e-8):
        raise ValueError("Root forward heading is undefined during entry")
    heading = heading / np.maximum(norm, 1e-8)
    entry_metrics = {"episode_interval_s": [0., ENTRY_S], "observed_samples": int(entry.sum())}
    for index, name in ((2, "root"), (3, "torso")):
        pitch, lean = pitch_and_lean(rotation[entry, index], heading[entry])
        entry_metrics[name] = {"pitch_deg": stats(pitch), "signed_backward_lean_deg": stats(lean)}
    return {"samples": len(poses), "duration_s": (len(poses)-1)*dt,
            "metrics": {**metrics, "entry": entry_metrics}}


def expert_comparison(path, model, bodies, centers, radii, geometry):
    """Inspect only explicitly listed local CSVs; no resampling or floor alignment."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Expert config must be a JSON object")
    names, files = cfg.get("joint_names"), cfg.get("files")
    expected = geometry["qpos_order"][7:]
    if (cfg.get("version") != 1 or cfg.get("quaternion_order") != "xyzw"
            or not isinstance(names, list) or any(not isinstance(name, str) for name in names)
            or sorted(names) != sorted(expected)
            or not isinstance(files, list) or not files or len(files) > 64
            or any(not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name
                   or Path(name).name != name for name in files) or len(set(files)) != len(files)
            or not isinstance(cfg.get("root"), str) or not cfg["root"]):
        raise ValueError("Expert config requires G1 joints, XYZW, and 1..64 unique local CSV basenames")
    fps = finite(cfg.get("source_fps"), "expert source_fps")
    if fps.ndim or fps <= 0:
        raise ValueError("Expert source_fps must be a positive scalar")
    root, clips = (path.parent / cfg["root"]).resolve(), []
    for name in files:
        clip = root / name
        values = finite(np.loadtxt(clip, delimiter=",", ndmin=2), "expert CSV")
        if values.ndim != 2 or values.shape[1] != 36 or len(values) < 2:
            raise ValueError("Expert CSV requires (T>=2, 36)")
        poses = np.empty_like(values)
        poses[:, :3], poses[:, 3:7] = values[:, :3], values[:, [6, 3, 4, 5]]
        poses[:, 7:] = values[:, 7 + np.array([names.index(joint) for joint in expected])]
        quat_matrix(poses[:, 3:7])
        clips.append({"file": str(clip), "sha256": sha256(clip),
                      **measure_poses(poses, 1/float(fps), 0, model, bodies, centers, radii)})
    return {"config": str(path), "config_sha256": sha256(path), "source_fps": float(fps),
            "input_order": "XYZ, XYZW, joint_names", "joint_names": names, "clips": clips,
            "scope": "Source comparison only; clip-time entry is not a standing-to-motion transition. "
                     "No floor alignment, resampling, training-pool membership or style-success assertion."}


def audit(path, lane=0, start_second=0., end_second=None, expert_config=None):
    poses, dt, source = load_record(path, lane, start_second, end_second)
    model, bodies, centers, radii, geometry = g1_geometry()
    comparison = ({"expert_source_comparison": expert_comparison(expert_config, model, bodies, centers, radii, geometry)}
                  if expert_config is not None else {})
    return {"schema": "stealth-trajectory-audit-v1", "trajectory": str(Path(path).resolve()),
            "trajectory_sha256": sha256(path), "source": source, "model": geometry,
            "control_dt_s": dt,
            **measure_poses(poses, dt, source["frame_range_inclusive"][0], model, bodies, centers, radii),
            **comparison,
            "definitions": {"lift": "min(all sphere bottom heights) > 0.02 m for observed span >= 0.08 s",
                            "bout_indices": "relative to selected window; censored flags denote window edges",
                            "both_down_duration": "intervals with both feet <= 0.02 m at both endpoint samples",
                            "pitch": "ZYX Euler pitch; positive Y tilts local up forward",
                            "backward_lean": "atan2(-body up projected on root horizontal heading, body up Z)",
                            "slip": "horizontal v_body + omega_world x pole_arm; numpy first/central differences; "
                                    "pole support proxy: sphere bottom <= 0.02 m"},
            "limitations": ["Flat stealth only: heights relative to world Z=0, not terrain-relative.",
                            "Geometric support is not measured contact force; slip is a sampled FK estimate.",
                            "Configured original XML only; no reconstruction of randomized dynamics.",
                            "No physics stepping, GPU, policy inference, or naturalness/safety assertion."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--start-second", type=float, default=0.)
    parser.add_argument("--end-second", type=float)
    parser.add_argument("--expert-config", type=Path, help="Local source-clip comparison only")
    args = parser.parse_args()
    if args.output.resolve() == args.trajectory.resolve():
        parser.error("--output must differ from --trajectory")
    result = audit(args.trajectory, args.lane, args.start_second, args.end_second, args.expert_config)
    if args.expert_config is not None and args.output.resolve() in {
            args.expert_config.resolve(), *(Path(clip["file"]).resolve()
            for clip in result["expert_source_comparison"]["clips"])}:
        parser.error("--output must differ from expert source files")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

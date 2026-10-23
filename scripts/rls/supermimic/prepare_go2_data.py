"""Generate Go2 gait CSVs with bounded MuJoCo-Jacobian inverse kinematics.

These are procedural kinematic references, not captured animal motion or LAFAN1
retargets. Training learns their physical execution in IsaacLab.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from bootstrap import configure_paths
configure_paths()
from trackerTask.supermimic.go2_gaits import GAITS
from trackerTask.supermimic.go2_robot import LEGS, FOOT_RADIUS
from trackerTask.supermimic.robots import get_robot

REPO = Path(__file__).resolve().parents[3]
FPS = 50


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def foot_targets(gait, phase):
    phases = (phase + np.array([0.0, 0.5, 0.5, 0.0])) % 1
    stance = phases < gait.duty_factor
    stride = gait.speed * gait.period * gait.duty_factor
    x = stride * (0.5 - phases / gait.duty_factor)
    u = np.clip((phases - gait.duty_factor) / (1 - gait.duty_factor), 0, 1)
    smooth = 10*u**3 - 15*u**4 + 6*u**5
    swing_x = -stride/2 - gait.speed*gait.period*(1-gait.duty_factor)*u + gait.speed*gait.period*smooth
    x = np.where(stance, x, swing_x)
    z = FOOT_RADIUS + gait.clearance * 16*u**2*(1-u)**2
    return np.stack((
        x + np.array([0.1934, 0.1934, -0.1934, -0.1934]) + gait.forward_foot_offset,
        np.array([1, -1, 1, -1]) * gait.lateral_foot_position,
        z,
    ), -1), stance


def generate(gait, model, spec):
    data = mujoco.MjData(model)
    jids = [model.joint(n).id for n in spec.joint_names]
    qids, vids = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
    gids = [model.geom(leg).id for leg in LEGS]
    lower, upper = model.jnt_range[jids].T
    cycle = round(gait.period * FPS)
    if not np.isclose(cycle, gait.period * FPS):
        raise ValueError("Gait period must contain an integer number of control steps")
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    lower, upper = mid - 0.9*half, mid + 0.9*half
    data.qpos[:7] = [0, 0, gait.root_height, 1, 0, 0, 0]
    data.qpos[qids] = spec.default_q
    joint_cycle, foot_cycle, stance_cycle, errors = [], [], [], []
    for frame in range(cycle):
        targets, stance = foot_targets(gait, frame / cycle)
        for leg, gid in enumerate(gids):
            qi, vi = qids[3*leg:3*leg+3], vids[3*leg:3*leg+3]

            def residual(q):
                data.qpos[qi] = q
                mujoco.mj_forward(model, data)
                return data.geom_xpos[gid] - targets[leg]

            def jacobian(q):
                residual(q)
                jac = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac, None, data.geom_xpos[gid], model.geom_bodyid[gid])
                return jac[:, vi]

            fit = least_squares(residual, data.qpos[qi].copy(), jac=jacobian,
                                bounds=(lower[3*leg:3*leg+3], upper[3*leg:3*leg+3]),
                                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100)
            errors.append(float(np.linalg.norm(residual(fit.x))))
        joint_cycle.append(data.qpos[qids].copy())
        foot_cycle.append(data.geom_xpos[gids].copy())
        stance_cycle.append(stance)
    count = round(gait.duration * FPS) + 1
    index = np.arange(count) % cycle
    q = np.asarray(joint_cycle)[index]
    root = np.zeros((count, 3))
    root[:, 0], root[:, 2] = np.arange(count) * gait.speed / FPS, gait.root_height
    quat = np.tile([0, 0, 0, 1], (count, 1))
    feet_world = np.asarray(foot_cycle)[index] + root[:, None]
    feet_world[:, :, 2] -= gait.root_height
    stance = np.asarray(stance_cycle)[index]
    consecutive_stance = stance[1:] & stance[:-1]
    slip = np.linalg.norm(np.diff(feet_world, axis=0)[..., :2] * FPS, axis=-1)
    max_slip = float(slip[consecutive_stance].max(initial=0))
    max_error = max(errors)
    velocity = np.abs(np.gradient(q, 1/FPS, axis=0))
    if max_error > 1e-5 or max_slip > 1e-4 or velocity.max() > 30:
        raise ValueError(f"Invalid gait: IK {max_error}, stance slip {max_slip}, joint velocity {velocity.max()}")
    return np.concatenate((root, quat, q), -1), {
        "ik_max_position_error_m": max_error,
        "stance_max_horizontal_speed_m_s": max_slip,
        "joint_max_velocity_rad_s": float(velocity.max()),
        "joint_soft_limit_min_margin_rad": float(np.minimum(q-lower, upper-q).min()),
        "foot_center_min_height_m": float(feet_world[..., 2].min()),
        "foot_center_max_height_m": float(feet_world[..., 2].max()),
    }


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    names = {entry["file"] for entry in manifest["files"]}
    if {p.name for p in directory.glob("*.csv")} != names:
        raise ValueError("CSV files differ from the recorded Go2 gait library")
    for entry in manifest["files"]:
        file = directory / entry["file"]
        if digest(file) != entry["sha256"]:
            raise ValueError(f"Gait data hash mismatch: {file}")
        values = np.loadtxt(file, delimiter=",", ndmin=2)
        if values.shape != (entry["frames"], 19) or not np.isfinite(values).all():
            raise ValueError(f"Malformed gait CSV: {file}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "data/retargeted/go2_gaits")
    parser.add_argument("--motion_config", type=Path, default=REPO / "data/configs/go2_gaits.json")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.motion_config = args.motion_config.expanduser().resolve()
    spec = get_robot("go2")
    if args.verify_only or (args.output / "manifest.json").exists():
        manifest = verify(args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise ValueError("Use an empty data directory to preserve existing files")
        model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
        manifest = {
            "version": 1, "task": "go2", "source": "procedural foot trajectories + bounded numerical IK",
            "retargeted_from_mocap": False, "dynamic_feasibility": "measured by policy rollouts, not assumed by IK",
            "source_fps": FPS, "quaternion_order": "xyzw", "joint_names": spec.joint_names,
            "mjcf_sha256": digest(spec.mjcf_path),
            "generator_sha256": digest(__file__),
            "gait_config_sha256": digest(REPO / "source/trackerTask/trackerTask/supermimic/go2_gaits.py"),
            "mujoco_version": mujoco.__version__, "files": [],
        }
        for gait in GAITS:
            values, metrics = generate(gait, model, spec)
            path = args.output / (gait.name + ".csv")
            np.savetxt(path, values, delimiter=",", fmt="%.10f")
            manifest["files"].append({
                "file": path.name, "frames": len(values), "sha256": digest(path),
                "specification": asdict(gait), "validation": metrics,
            })
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(args.output)
    config = {
        "version": 1, "root": os.path.relpath(args.output, args.motion_config.parent).replace("\\", "/"),
        "source_fps": FPS, "quaternion_order": "xyzw", "joint_names": list(spec.joint_names),
        "files": [entry["file"] for entry in manifest["files"]],
    }
    if args.verify_only:
        if json.loads(args.motion_config.read_text()) != config:
            raise ValueError("Motion configuration differs from verified data")
    else:
        args.motion_config.parent.mkdir(parents=True, exist_ok=True)
        args.motion_config.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"motion_config": str(args.motion_config), **manifest}, indent=2))


if __name__ == "__main__":
    main()

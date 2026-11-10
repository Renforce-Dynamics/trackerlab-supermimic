"""Retarget the selected current Hangar human pose with anatomical frame axes."""

import argparse
import hashlib
import json
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bootstrap import configure_paths
configure_paths()
from retarget_lafan_aiming import morphology_targets
from retarget_gait_references import flat_model, ROLES


NAMES = ("Hips", "Spine2", "LeftUpLeg", "LeftLeg", "LeftFoot",
         "RightUpLeg", "RightLeg", "RightFoot", "LeftArm", "LeftForeArm",
         "LeftHand", "RightArm", "RightForeArm", "RightHand")


def unit(v):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-7:
        raise ValueError("Selected anatomy has a degenerate direction")
    return v/n


def body_frame(left, up):
    up = unit(up)
    left = unit(left-up*np.dot(left, up))
    forward = unit(np.cross(left, up))
    return np.stack((forward, left, up), -1)


def retarget(canonical, output, frame=68):
    canonical, output = Path(canonical).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = canonical.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw)
    if (document.get("schemaVersion") != "canonical-human-motion/1"
            or document.get("frameCount") != 180 or document.get("fps") != 30
            or source_hash != "6095c299915e19c8afd04034eb3c4d3b54d8bf60854648e6a2e53e7019980766"
            or not 0 <= frame < len(document["frames"])):
        raise ValueError("Canonical input differs from the selected current asset")
    f = document["frames"][frame]
    p = np.array([f[name]["position"] for name in NAMES], dtype=float)
    pelvis_frame = body_frame(p[2]-p[5], p[1]-p[0])
    torso_frame = body_frame(p[8]-p[11], np.asarray(f["Neck"]["position"])-p[1])
    heading = np.arctan2(pelvis_frame[1, 0], pelvis_frame[0, 0])
    H = np.array(((np.cos(heading), -np.sin(heading), 0.),
                  (np.sin(heading), np.cos(heading), 0.), (0., 0., 1.)))
    origin = p[0].copy()
    origin[2] = 0.
    p = (p-origin) @ H
    root_R, torso_R = H.T@pelvis_frame, H.T@torso_frame
    model, spec, _, qids, *_ = flat_model("g1")
    arm_limits = {}
    for name in spec.joint_names:
        if any(part in name.split("_") for part in ("shoulder", "elbow", "wrist")):
            jid = model.joint(name).id
            lo, hi = model.jnt_range[jid]
            midpoint, radius = (lo+hi)/2, (hi-lo)*.45
            model.jnt_range[jid] = (midpoint-radius, midpoint+radius)
            arm_limits[name] = model.jnt_range[jid].tolist()
    data = mujoco.MjData(model)
    data.qpos[:3], data.qpos[qids] = (0., 0., .76), spec.default_q
    mujoco.mj_forward(model, data)
    ids = [model.body(name).id for name in ROLES["g1"]]
    rest = data.xpos[ids].copy()
    # morphology_targets expects the raw BVH basis for its first two rotations.
    from retarget_lafan_aiming import BODY_BASIS, robot_hand_axes, make_hand_direction_task
    rotations = np.broadcast_to(np.eye(3), (14, 3, 3)).copy()
    rotations[0], rotations[1] = root_R@BODY_BASIS.T, torso_R@BODY_BASIS.T
    target_p, target_R, scale = morphology_targets(p, rotations, rest)
    config = mink.Configuration(model, data.qpos.copy())
    q = config.q.copy()
    q[:3] = target_p[0]
    mujoco.mju_mat2Quat(q[3:7], root_R.ravel())
    config.update(q)
    tasks = []
    for i, name in enumerate(ROLES["g1"]):
        arm = i in (8, 9, 10, 11, 12, 13)
        task = mink.FrameTask(
            name, "body", position_cost=35. if i in (4, 7) else 20. if arm else 6.,
            orientation_cost=12. if i < 2 else 0., lm_damping=1e-3)
        task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(target_R[i]), target_p[i]))
        tasks.append(task)
    robot_axes, _ = robot_hand_axes(model)
    hand_directions = []
    for i, (side, sign) in enumerate((("Left", 1.), ("Right", -1.))):
        quaternion = np.asarray(f[side+"Hand"]["quaternion"])
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        direction = H.T @ (rotation[:, 0]*sign)
        hand_directions.append(direction.tolist())
        task = make_hand_direction_task(
            ROLES["g1"][10 if i == 0 else 13], robot_axes[i], .7)
        task.set_target(direction)
        tasks.append(task)
    posture = mink.PostureTask(model, cost=.015, lm_damping=1e-3)
    posture.set_target(config.q.copy())
    tasks += [posture, mink.DampingTask(model, cost=.005)]
    for _ in range(120):
        velocity = mink.solve_ik(config, tasks, .02, "daqp", damping=1e-5,
                                 limits=[mink.ConfigurationLimit(model)])
        config.integrate_inplace(velocity, .02)
    q = config.q.copy()
    foot_ids = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    bottom = min(float(config.data.geom_xpos[g, 2]-model.geom_size[g, 0])
                 for g in range(model.ngeom)
                 if model.geom_bodyid[g] in foot_ids
                 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                 and (model.geom_contype[g] or model.geom_conaffinity[g]))
    floor_shift = max(0., .003-bottom)
    q[2] += floor_shift
    values = np.r_[q[:3], q[[4, 5, 6, 3]], q[qids]][None]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, values, delimiter=",", fmt="%.17g")
    metrics = {
        "schema": "selected-canonical-aim-g1-v1",
        "source": {"asset_id": "motion_9fbbcfe3539e42d0",
                   "action_data_id": "actiondata_58b86b8139a54265",
                   "canonical_sha256": source_hash, "source_frame": frame,
                   "source_time_s": frame/30, "owner_screenshot_time_s": 2.27,
                   "source_name": document.get("name"),
                   "source_input_sha256": document.get("sourceInput", {}).get("sourceSha256"),
                   "source_family": document.get("sourceFamily"),
                   "frame_convention": "Anatomical hip/shoulder lines and spine directions; worldZup."},
        "root_scale": scale, "root_R": root_R.tolist(), "torso_R": torso_R.tolist(),
        "floor_shift_m": floor_shift,
        "ik_arm_joint_limits": arm_limits,
        "hand_direction": {"world_targets_heading_frame": hand_directions,
            "convention": "SMPL-X T-pose hand length axes: left+X/right-X; robot fixed-hand PCA axes.",
            "scope": "Length direction only; finger articulation is absent on the G1 model."},
        "target_positions": dict(zip(ROLES["g1"], target_p.tolist())),
        "position_error_m": dict(zip(ROLES["g1"], np.linalg.norm(config.data.xpos[ids]-target_p, axis=-1).tolist())),
        "file": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "joint_names": list(spec.joint_names), "output_frames": 1,
        "model_sha256": hashlib.sha256(Path(spec.mjcf_path).read_bytes()).hexdigest(),
        "scope": "Selected pose only; kinematic target with no physical policy acceptance.",
    }
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=68)
    args = parser.parse_args()
    print(json.dumps(retarget(args.canonical, args.output, args.frame), indent=2))


if __name__ == "__main__":
    main()

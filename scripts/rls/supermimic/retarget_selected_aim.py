"""Retarget the selected human pose with bilateral wrists and full palm frames."""

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
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-7:
        raise ValueError("Selected anatomy has a degenerate direction")
    return v/n


def body_frame(left, up):
    up = unit(up)
    left = unit(left-up*np.dot(left, up))
    forward = unit(np.cross(left, up))
    return np.stack((forward, left, up), -1)


def palm_frame(length, thumbward):
    length = unit(np.asarray(length, dtype=float))
    thumbward = np.asarray(thumbward, dtype=float)
    thumbward = unit(thumbward-length*np.dot(length, thumbward))
    return np.stack((length, thumbward, np.cross(length, thumbward)), -1)


def source_palms(document, body_model):
    """Use the selected body's wrist and metacarpal roots, not finger curl."""
    body_model = Path(body_model)
    model_hash = hashlib.sha256(body_model.read_bytes()).hexdigest()
    if model_hash != document["sourceCorrection"]["bodyModelSha256"]:
        raise ValueError("SMPL-X palm calibration requires the exact source body")
    with np.load(body_model, allow_pickle=False) as model:
        betas = np.asarray(document["surface"]["betas"])
        shaped = model["v_template"] + np.einsum(
            "vci,i->vc", model["shapedirs"][:, :, :len(betas)], betas)
        rest = model["J_regressor"] @ shaped
    frames = [palm_frame(rest[middle]-rest[wrist], rest[index]-rest[pinky])
              for wrist, index, middle, pinky in ((20, 25, 28, 31), (21, 40, 43, 46))]
    return np.array(frames), {"body_model_sha256": model_hash,
        "basis": "Shaped SMPL-X wrist->middle MCP and pinky MCP->index MCP.",
        "local_frames": np.array(frames).tolist(),
        "finger_channels": "Not used to infer a rigid palm orientation."}


def robot_palms(model):
    """Measure flat proximal palms; curled fingers bias whole-hand length PCA."""
    frames, geoms, evidence = [], [], {}
    for side in ("left", "right"):
        mesh = model.mesh(side+"_rubber_hand").id
        wrist = model.body(side+"_wrist_yaw_link").id
        found = np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                               & (model.geom_dataid == mesh)
                               & (model.geom_bodyid == wrist))
        if len(found) != 1:
            raise ValueError(f"Expected one fitted {side} palm")
        geom = int(found[0])
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[geom])
        vertices = vertices @ R.reshape(3, 3).T + model.geom_pos[geom]
        core = vertices[(vertices[:, 0] > .055) & (vertices[:, 0] < .12)
                        & (vertices[:, 2] > -.05) & (vertices[:, 2] < .04)]
        if len(core) < 20:
            raise ValueError("G1 proximal palm calibration region is missing")
        variances, axes = np.linalg.eigh(np.cov(core.T))
        normal = axes[:, 0]
        if variances[1] < 2*variances[0] or abs(normal[1]) < .8:
            raise ValueError("G1 proximal palm plane is ambiguous")
        length = unit(np.array((1., 0., 0.))-normal*normal[0])
        thumb = vertices[vertices[:, 2] > .045].mean(0)-core.mean(0)
        frame = palm_frame(length, thumb-normal*np.dot(thumb, normal))
        frames.append(frame)
        geoms.append(geom)
        evidence[side] = {"local_frame": frame.tolist(),
            "proximal_plane_variances_m2": variances.tolist(),
            "core_vertices": len(core),
            "mesh_vertices_sha256": hashlib.sha256(
                np.asarray(vertices, dtype="<f8").tobytes()).hexdigest(),
            "basis": "Proximal palm plane, wrist-forward projection and thumb mesh lobe."}
    return np.array(frames), geoms, evidence


def coupled_arm_targets(source_p, target_p, rest_p, torso_R):
    """Transfer one shoulder-relative hand midpoint and a shared wrist delta."""
    source_p, target_p = np.asarray(source_p), np.array(target_p, copy=True)
    chains = ((8, 9, 10), (11, 12, 13))
    scale = float(np.mean([
        sum(np.linalg.norm(rest_p[b]-rest_p[a]) for a, b in ((s, e), (e, w)))
        / sum(np.linalg.norm(source_p[b]-source_p[a]) for a, b in ((s, e), (e, w)))
        for s, e, w in chains]))
    shoulder = source_p[[8, 11]].mean(0)
    robot_shoulder = target_p[[8, 11]].mean(0)
    midpoint = robot_shoulder + (source_p[[10, 13]].mean(0)-shoulder)*scale
    delta = (source_p[10]-source_p[13])*scale
    target_p[10], target_p[13] = midpoint+delta/2, midpoint-delta/2
    for shoulder_i, elbow, _ in chains:
        target_p[elbow] = (target_p[shoulder_i]
                          + (source_p[elbow]-source_p[shoulder_i])*scale)
    return target_p, {"arm_scale": scale,
        "midpoint_torso_m": (torso_R.T @ (midpoint-target_p[1])).tolist(),
        "relative_wrist_torso_m": (torso_R.T @ delta).tolist(),
        "wrist_distance_m": float(np.linalg.norm(delta))}


def palm_clearance_limit(model, hand_geoms):
    # Mink filters visual-only meshes at construction; include palms for this IK
    # constraint, then restore the simulation model's original collision masks.
    original_masks = (model.geom_contype[hand_geoms].copy(),
                      model.geom_conaffinity[hand_geoms].copy())
    try:
        model.geom_contype[hand_geoms] = 1
        model.geom_conaffinity[hand_geoms] = 1
        limit = mink.CollisionAvoidanceLimit(
            model, [([hand_geoms[0]], [hand_geoms[1]])],
            minimum_distance_from_collisions=.004, collision_detection_distance=.06)
    finally:
        model.geom_contype[hand_geoms], model.geom_conaffinity[hand_geoms] = original_masks
    if len(limit.geom_id_pairs) != 1:
        raise ValueError("Palm geometry must participate in IK clearance")
    return limit


def retarget(canonical, output, body_model, frame=68):
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
    from retarget_lafan_aiming import BODY_BASIS
    rotations = np.broadcast_to(np.eye(3), (14, 3, 3)).copy()
    rotations[0], rotations[1] = root_R@BODY_BASIS.T, torso_R@BODY_BASIS.T
    target_p, target_R, scale = morphology_targets(p, rotations, rest)
    target_p, bilateral = coupled_arm_targets(p, target_p, rest, torso_R)
    human_palms, source_evidence = source_palms(document, body_model)
    fitted_palms, hand_geoms, robot_evidence = robot_palms(model)
    for i, (side, role) in enumerate((("Left", 10), ("Right", 13))):
        quaternion = np.asarray(f[side+"Hand"]["quaternion"])
        source_R = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        target_R[role] = H.T @ source_R @ human_palms[i] @ fitted_palms[i].T
    config = mink.Configuration(model, data.qpos.copy())
    q = config.q.copy()
    q[:3] = target_p[0]
    mujoco.mju_mat2Quat(q[3:7], root_R.ravel())
    config.update(q)
    tasks = []
    for i, name in enumerate(ROLES["g1"]):
        arm = i in (8, 9, 10, 11, 12, 13)
        task = mink.FrameTask(
            name, "body", position_cost=(35. if i in (4, 7) else
                35. if i in (10, 13) else 2. if arm else 6.),
            orientation_cost=12. if i < 2 else 3. if i in (10, 13) else 0.,
            lm_damping=1e-3)
        task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(target_R[i]), target_p[i]))
        tasks.append(task)
    posture = mink.PostureTask(model, cost=.015, lm_damping=1e-3)
    posture.set_target(config.q.copy())
    tasks += [posture, mink.DampingTask(model, cost=.005)]
    collision_limit = palm_clearance_limit(model, hand_geoms)
    for _ in range(180):
        velocity = mink.solve_ik(config, tasks, .02, "daqp", damping=1e-5,
                                 limits=[mink.ConfigurationLimit(model), collision_limit])
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
    target_p[:, 2] += floor_shift
    config.update(q)
    actual_R = config.data.xmat[ids].reshape(-1, 3, 3)
    wrist_errors = np.linalg.norm(config.data.xpos[np.array(ids)[[10, 13]]]-target_p[[10, 13]], axis=-1)
    rotation_errors = Rotation.from_matrix(
        target_R[[10, 13]].transpose(0, 2, 1) @ actual_R[[10, 13]]).magnitude()
    palm_clearance = float(mujoco.mj_geomDistance(
        model, config.data, *hand_geoms, 1., np.empty(6)))
    actual_delta = config.data.xpos[ids[10]]-config.data.xpos[ids[13]]
    bilateral.update(actual_relative_wrist_torso_m=(actual_R[1].T @ actual_delta).tolist(),
        actual_wrist_distance_m=float(np.linalg.norm(actual_delta)),
        wrist_position_error_m=wrist_errors.tolist(),
        wrist_rotation_error_rad=rotation_errors.tolist(),
        visual_palm_convex_clearance_m=palm_clearance,
        clearance_scope="Explicit convex mesh distance, including visual-only palms; no dynamics acceptance.")
    geometry_accepted = bool(max(wrist_errors) <= .02
                             and max(rotation_errors) <= .15 and palm_clearance >= .0035)
    values = np.r_[q[:3], q[[4, 5, 6, 3]], q[qids]][None]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, values, delimiter=",", fmt="%.17g")
    metrics = {
        "schema": "selected-canonical-aim-g1-v2",
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
        "palm_frames": {"source": source_evidence, "robot": robot_evidence,
            "target_wrist_rotations": target_R[[10, 13]].tolist(),
            "scope": "Full rigid-palm orientation; G1 finger articulation is absent."},
        "bilateral_hands": bilateral,
        "geometry_acceptance": {"passed": geometry_accepted,
            "max_wrist_position_error_m": .02, "max_wrist_rotation_error_rad": .15,
            "minimum_visual_palm_clearance_m": .0035},
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
    parser.add_argument("--body-model", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=68)
    args = parser.parse_args()
    report = retarget(args.canonical, args.output, args.body_model, args.frame)
    print(json.dumps(report, indent=2))
    if not report["geometry_acceptance"]["passed"]:
        raise SystemExit("Candidate saved for review; palm geometry acceptance failed.")


if __name__ == "__main__":
    main()

"""Offline G1 posture retargeting and independent reference geometry checks."""

from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from audit_stealth_trajectory import sole_metrics


FPS = 50
STAGGERED_IDLE = {
    "schema": "g1-selected-staggered-idle-v1",
    "fore_aft_separation_m": .28,
    "lateral_separation_m": .18,
    "sole_clearance_m": .003,
    "minimum_com_support_margin_m": .01,
    "stationary": True,
}
CRITERIA = {
    "max_foot_position_error_m": .006,
    "max_foot_rotation_error_rad": .005,
    "max_sole_vertical_error_m": .003,
    "max_support_slip_increase_m_s": .005,
    "max_torso_rotation_error_rad": .01,
    "max_root_height_error_m": .005,
    "max_root_xy_error_m": .05,
    "max_joint_limit_violation_rad": 1e-6,
    "max_joint_velocity_ratio": 1.0001,
    "max_arm_coordinate_error_rad": 1e-8,
    "minimum_sole_height_m": -.0005,
    "max_support_gap_increase_m": .003,
    "max_self_penetration_m": .001,
}


def validate_motion(values):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 36 or len(values) < 11
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Expected at least eleven finite G1 XYZ/XYZW/q29 frames at 50 Hz")
    return values


def heading(rotation):
    return Rotation.from_euler("z", np.arctan2(rotation[1, 0], rotation[0, 0])).as_matrix()


def csv_to_qpos(values, qids):
    values = np.asarray(values, dtype=np.float64)
    poses = np.empty_like(values)
    poses[:, :3], poses[:, 3:7] = values[:, :3], values[:, [6, 3, 4, 5]]
    poses[:, qids] = values[:, 7:]
    return poses


def layout(model, spec):
    arms = [i for i, name in enumerate(spec.joint_names)
            if any(part in name.split("_") for part in ("shoulder", "elbow", "wrist"))]
    if len(arms) != 14 or spec.num_joints != 29 or spec.name != "g1":
        raise ValueError("Full-body reference correction requires the named G1 29-joint contract")
    joints = np.array([model.joint(name).id for name in spec.joint_names])
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    return arms, joints, qids, vids, feet, model.body("torso_link").id


def selected_geometry(model, spec, reference):
    reference = np.asarray(reference, dtype=np.float64)
    if (reference.shape != (36,) or not np.isfinite(reference).all()
            or not math.isclose(float(np.linalg.norm(reference[3:7])), 1., abs_tol=1e-5)):
        raise ValueError("The selected pose must be finite XYZ/XYZW/q29")
    arms, joints, qids, vids, feet, torso = layout(model, spec)
    limits = model.jnt_range[joints]
    if np.any(reference[7:] < limits[:, 0] - 1e-7) or np.any(reference[7:] > limits[:, 1] + 1e-7):
        raise ValueError("The selected pose exceeds robot joint limits")
    data = mujoco.MjData(model)
    data.qpos[:] = csv_to_qpos(reference[None], qids)[0]
    mujoco.mj_forward(model, data)
    root_R = Rotation.from_quat(reference[3:7]).as_matrix()
    H = heading(root_R)
    return H.T @ root_R, H.T @ data.xmat[torso].reshape(3, 3)


def _fixed_targets(targets):
    if targets is None:
        return None
    if not isinstance(targets, (tuple, list)) or len(targets) != 2:
        raise ValueError("Fixed foot targets require positions and rotations")
    positions, rotations = (np.asarray(value, dtype=np.float64) for value in targets)
    if (positions.shape != (2, 3) or rotations.shape != (2, 3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-8)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-8)):
        raise ValueError("Invalid finite foot positions or proper rotation matrices")
    return positions, rotations


def solve_reference(values, reference, model, spec, motors, *, foot_targets=None):
    """Use Mink/DAQP to fit feet, torso and world height with bounded actuators.

    Arm coordinates are fixed only during offline reference construction.
    Root, waist and leg coordinates remain IK variables. No runtime action
    controller is created by this function.
    """
    import mink

    class RootHeightTask(mink.Task):
        # A tilted FrameTask's local translation error does not isolate world Z.
        def __init__(self, body_id):
            super().__init__(cost=np.array([1000.]), gain=1., lm_damping=1e-4)
            self.body_id, self.target = body_id, 0.

        def compute_error(self, configuration):
            return np.array([configuration.data.xpos[self.body_id, 2] - self.target])

        def compute_jacobian(self, configuration):
            jacobian = np.zeros((3, configuration.nv))
            mujoco.mj_jacBody(
                configuration.model, configuration.data, jacobian, None, self.body_id)
            return jacobian[2:3]

    values = validate_motion(values)
    foot_targets = _fixed_targets(foot_targets)
    reference = np.asarray(reference, dtype=np.float64)
    arms, _, qids, vids, feet, torso = layout(model, spec)
    upper = [i for i, name in enumerate(spec.joint_names)
             if name.startswith("waist_") or i in arms]
    if len(upper) != 17:
        raise ValueError("Expected the three waist and fourteen arm joints")
    root_local, torso_local = selected_geometry(model, spec, reference)
    frozen = mink.DofFreezingTask(model, vids[arms].tolist())
    feet_tasks = [mink.FrameTask(model.body(body).name, "body", position_cost=1000.,
                                 orientation_cost=1000., lm_damping=1e-4) for body in feet]
    root_task = mink.FrameTask(spec.root_body, "body", position_cost=5.,
                               orientation_cost=5., lm_damping=1e-4)
    torso_task = mink.FrameTask("torso_link", "body", position_cost=0.,
                                orientation_cost=1000., lm_damping=1e-4)
    posture = mink.PostureTask(model, cost=.01, lm_damping=1e-4)
    height_task = RootHeightTask(model.body(spec.root_body).id)
    tasks = [*feet_tasks, root_task, torso_task, posture, height_task]
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model, {name: motors[i][1] for i, name in enumerate(spec.joint_names)
                                  if i not in arms}),
        mink.FreeJointVelocityLimit(
            model, max_linear_velocity=np.full(3, .8), max_angular_velocity=np.full(3, 3.)),
    ]
    data = mujoco.MjData(model)
    source_poses = csv_to_qpos(values, qids)
    height_offset = float(reference[2] - values[:, 2].mean())
    result, previous, configuration = [], None, None

    def target(position, rotation):
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        return mink.SE3.from_rotation_and_translation(mink.SO3(quaternion), position)

    for frame, source in enumerate(source_poses):
        data.qpos[:] = source
        mujoco.mj_forward(model, data)
        for index, (task, body) in enumerate(zip(feet_tasks, feet)):
            position, rotation = (
                (data.xpos[body], data.xmat[body].reshape(3, 3)) if foot_targets is None else
                (foot_targets[0][index], foot_targets[1][index]))
            task.set_target(target(position, rotation))
        H = heading(Rotation.from_quat(values[frame, 3:7]).as_matrix())
        q = source.copy()
        q[2] += height_offset
        q[3:7] = Rotation.from_matrix(H @ root_local).as_quat()[[3, 0, 1, 2]]
        q[qids[upper]] = reference[7 + np.array(upper)]
        root_task.set_target(target(q[:3], H @ root_local))
        torso_task.set_target(target(data.xpos[torso], H @ torso_local))
        posture.set_target(q)
        height_task.target = float(q[2])
        if previous is not None:
            q = previous.copy()
        if configuration is None:
            configuration = mink.Configuration(model, q)
        else:
            configuration.update(q)
        count, dt = (100, .02) if frame == 0 else (20, .02 / 20)
        for _ in range(count):
            velocity = mink.solve_ik(
                configuration, tasks, dt, "daqp", damping=1e-7,
                limits=limits, constraints=[frozen])
            configuration.integrate_inplace(velocity, dt)
        previous = configuration.q.copy()
        result.append(np.r_[previous[:3], previous[[4, 5, 6, 3]], previous[qids]])
    return np.asarray(result)


def _foot_geometry(model, feet):
    centers, radii = [], []
    for body in feet:
        ids = np.flatnonzero((model.geom_bodyid == body)
                            & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        if len(ids) != 4 or np.any(model.geom_type[ids] != mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError("G1 sole audit requires the four physical spheres on each foot")
        centers.append(model.geom_pos[ids].copy())
        radii.append(model.geom_size[ids, 0].copy())
    return np.asarray(centers), np.asarray(radii)


def assess_reference(values, source, reference, model, spec, motors, *, foot_targets=None):
    """Recompute geometry from actual candidate/source bytes, without solving."""
    values, source = validate_motion(values), validate_motion(source)
    foot_targets = _fixed_targets(foot_targets)
    if values.shape != source.shape:
        raise ValueError("Reference correction must preserve source frame count and timing")
    reference = np.asarray(reference, dtype=np.float64)
    arms, joints, qids, _, feet, torso = layout(model, spec)
    _, torso_local = selected_geometry(model, spec, reference)
    centers, radii = _foot_geometry(model, feet)
    data = mujoco.MjData(model)
    positions, rotations, torso_errors, penetrations = [], [], [], []
    for case, rows in enumerate((source, values)):
        p, R = [], []
        for frame, pose in enumerate(csv_to_qpos(rows, qids)):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            p.append(data.xpos[feet].copy())
            R.append(data.xmat[feet].reshape(2, 3, 3).copy())
            if case == 1:
                source_H = heading(Rotation.from_quat(source[frame, 3:7]).as_matrix())
                desired = source_H @ torso_local
                torso_errors.append(Rotation.from_matrix(
                    data.xmat[torso].reshape(3, 3) @ desired.T).magnitude())
                penetrations.extend(
                    max(0., -float(contact.dist))
                    for contact in data.contact[:data.ncon]
                    if model.geom_bodyid[contact.geom1] != 0 and model.geom_bodyid[contact.geom2] != 0
                )
        positions.append(np.asarray(p))
        rotations.append(np.asarray(R))
    source_p, candidate_p = positions
    source_R, candidate_R = rotations
    if foot_targets is not None:
        source_p = np.broadcast_to(foot_targets[0], source_p.shape).copy()
        source_R = np.broadcast_to(foot_targets[1], source_R.shape).copy()
        positions[0], rotations[0] = source_p, source_R
    poles = []
    for p, R in zip(positions, rotations):
        points = p[:, :, None, :] + np.einsum("tfij,fsj->tfsi", R, centers)
        points[..., 2] -= radii
        poles.append(points)
    old_sole, new_sole = poles
    old_metrics = sole_metrics(source_p, source_R, centers, radii, 1 / FPS)
    new_metrics = sole_metrics(candidate_p, candidate_R, centers, radii, 1 / FPS)
    if any(metrics["feet"][side]["stance_pole_material_speed_m_s"] is None
           for metrics in (old_metrics, new_metrics) for side in ("left", "right")):
        raise ValueError("Each stealth foot needs supported samples for slip comparison")
    slip_increases = [
        new_metrics["feet"][side]["stance_pole_material_speed_m_s"]["mean"]
        - old_metrics["feet"][side]["stance_pole_material_speed_m_s"]["mean"]
        for side in ("left", "right")
    ]
    limits = model.jnt_range[joints]
    joint_violation = np.maximum(values[:, 7:] - limits[:, 1], limits[:, 0] - values[:, 7:])
    velocity_ratio = np.abs(np.diff(values[:, 7:], axis=0) * FPS) / np.array([m[1] for m in motors])
    desired_height = reference[2] + source[:, 2] - source[:, 2].mean()
    foot_position_error = float(np.linalg.norm(candidate_p - source_p, axis=-1).max())
    report = {
        "frames": len(values),
        "target_clearance_m": float(reference[2]),
        "root_height_mean_m": float(values[:, 2].mean()),
        "root_height_range_m": [float(values[:, 2].min()), float(values[:, 2].max())],
        "max_foot_position_error_m": foot_position_error,
        "strict_rigid_foot_1mm_passed": foot_position_error <= .001,
        "max_foot_rotation_error_rad": float(Rotation.from_matrix(
            (candidate_R @ source_R.transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).magnitude().max()),
        "max_sole_vertical_error_m": float(np.abs(new_sole[..., 2] - old_sole[..., 2]).max()),
        "max_support_slip_increase_m_s": max(0., *slip_increases),
        "max_torso_rotation_error_rad": float(max(torso_errors)),
        "max_root_height_error_m": float(np.abs(values[:, 2] - desired_height).max()),
        "max_root_xy_error_m": float(np.linalg.norm(values[:, :2] - source[:, :2], axis=-1).max()),
        "max_joint_limit_violation_rad": max(0., float(joint_violation.max())),
        "max_joint_velocity_ratio": float(velocity_ratio.max()),
        "max_arm_coordinate_error_rad": float(np.abs(values[:, 7 + np.array(arms)]
                                                       - reference[7 + np.array(arms)]).max()),
        "minimum_sole_height_m": float(new_sole[..., 2].min()),
        "max_support_gap_increase_m": max(0., float(
            new_sole[..., 2].min((1, 2)).max() - old_sole[..., 2].min((1, 2)).max())),
        "max_self_penetration_m": max([0., *penetrations]),
        "source_support_slip_m_s": [
            old_metrics["feet"][side]["stance_pole_material_speed_m_s"]["mean"]
            for side in ("left", "right")],
        "candidate_support_slip_m_s": [
            new_metrics["feet"][side]["stance_pole_material_speed_m_s"]["mean"]
            for side in ("left", "right")],
        "source_lifts": [old_metrics["feet"][side]["sustained_lift_count"] for side in ("left", "right")],
        "candidate_lifts": [new_metrics["feet"][side]["sustained_lift_count"] for side in ("left", "right")],
        "definitions": {
            "support": "Sphere-pole height <=2cm; a geometric proxy, not contact force.",
            "foot_fidelity": "At most6mm origin error and3mm vertical pole error; root-height fit is explicit.",
            "role": "Kinematic data acceptance only; policy dynamics and naturalness remain unproved.",
        },
    }
    report["passed"], report["failures"] = accept_report(report)
    return report


def staggered_foot_targets(reference, model, spec):
    """Keep the selected front-foot order in a 28cm by18cm planted stance."""
    _, _, qids, _, feet, _ = layout(model, spec)
    selected_geometry(model, spec, reference)
    centers, radii = _foot_geometry(model, feet)
    data = mujoco.MjData(model)
    data.qpos[:] = csv_to_qpos(np.asarray(reference)[None], qids)[0]
    mujoco.mj_forward(model, data)
    positions = data.xpos[feet].copy()
    H = heading(Rotation.from_quat(reference[3:7]).as_matrix())
    front_sign = 1. if (H.T @ (positions[0] - positions[1]))[0] > 0 else -1.
    for index, side in enumerate((1., -1.)):
        positions[index, :2] = np.asarray(reference[:2]) + H[:2, :2] @ np.array((
            side * front_sign * STAGGERED_IDLE["fore_aft_separation_m"] / 2,
            side * STAGGERED_IDLE["lateral_separation_m"] / 2))
    rotations = np.array([heading(data.xmat[body].reshape(3, 3)) for body in feet])
    for index in range(2):
        offsets = (rotations[index] @ centers[index].T).T[:, 2] - radii[index]
        positions[index, 2] = STAGGERED_IDLE["sole_clearance_m"] - offsets.min()
    return positions, rotations


def solve_staggered_idle(reference, frames, model, spec, motors):
    if type(frames) is not int or frames < 11:
        raise ValueError("Stationary AMP history requires at least eleven frames")
    targets = staggered_foot_targets(reference, model, spec)
    source = np.repeat(np.asarray(reference)[None], frames, axis=0)
    settled = solve_reference(
        source[:11], reference, model, spec, motors, foot_targets=targets)[-1]
    return np.repeat(settled[None], frames, axis=0), source


def assess_staggered_idle(values, source, reference, model, spec, motors):
    targets = staggered_foot_targets(reference, model, spec)
    values, source = validate_motion(values), validate_motion(source)
    if not np.array_equal(source, np.repeat(np.asarray(reference)[None], len(source), axis=0)):
        raise ValueError("Idle donor must be the exact selected-pose hold")
    result = assess_reference(
        values, source, reference, model, spec, motors, foot_targets=targets)
    _, _, qids, _, feet, _ = layout(model, spec)
    centers, radii = _foot_geometry(model, feet)
    data = mujoco.MjData(model)
    data.qpos[:] = csv_to_qpos(values[:1], qids)[0]
    mujoco.mj_forward(model, data)
    R = data.xmat[feet].reshape(2, 3, 3)
    poles = data.xpos[feet, None, :] + np.einsum("fij,fsj->fsi", R, centers)
    poles[..., 2] -= radii
    hull = ConvexHull(poles[..., :2].reshape(-1, 2))
    com = data.subtree_com[model.body(spec.root_body).id]
    distances = (hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]) / np.linalg.norm(
        hull.equations[:, :2], axis=1)
    H = heading(Rotation.from_quat(reference[3:7]).as_matrix())
    foot_delta = H.T @ (data.xpos[feet[0]] - data.xpos[feet[1]])
    idle = {
        "pose_change_max": float(np.abs(values - values[0]).max()),
        "fore_aft_separation_m": float(abs(foot_delta[0])),
        "lateral_separation_m": float(abs(foot_delta[1])),
        "front_foot": "left" if foot_delta[0] > 0 else "right",
        "all_sole_poles_height_range_m": [float(poles[..., 2].min()), float(poles[..., 2].max())],
        "com_support_margin_m": float(-distances.max()),
        "scope": "Static nominal COM inside the sole-pole hull; not randomized dynamic balance.",
    }
    failures = []
    if idle["pose_change_max"] > 1e-10:
        failures.append("Idle reference must retain a single stationary posture")
    if idle["fore_aft_separation_m"] < .15 or idle["lateral_separation_m"] < .08:
        failures.append("Idle feet must retain selected fore/aft and lateral separation")
    if idle["all_sole_poles_height_range_m"][1] > .006:
        failures.append("Both idle soles must be planted within6mm of the floor")
    if idle["com_support_margin_m"] < STAGGERED_IDLE["minimum_com_support_margin_m"]:
        failures.append("Idle COM needs at least1cm nominal sole-hull margin")
    result["idle_stance"] = idle
    result["failures"].extend(failures)
    result["passed"] = result["passed"] and not failures
    result["definitions"]["source_foot_targets"] = (
        "Selected front-foot order/yaw;28cm fore/aft,18cm lateral;flat soles at3mm.")
    return result


def accept_report(report):
    failures = []
    for name, limit in CRITERIA.items():
        value = report.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            failures.append(f"{name}: missing or nonfinite")
        elif (value < limit if name == "minimum_sole_height_m" else value > limit):
            failures.append(f"{name}: {value} exceeds acceptance bound {limit}")
        elif name != "minimum_sole_height_m" and value < 0:
            failures.append(f"{name}: negative error magnitude")
    return not failures, failures

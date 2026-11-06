"""Retarget original LAFAN1 aiming/obstacles BVHs to 50 Hz G1 CSVs.

Requires bvh==0.3 plus requirements-gait.txt (Mink 1.4.0 / DAQP).
Input data are read-only. Derivatives inside this checkout must be Git-ignored.
Only body motion is retargeted; no carried object is instantiated.
Optional hand-forward tracking uses skeleton/model geometry. Original LAFAN1
has zero hand End Sites and no thumb landmarks, so palm roll is unobserved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

# The Windows CLI combines conda NumPy and torch through flat_model. Sequential
# MKL avoids loading a second Intel OpenMP runtime; never allow duplicate OMP.
if __name__ == "__main__" and sys.platform == "win32":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SOURCE_FPS = 30.0
OUTPUT_FPS = 50.0
MAX_SOURCE_FRAMES = 500
OFFICIAL_REVISION = "94084601bacdf9cc3764b5c73daaeccae6035fac"
ARCHIVE_SHA256 = "ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977"
REPO = Path(__file__).resolve().parents[3]
WORLD_BASIS = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
# LAFAN's Hips/Spine bone axes: forward=Y, left=Z, up=X.
BODY_BASIS = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
# Toe frames use X forward, Y up and Z right.
FOOT_BASIS = np.array(((1., 0., 0.), (0., 0., 1.), (0., -1., 0.)))
SOURCE_ROLES = (
    "Hips", "Spine2",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
)
CHAINS = ((2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class BvhMotion:
    names: tuple[str, ...]
    parents: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    quaternions: np.ndarray
    channels: tuple[tuple[str, ...], ...]
    frame_time: float
    start_frame: int
    total_frames: int
    end_sites: tuple[tuple[tuple[float, ...], ...], ...] = ()


def parse_bvh(text, start_frame=0, max_frames=300):
    """Use bvh-python's hierarchy parser and intrinsic, declared Euler order.

    Ubisoft's root position channels replace its OFFSET, which already contains
    the first recorded translation. Adding it would translate the root twice.
    """
    from bvh import Bvh

    if (isinstance(start_frame, bool) or not isinstance(start_frame, int)
            or start_frame < 0):
        raise ValueError("start_frame must be a nonnegative integer")
    if (isinstance(max_frames, bool) or not isinstance(max_frames, int)
            or not 1 <= max_frames <= MAX_SOURCE_FRAMES):
        raise ValueError("max_frames must lie in 1..500")
    parsed = Bvh(text.rstrip() + "\n")
    names = tuple(parsed.get_joints_names())
    if not names or len(set(names)) != len(names):
        raise ValueError("BVH joint names must be unique")
    parents = np.array([parsed.joint_parent_index(name) for name in names])
    if parents[0] != -1 or any(not 0 <= parents[i] < i for i in range(1, len(names))):
        raise ValueError("BVH must contain one topologically ordered root")
    frame_time = parsed.frame_time
    if not math.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("BVH frame time must be positive")
    if not np.isclose(1 / frame_time, SOURCE_FPS, rtol=1e-3):
        raise ValueError("Original LAFAN1 BVHs must be approximately 30 fps")
    if len(parsed.frames) != parsed.nframes or start_frame >= parsed.nframes:
        raise ValueError("BVH frame count or start_frame is invalid")
    stop = min(parsed.nframes, start_frame + max_frames)
    channels = tuple(tuple(parsed.joint_channels(name)) for name in names)
    count = sum(map(len, channels))
    rows = np.array(parsed.frames[start_frame:stop], dtype=np.float64)
    if rows.shape != (stop - start_frame, count) or not np.isfinite(rows).all():
        raise ValueError("BVH channel rows must be finite and match the hierarchy")
    offsets = np.array([parsed.joint_offset(name) for name in names])
    if not np.isfinite(offsets).all():
        raise ValueError("BVH offsets must be finite")
    end_sites = tuple(tuple(tuple(map(float, node["OFFSET"]))
                            for node in parsed.get_joint(name).filter("End"))
                      for name in names)
    if any(len(offset) != 3 or not np.isfinite(offset).all()
           for sites in end_sites for offset in sites):
        raise ValueError("BVH End Site offsets must be finite three-vectors")
    positions = np.broadcast_to(offsets, (len(rows), *offsets.shape)).copy()
    rotations = np.broadcast_to(np.eye(3), (len(rows), len(names), 3, 3)).copy()
    quaternions = np.zeros((len(rows), len(names), 4))
    quaternions[..., 3] = 1
    cursor = 0
    for joint, declared in enumerate(channels):
        if len(set(declared)) != len(declared):
            raise ValueError("Duplicate BVH channels")
        rotation_columns, axes = [], []
        for column, channel in enumerate(declared):
            if channel in ("Xposition", "Yposition", "Zposition"):
                positions[:, joint, "XYZ".index(channel[0])] = rows[:, cursor + column]
            elif channel in ("Xrotation", "Yrotation", "Zrotation"):
                axes.append(channel[0])
                rotation_columns.append(cursor + column)
            else:
                raise ValueError(f"Unsupported BVH channel {channel}")
        if len(axes) not in (0, 3) or len(set(axes)) != len(axes):
            raise ValueError("BVH rotations must contain all three distinct axes")
        if axes:
            rotation = Rotation.from_euler("".join(axes), rows[:, rotation_columns], degrees=True)
            rotations[:, joint] = rotation.as_matrix()
            quaternions[:, joint] = rotation.as_quat()
        cursor += len(declared)
    return BvhMotion(names, parents, offsets, positions, rotations, quaternions, channels,
                     frame_time, start_frame, parsed.nframes, end_sites)


def resample_local(motion):
    """Interpolate local translations and SLERP local rotations before IK/FK."""
    source_time = np.arange(len(motion.positions)) / SOURCE_FPS
    count = math.floor(source_time[-1] * OUTPUT_FPS + 1e-9) + 1
    target_time = np.arange(count) / OUTPUT_FPS
    if len(source_time) == 1:
        return motion.positions.copy(), motion.rotations.copy(), target_time
    positions = np.empty((count, len(motion.names), 3))
    rotations = np.empty((count, len(motion.names), 3, 3))
    for joint in range(len(motion.names)):
        for axis in range(3):
            positions[:, joint, axis] = np.interp(
                target_time, source_time, motion.positions[:, joint, axis])
        rotations[:, joint] = Slerp(
            source_time, Rotation.from_quat(motion.quaternions[:, joint]),
        )(target_time).as_matrix()
    return positions, rotations, target_time


def forward_kinematics(positions, rotations, parents):
    world_p, world_R = positions.copy(), rotations.copy()
    for joint, parent in enumerate(parents):
        if parent >= 0:
            world_p[:, joint] = world_p[:, parent] + np.einsum(
                "nij,nj->ni", world_R[:, parent], positions[:, joint])
            world_R[:, joint] = np.einsum(
                "nij,njk->nik", world_R[:, parent], rotations[:, joint])
    return world_p, world_R


def human_to_robot(positions_cm, rotations):
    return (np.einsum("ij,...j->...i", WORLD_BASIS, positions_cm) * .01,
            np.einsum("ij,...jk->...ik", WORLD_BASIS, rotations))


def vector_norm(value, axis=-1):
    return np.sqrt(np.square(value).sum(axis=axis))


def role_indices(names):
    missing = set((*SOURCE_ROLES, "LeftToe", "RightToe")).difference(names)
    if missing:
        raise ValueError(f"Missing original LAFAN1 joints: {sorted(missing)}")
    return np.array([names.index(name) for name in SOURCE_ROLES])


def _unit_direction(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or vector_norm(value) < 1e-8:
        raise ValueError("Hand direction requires a finite nonzero three-vector")
    return value / vector_norm(value)


def source_hand_axes(motion):
    """Hand-local forward axes from the BVH zero-channel rest hierarchy.

    A nonzero terminal End Site specifies wrist-to-hand forward. Ubisoft's
    original 22-joint files instead have zero End Sites. For these, extend the
    incoming ForeArm->Hand OFFSET in the zero-channel rest pose: parent and
    child rotation frames coincide in that pose. This is an explicit straight
    hand rest convention, not a measured fingertip or palm-normal direction.
    Applying the animated Hand rotation retains wrist flexion/deviation.

    Both sides use their own distal offsets without a sign flip. No transverse
    axis can be recovered from a single bone; never invent a thumb/palm axis.
    """
    axes, evidence = [], {}
    for side in ("Left", "Right"):
        name, parent = side + "Hand", side + "ForeArm"
        if name not in motion.names or parent not in motion.names:
            raise ValueError(f"Hand calibration requires {parent} and {name}")
        joint = motion.names.index(name)
        if motion.parents[joint] != motion.names.index(parent):
            raise ValueError(f"{name} must be a direct child of {parent}")
        if np.any(motion.parents == joint):
            raise ValueError(f"{name} must be terminal for hand-forward calibration")
        if len(motion.end_sites) != len(motion.names) or len(motion.end_sites[joint]) != 1:
            raise ValueError(f"{name} requires one explicit BVH End Site")
        end = np.asarray(motion.end_sites[joint][0], dtype=float)
        if end.shape != (3,) or not np.isfinite(end).all():
            raise ValueError(f"{name} has an invalid End Site")
        has_end_direction = vector_norm(end) > 1e-8
        axis = _unit_direction(end if has_end_direction else motion.offsets[joint])
        axes.append(axis)
        evidence[name] = {
            "local_forward": axis.tolist(), "end_site_offset_cm": end.tolist(),
            "basis": ("terminal End Site" if has_end_direction else
                      "zero-channel rest extension of ForeArm->Hand OFFSET"),
            "palm_roll_observed": False,
        }
    return np.array(axes), evidence


def robot_hand_axes(model):
    """Calibrate fixed rubber-palm forward from its mesh in the wrist frame.

    The longest mesh principal axis is a fixed-hand length proxy. The wrist_pitch ->
    wrist_yaw direction resolves PCA's sign in the robot's straight-link rest
    posture. This measures the actual fitted hand mesh, including its mounting
    angle, rather than treating the wrist hinge axis as the palm direction.
    """
    import mujoco

    rest = mujoco.MjData(model)
    mujoco.mj_kinematics(model, rest)
    axes, evidence = [], {}
    for side in ("left", "right"):
        wrist = model.body(side + "_wrist_yaw_link").id
        pitch = model.body(side + "_wrist_pitch_link").id
        if model.body_parentid[wrist] != pitch:
            raise ValueError("Hand calibration requires the G1 pitch->yaw wrist chain")
        mesh_id = model.mesh(side + "_rubber_hand").id
        geoms = np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                               & (model.geom_dataid == mesh_id)
                               & (model.geom_bodyid == wrist))
        if len(geoms) != 1:
            raise ValueError(f"Expected one fitted {side} rubber-hand mesh")
        geom = int(geoms[0])
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh_id]:model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]
        ]
        if len(vertices) < 3:
            raise ValueError("Rubber-hand mesh must contain at least three vertices")
        geom_R = np.empty(9)
        mujoco.mju_quat2Mat(geom_R, model.geom_quat[geom])
        body_vertices = vertices @ geom_R.reshape(3, 3).T + model.geom_pos[geom]
        covariance = np.cov(body_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, -1]
        rotation = rest.xmat[wrist].reshape(3, 3)
        wrist_forward = _unit_direction(
            rotation.T @ (rest.xpos[wrist] - rest.xpos[pitch]))
        alignment = float(principal @ wrist_forward)
        if abs(alignment) < .5 or eigenvalues[-1] <= eigenvalues[-2]:
            raise ValueError(f"{side} rubber-hand principal axis is ambiguous")
        if alignment < 0:
            principal = -principal
            alignment = -alignment
        axes.append(_unit_direction(principal))
        evidence[side] = {
            "local_forward": axes[-1].tolist(),
            "mesh_vertices_sha256": hashlib.sha256(
                np.asarray(body_vertices, dtype="<f8").tobytes()).hexdigest(),
            "principal_variance_m2": float(eigenvalues[-1]),
            "next_variance_m2": float(eigenvalues[-2]),
            "principal_axis_to_wrist_axis_deg": float(np.degrees(np.arccos(
                np.clip(alignment, -1, 1)))),
            "sign_basis": "model.qpos0 wrist_pitch->wrist_yaw direction",
            "geometry_note": "Vertex PCA length proxy; depends on fixed-mesh tessellation",
        }
    return np.array(axes), evidence


def hand_forward_directions(hand_rotations, local_axes):
    """Apply calibrated left/right local hand axes to world rotations."""
    rotations = np.asarray(hand_rotations, dtype=float)
    axes = np.asarray(local_axes, dtype=float)
    if rotations.ndim < 3 or rotations.shape[-3:] != (2, 3, 3):
        raise ValueError("hand_rotations must end in (2, 3, 3)")
    if not np.isfinite(rotations).all() or axes.shape != (2, 3):
        raise ValueError("Hand rotations and two local axes must be finite")
    axes = np.array([_unit_direction(axis) for axis in axes])
    return np.einsum("...sij,sj->...si", rotations, axes)


def make_hand_direction_task(frame_name, local_axis, cost):
    """Two-DOF direction task with an exact world-frame vector Jacobian.

    e = R(q) a - d, J = -[R(q) a]x J_angular_world. Isotropic vector weights
    preserve world-rotation invariance and leave roll about a in the nullspace.
    Reject the antipodal stationary point of this chordal residual explicitly.
    Mink remains optional until IK is requested.
    """
    import mink

    if isinstance(cost, bool) or not math.isfinite(cost) or cost <= 0:
        raise ValueError("Hand orientation cost must be finite and positive")

    class HandDirectionTask(mink.Task):
        def __init__(self):
            super().__init__(cost=np.full(3, cost), lm_damping=1e-3)
            self.local_axis = _unit_direction(local_axis)
            self.target_direction = None

        def set_target(self, direction):
            self.target_direction = _unit_direction(direction)

        def _rotation(self, configuration):
            return configuration.get_transform_frame_to_world(
                frame_name, "body").rotation().as_matrix()

        def compute_error(self, configuration):
            if self.target_direction is None:
                raise ValueError("Hand direction target has not been set")
            direction = self._rotation(configuration) @ self.local_axis
            if direction @ self.target_direction <= -1 + 1e-8:
                raise ValueError("Antipodal hand directions require a closer IK initial pose")
            return direction - self.target_direction

        def compute_jacobian(self, configuration):
            rotation = self._rotation(configuration)
            angular = rotation @ configuration.get_frame_jacobian(frame_name, "body")[3:]
            return np.cross(angular.T, rotation @ self.local_axis).T

    return HandDirectionTask()


def validate_hand_options(hand_orientation, hand_orientation_cost):
    if hand_orientation not in ("none", "forward"):
        raise ValueError("hand_orientation must be 'none' or 'forward'")
    if (isinstance(hand_orientation_cost, bool)
            or not math.isfinite(hand_orientation_cost) or hand_orientation_cost <= 0):
        raise ValueError("Hand orientation cost must be finite and positive")


def morphology_targets(source_p, source_R, rest_p):
    """Preserve limb directions using each G1 segment's calibrated length."""
    if source_p.shape != (14, 3) or rest_p.shape != (14, 3):
        raise ValueError("Morphology targets require fourteen named body roles")
    root_R, trunk_R = np.einsum("nij,jk->nik", source_R[:2], BODY_BASIS)
    human_legs = np.array([vector_norm(source_p[k] - source_p[h])
                          + vector_norm(source_p[f] - source_p[k])
                          for h, k, f in CHAINS[:2]])
    robot_legs = np.array([vector_norm(rest_p[k] - rest_p[h])
                          + vector_norm(rest_p[f] - rest_p[k])
                          for h, k, f in CHAINS[:2]])
    if np.any(human_legs < 1e-6) or np.any(robot_legs < 1e-6):
        raise ValueError("Degenerate leg segment in source or robot")
    scale = float(np.mean(robot_legs / human_legs))
    target_p = np.empty_like(source_p)
    target_p[0] = source_p[0] * scale
    target_p[1] = target_p[0] + np.einsum("ij,j->i", root_R, rest_p[1] - rest_p[0])
    for proximal, middle, distal in CHAINS:
        anchor = 0 if proximal < 8 else 1
        orientation = root_R if anchor == 0 else trunk_R
        target_p[proximal] = target_p[anchor] + np.einsum(
            "ij,j->i", orientation, rest_p[proximal] - rest_p[anchor])
        for parent, child in ((proximal, middle), (middle, distal)):
            direction = source_p[child] - source_p[parent]
            length = vector_norm(direction)
            if length < 1e-6:
                raise ValueError("Degenerate human limb segment")
            target_p[child] = target_p[parent] + direction / length * vector_norm(
                rest_p[child] - rest_p[parent])
    target_R = np.broadcast_to(np.eye(3), (14, 3, 3)).copy()
    target_R[0], target_R[1] = root_R, trunk_R
    return target_p, target_R, scale


def validate_output(values, joint_limits):
    expected = 7 + len(joint_limits)
    if values.ndim != 2 or values.shape[1] != expected or not len(values):
        raise ValueError(f"Output must contain {expected} columns")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite retarget output")
    norms = vector_norm(values[:, 3:7], axis=1)
    if not np.allclose(norms, 1, atol=1e-6):
        raise ValueError("Root quaternions must be unit XYZW")
    violation = np.maximum(joint_limits[:, 0] - values[:, 7:],
                           values[:, 7:] - joint_limits[:, 1]).clip(0)
    if np.max(violation) > 1e-6:
        raise ValueError("Retarget output exceeds robot joint limits")
    return {"finite": True, "unit_quaternion_max_error": float(np.max(abs(norms - 1))),
            "joint_limit_violation_max_rad": float(np.max(violation))}


def align_floor(values, task_group):
    from prepare_gait_references import align_reference_floor, physical_floor_distances

    if task_group == "aiming":
        values, report = align_reference_floor(values, "g1")
    else:
        distances = physical_floor_distances(values, "g1")
        shift = max(0., .003 - float(distances.min()))
        values = values.copy()
        values[:, 2] += shift
        report = {"method": "One global upward translation; flight/elevation retained",
                  "root_height_shift_m": shift,
                  "assumption": "Flat-floor nonpenetration only; obstacle contacts are unmodeled"}
    distances = physical_floor_distances(values, "g1")
    if distances.min() < -1e-5:
        raise ValueError("Physical foot collider penetrates the flat floor")
    report["physical_foot_floor_distance_m"] = {
        "min": float(distances.min()), "max": float(distances.max()),
        "p95": float(np.quantile(distances, .95))}
    return values, report


def retarget_motion(motion, task_group="aiming", *, hand_orientation="none",
                    hand_orientation_cost=1.):
    validate_hand_options(hand_orientation, hand_orientation_cost)
    import mink
    import mujoco
    from bootstrap import configure_paths

    configure_paths()
    from retarget_gait_references import flat_model, ROLES
    from trackerTask.supermimic.terrain import robot_parameters

    model, spec, height, qids, *_ = flat_model("g1")
    _, _, motors = robot_parameters("g1")
    body_names = ROLES["g1"]
    bodies = np.array([model.body(name).id for name in body_names])
    joint_ids = np.array([model.joint(name).id for name in spec.joint_names])
    limits_q = model.jnt_range[joint_ids].copy()
    data = mujoco.MjData(model)
    data.qpos[:3], data.qpos[qids] = (0, 0, height), spec.default_q
    mujoco.mj_forward(model, data)
    rest_p = data.xpos[bodies].copy()
    local_p, local_R, times = resample_local(motion)
    human_p, human_R = human_to_robot(
        *forward_kinematics(local_p, local_R, motion.parents))
    indices = role_indices(motion.names)
    initial_R = np.einsum("ij,jk->ik", human_R[0, 0], BODY_BASIS)
    yaw = math.atan2(initial_R[1, 0], initial_R[0, 0])
    heading = Rotation.from_euler("z", -yaw).as_matrix()
    origin = human_p[0, 0].copy()
    origin[2] = 0
    human_p = np.einsum("ij,...j->...i", heading, human_p - origin)
    human_R = np.einsum("ij,...jk->...ik", heading, human_R)
    configuration = mink.Configuration(model, data.qpos.copy())
    hand_tasks, hand_errors = [], []
    hand_report = {"mode": hand_orientation, "cost": 0.}
    if hand_orientation == "forward":
        source_axes, evidence = source_hand_axes(motion)
        robot_axes, robot_evidence = robot_hand_axes(model)
        hand_indices = [motion.names.index(side + "Hand") for side in ("Left", "Right")]
        hand_targets = hand_forward_directions(
            human_R[:, hand_indices], source_axes)
        hand_tasks = [make_hand_direction_task(body_names[role], axis, hand_orientation_cost)
                      for role, axis in zip((10, 13), robot_axes)]
        hand_report = {
            "mode": hand_orientation, "cost": hand_orientation_cost,
            "source": evidence, "robot": robot_evidence,
            "robot_local_forward": robot_axes.tolist(),
            "robot_basis": "principal fixed rubber-hand mesh axis in wrist_yaw body",
            "residual": "world-frame unit direction difference; two rotational DOFs",
            "palm_roll_observed": False,
            "limitations": ("Zero End Sites use a straight-hand rest convention. "
                            "Robot forward is a fixed-mesh PCA length proxy. "
                            "Palm roll and finger articulation are not reconstructed. "
                            "Collision avoidance is not part of this IK objective."),
        }
    tasks = []
    for role, body in enumerate(body_names):
        foot, hand = role in (4, 7), role in (10, 13)
        tasks.append(mink.FrameTask(
            body, "body",
            position_cost=30. if foot else 12. if hand else 10. if role == 0 else 3.,
            orientation_cost=8. if role == 0 else 3. if foot or role == 1 else 0.,
            lm_damping=1e-3,
        ))
    posture = mink.PostureTask(model, cost=.03, lm_damping=1e-3)
    posture.set_target(data.qpos.copy())
    all_tasks = [*tasks, *hand_tasks, posture, mink.DampingTask(model, cost=.005)]
    limits = [mink.ConfigurationLimit(model), mink.VelocityLimit(
        model, {name: motor[1] for name, motor in zip(spec.joint_names, motors)})]
    output, errors, scales = [], [], []
    for frame in range(len(times)):
        targets_p, targets_R, scale = morphology_targets(
            human_p[frame, indices], human_R[frame, indices], rest_p)
        for role, toe in ((4, "LeftToe"), (7, "RightToe")):
            targets_R[role] = np.einsum(
                "ij,jk->ik", human_R[frame, motion.names.index(toe)], FOOT_BASIS)
        for task, p, R in zip(tasks, targets_p, targets_R):
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(R), p))
        for side, task in enumerate(hand_tasks):
            task.set_target(hand_targets[frame, side])
        if frame == 0:
            q = configuration.q.copy()
            q[:3] = targets_p[0]
            mujoco.mju_mat2Quat(q[3:7], targets_R[0].ravel())
            configuration.update(q)
        iterations = 40 if frame == 0 else 6
        dt = .02 if frame == 0 else .02 / iterations
        for _ in range(iterations):
            velocity = mink.solve_ik(
                configuration, all_tasks, dt, "daqp", damping=1e-5, limits=limits)
            configuration.integrate_inplace(velocity, dt)
        q = configuration.q.copy()
        output.append(np.r_[q[:3], q[[4, 5, 6, 3]], q[qids]])
        errors.append(vector_norm(configuration.data.xpos[bodies] - targets_p, axis=1))
        if hand_tasks:
            actual = np.einsum("sij,sj->si", configuration.data.xmat[
                bodies[[10, 13]]].reshape(2, 3, 3), robot_axes)
            hand_errors.append(np.degrees(np.arccos(
                np.clip(np.sum(actual * hand_targets[frame], axis=1), -1, 1))))
        scales.append(scale)
        if (frame + 1) % 100 == 0:
            print(f"IK {frame + 1}/{len(times)}", flush=True)
    values = np.array(output)
    raw_violation = float(np.maximum(
        limits_q[:, 0] - values[:, 7:], values[:, 7:] - limits_q[:, 1]).clip(0).max())
    if raw_violation > 1e-4:
        raise ValueError(f"IK joint-limit violation: {raw_violation:.6g} rad")
    values[:, 7:] = np.clip(values[:, 7:], *limits_q.T)
    for frame in range(1, len(values)):
        if np.sum(values[frame - 1, 3:7] * values[frame, 3:7]) < 0:
            values[frame, 3:7] *= -1
    values, floor = align_floor(values, task_group)
    if hand_tasks:
        hand_report["forward_error_deg_mean"] = dict(zip(
            ("LeftHand", "RightHand"), np.mean(hand_errors, axis=0).tolist()))
        hand_report["forward_error_deg_p95"] = dict(zip(
            ("LeftHand", "RightHand"), np.quantile(hand_errors, .95, axis=0).tolist()))
    report = {
        "solver": {"mink": version("mink"), "daqp": version("daqp"),
                   "qp_solver": "daqp", "first_frame_iterations": 40,
                   "subsequent_frame_iterations": 6, "damping": 1e-5},
        "joint_names": list(spec.joint_names), "target_model_sha256": digest(spec.mjcf_path),
        "source_to_target_bodies": dict(zip(SOURCE_ROLES, body_names)),
        "root_scale": float(np.mean(scales)),
        "world_basis": WORLD_BASIS.tolist(), "pelvis_trunk_basis": BODY_BASIS.tolist(),
        "foot_basis": FOOT_BASIS.tolist(),
        "foot_orientation": "Toe global orientation mapped to G1 ankle; pitch/roll retained",
        "hand_orientation": hand_report,
        "initial_yaw_removed_rad": yaw,
        "ik_position_error_m_mean": dict(zip(body_names, np.mean(errors, axis=0).tolist())),
        "ik_position_error_m_p95": dict(zip(body_names, np.quantile(errors, .95, axis=0).tolist())),
        "raw_joint_limit_violation_max_rad": raw_violation,
        "ground_alignment": floor, "validation": validate_output(values, limits_q),
        "scope": "Kinematic reference; physical policy tracking remains untested",
    }
    return values, report


def source_identity(path, task_group):
    match = re.fullmatch(r"(aiming|obstacles)(\d+)_subject(\d+)\.bvh",
                         Path(path).name, flags=re.IGNORECASE)
    if not match or match[1].lower() != task_group:
        raise ValueError("BVH basename must match the selected original task group")
    take, subject = int(match[2]), int(match[3])
    return {"source_group": f"{task_group}{take}", "source_take": take,
            "source_subject": subject, "split": "test" if take == 1 else "train"}


def validate_paths(sources, outputs, report):
    paths = [*outputs, report]
    if len(set(paths)) != len(paths) or set(paths).intersection(sources):
        raise ValueError("Sources, CSV outputs and report must have distinct paths")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
        if path.is_relative_to(REPO):
            check = subprocess.run(
                ["git", "-C", str(REPO), "check-ignore", "--quiet", "--no-index", str(path)],
                capture_output=True, check=False)
            if check.returncode:
                raise ValueError("Motion derivatives and reports must be Git-ignored")


def convert(bvhs, output, task_group="aiming", start_frame=0, max_frames=300, report=None,
            start_frames=None, *, hand_orientation="none", hand_orientation_cost=1.):
    validate_hand_options(hand_orientation, hand_orientation_cost)
    sources = [Path(path).expanduser().resolve() for path in bvhs]
    output = Path(output).expanduser().resolve()
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Supply distinct BVH inputs")
    starts = [start_frame] * len(sources) if start_frames is None else list(start_frames)
    if (len(starts) != len(sources)
            or any(type(value) is not int or value < 0 for value in starts)):
        raise ValueError("Supply one nonnegative source start frame per BVH")
    identities = [source_identity(path, task_group) for path in sources]
    single_csv = len(sources) == 1 and output.suffix.lower() == ".csv"
    if not single_csv and output.suffix.lower() == ".csv":
        raise ValueError("Multiple BVHs require an output directory")
    outputs = [output] if single_csv else [output / (path.stem + ".csv") for path in sources]
    report_path = Path(report).expanduser().resolve() if report else (
        output.with_suffix(".json") if single_csv else output / "origin_manifest.json")
    validate_paths(sources, outputs, report_path)
    entries = []
    revision = digest(__file__)
    for source, destination, identity, source_start in zip(sources, outputs, identities, starts):
        source_hash = digest(source)
        motion = parse_bvh(source.read_text(encoding="utf-8-sig"), source_start, max_frames)
        values, metrics = retarget_motion(
            motion, task_group, hand_orientation=hand_orientation,
            hand_orientation_cost=hand_orientation_cost)
        if digest(source) != source_hash:
            raise ValueError("Source BVH changed during conversion")
        config = {"task_group": task_group, "start_frame": source_start,
                  "max_frames": max_frames, "source_fps": SOURCE_FPS,
                  "output_fps": OUTPUT_FPS, "retargeter_sha256": revision,
                  "hand_orientation": hand_orientation,
                  "hand_orientation_cost": hand_orientation_cost if hand_orientation != "none" else 0.}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="") as stream:
            np.savetxt(stream, values, delimiter=",", fmt="%.10g")
        entry = {
            "file": destination.name, "sha256": digest(destination),
            "source_file": source.name, "source_sha256": source_hash, **identity,
            "source_fps": SOURCE_FPS, "output_fps": OUTPUT_FPS,
            "source_frame_time_declared": motion.frame_time,
            "source_total_frames": motion.total_frames,
            "source_frame_range": [source_start, source_start + len(motion.positions)],
            "frame_range_semantics": "zero-based, stop-exclusive",
            "output_frames": len(values), "duration_seconds": (len(values) - 1) / OUTPUT_FPS,
            "retargeter": "retarget_lafan_aiming.py / morphology-scaled Mink IK",
            "retargeter_revision": revision,
            "retarget_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode("ascii")).hexdigest(),
            "parser": {"package": "bvh", "version": version("bvh"),
                       "rotation_convention": "SciPy intrinsic declared channel order",
                       "channels": dict(zip(motion.names, motion.channels))},
            "interpolation": "30 -> 50 Hz local-translation linear + rotation SLERP before IK",
            **metrics,
        }
        entries.append(entry)
        print(f"Retargeted {source.name}: {len(values)} frames -> {destination}", flush=True)
    document = {
        "schema": "lafan1-g1-origin-v1", "dataset": "ubisoft/ubisoft-laforge-animation-dataset",
        "revision": OFFICIAL_REVISION, "archive_sha256": ARCHIVE_SHA256,
        "license": "CC BY-NC-ND 4.0; original data retained, derivatives local-only",
        "archive_validation": "Archive pin supplied by owner; each input BVH is hashed locally",
        "task_group": task_group, "source_fps": SOURCE_FPS, "output_fps": OUTPUT_FPS,
        "split_policy": "Take 1 held out across subjects; take 2+ training",
        "files": entries,
        "motion_config": {"version": 1, "root": ".", "source_fps": OUTPUT_FPS,
                          "quaternion_order": "xyzw", "joint_names": entries[0]["joint_names"],
                          "files": [path.name for path in outputs]},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="One CSV, or a directory for multiple BVHs")
    parser.add_argument("--task-group", choices=("aiming", "obstacles"), default="aiming")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--start-frames", type=int, nargs="+",
                        help="One zero-based start frame for each --bvh input")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--report", "--output-report", dest="report", type=Path)
    parser.add_argument("--hand-orientation", choices=("none", "forward"), default="none",
                        help="Optional skeleton-calibrated hand forward; palm roll remains free")
    parser.add_argument("--hand-orientation-cost", type=float, default=1.,
                        help="Positive direction-task weight when --hand-orientation forward")
    args = parser.parse_args(argv)
    convert(args.bvh, args.output, args.task_group, args.start_frame, args.max_frames, args.report,
            args.start_frames, hand_orientation=args.hand_orientation,
            hand_orientation_cost=args.hand_orientation_cost)


if __name__ == "__main__":
    main()

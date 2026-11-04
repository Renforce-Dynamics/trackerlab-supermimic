"""Retarget the audited G1 walking recordings to each humanoid with Mink IK."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from bootstrap import configure_paths
REPO = configure_paths()
from prepare_gait_references import align_reference_floor, digest, motion_config, reference_audit, write_json
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerTask.supermimic.terrain import ROBOT_HEIGHTS, generate, robot_parameters

ROLES = {
    "g1": ("pelvis", "torso_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
           "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
           "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
           "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"),
    "a3": ("pelvis_link", "torso_Link", "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
           "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link",
           "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
           "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link"),
    "t1": ("trunk", "waist_yaw_link", "left_hip_roll_link", "left_knee_pitch_link", "left_ankle_roll_link",
           "right_hip_roll_link", "right_knee_pitch_link", "right_ankle_roll_link",
           "left_shoulder_roll_link", "left_elbow_pitch_link", "left_elbow_yaw_link",
           "right_shoulder_roll_link", "right_elbow_pitch_link", "right_elbow_yaw_link"),
    "bumi": ("base_link", "waist_yaw_link", "l_leg_roll_link", "l_knee_pitch_link", "l_ankle_roll_link",
             "r_leg_roll_link", "r_knee_pitch_link", "r_ankle_roll_link",
             "l_arm_roll_link", "l_elbow_pitch_link", "l_elbow_pitch_link",
             "r_arm_roll_link", "r_elbow_pitch_link", "r_elbow_pitch_link"),
    "pi_plus": ("base_link", "base_link", "l_hip_roll_link", "l_calf_link", "l_ankle_roll_link",
                "r_hip_roll_link", "r_calf_link", "r_ankle_roll_link",
                "l_shoulder_roll_link", "l_elbow_link", "l_wrist_link",
                "r_shoulder_roll_link", "r_elbow_link", "r_wrist_link"),
}


def flat_model(robot):
    field = generate("stairs", ROBOT_HEIGHTS[robot], 0)
    return build_model(robot, replace(field, heights=np.zeros_like(field.heights)), ground_plane=True)


def retarget(values, robot):
    import mink

    donor, donor_spec, *_ = flat_model("g1")
    model, spec, _, qpos_ids, _, root_id, _, _, _ = flat_model(robot)
    _, _, motors = robot_parameters(robot)
    donor_qpos = np.array([donor.joint(name).qposadr[0] for name in donor_spec.joint_names])
    donor_bodies = np.array([donor.body(name).id for name in ROLES["g1"]])
    bodies = np.array([model.body(name).id for name in ROLES[robot]])
    donor_data, calibration = mujoco.MjData(donor), mujoco.MjData(model)
    donor_data.qpos[:3] = (0, 0, ROBOT_HEIGHTS["g1"])
    donor_data.qpos[donor_qpos] = donor_spec.default_q
    calibration.qpos[:3] = (0, 0, ROBOT_HEIGHTS[robot])
    calibration.qpos[qpos_ids] = spec.default_q
    mujoco.mj_forward(donor, donor_data)
    mujoco.mj_forward(model, calibration)
    donor_rest_p = donor_data.xpos[donor_bodies].copy()
    donor_rest_R = donor_data.xmat[donor_bodies].copy().reshape(-1, 3, 3)
    rest_p = calibration.xpos[bodies].copy()
    rest_R = calibration.xmat[bodies].copy().reshape(-1, 3, 3)
    scale = ROBOT_HEIGHTS[robot] / ROBOT_HEIGHTS["g1"]
    configuration = mink.Configuration(model, calibration.qpos.copy())
    tasks, role_tasks = [], []
    seen = set()
    for role, body in enumerate(ROLES[robot]):
        if body in seen:
            continue
        seen.add(body)
        is_foot = role in (4, 7)
        task = mink.FrameTask(
            body, "body", position_cost=30. if is_foot else 8. if role == 0 else 2.,
            orientation_cost=4. if is_foot or role < 2 else .3,
            lm_damping=1e-3)
        tasks.append(task)
        role_tasks.append((role, task))
    posture = mink.PostureTask(model, cost=.05, lm_damping=1e-3)
    tasks.append(posture)
    tasks.append(mink.DampingTask(model, cost=.005))
    limits = [mink.ConfigurationLimit(model),
              mink.VelocityLimit(model, {name: motor[1] for name, motor in zip(spec.joint_names, motors)})]
    chains = ((2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13))
    output, errors = [], []
    for frame, source in enumerate(values):
        donor_data.qpos[:3] = source[:3]
        donor_data.qpos[3:7] = source[[6, 3, 4, 5]]
        donor_data.qpos[donor_qpos] = source[7:]
        mujoco.mj_forward(donor, donor_data)
        p = donor_data.xpos[donor_bodies].copy()
        R = donor_data.xmat[donor_bodies].copy().reshape(-1, 3, 3)
        target_p = source[:3] * scale + (p - p[0]) * scale
        target_p[1] = target_p[0] + R[0] @ (rest_p[1] - rest_p[0])
        for hip, knee, ankle in chains:
            target_p[hip] = target_p[0] + R[0] @ (rest_p[hip] - rest_p[0])
            for parent, child in ((hip, knee), (knee, ankle)):
                length = np.linalg.norm(rest_p[child] - rest_p[parent])
                direction = p[child] - p[parent]
                if length > 1e-6 and np.linalg.norm(direction) > 1e-6:
                    target_p[child] = target_p[parent] + direction / np.linalg.norm(direction) * length
        target_R = R @ donor_rest_R.transpose(0, 2, 1) @ rest_R
        for role, task in role_tasks:
            quat = Rotation.from_matrix(target_R[role]).as_quat()[[3, 0, 1, 2]]
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat), target_p[role]))
        posture.set_target(configuration.q)
        count = 40 if frame == 0 else 6
        dt = .02 if frame == 0 else .02 / count
        for _ in range(count):
            velocity = mink.solve_ik(configuration, tasks, dt, "daqp", damping=1e-5, limits=limits)
            configuration.integrate_inplace(velocity, dt)
        q = configuration.q.copy()
        output.append(np.r_[q[:3], q[[4, 5, 6, 3]], q[qpos_ids]])
        errors.append([np.linalg.norm(configuration.data.xpos[bodies[index]] - target_p[index]) for index in (4, 7)])
        if (frame + 1) % 200 == 0:
            print(f"{robot} retargeted {frame + 1}/{len(values)} frames", flush=True)
    output = np.asarray(output)
    # Remove framewise IK noise without changing the gait's timestamps or phase.
    output[:, :3] = savgol_filter(output[:, :3], 7, 2, axis=0)
    output[:, 7:] = savgol_filter(output[:, 7:], 7, 2, axis=0)
    joints = np.array([model.joint(name).id for name in spec.joint_names])
    output[:, 7:] = np.clip(output[:, 7:], *model.jnt_range[joints].T)
    return output, {"solver": "Mink 1.4.0 / DAQP", "foot_ik_error_m_mean": np.mean(errors, axis=0).tolist(),
                    "foot_ik_error_m_p95": np.quantile(errors, .95, axis=0).tolist(),
                    "target_model_sha256": digest(spec.mjcf_path),
                    "source": "Audited G1 BONES-SEED walking; morphology-scaled body/foot tasks",
                    "smoothing": "7-frame order-2 Savitzky-Golay for root translation and joints"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("a3", "t1", "bumi", "pi_plus"), required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    donor = args.donor.resolve()
    audit = json.loads((donor / "gait_audit.json").read_text())
    if audit["status"] != "passed" or audit["robot"] != "g1":
        raise ValueError("The donor walking reference audit must pass")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new retarget directory")
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    train, test = audit["training_files"], audit["evaluation_files"]
    for name in [*train, *test]:
        if digest(donor / name) != audit["records"][name]["sha256"]:
            raise ValueError("Donor motion changed after audit")
        values = np.loadtxt(donor / name, delimiter=",", ndmin=2)
        result, report = retarget(values, args.robot)
        result, alignment = align_reference_floor(result, args.robot)
        report["ground_alignment"] = alignment
        np.savetxt(output / name, result, delimiter=",", fmt="%.9g")
        sources[name] = {"donor": str(donor / name), "donor_sha256": digest(donor / name), **report}
        print(json.dumps({"robot": args.robot, "clip": name, **report}), flush=True)
    write_json(output / "train.json", motion_config(args.robot, train))
    write_json(output / "test.json", motion_config(args.robot, test))
    result = reference_audit(output, args.robot, train, test, sources)
    write_json(output / "gait_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "passed":
        raise SystemExit("Retargeted gait audit failed")


if __name__ == "__main__":
    main()

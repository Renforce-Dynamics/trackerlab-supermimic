"""Compare Kine2Go foot positions against Go2 MJCF forward kinematics."""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path

import mujoco
import numpy as np

from bootstrap import configure_paths

configure_paths()
from trackerTask.supermimic.robots import get_robot


MODEL_LEG_NAMES = ("FL", "FR", "RL", "RR")
DATASET_FOOT_ORDER = ("FL", "RL", "FR", "RR")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    motion = np.load(args.motion, allow_pickle=False)
    if motion.ndim != 2 or motion.shape[1] != 61 or not np.isfinite(motion).all():
        raise ValueError(f"Expected finite Kine2Go (T, 61) array, got {motion.shape}")
    frame_ids = np.linspace(0, len(motion) - 1, min(args.frames, len(motion)), dtype=int)
    rows = motion[frame_ids]
    spec = get_robot("go2")
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    qpos_ids = np.array([model.joint(name).id for name in spec.joint_names])
    qpos_ids = model.jnt_qposadr[qpos_ids]
    calves = {name: model.body(name + "_calf").id for name in MODEL_LEG_NAMES}
    target = rows[:, 36:48].reshape(-1, 4, 3)
    target = target[:, [DATASET_FOOT_ORDER.index(name) for name in MODEL_LEG_NAMES]]
    scores = []

    # The published array groups the 12 coordinates by joint type. Test each
    # shared leg permutation against the separately stored foot trajectories.
    for leg_perm in permutations(range(4)):
        q = np.stack(
            [rows[:, 6 + group * 4 + np.array(leg_perm)] for group in range(3)],
            axis=-1,
        ).reshape(len(rows), 12)
        errors = []
        for frame_idx, row in enumerate(rows):
            data.qpos[:3] = row[48:51]
            data.qpos[3:7] = row[51:55]
            data.qpos[qpos_ids] = q[frame_idx]
            mujoco.mj_forward(model, data)
            actual = np.stack([
                data.xpos[calves[name]]
                + data.xmat[calves[name]].reshape(3, 3) @ np.array([0., 0., -.213])
                for name in MODEL_LEG_NAMES
            ])
            errors.append(np.linalg.norm(actual - target[frame_idx], axis=-1))
        err = np.asarray(errors)
        scores.append({
            "raw_slots_for_model_leg_order": list(leg_perm),
            "mean_foot_error_m": float(err.mean()),
            "median_foot_error_m": float(np.median(err)),
            "p95_foot_error_m": float(np.quantile(err, 0.95)),
            "max_foot_error_m": float(err.max()),
        })
    scores.sort(key=lambda item: item["mean_foot_error_m"])
    result = {
        "motion": str(args.motion),
        "dataset_shape": list(motion.shape),
        "frames_checked": frame_ids.tolist(),
        "interpretation": "same raw leg-slot permutation applied to hip, thigh and calf groups",
        "model_leg_order": MODEL_LEG_NAMES,
        "foot_reference": "foot link origin, calf-local [0, 0, -0.213]; collision geom differs by 0.002 m",
        "best_candidates": scores[:8],
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

"""Booster T1 limits/armatures from booster_assets at the pinned revision.

PD gains use the existing G1 preset's 10 Hz, damping-ratio-2 design rule with
T1's own armatures. These gains are a simulation preset, not vendor gains.
"""

import math
from trackerTask.supermimic.humanoid_asset import asset_dir, make_articulation

ASSET_DIR = asset_dir("t1", "booster")
MJCF_PATH = str(ASSET_DIR / "mjcf/t1.xml")
JOINT_NAMES = (
    "aahead_yaw_joint", "aahead_pitch_joint",
    *(f"{side}_{part}_joint" for side in ("left", "right")
      for part in ("shoulder_pitch", "shoulder_roll", "elbow_pitch", "elbow_yaw")),
    "waist_yaw_joint",
    *(f"{side}_{part}_joint" for side in ("left", "right")
      for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")),
)
TRACKED_BODIES = (
    "trunk", "waist_yaw_link",
    *(f"{side}_{part}_link" for side in ("left", "right")
      for part in ("hip_roll", "knee_pitch", "ankle_roll", "shoulder_roll", "elbow_pitch", "elbow_yaw")),
)


def motor(name):
    part = name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    if part.startswith("aahead"):
        effort, velocity, arm = 7., 41.89, .0018
    elif part.startswith(("shoulder", "elbow")):
        effort, velocity, arm = 38.3, 17.59, .0282528
    else:
        effort, velocity, arm = {
            "waist_yaw": (68, 14.66, .0478125), "hip_pitch": (98.8, 15.29, .0523908),
            "hip_roll": (68, 14.66, .0478125), "hip_yaw": (68, 14.66, .0478125),
            "knee_pitch": (130.5, 14.76, .0636012),
            "ankle_pitch": (73.1, 12.57, .0679104), "ankle_roll": (17.2, 12.57, .02037312),
        }[part]
    omega = 10 * 2 * math.pi
    return effort, velocity, arm * omega**2, 4 * arm * omega, arm


def default(name):
    part = name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    return {"hip_pitch": -.2, "knee_pitch": .4, "ankle_pitch": -.2,
            "shoulder_roll": -1.4 if name.startswith("left") else 1.4}.get(part, 0.)


DEFAULT_Q = tuple(default(n) for n in JOINT_NAMES)
ACTION_SCALE = tuple(.25 * motor(n)[0] / motor(n)[2] for n in JOINT_NAMES)


def make_cfg():
    return make_articulation("t1", ASSET_DIR, JOINT_NAMES, DEFAULT_Q,
                             tuple(motor(n) for n in JOINT_NAMES), .69)

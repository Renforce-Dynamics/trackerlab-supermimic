"""Bumi V3 model and PD gains from the pinned Noetix vendor configuration."""

from trackerTask.supermimic.humanoid_asset import asset_dir, make_articulation

ASSET_DIR = asset_dir("bumi", "noetix")
MJCF_PATH = str(ASSET_DIR / "mjcf/bumi.xml")
JOINT_NAMES = (
    "waist_yaw_joint",
    *(f"{side}_{part}_joint" for side in ("l", "r")
      for part in ("arm_pitch", "arm_roll", "arm_yaw", "elbow_pitch")),
    *(f"{side}_{part}_joint" for side in ("l", "r")
      for part in ("leg_pitch", "leg_roll", "leg_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")),
)
TRACKED_BODIES = (
    "base_link", "waist_yaw_link",
    *(f"{side}_{part}_link" for side in ("l", "r")
      for part in ("leg_roll", "knee_pitch", "ankle_roll", "arm_roll", "elbow_pitch")),
)


def motor(name):
    part = name.removeprefix("l_").removeprefix("r_").removesuffix("_joint")
    if part == "waist_yaw":
        return 27., 9., 53., 3.4, 0.
    if part == "leg_yaw":
        return 27., 9., 60., 2.5, 0.
    if part == "leg_roll":
        return 27., 12., 60., 3., 0.
    if part == "leg_pitch":
        return 60., 12., 60., 3., 0.
    if part == "knee_pitch":
        return 60., 12., 45., 2., 0.
    if part.startswith("ankle"):
        return 13., 12., 10., .5, .01
    return 5.5, 50., 12., .4, 0.


def default(name):
    part = name.removeprefix("l_").removeprefix("r_").removesuffix("_joint")
    return {"leg_pitch": -.1495, "knee_pitch": .3215, "ankle_pitch": -.172,
            "arm_roll": .3 if name.startswith("l_") else -.3}.get(part, 0.)


DEFAULT_Q = tuple(default(n) for n in JOINT_NAMES)
ACTION_SCALE = tuple(.25 * motor(n)[0] / motor(n)[2] for n in JOINT_NAMES)


def make_cfg():
    return make_articulation("bumi", ASSET_DIR, JOINT_NAMES, DEFAULT_Q,
                             tuple(motor(n) for n in JOINT_NAMES), .65)

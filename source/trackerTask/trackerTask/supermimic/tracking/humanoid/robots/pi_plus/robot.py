"""Vendor Pi Plus model: 22 moving joints despite the pi_20dof.xml filename."""

from trackerTask.supermimic.humanoid_asset import asset_dir, make_articulation

ASSET_DIR = asset_dir("pi_plus", "hightorque")
MJCF_PATH = str(ASSET_DIR / "mjcf/pi_plus.xml")
JOINT_NAMES = (
    *(f"{side}_{part}_joint" for side in ("l", "r")
      for part in ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")),
    *(f"{side}_{part}_joint" for side in ("l", "r")
      for part in ("shoulder_pitch", "shoulder_roll", "upper_arm", "elbow", "wrist")),
)
TRACKED_BODIES = (
    "base_link",
    *(f"{side}_{part}_link" for side in ("l", "r")
      for part in ("hip_roll", "calf", "ankle_roll", "shoulder_roll", "elbow", "wrist")),
)


def motor(name):
    part = name.removeprefix("l_").removeprefix("r_").removesuffix("_joint")
    if part in ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll"):
        return 30., 17. if part.startswith("ankle") else 8., 80., 1.1, .044277
    return 20., 17., 30., .6, .008419


DEFAULT_Q = tuple({"hip_pitch": -.25, "calf": .65, "ankle_pitch": -.4}.get(
    n.removeprefix("l_").removeprefix("r_").removesuffix("_joint"), 0.) for n in JOINT_NAMES)
ACTION_SCALE = tuple(.25 * motor(n)[0] / motor(n)[2] for n in JOINT_NAMES)


def make_cfg():
    return make_articulation("pi_plus", ASSET_DIR, JOINT_NAMES, DEFAULT_Q,
                             tuple(motor(n) for n in JOINT_NAMES), .351)

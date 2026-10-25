"""A3 actuator settings from AgibotTech/sonic_for_a3, a3_old.py.

The serial a3_t2d5 geometry is used with its zero-valued head joints fixed.
Passive compliant-foot and parallel-motor models are outside this preset.
"""

from trackerTask.supermimic.humanoid_asset import asset_dir, make_articulation

ASSET_DIR = asset_dir("a3", "agibot")
MJCF_PATH = str(ASSET_DIR / "mjcf/a3.xml")
JOINT_NAMES = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *(f"{side}_{part}_joint" for side in ("left", "right")
      for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                   "wrist_roll", "wrist_pitch", "wrist_yaw")),
    *(f"{side}_{part}_joint" for side in ("left", "right")
      for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
)
TRACKED_BODIES = (
    "pelvis_link", "torso_Link",
    *(f"{side}_{part}_Link" for side in ("left", "right")
      for part in ("hip_roll", "knee", "ankle_roll", "shoulder_roll", "elbow", "wrist_yaw")),
)
P78, P93 = 30.118e-6 * 20.03**2, 138.5069e-6 * 21.906**2
P41, P59, P110 = 1.359e-6 * 24.415**2, 10.1374e-6 * 22.136**2, 300.851e-6 * 20**2
MOTORS = {
    "hip_pitch": (220, 12.04277184, 80, 3, P93),
    "hip_yaw": (220, 12.04277184, 80, 3, P93),
    "hip_roll": (220, 12.04277184, 120, 4, P93),
    "knee": (320, 14.66076572, 250, 8, P110),
    "ankle_pitch": (118.2, 10.8, 50, 2, P78 * 5.333),
    "ankle_roll": (54.75, 19.37, 50, 2, P78 * 1.66562),
    "waist_yaw": (220, 12.04277184, 85, 3, P93),
    "waist_roll": (46, 22.7, 50, 2, P78 * 1.21),
    "waist_pitch": (115, 9.24785, 50, 2, P78 * 7.3),
    "shoulder_pitch": (60, 13.61356817, 40, 3, P78),
    "shoulder_roll": (60, 13.61356817, 40, 3, P78),
    "shoulder_yaw": (24, 15.70796327, 30, 2, P59),
    "elbow": (24, 15.70796327, 30, 2, P59),
    "wrist_roll": (24, 15.70796327, 30, 2, P59),
    "wrist_pitch": (6, 12.77581012, 20, 2, P41),
    "wrist_yaw": (6, 12.77581012, 20, 2, P41),
}


def motor(name):
    return MOTORS[name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")]


def default(name):
    part = name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
    sign = -1 if name.startswith("right") else 1
    return {"hip_pitch": -.1311, "hip_roll": .0056 * sign, "hip_yaw": -.0348 * sign,
            "knee": .2468, "ankle_pitch": -.1204, "ankle_roll": -.0078 * sign,
            "shoulder_pitch": .3, "shoulder_roll": .12 * sign, "elbow": .8}.get(part, 0.)


DEFAULT_Q = tuple(default(n) for n in JOINT_NAMES)
ACTION_SCALE = tuple(.25 * motor(n)[0] / motor(n)[2] for n in JOINT_NAMES)


def make_cfg():
    return make_articulation("a3", ASSET_DIR, JOINT_NAMES, DEFAULT_Q,
                             tuple(motor(n) for n in JOINT_NAMES), 1.09)

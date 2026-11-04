"""Locomotion foundation contracts and advancement criteria."""

from trackerTask.supermimic.robots import TRACKING_TASKS

ROBOTS = TRACKING_TASKS
FOOT_BODIES = {
    "g1": ("left_ankle_roll_link", "right_ankle_roll_link"),
    "go2": ("FL_calf", "FR_calf", "RL_calf", "RR_calf"),
    "a3": ("left_ankle_roll_Link", "right_ankle_roll_Link"),
    "t1": ("left_ankle_roll_link", "right_ankle_roll_link"),
    "bumi": ("l_ankle_roll_link", "r_ankle_roll_link"),
    "pi_plus": ("l_ankle_roll_link", "r_ankle_roll_link"),
}
FOOT_OFFSETS = {"go2": (-.002, 0., -.213)}
FOUNDATION_VERSION = 1
QUALITY_LIMITS = {
    "full_clip_survival": .95,
    "mean_body_position_error_scale_m": .08,
    "joint_rmse_rad": .18,
    "root_tilt_error_rad": .15,
    "stance_foot_slip_scale_m_s": .12,
    "foot_position_rmse_scale_m": .06,
}


def quality_passes(metrics):
    """Advancement needs every measured gait property, including completed clips."""
    import math
    for name, threshold in QUALITY_LIMITS.items():
        value = metrics.get(name)
        if value is None or not math.isfinite(value):
            return False
        if name == "full_clip_survival":
            if value < threshold:
                return False
        elif value > threshold:
            return False
    return True

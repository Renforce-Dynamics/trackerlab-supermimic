"""Booster T1 public dance motion tracking."""

from isaaclab.utils import configclass
from trackerTask.supermimic.task_cfg import SuperMimicTaskCfg


@configclass
class SuperMimicT1TaskCfg(SuperMimicTaskCfg):
    robot_name: str = "t1"
    observation_mode: str = "relative_orientation"
    torso_com_offset_range: tuple[float, float, float] = (0.01, 0.02, 0.02)

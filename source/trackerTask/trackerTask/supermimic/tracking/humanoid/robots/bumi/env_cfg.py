"""Bumi V3 vendor reference tracking."""

from isaaclab.utils import configclass
from trackerTask.supermimic.task_cfg import SuperMimicTaskCfg


@configclass
class SuperMimicBumiTaskCfg(SuperMimicTaskCfg):
    robot_name: str = "bumi"
    observation_mode: str = "relative_orientation"
    torso_com_offset_range: tuple[float, float, float] = (0.01, 0.02, 0.02)

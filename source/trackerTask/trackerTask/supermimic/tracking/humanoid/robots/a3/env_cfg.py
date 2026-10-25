"""A3 public SOMA motion tracking with fixed head and serial rigid feet."""

from isaaclab.utils import configclass
from trackerTask.supermimic.task_cfg import SuperMimicTaskCfg


@configclass
class SuperMimicA3TaskCfg(SuperMimicTaskCfg):
    robot_name: str = "a3"
    observation_mode: str = "relative_orientation"
    torso_com_offset_range: tuple[float, float, float] = (0.025, 0.05, 0.05)

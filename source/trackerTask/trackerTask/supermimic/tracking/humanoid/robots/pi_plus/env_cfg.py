"""Pi Plus public vendor reference tracking."""

from isaaclab.utils import configclass
from trackerTask.supermimic.task_cfg import SuperMimicTaskCfg


@configclass
class SuperMimicPiPlusTaskCfg(SuperMimicTaskCfg):
    robot_name: str = "pi_plus"
    observation_mode: str = "relative_orientation"
    torso_com_offset_range: tuple[float, float, float] = (0.01, 0.01, 0.01)

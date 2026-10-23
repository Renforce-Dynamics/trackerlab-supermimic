"""TrackerLab task configuration for SuperMimic on the Unitree G1 29-DoF robot."""

from isaaclab.utils import configclass
from .task_cfg import SuperMimicTaskCfg


@configclass
class SuperMimicG1TaskCfg(SuperMimicTaskCfg):
    robot_name: str = "g1"

"""Go2 periodic-gait imitation task; uses the shared SuperMimic learning flow."""

from isaaclab.utils import configclass
from .task_cfg import SuperMimicTaskCfg


@configclass
class SuperMimicGo2TaskCfg(SuperMimicTaskCfg):
    robot_name: str = "go2"
    observation_mode: str = "relative_orientation"
    torso_com_offset_range: tuple[float, float, float] = (0.01, 0.01, 0.01)
    reset_position: tuple[float, float, float] = (0.02, 0.02, 0.005)
    reset_rotation: tuple[float, float, float] = (0.05, 0.05, 0.1)
    reset_linvel: tuple[float, float, float] = (0.2, 0.1, 0.1)
    reset_angvel: tuple[float, float, float] = (0.2, 0.2, 0.3)
    reset_joint: float = 0.05
    reward_overrides: dict = {
        "body_pos_std": 0.15, "body_linvel_std": 0.5,
        "root_pos_std": 0.2, "term_height": 0.16,
    }

"""Simulator-free contracts for the command/style MuJoCo-Warp task family."""

from dataclasses import dataclass


@dataclass
class CommandTaskCfg:
    task_id: str
    robot_name: str
    profile: str
    num_envs: int = 8096
    control_dt: float = .02
    physics_dt: float = .005
    observation_history: int = 10
    imagination_horizon: int = 32
    scenario_dim: int = 7
    goal_dim: int = 12
    perception_rays: int = 187
    runtime_reference_motion: bool = False
    style_reward_enabled: bool = True
    algorithm: str = "supermimic-command-amp"
    simulator: str = "mujoco-warp"

    def __post_init__(self):
        from trackerTask.supermimic.terrain import get_task
        task = get_task(self.task_id)
        if task.robot != self.robot_name or task.profile != self.profile:
            raise ValueError("Command configuration differs from its registered task")

    @property
    def obs_dim(self):
        from trackerTask.supermimic.robots import get_robot
        return 3 * get_robot(self.robot_name).num_joints + 196

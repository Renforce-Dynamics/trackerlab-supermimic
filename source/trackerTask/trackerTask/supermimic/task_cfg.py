"""Shared simulation and learning defaults; robot-specific tasks override fields."""

from isaaclab.utils import configclass


@configclass
class SuperMimicTaskCfg:
    robot_name: str = "g1"
    num_envs: int = 4096
    env_spacing: float = 2.0
    physics_dt: float = 0.005
    control_decimation: int = 4
    control_dt: float = 0.02
    max_episode_steps: int = 500
    reset_noise: bool = True
    gpu_max_rigid_patch_count: int = 2**21
    observation_mode: str = "relative_ori99"
    observation_history: int = 10
    goal_frames: int = 10
    imagination_horizon: int = 32
    replay_buffer_steps: int = 1000
    world_model_window: int = 8
    world_model_updates_per_iteration: int = 10
    world_model_batch_size: int = 2048
    policy_batch_size: int = 4096
    friction_range: tuple[float, float] = (0.3, 1.2)
    joint_offset_range: float = 0.01
    torso_com_offset_range: tuple[float, float, float] = (0.025, 0.05, 0.05)
    reset_position: tuple[float, float, float] = (0.05, 0.05, 0.01)
    reset_rotation: tuple[float, float, float] = (0.1, 0.1, 0.2)
    reset_linvel: tuple[float, float, float] = (0.5, 0.5, 0.2)
    reset_angvel: tuple[float, float, float] = (0.52, 0.52, 0.78)
    reset_joint: float = 0.1
    reward_overrides: dict = {}

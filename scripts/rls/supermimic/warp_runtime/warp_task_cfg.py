"""Isaac-free task settings copied from the current G1 training task contract."""
from dataclasses import dataclass, field


@dataclass
class WarpTaskCfg:
    robot_name: str = "g1"
    num_envs: int = 4096
    env_spacing: float = 0.
    physics_dt: float = .005
    control_decimation: int = 4
    control_dt: float = .02
    max_episode_steps: int = 500
    reset_noise: bool = True
    observation_mode: str = "relative_ori99"
    observation_history: int = 10
    goal_frames: int = 10
    imagination_horizon: int = 32
    replay_buffer_steps: int = 1000
    world_model_window: int = 8
    world_model_updates_per_iteration: int = 10
    world_model_batch_size: int = 2048
    policy_batch_size: int = 4096
    friction_range: tuple[float, float] = (.3, 1.2)
    joint_offset_range: float = .01
    torso_com_offset_range: tuple[float, float, float] = (.025, .05, .05)
    reset_position: tuple[float, float, float] = (.05, .05, .01)
    reset_rotation: tuple[float, float, float] = (.1, .1, .2)
    reset_linvel: tuple[float, float, float] = (.5, .5, .2)
    reset_angvel: tuple[float, float, float] = (.52, .52, .78)
    reset_joint: float = .1
    reward_overrides: dict = field(default_factory=dict)
    # Physics backend choices are explicit, separate from learned-model settings.
    warp_nconmax: int = 128
    warp_njmax: int = 500
    warp_solver_iterations: int = 50
    warp_solver_ls_iterations: int = 50
    warp_use_graph: bool = True


def get_task_cfg(name="g1"):
    if name != "g1":
        raise ValueError("The initial isolated Warp migration supports G1 only")
    return WarpTaskCfg()

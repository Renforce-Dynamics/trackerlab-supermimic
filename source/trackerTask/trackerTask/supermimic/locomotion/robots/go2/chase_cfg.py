"""Go2 target following with moving high/low bars."""

from dataclasses import dataclass


@dataclass
class ChaseDodgeCfg:
    episode_steps: int = 400
    normal_height: float = 0.29
    bar_half_width: float = 3.0
    bar_half_thickness: float = 0.025
    curriculum_iterations: int = 5000
    max_speed: float = 1.5
    min_speed: float = 0.65
    target_distance: float = 0.8
    high_bar_height: float = 0.34
    low_bar_height: float = 0.12
    task_feature_dim: int = 18
    scenario_dim: int = 12
    max_step_reward: float = 0.1
    termination_height: float = 0.10
    termination_upright: float = 0.25


def make_task_cfg():
    from trackerTask.supermimic.tracking.quadruped.robots.go2.env_cfg import SuperMimicGo2TaskCfg

    cfg = SuperMimicGo2TaskCfg()
    cfg.max_episode_steps = ChaseDodgeCfg().episode_steps
    cfg.env_spacing = 8.0
    return cfg

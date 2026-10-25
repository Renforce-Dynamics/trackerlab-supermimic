"""Body-velocity commands and repeated, oriented bar encounters."""

from dataclasses import dataclass
import math
from typing import ClassVar


@dataclass(frozen=True)
class VelocityDodgeCfg:
    minimum_period: ClassVar[float] = 1.5
    enforce_period_clearance: ClassVar[bool] = False
    flat_warmup: ClassVar[bool] = True
    front_only: ClassVar[bool] = False
    max_bar_speed: ClassVar[float] = math.inf
    autonomous_passage: ClassVar[bool] = False
    control_dt: float = 0.02
    episode_steps: int = 800
    normal_height: float = 0.29
    bar_half_width: float = 3.0
    bar_half_thickness: float = 0.025
    respawn_clearance: float = 0.8
    curriculum_delay: int = 300
    curriculum_iterations: int = 6000
    initial_vx: float = 0.83
    vx_range: tuple[float, float] = (-0.6, 1.5)
    vy_max: float = 0.8
    yaw_rate_max: float = 1.5
    initial_bar_speed: tuple[float, float] = (0.6, 1.0)
    bar_speed_range: tuple[float, float] = (1.8, 3.6)
    initial_period: tuple[float, float] = (4.0, 5.0)
    period_range: tuple[float, float] = (2.2, 3.0)
    initial_high_bar_height: float = 0.65
    initial_low_bar_height: float = 0.04
    high_bar_range: tuple[float, float] = (0.31, 0.39)
    low_bar_range: tuple[float, float] = (0.08, 0.18)
    minimum_approach_speed: float = 1.0
    flat_fraction: float = 0.2
    jump_only: bool = False
    command_hold_encounters: int = 1
    lead_fraction: float = 0.45
    task_feature_dim: int = 24
    scenario_dim: int = 15
    termination_height: float = 0.10
    termination_upright: float = 0.25
    velocity_weight: float = 0.04
    yaw_rate_weight: float = 0.02
    upright_weight: float = 0.01
    height_weight: float = 0.015
    jump_clearance_weight: float = 0.0
    gait_style_weight: float = 0.005
    alive_weight: float = 0.01

    def __post_init__(self):
        for name in ("vx_range", "initial_bar_speed", "bar_speed_range",
                     "initial_period", "period_range", "high_bar_range", "low_bar_range"):
            values = getattr(self, name)
            if len(values) != 2 or not all(math.isfinite(x) for x in values) or values[0] > values[1]:
                raise ValueError(f"Invalid {name}")
        for name in ("control_dt", "normal_height", "bar_half_width", "bar_half_thickness",
                     "respawn_clearance", "minimum_approach_speed", "vy_max", "yaw_rate_max",
                     "initial_high_bar_height", "initial_low_bar_height"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.control_dt != 0.02:
            raise ValueError("The transferred dynamics checkpoint requires 50 Hz control")
        if self.episode_steps < 1 or self.curriculum_iterations < 1 or self.curriculum_delay < 0:
            raise ValueError("Invalid episode or curriculum duration")
        if self.task_feature_dim != 24 or self.scenario_dim != 15:
            raise ValueError("Velocity/dodge feature and scenario layouts are fixed")
        if not 0 <= self.flat_fraction < 1 or not 0 < self.termination_upright < 1:
            raise ValueError("Invalid flat fraction or termination tilt")
        if not 0 < self.termination_height < self.normal_height:
            raise ValueError("Termination height must be below normal height")
        if min(self.initial_bar_speed[0], self.bar_speed_range[0]) <= 0:
            raise ValueError("Bar speeds must be positive")
        if min(self.initial_period[0], self.period_range[0]) < self.minimum_period:
            raise ValueError(f"Encounter periods must be at least {self.minimum_period} seconds")
        if (isinstance(self.command_hold_encounters, bool) or
                not isinstance(self.command_hold_encounters, int) or self.command_hold_encounters < 1):
            raise ValueError("command_hold_encounters must be a positive integer")
        if not math.isfinite(self.lead_fraction) or not 0 < self.lead_fraction < 1:
            raise ValueError("lead_fraction must lie strictly between zero and one")
        if self.low_bar_range[0] <= self.bar_half_thickness:
            raise ValueError("Low bars must clear the ground")
        if self.low_bar_range[1] >= self.high_bar_range[0]:
            raise ValueError("Low and high bar height ranges must be disjoint")
        if self.initial_low_bar_height <= self.bar_half_thickness:
            raise ValueError("The initial low bar must clear the ground")
        if self.initial_high_bar_height <= self.initial_low_bar_height:
            raise ValueError("The initial high bar must be above the initial low bar")
        if not self.vx_range[0] <= self.initial_vx <= self.vx_range[1]:
            raise ValueError("Initial forward speed must lie in vx_range")
        if not all(math.isfinite(w) and w >= 0 for w in self.positive_weights):
            raise ValueError("Positive reward weights must be finite and nonnegative")
        if self.max_step_reward <= 0:
            raise ValueError("At least one positive reward weight is required")
        if self.enforce_period_clearance:
            # Every projected twist displacement is bounded by planar_speed*time.
            # Reject incompatible ranges rather than clipping emitted trajectories.
            max_vx = max(abs(self.initial_vx), *(abs(x) for x in self.vx_range))
            max_planar_speed = math.hypot(max_vx, self.vy_max)
            min_lead = self.lead_fraction * self.minimum_period
            min_tail = (1 - self.lead_fraction) * self.minimum_period
            speed_bound = max(
                self.initial_bar_speed[1],
                self.bar_speed_range[1],
                self.minimum_approach_speed + max_planar_speed,
                (self.respawn_clearance + 0.3) / min_lead + max_planar_speed,
                (self.respawn_clearance + 0.1) / min_tail + max_planar_speed,
            )
            if not math.isfinite(self.max_bar_speed) or speed_bound > self.max_bar_speed:
                raise ValueError(
                    f"Rapid clearance geometry requires up to {speed_bound:.3f} m/s, "
                    f"above max_bar_speed={self.max_bar_speed:.3f} m/s"
                )

    @property
    def positive_weights(self):
        return (self.velocity_weight, self.yaw_rate_weight, self.upright_weight,
                self.height_weight, self.jump_clearance_weight, self.gait_style_weight,
                self.alive_weight)

    @property
    def max_step_reward(self):
        return sum(self.positive_weights)

    def value_bound(self, gamma):
        if not 0 < gamma < 1:
            raise ValueError("gamma must lie between zero and one")
        return self.max_step_reward / (1 - gamma)

    def difficulty(self, iteration):
        return min(1.0, max(0.0, (iteration - self.curriculum_delay) / self.curriculum_iterations))


@dataclass(frozen=True)
class RapidDodgeCfg(VelocityDodgeCfg):
    """Short encounter periods with command holds and geometric speed floors.

    Speed ranges are base draws, not upper bounds. Ensuring 1.1 m spawn and
    0.9 m exit clearance at a 0.5 s period can raise speed to about 6-7 m/s.
    High/low bar centers start at 0.38/0.08 m, then progress to their endpoint
    ranges. Initial periods span 1.0-1.2 s before progressing to 0.5-1.0 s.
    The clear gate still defers a respawn if actual motion differs from the
    constant-body-twist prediction.
    """

    minimum_period: ClassVar[float] = 0.5
    enforce_period_clearance: ClassVar[bool] = True
    flat_warmup: ClassVar[bool] = False
    max_bar_speed: ClassVar[float] = 7.0
    curriculum_delay: int = 100
    curriculum_iterations: int = 4000
    initial_bar_speed: tuple[float, float] = (1.5, 2.5)
    bar_speed_range: tuple[float, float] = (3.0, 4.5)
    initial_period: tuple[float, float] = (1.0, 1.2)
    period_range: tuple[float, float] = (0.5, 1.0)
    initial_high_bar_height: float = 0.38
    initial_low_bar_height: float = 0.08
    flat_fraction: float = 0.0
    command_hold_encounters: int = 4


@dataclass(frozen=True)
class JumpDodgeCfg(VelocityDodgeCfg):
    """Low, slow, forward-only jump encounters with a clearance curriculum."""

    minimum_period: ClassVar[float] = 1.0
    flat_warmup: ClassVar[bool] = False
    front_only: ClassVar[bool] = True
    curriculum_delay: int = 0
    curriculum_iterations: int = 5000
    initial_vx: float = 0.3
    vx_range: tuple[float, float] = (-0.2, 0.8)
    vy_max: float = 0.25
    yaw_rate_max: float = 0.5
    initial_bar_speed: tuple[float, float] = (0.5, 0.8)
    bar_speed_range: tuple[float, float] = (0.8, 1.4)
    initial_period: tuple[float, float] = (2.5, 3.0)
    period_range: tuple[float, float] = (1.5, 2.0)
    initial_low_bar_height: float = 0.08
    low_bar_range: tuple[float, float] = (0.12, 0.18)
    minimum_approach_speed: float = 0.7
    flat_fraction: float = 0.0
    jump_only: bool = True
    command_hold_encounters: int = 1
    height_weight: float = 0.005
    jump_clearance_weight: float = 0.010


@dataclass(frozen=True)
class AdaptiveDodgeCfg(VelocityDodgeCfg):
    """Mixed-height bars; the policy chooses its own collision-free maneuver."""

    minimum_period: ClassVar[float] = 1.0
    enforce_period_clearance: ClassVar[bool] = True
    flat_warmup: ClassVar[bool] = False
    autonomous_passage: ClassVar[bool] = True
    max_bar_speed: ClassVar[float] = 3.0
    curriculum_delay: int = 0
    curriculum_iterations: int = 20000
    initial_vx: float = 0.3
    vx_range: tuple[float, float] = (0.0, 0.6)
    vy_max: float = 0.15
    yaw_rate_max: float = 0.35
    initial_bar_speed: tuple[float, float] = (1.0, 1.4)
    bar_speed_range: tuple[float, float] = (1.0, 1.8)
    initial_period: tuple[float, float] = (1.0, 2.0)
    period_range: tuple[float, float] = (1.0, 2.0)
    respawn_clearance: float = 0.55
    initial_high_bar_height: float = 0.43
    initial_low_bar_height: float = 0.045
    high_bar_range: tuple[float, float] = (0.30, 0.46)
    low_bar_range: tuple[float, float] = (0.04, 0.15)
    minimum_approach_speed: float = 0.8
    flat_fraction: float = 0.0
    command_hold_encounters: int = 4
    velocity_weight: float = 0.015
    yaw_rate_weight: float = 0.005
    upright_weight: float = 0.010
    height_weight: float = 0.002
    jump_clearance_weight: float = 0.060
    gait_style_weight: float = 0.003
    alive_weight: float = 0.005


def get_task_cfg(profile="standard"):
    """Return pure task signals configuration; simulation wiring is separate."""
    if profile == "standard":
        return VelocityDodgeCfg()
    if profile == "rapid":
        return RapidDodgeCfg()
    if profile == "jump":
        return JumpDodgeCfg()
    if profile == "adaptive":
        return AdaptiveDodgeCfg()
    raise ValueError(f"Unknown velocity/dodge profile {profile!r}")


def make_task_cfg():
    from trackerTask.supermimic.tracking.quadruped.robots.go2.env_cfg import SuperMimicGo2TaskCfg

    task = VelocityDodgeCfg()
    cfg = SuperMimicGo2TaskCfg()
    cfg.max_episode_steps = task.episode_steps
    cfg.env_spacing = 12.0
    return cfg

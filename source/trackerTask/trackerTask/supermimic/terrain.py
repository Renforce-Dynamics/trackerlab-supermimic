"""Seeded terrain profiles and robot contracts for SuperMimic locomotion."""

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from trackerTask.supermimic.robots import TRACKING_TASKS, get_robot


PROFILES = ("stairs", "mountain", "parkour")
COMMAND_PROFILES = ("velocity", "stealth")
ROBOTS = TRACKING_TASKS
ROBOT_HEIGHTS = {
    "g1": .76, "go2": .30, "a3": 1.09, "t1": .69, "bumi": .65, "pi_plus": .351,
}


@dataclass(frozen=True)
class TerrainTask:
    id: str
    robot: str
    profile: str

    @property
    def height(self):
        return ROBOT_HEIGHTS[self.robot]

    def describe(self):
        return {"id": self.id, "robot": self.robot, "profile": self.profile,
                "height_m": self.height, "algorithm": "supermimic",
                "objective": "command-driven-locomotion", "objective_version": 4,
                "scenario_dim": 7, "runtime_reference_motion": False,
                "terrain_navigation": self.profile in PROFILES}


TASKS = {
    f"locomotion/{robot}/{profile}": TerrainTask(
        f"locomotion/{robot}/{profile}", robot, profile,
    )
    for robot in ROBOTS for profile in PROFILES
}
COMMAND_TASKS = {
    f"locomotion/{robot}/velocity": TerrainTask(
        f"locomotion/{robot}/velocity", robot, "velocity",
    )
    for robot in ROBOTS
}
STEALTH_TASKS = {
    f"locomotion/{robot}/stealth": TerrainTask(
        f"locomotion/{robot}/stealth", robot, "stealth",
    )
    for robot in ROBOTS if robot != "go2"
}
ALL_TASKS = TASKS | COMMAND_TASKS | STEALTH_TASKS


def get_task(name):
    try:
        return ALL_TASKS[name]
    except KeyError as error:
        raise ValueError(f"Unknown SuperMimic locomotion task {name!r}; choices: {tuple(ALL_TASKS)}") from error


def robot_parameters(name):
    """Return the existing named-joint and motor contract for one robot."""
    spec = get_robot(name)
    height = ROBOT_HEIGHTS[name]
    if name == "g1":
        from trackerTask.supermimic.robot import motor_of, _kp, _kd
        motors = tuple((motor_of(joint)[0], motor_of(joint)[1],
                        _kp(motor_of(joint)[2]), _kd(motor_of(joint)[2]),
                        motor_of(joint)[2]) for joint in spec.joint_names)
    elif name == "go2":
        motors = ((23.5, 30., 25., .5, .01),) * spec.num_joints
    else:
        module = import_module(
            f"trackerTask.supermimic.tracking.humanoid.robots.{name}.robot")
        motors = tuple(module.motor(joint) for joint in spec.joint_names)
    return spec, height, motors


@dataclass
class Terrain:
    heights: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    centers: tuple[float, ...]
    goal_x: float
    seed: int
    profile: str

    @property
    def size(self):
        return ((self.x_max - self.x_min) / 2, (self.y_max - self.y_min) / 2,
                max(float(np.ptp(self.heights)), .01), .1)

    @property
    def position(self):
        return ((self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2,
                float(self.heights.min()))

    @property
    def normalized(self):
        return ((self.heights - self.heights.min()) / self.size[2]).astype(np.float32)


def generate(profile, height, seed=0):
    if profile not in (*PROFILES, *COMMAND_PROFILES):
        raise ValueError(f"Unknown terrain profile {profile!r}")
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Robot height must be finite and positive")
    scale = height / .76
    if profile in COMMAND_PROFILES:
        extent = 32. * scale
        return Terrain(np.zeros((129, 129), dtype=np.float32), -extent, extent,
                       -extent, extent, (0.,) * 6, 24. * scale, seed, profile)
    x = np.linspace(-2, 14, 129) * scale
    y = np.linspace(-12, 12, 193) * scale
    xx, yy = np.meshgrid(x / scale, y / scale)
    centers = np.arange(-10, 11, 4)
    out = np.zeros_like(xx)
    rng = np.random.default_rng(seed)
    for level, center in enumerate(centers):
        strength = level / (len(centers) - 1)
        lane = np.abs(yy - center) <= 2
        active_x = np.maximum(xx - 1.25, 0)
        if profile == "stairs":
            tread = rng.uniform(.42, .58)
            rise = rng.uniform(.11, .17)
            steps = rng.integers(7, 10)
            surface = np.minimum(np.floor(active_x / tread), steps) * rise * strength
        elif profile == "mountain":
            phase = rng.uniform(-np.pi, np.pi, 3)
            surface = strength * (
                .24 * np.sin(active_x * 1.1 + phase[0])
                + .10 * np.sin(active_x * 3.4 + (yy - center) * 2 + phase[1])
                + .045 * np.sin(active_x * 8.1 + phase[2])
                + .045 * active_x
            )
            surface *= np.clip(active_x / .7, 0, 1)
        else:
            spacing = rng.uniform(.68, .84)
            tread_width = rng.uniform(.48, .62) - .1 * strength
            phase = rng.uniform(0, spacing)
            index = np.floor((active_x + phase) / spacing).astype(int)
            tread = ((active_x + phase) % spacing) < tread_width
            block = .08 + .06 * (index % 3)
            surface = np.where(tread, block, -.22) * strength
            surface = np.where(active_x <= 0, 0, surface)
        out[lane] = surface[lane] * scale
    return Terrain(out.astype(np.float32), float(x[0]), float(x[-1]),
                   float(y[0]), float(y[-1]), tuple(centers * scale),
                   9. * scale, seed, profile)

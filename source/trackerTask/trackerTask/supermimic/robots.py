"""Immutable robot contracts shared by data preparation, learning and simulation."""

from dataclasses import dataclass
from importlib import import_module
from .registry import get_task


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]
    tracked_bodies: tuple[str, ...]
    root_body: str
    com_body: str
    mjcf_path: str
    default_q: tuple[float, ...]
    action_scale: tuple[float, ...]

    @property
    def num_joints(self):
        return len(self.joint_names)

    @property
    def state_dim(self):
        return 13 + 2 * self.num_joints

    @property
    def context_dim(self):
        return self.num_joints + 4

    @property
    def goal_frame_dim(self):
        return 15 + 2 * self.num_joints

    @property
    def critic_extra(self):
        return self.num_joints + 3 * len(self.tracked_bodies) + 9 + self.context_dim

    def make_articulation(self):
        if self.name == "g1":
            from .robot import make_g1_cfg
            return make_g1_cfg()
        if self.name == "go2":
            from .go2_robot import make_go2_cfg
            return make_go2_cfg()
        if self.name in ("a3", "t1"):
            module = import_module(f".tracking.humanoid.robots.{self.name}.robot", __package__)
            return module.make_cfg()
        raise ValueError(f"Unknown robot {self.name}")


TASKS = ("g1", "go2", "a3", "t1", "go2_chase_dodge", "go2_velocity_dodge")


def get_robot(name="g1"):
    name = get_task(name).robot
    if name == "g1":
        from .robot import G1_JOINT_NAMES, TRACKED_BODIES, MJCF_PATH, default_joint_pos_list, action_scale
        return RobotSpec(name, tuple(G1_JOINT_NAMES), tuple(TRACKED_BODIES),
                         "pelvis", "torso_link", MJCF_PATH, tuple(default_joint_pos_list()),
                         tuple(action_scale(n) for n in G1_JOINT_NAMES))
    if name == "go2":
        from .go2_robot import GO2_JOINT_NAMES, TRACKED_BODIES, MJCF_PATH, DEFAULT_Q
        return RobotSpec(name, tuple(GO2_JOINT_NAMES), tuple(TRACKED_BODIES),
                         "base", "base", MJCF_PATH, tuple(DEFAULT_Q), (0.25,) * 12)
    module = import_module(f".tracking.humanoid.robots.{name}.robot", __package__)
    root, com = ("pelvis_link", "torso_Link") if name == "a3" else ("trunk", "trunk")
    return RobotSpec(name, module.JOINT_NAMES, module.TRACKED_BODIES, root, com,
                     module.MJCF_PATH, module.DEFAULT_Q, module.ACTION_SCALE)


def get_task_cfg(name="g1"):
    return get_task(name).make_env_cfg()

"""Immutable robot contracts shared by data preparation, learning and simulation."""

from dataclasses import dataclass


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
        from .go2_robot import make_go2_cfg
        return make_go2_cfg()


TASKS = ("g1", "go2")


def get_robot(name="g1"):
    if name == "g1":
        from .robot import G1_JOINT_NAMES, TRACKED_BODIES, MJCF_PATH, default_joint_pos_list, action_scale
        return RobotSpec(name, tuple(G1_JOINT_NAMES), tuple(TRACKED_BODIES),
                         "pelvis", "torso_link", MJCF_PATH, tuple(default_joint_pos_list()),
                         tuple(action_scale(n) for n in G1_JOINT_NAMES))
    if name == "go2":
        from .go2_robot import GO2_JOINT_NAMES, TRACKED_BODIES, MJCF_PATH, DEFAULT_Q
        return RobotSpec(name, tuple(GO2_JOINT_NAMES), tuple(TRACKED_BODIES),
                         "base", "base", MJCF_PATH, tuple(DEFAULT_Q), (0.25,) * 12)
    raise ValueError(f"Unknown task {name!r}; expected one of {TASKS}")


def get_task_cfg(name="g1"):
    if name == "g1":
        from .g1_env_cfg import SuperMimicG1TaskCfg
        return SuperMimicG1TaskCfg()
    if name == "go2":
        from .go2_env_cfg import SuperMimicGo2TaskCfg
        return SuperMimicGo2TaskCfg()
    raise ValueError(f"Unknown task {name!r}; expected one of {TASKS}")

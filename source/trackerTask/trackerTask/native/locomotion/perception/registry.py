"""Simulator-free task discovery and robot-specific locomotion parameters."""

from dataclasses import asdict, dataclass
from importlib import import_module

from trackerTask.supermimic.robots import TRACKING_TASKS, get_robot

PROFILES = ("stairs", "mountain", "parkour")
ROBOTS = TRACKING_TASKS


@dataclass(frozen=True)
class PerceptionTask:
    id: str
    robot: str
    profile: str
    backend: str = "mujoco-warp"
    algorithm: str = "rsl-rl-ppo"
    env_entry: str = "trackerLab.tracker_env.perception.env:PerceptionEnv"

    def describe(self):
        return asdict(self)


TASKS = {
    f"locomotion/{robot}/{profile}": PerceptionTask(
        f"locomotion/{robot}/{profile}", robot, profile,
    )
    for robot in TRACKING_TASKS for profile in PROFILES
}


def get_task(name):
    try:
        return TASKS[name]
    except KeyError as error:
        raise ValueError(f"Unknown perception task {name!r}; choices: {tuple(TASKS)}") from error


def robot_parameters(name):
    """Reuse each existing robot's named joints, geometry and motor settings."""
    spec = get_robot(name)
    height = {"g1": .76, "go2": .30, "a3": 1.09, "t1": .69,
              "bumi": .65, "pi_plus": .351}[name]
    if name == "g1":
        from trackerTask.supermimic.robot import motor_of, _kp, _kd
        motors = tuple((motor_of(n)[0], motor_of(n)[1], _kp(motor_of(n)[2]),
                        _kd(motor_of(n)[2]), motor_of(n)[2]) for n in spec.joint_names)
    elif name == "go2":
        motors = ((23.5, 30., 25., .5, .01),) * spec.num_joints
    else:
        module = import_module(f"trackerTask.supermimic.tracking.humanoid.robots.{name}.robot")
        motors = tuple(module.motor(n) for n in spec.joint_names)
    return spec, height, motors


def ppo_config():
    return {
        "num_steps_per_env": 24, "save_interval": 500,
        "empirical_normalization": True, "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic", "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": .6, "noise_std_type": "log",
        },
        "algorithm": {
            "class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4,
            "clip_param": .2, "gamma": .99, "lam": .95,
            "value_loss_coef": 1., "entropy_coef": .01, "learning_rate": 3e-4,
            "max_grad_norm": 1., "use_clipped_value_loss": True,
            "schedule": "adaptive", "desired_kl": .01,
        },
    }

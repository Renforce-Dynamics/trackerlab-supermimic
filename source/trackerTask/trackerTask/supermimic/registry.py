"""Simulator-free task discovery, following TrackerLab's category/robot layout."""

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path


@dataclass(frozen=True)
class TaskEntry:
    id: str
    robot: str
    runtime_task: str
    env_entry: str
    preset: str
    motion_config: str
    profile: str = "standard"

    def make_env_cfg(self):
        module, name = self.env_entry.split(":")
        factory = getattr(import_module(module), name)
        if self.runtime_task in ("terrain", "command"):
            return factory(self.id, self.robot, self.profile)
        return factory()

    def describe(self):
        value = asdict(self)
        if self.runtime_task in ("terrain", "command"):
            value.update(objective="command-driven-locomotion", objective_version=3,
                         scenario_dim=7, runtime_reference_motion=False)
        return value


_ROOT = "trackerTask.supermimic"
_HUMANOID = _ROOT + ".tracking.humanoid.robots"
_QUADRUPED = _ROOT + ".tracking.quadruped.robots"
_LOCOMOTION = _ROOT + ".locomotion.robots.go2"
_TASK_DIR = Path(__file__).resolve().parent


def _preset(relative):
    return str(_TASK_DIR / relative / "agents/supermimic.json")


_ENTRIES = (
    TaskEntry("tracking/g1", "g1", "g1",
              _HUMANOID + ".g1.env_cfg:SuperMimicG1TaskCfg",
              "scripts/rls/supermimic/configs/lafan1_8096.json", "data/configs/supermimic_lafan1.json"),
    TaskEntry("tracking/go2", "go2", "go2",
              _QUADRUPED + ".go2.env_cfg:SuperMimicGo2TaskCfg",
              "scripts/rls/supermimic/configs/go2_gait_8096.json", "data/configs/go2_kine2go.json"),
    TaskEntry("tracking/a3", "a3", "a3",
              _HUMANOID + ".a3.env_cfg:SuperMimicA3TaskCfg",
              _preset("tracking/humanoid/robots/a3"), "data/configs/a3_sonic.json"),
    TaskEntry("tracking/t1", "t1", "t1",
              _HUMANOID + ".t1.env_cfg:SuperMimicT1TaskCfg",
              _preset("tracking/humanoid/robots/t1"), "data/configs/t1_dance.json"),
    TaskEntry("tracking/bumi", "bumi", "bumi",
              _HUMANOID + ".bumi.env_cfg:SuperMimicBumiTaskCfg",
              _preset("tracking/humanoid/robots/bumi"), "data/configs/bumi_vendor.json"),
    TaskEntry("tracking/pi_plus", "pi_plus", "pi_plus",
              _HUMANOID + ".pi_plus.env_cfg:SuperMimicPiPlusTaskCfg",
              _preset("tracking/humanoid/robots/pi_plus"), "data/configs/pi_plus_vendor.json"),
    TaskEntry("locomotion/go2/chase", "go2", "go2_chase_dodge",
              _LOCOMOTION + ".env_cfg:make_chase_cfg",
              "scripts/rls/supermimic/configs/go2_chase_dodge_8096.json", "data/configs/go2_kine2go.json"),
    *(TaskEntry(f"locomotion/go2/{profile}", "go2", "go2_velocity_dodge",
               _LOCOMOTION + ".env_cfg:make_velocity_cfg",
               f"scripts/rls/supermimic/configs/go2_{filename}_8096.json",
               "data/configs/go2_kine2go.json", profile)
      for profile, filename in (("standard", "velocity_dodge"), ("rapid", "rapid_dodge"),
                                ("jump", "jump_dodge"), ("adaptive", "adaptive_dodge"))),
    *(TaskEntry(f"locomotion/{robot}/{profile}", robot, "terrain",
               _ROOT + ".locomotion.command_cfg:CommandTaskCfg",
               f"scripts/rls/supermimic/configs/terrain_{profile}.json", "procedural heightfield", profile)
      for robot in ("g1", "go2", "a3", "t1", "bumi", "pi_plus")
      for profile in ("stairs", "mountain", "parkour")),
    *(TaskEntry(f"locomotion/{robot}/{profile}", robot, "command",
               _ROOT + ".locomotion.command_cfg:CommandTaskCfg",
               f"scripts/rls/supermimic/configs/command_{profile}.json",
               "none (command locomotion)", profile)
      for robot in ("g1", "go2", "a3", "t1", "bumi", "pi_plus")
      for profile in ("velocity", "stealth") if profile != "stealth" or robot != "go2"),
    *(TaskEntry(f"gait/{robot}", robot, "gait-foundation",
               (_QUADRUPED + ".go2.env_cfg:SuperMimicGo2TaskCfg" if robot == "go2" else
                _HUMANOID + f".{robot}.env_cfg:SuperMimic{robot.title().replace('_', '')}TaskCfg"),
               "scripts/rls/supermimic/configs/gait_foundation.json",
               "audited robot-specific locomotion references",
               "foundation")
      for robot in ("g1", "go2", "a3", "t1", "bumi", "pi_plus")),
)
TASK_REGISTRY = {entry.id: entry for entry in _ENTRIES}
ALIASES = {"g1": "tracking/g1", "go2": "tracking/go2", "a3": "tracking/a3",
           "t1": "tracking/t1", "bumi": "tracking/bumi", "pi_plus": "tracking/pi_plus",
           "pi+": "tracking/pi_plus", "go2_chase_dodge": "locomotion/go2/chase",
           "go2_velocity_dodge": "locomotion/go2/standard"}


def get_task(name):
    try:
        return TASK_REGISTRY[ALIASES.get(name, name)]
    except KeyError as error:
        raise ValueError(f"Unknown task {name!r}; choices: {tuple(TASK_REGISTRY)}") from error

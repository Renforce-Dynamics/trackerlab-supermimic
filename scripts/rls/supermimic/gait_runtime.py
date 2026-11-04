"""Multi-robot flat-ground Warp simulation for reference gait foundations."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from bootstrap import configure_paths
REPO = configure_paths()
sys.path.insert(0, str(Path(__file__).with_name("warp_runtime")))
from warp_env import TrackingSim as _TrackingSim, launch_simulation, make_sim
from warp_task_cfg import WarpTaskCfg
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerTask.supermimic.terrain import ROBOT_HEIGHTS, generate


def task_settings(robot):
    scale = ROBOT_HEIGHTS[robot] / .76
    return WarpTaskCfg(
        robot_name=robot, observation_mode="relative_orientation", max_episode_steps=1000,
        reset_position=(.015 * scale, .015 * scale, .005 * scale),
        reset_rotation=(.04, .04, .08), reset_linvel=(.12 * scale, .12 * scale, .08 * scale),
        reset_angvel=(.15, .15, .2), reset_joint=.04,
        reward_overrides={"body_pos_std": .15 * scale, "root_pos_std": .2 * scale,
                          "body_linvel_std": .7 * scale, "term_height": .2 * scale,
                          "term_tilt": .5},
    )


def prepare_model(spec, cfg):
    terrain = generate("stairs", ROBOT_HEIGHTS[spec.name], 0)
    terrain = replace(terrain, heights=np.zeros_like(terrain.heights))
    model, _, _, qpos, qvel, _, com, actuators, contract = build_model(
        spec.name, terrain, cfg.warp_solver_iterations, ground_plane=True)
    model.opt.ls_iterations = cfg.warp_solver_ls_iterations
    joints = np.array([model.joint(name).id for name in spec.joint_names])
    contract.update(training_stage="reference-gait-foundation", flat_ground=True)
    return model, contract, joints, qpos, qvel, actuators, com


class TrackingSim(_TrackingSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, model_factory=prepare_model)

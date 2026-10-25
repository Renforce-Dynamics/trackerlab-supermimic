"""Shared import path for robot-only MJCFs prepared with recorded provenance."""

import os
from pathlib import Path


def asset_dir(robot, vendor):
    repo = Path(os.environ.get("SUPERMIMIC_REPO_ROOT", Path(__file__).resolve().parents[4]))
    root = Path(os.environ.get("SUPERMIMIC_ASSETLIB_ROOT", repo / "data/assetslib"))
    return Path(os.environ.get(f"SUPERMIMIC_{robot.upper()}_ASSET_DIR", root / vendor / robot))


def make_articulation(name, directory, joint_names, default_q, motors, root_height):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaacsim.core.utils.extensions import enable_extension

    if not enable_extension("isaacsim.asset.importer.mjcf"):
        raise RuntimeError("Isaac Sim MJCF importer could not be enabled")

    path = directory / "mjcf" / f"{name}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"{path}; run prepare_resources.py --task {name}")
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from pxr import Usd, UsdPhysics

    converted = MjcfConverter(MjcfConverterCfg(
        asset_path=str(path), fix_base=False, import_inertia_tensor=True,
        self_collision=False, make_instanceable=True,
        usd_dir=str(directory / "usd"), usd_file_name=f"{name}.usd",
    ))
    stage = Usd.Stage.Open(converted.usd_path)
    # Isaac Sim 5.1 adds an empty worldBody articulation beside the floating robot.
    for empty_world in [p for p in stage.Traverse() if p.GetName() == "worldBody"]:
        if list(empty_world.GetChildren()) or empty_world.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Importer worldBody contains physics; automatic removal is unsafe")
        empty_world.SetActive(False)
        stage.GetRootLayer().Save()
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise ValueError(f"Expected one imported articulation, got {len(roots)}")
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=converted.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False,
                linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8, solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, root_height), joint_pos=dict(zip(joint_names, default_q)),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            joint: ImplicitActuatorCfg(
                joint_names_expr=[joint], effort_limit_sim=effort, velocity_limit_sim=velocity,
                stiffness=kp, damping=kd, armature=armature,
            )
            for joint, (effort, velocity, kp, kd, armature) in zip(joint_names, motors)
        },
    )

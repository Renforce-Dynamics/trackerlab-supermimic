"""Go2 resources and IsaacLab's Go2 DC motor settings (IsaacLab v2.3.2)."""

import os
from pathlib import Path

REPO = Path(os.environ.get("SUPERMIMIC_REPO_ROOT", Path(__file__).resolve().parents[8]))
ASSET_DIR = Path(os.environ.get("SUPERMIMIC_GO2_ASSET_DIR", Path(
    os.environ.get("SUPERMIMIC_ASSETLIB_ROOT", REPO / "data/assetslib")
) / "unitree/unitree_go2"))
MJCF_PATH = str(ASSET_DIR / "mjcf/go2.xml")
USD_PATH = str(ASSET_DIR / "usd/go2.usd")
LEGS = ("FL", "FR", "RL", "RR")
GO2_JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")]
TRACKED_BODIES = ["base", *(f"{leg}_{body}" for leg in LEGS for body in ("hip", "thigh", "calf"))]
DEFAULT_Q = [0.0, 0.9, -1.8] * 4
FOOT_OFFSET = (-0.002, 0.0, -0.213)
FOOT_RADIUS = 0.022


def make_go2_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import DCMotorCfg

    if not Path(USD_PATH).is_file():
        raise FileNotFoundError(f"Go2 USD missing: {USD_PATH}; run prepare_resources.py --task go2")
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH, activate_contact_sensors=True,
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
            pos=(0.0, 0.0, 0.30), joint_pos=dict(zip(GO2_JOINT_NAMES, DEFAULT_Q)),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={"legs": DCMotorCfg(
            joint_names_expr=[".*_joint"], effort_limit=23.5, saturation_effort=23.5,
            velocity_limit=30.0, stiffness=25.0, damping=0.5, friction=0.0,
        )},
    )

"""Build robot MJCF models with shared SuperMimic terrain and motor settings."""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from trackerTask.supermimic.terrain import robot_parameters


def build_model(robot, terrain, solver_iterations=50, *, render_size=None, ground_plane=False,
                self_collisions=False):
    import mujoco

    if not isinstance(self_collisions, bool):
        raise ValueError("self_collisions must be a boolean")
    spec, height, motors = robot_parameters(robot)
    path = Path(spec.mjcf_path).resolve()
    document = ET.parse(path).getroot()
    compiler = document.find("compiler")
    directories = {}
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            directories[key] = (path.parent / compiler.get(key, ".")).resolve()
            compiler.attrib.pop(key, None)
    for section in ("actuator", "sensor", "keyframe", "contact", "equality"):
        for node in document.findall(section):
            document.remove(node)
    world = document.find("worldbody")
    for node in list(world):
        if node.tag != "body":
            world.remove(node)
    for parent in document.iter():
        for child in list(parent):
            if child.tag in ("light", "camera"):
                parent.remove(child)
    asset = document.find("asset")
    if asset is None:
        asset = ET.SubElement(document, "asset")
    virtual_files = {}
    for index, node in enumerate(asset):
        filename = node.get("file")
        if filename:
            if node.tag == "mesh" and not node.get("name"):
                node.set("name", Path(filename).stem)
            folder = directories.get("meshdir" if node.tag == "mesh" else "texturedir", path.parent)
            original = (folder / filename).resolve()
            safe_name = f"terrain_robot_asset_{index}" + original.suffix
            virtual_files[safe_name] = original.read_bytes()
            node.set("file", safe_name)
    ET.SubElement(asset, "hfield", {
        "name": "supermimic_terrain", "nrow": str(terrain.heights.shape[0]),
        "ncol": str(terrain.heights.shape[1]), "size": " ".join(map(str, terrain.size)),
    })
    ground = ({"type": "plane", "size": "0 0 .1", "pos": "0 0 0"} if ground_plane
              else {"type": "hfield", "hfield": "supermimic_terrain",
                    "pos": " ".join(map(str, terrain.position))})
    ET.SubElement(world, "geom", {
        "name": "supermimic_ground", **ground, "group": "5",
        "rgba": ".32 .43 .33 1", "contype": "1", "conaffinity": "2",
        "friction": "1 .005 .0001", "condim": "3", "priority": "1",
        "solref": ".04 1", "solimp": ".9 .95 .001 .5 2",
    })
    if render_size is not None:
        width, render_height = render_size
        visual = document.find("visual")
        if visual is None:
            visual = ET.SubElement(document, "visual")
        settings = visual.find("global")
        if settings is None:
            settings = ET.SubElement(visual, "global")
        settings.set("offwidth", str(width))
        settings.set("offheight", str(render_height))
        ET.SubElement(asset, "texture", {
            "name": "terrain_demo_sky", "type": "skybox", "builtin": "gradient",
            "rgb1": ".72 .83 .90", "rgb2": ".94 .97 .98", "width": "256", "height": "1536",
        })
        ET.SubElement(asset, "texture", {
            "name": "terrain_demo_grid", "type": "2d", "builtin": "checker",
            "rgb1": ".53 .64 .53", "rgb2": ".64 .73 .62", "width": "512", "height": "512",
        })
        ET.SubElement(asset, "material", {
            "name": "terrain_demo_ground", "texture": "terrain_demo_grid",
            "texrepeat": "12 18", "texuniform": "false", "reflectance": "0",
        })
        world.find("geom[@name='supermimic_ground']").set("material", "terrain_demo_ground")
        ET.SubElement(world, "light", {
            "pos": "0 0 8", "dir": ".1 .2 -1", "directional": "true",
            "diffuse": ".65 .65 .65", "ambient": ".25 .25 .25",
        })
    actuators = ET.SubElement(document, "actuator")
    for name, (effort, _velocity, kp, kd, _armature) in zip(spec.joint_names, motors):
        ET.SubElement(actuators, "general", {
            "name": "supermimic_" + name, "joint": name, "dyntype": "none",
            "gaintype": "fixed", "biastype": "affine", "gainprm": str(kp),
            "biasprm": f"0 {-kp} {-kd}", "gear": "1",
            "forcelimited": "true", "forcerange": f"{-effort} {effort}",
            "ctrllimited": "false",
        })
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(document, encoding="unicode"), virtual_files)
    if (model.nq, model.nv, model.nu) != (
            spec.num_joints + 7, spec.num_joints + 6, spec.num_joints):
        raise ValueError(f"{robot}: expected one free root and named policy actuators")
    joint_ids = np.array([model.joint(name).id for name in spec.joint_names])
    qpos_ids = model.jnt_qposadr[joint_ids].copy()
    qvel_ids = model.jnt_dofadr[joint_ids].copy()
    root_id = model.body(spec.root_body).id
    com_id = model.body(spec.com_body).id
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if (len(free) != 1 or model.jnt_bodyid[free[0]] != root_id
            or not np.all(model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError("Robot root or controlled-joint layout differs from its contract")
    actuator_ids = []
    for joint_id in joint_ids:
        candidates = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(candidates) != 1:
            raise ValueError("Each policy joint must have exactly one actuator")
        actuator_ids.append(int(candidates[0]))
    for dof_id, motor in zip(qvel_ids, motors):
        model.dof_armature[dof_id], model.dof_damping[dof_id], model.dof_frictionloss[dof_id] = motor[4], 0, 0
    physical = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    robot_geoms = physical & (model.geom_bodyid != 0)
    model.geom_contype[robot_geoms] = 2
    model.geom_conaffinity[robot_geoms] = 3 if self_collisions else 1
    model.geom_group[model.geom_bodyid != 0] = 3
    model.opt.timestep = .005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.impratio = 1
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = solver_iterations
    model.opt.ls_iterations = 50
    model.hfield_data[:] = terrain.normalized.ravel()
    mujoco.mj_setConst(model, mujoco.MjData(model))
    contract = {
        "backend": "mujoco-warp", "algorithm": "supermimic-world-model-imagination-critic",
        "robot": robot, "source_mjcf": str(path),
        "source_mjcf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "joint_names": list(spec.joint_names), "default_q": list(spec.default_q),
        "action_scale": list(spec.action_scale), "motors_effort_velocity_kp_kd_armature": motors,
        "height": height, "physics_dt": .005, "control_substeps": 4,
        "solver_iterations": solver_iterations, "solver_ls_iterations": 50,
        "self_collisions": self_collisions,
        "terrain": {
            "profile": terrain.profile, "seed": terrain.seed,
            "height_sha256": hashlib.sha256(terrain.heights.tobytes()).hexdigest(),
            "shape": list(terrain.heights.shape), "size": terrain.size,
            "position": terrain.position, "centers": terrain.centers, "goal_x": terrain.goal_x,
        },
        "ground_plane": ground_plane,
    }
    return model, spec, height, qpos_ids, qvel_ids, root_id, com_id, actuator_ids, contract

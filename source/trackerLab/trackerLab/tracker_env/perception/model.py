"""MuJoCo adaptation of the repository's existing six robot motor contracts."""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from trackerTask.native.locomotion.perception.registry import robot_parameters


def build_model(robot, terrain, solver_iterations=50):
    import mujoco

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
    # Virtual files also support Windows workspaces with non-ASCII path components.
    virtual_files = {}
    for index, node in enumerate(asset):
        filename = node.get("file")
        if filename:
            if node.tag == "mesh" and not node.get("name"):
                node.set("name", Path(filename).stem)
            folder = directories.get("meshdir" if node.tag == "mesh" else "texturedir", path.parent)
            original = (folder / filename).resolve()
            safe_name = f"perception_asset_{index}" + original.suffix
            virtual_files[safe_name] = original.read_bytes()
            node.set("file", safe_name)
    ET.SubElement(asset, "hfield", {
        "name": "perception_terrain", "nrow": str(terrain.heights.shape[0]),
        "ncol": str(terrain.heights.shape[1]), "size": " ".join(map(str, terrain.size)),
    })
    ET.SubElement(world, "geom", {
        "name": "perception_ground", "type": "hfield", "hfield": "perception_terrain",
        "pos": " ".join(map(str, terrain.position)), "group": "5",
        "rgba": ".32 .43 .33 1", "contype": "1", "conaffinity": "2",
        "friction": "1 .005 .0001", "condim": "3", "priority": "1",
        "solref": ".04 1", "solimp": ".9 .95 .001 .5 2",
    })
    actuators = ET.SubElement(document, "actuator")
    for name, (effort, velocity, kp, kd, armature) in zip(spec.joint_names, motors):
        ET.SubElement(actuators, "general", {
            "name": "perception_" + name, "joint": name, "dyntype": "none",
            "gaintype": "fixed", "biastype": "affine", "gainprm": str(kp),
            "biasprm": f"0 {-kp} {-kd}", "gear": "1",
            "forcelimited": "true", "forcerange": f"{-effort} {effort}",
            "ctrllimited": "false",
        })
    model = mujoco.MjModel.from_xml_string(ET.tostring(document, encoding="unicode"), virtual_files)
    if (model.nq, model.nv, model.nu) != (spec.num_joints + 7, spec.num_joints + 6, spec.num_joints):
        raise ValueError(f"{robot}: expected exactly one free root and the named controlled joints")
    jids = np.array([model.joint(n).id for n in spec.joint_names])
    qids, vids = model.jnt_qposadr[jids].copy(), model.jnt_dofadr[jids].copy()
    root_id = model.body(spec.root_body).id
    if (model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[0] != root_id
            or not np.all(model.jnt_type[jids] == mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError("Robot root or joint layout differs from its contract")
    for i, motor in zip(vids, motors):
        model.dof_armature[i], model.dof_damping[i], model.dof_frictionloss[i] = motor[4], 0, 0
    physical = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    robot_geoms = physical & (model.geom_bodyid != 0)
    model.geom_contype[robot_geoms], model.geom_conaffinity[robot_geoms] = 2, 1
    model.geom_group[model.geom_bodyid != 0] = 3
    model.opt.timestep = .005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.impratio = 1
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = solver_iterations
    model.opt.ls_iterations = 50
    model.hfield_data[:] = terrain.normalized.ravel()
    # Hinge qpos0 defines the geometric reference, not a reset/home pose.
    mujoco.mj_setConst(model, mujoco.MjData(model))
    contract = {
        "robot": robot, "source_mjcf": str(path),
        "source_mjcf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "joint_names": list(spec.joint_names), "default_q": list(spec.default_q),
        "action_scale": list(spec.action_scale), "motors_effort_velocity_kp_kd_armature": motors,
        "height": height, "physics_dt": .005, "control_substeps": 4,
        "solver_iterations": solver_iterations, "solver_ls_iterations": 50,
        "self_collisions": False,
        "velocity_limit_note": "MuJoCo force limits are enforced; joint velocities are not artificially clipped.",
        "terrain": {
            "profile": terrain.profile, "seed": terrain.seed, "shape": list(terrain.heights.shape),
            "height_sha256": hashlib.sha256(terrain.heights.tobytes()).hexdigest(),
            "size": terrain.size, "position": terrain.position, "centers": terrain.centers,
            "goal_x": terrain.goal_x,
        },
    }
    return model, spec, height, qids, vids, root_id, contract

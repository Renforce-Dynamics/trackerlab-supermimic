"""Direct MuJoCo-Warp backend for the existing TrackerLab learning stack.

No Isaac/Kit dependency. Existing motion manager, analytic MDP, state layout,
transition dictionary, observation builders and checkpoints are reused.
Physics changes from PhysX to MuJoCo; model-weight compatibility does not imply
identical transition distributions. Resume requires a fresh replay buffer.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from warp_task_cfg import get_task_cfg
from trackerTask.supermimic.robots import get_robot
from trackerLab.managers.supermimic import SuperMimicMotionManager
from trackerLab.tracker_env.mdp.supermimic import MAX_STEP_REWARD, flatten_state
from trackerLab.kinematics.supermimic.rotations import mat_to_quat, quat_to_mat, so3_exp

PHYSICS_DT = .005
CONTROL_DECIMATION = 4


@dataclass
class WarpSimulationCfg:
    device: str = "cuda:0"
    dt: float = PHYSICS_DT
    render_interval: int = CONTROL_DECIMATION
    log_dir: str | None = None


def make_sim(device, args=None, task_cfg=None):
    cfg = task_cfg or get_task_cfg()
    if args is not None:
        # These parser options must affect allocation or fail explicitly.
        # An arbitrary replacement robot asset would invalidate the current
        # kinematics/motor contract, so it remains unsupported.
        if getattr(args, "warp_model", None) is not None:
            raise ValueError("--warp_model is unsupported; use the RobotSpec asset")
        for key in ("warp_nconmax", "warp_njmax",
                    "warp_solver_iterations", "warp_solver_ls_iterations"):
            if hasattr(args, key):
                value = getattr(args, key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
                setattr(cfg, key, value)
    if (cfg.physics_dt, cfg.control_decimation) != (.005, 4):
        raise ValueError("Warp migration fixes 50 Hz control with four 5 ms substeps")
    return WarpSimulationCfg(device, cfg.physics_dt, cfg.control_decimation,
                             getattr(args, "log_dir", None))


def add_launcher_args(parser):
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)


@contextmanager
def launch_simulation(sim_cfg, args=None, app_launcher=None):
    if app_launcher is not None:
        raise ValueError("Standalone Warp cannot share an Isaac/Kit launcher")
    yield sim_cfg


def prepare_model(spec, task_cfg):
    """Use our existing G1 MJCF and current motor contract, with a ground plane."""
    import mujoco
    from trackerTask.supermimic.robot import motor_of, _kp, _kd

    path = Path(spec.mjcf_path).resolve()
    document = ET.fromstring(path.read_text())
    compiler = document.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            if key in compiler.attrib:
                compiler.set(key, str((path.parent / compiler.get(key)).resolve()))
    world = document.find("worldbody")
    if world is None:
        raise ValueError("G1 asset lacks a worldbody")
    # Assets retain their collision geometry and inertias. Remove unused sensors
    # and lights; no learner consumes sensor arrays or renderer outputs.
    sensor = document.find("sensor")
    if sensor is not None:
        document.remove(sensor)
    for parent in document.iter():
        for child in list(parent):
            if child.tag in ("light", "camera"):
                parent.remove(child)
    if any(geom.get("type") == "plane" for geom in world.findall("geom")):
        raise ValueError("Expected the robot-only source MJCF, not a second scene")
    ET.SubElement(world, "geom", {
        "name": "warp_floor", "type": "plane", "size": "0 0 .1",
        "pos": "0 0 0", "friction": "1 .005 .0001", "condim": "3",
        "contype": "1", "conaffinity": "2",
    })
    model = mujoco.MjModel.from_xml_string(ET.tostring(document, encoding="unicode"))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Expected G1 free root plus 29 hinge joints and actuators")
    joint_ids = np.array([model.joint(n).id for n in spec.joint_names])
    if not np.all(model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Policy joint names do not identify hinge joints")
    qpos_ids = model.jnt_qposadr[joint_ids].copy()
    qvel_ids = model.jnt_dofadr[joint_ids].copy()
    root_id = model.body(spec.root_body).id
    com_id = model.body(spec.com_body).id
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if (len(free) != 1 or model.jnt_bodyid[free[0]] != root_id
            or model.jnt_qposadr[free[0]] != 0 or model.jnt_dofadr[free[0]] != 0):
        raise ValueError("Root state does not match the checkpoint free-root layout")
    actuator_ids = []
    for joint_id in joint_ids:
        candidates = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(candidates) != 1:
            raise ValueError("Expected exactly one actuator per policy joint")
        actuator_ids.append(int(candidates[0]))
    effort, arm, kp, kd, velocity_limit = [], [], [], [], []
    for name in spec.joint_names:
        torque, velocity, inertia = motor_of(name)
        effort.append(torque)
        velocity_limit.append(velocity)
        arm.append(inertia)
        kp.append(_kp(inertia))
        kd.append(_kd(inertia))
    effort, arm, kp, kd = map(np.asarray, (effort, arm, kp, kd))
    model.opt.timestep = .005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = task_cfg.warp_solver_iterations
    model.opt.ls_iterations = task_cfg.warp_solver_ls_iterations
    # Disable self collisions, as in our current G1 articulation.
    physical = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    robot_geoms = physical & (model.geom_bodyid != 0)
    model.geom_contype[robot_geoms] = 2
    model.geom_conaffinity[robot_geoms] = 1
    model.geom_priority[physical] = 0
    model.geom_friction[physical, 0] = 1.
    model.dof_armature[qvel_ids] = arm
    model.dof_damping[qvel_ids] = 0.
    model.dof_frictionloss[qvel_ids] = 0.
    model.jnt_actfrclimited[joint_ids] = True
    model.jnt_actfrcrange[joint_ids] = np.stack((-effort, effort), -1)
    for index, actuator in enumerate(actuator_ids):
        model.actuator_dyntype[actuator] = mujoco.mjtDyn.mjDYN_NONE
        model.actuator_gaintype[actuator] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[actuator] = 0.
        model.actuator_gainprm[actuator, 0] = kp[index]
        model.actuator_biasprm[actuator] = 0.
        model.actuator_biasprm[actuator, 1:3] = [-kp[index], -kd[index]]
        model.actuator_gear[actuator] = [1., 0., 0., 0., 0., 0.]
        model.actuator_ctrllimited[actuator] = False
        model.actuator_forcelimited[actuator] = True
        model.actuator_forcerange[actuator] = [-effort[index], effort[index]]
    # Recompute fixed fields after motor armature changes before device transfer.
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    contract = {
        "backend": "direct-mujoco-warp",
        "source_mjcf": str(path),
        "source_mjcf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "joint_names": list(spec.joint_names), "root_body": spec.root_body,
        "com_body": spec.com_body, "qpos_indices": qpos_ids.tolist(),
        "qvel_indices": qvel_ids.tolist(), "actuator_indices": actuator_ids,
        "default_q": list(spec.default_q), "action_scale": list(spec.action_scale),
        "kp": kp.tolist(), "kd": kd.tolist(), "armature": arm.tolist(),
        "effort_limit": effort.tolist(), "training_velocity_limit": velocity_limit,
        "velocity_limit_difference": "MuJoCo has no PhysX hard joint-velocity constraint; no artificial qvel clipping applied",
        "actuation": "MuJoCo affine position servo with implicitfast damping and bounded force",
        "self_collisions": False, "physics_dt": .005, "control_substeps": 4,
        "physics_transition_equal_to_physx_claim": False,
    }
    return model, contract, joint_ids, qpos_ids, qvel_ids, actuator_ids, com_id


class TrackingSim:
    """Existing TrackerLab transition API, backed by independent Warp worlds."""

    def __init__(self, sim_cfg, num_envs, motion, mdp, gamma=.99,
                 eval_mode=False, task_cfg=None, env_spacing=None,
                 friction_range=None, joint_offset_range=None, com_offset_range=None):
        import mujoco_warp as mw
        import warp as wp

        if not torch.cuda.is_available() or not str(sim_cfg.device).startswith("cuda"):
            raise ValueError("Direct MuJoCo Warp simulation requires CUDA")
        if not 0. < gamma < 1. or num_envs < 1:
            raise ValueError("Invalid gamma or environment count")
        self.mw, self.wp = mw, wp
        self.device = str(torch.device(sim_cfg.device))
        self.num_envs = num_envs
        self.task_cfg = task_cfg or get_task_cfg()
        self.spec = get_robot(self.task_cfg.robot_name)
        self.num_joints = self.spec.num_joints
        if motion.num_joints != self.num_joints or mdp.num_joints != self.num_joints:
            raise ValueError("Task, motion and MDP joint counts differ")
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.physics_dt, self.control_decimation = sim_cfg.dt, sim_cfg.render_interval
        if (self.physics_dt, self.control_decimation) != (.005, 4):
            raise ValueError("Unexpected simulation clock")
        self.eval_mode = eval_mode
        self.reset_noise = self.task_cfg.reset_noise and not eval_mode
        self.max_episode_steps = 2**31 - 1 if eval_mode else self.task_cfg.max_episode_steps
        self.friction_range = friction_range or self.task_cfg.friction_range
        self.joint_offset_range = (self.task_cfg.joint_offset_range if joint_offset_range is None
                                   else joint_offset_range)
        self.com_offset_range = torch.tensor(
            com_offset_range or self.task_cfg.torso_com_offset_range, device=self.device)
        self.default_q = torch.tensor(self.spec.default_q, device=self.device)
        self.action_scale = torch.tensor(self.spec.action_scale, device=self.device)
        self.env_origins = torch.zeros(num_envs, 3, device=self.device)
        self.motion_manager = SuperMimicMotionManager(motion, num_envs, self.device, gamma, MAX_STEP_REWARD)
        self.ep_step = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.ep_return = torch.zeros(num_envs, device=self.device)
        self.ep_discounted_return = torch.zeros_like(self.ep_return)
        self.a_prev = torch.zeros(num_envs, self.num_joints, device=self.device)
        self.xi = torch.zeros(num_envs, self.spec.context_dim, device=self.device)
        self.done_eval = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self._reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._model, self.contract, _, qpos_ids, qvel_ids, actuator_ids, self._com_id = prepare_model(
            self.spec, self.task_cfg)
        self._qpos_ids = torch.tensor(qpos_ids, device=self.device)
        self._qvel_ids = torch.tensor(qvel_ids, device=self.device)
        self._actuator_ids = torch.tensor(actuator_ids, device=self.device)
        wp.init()
        # CUDA capture cannot use PyTorch's null/default stream on this Warp
        # version. A dedicated nondefault stream has explicit producer/consumer
        # waits, without changing the learner's current stream.
        self._torch_stream = torch.cuda.Stream(device=self.device)
        self._stream = wp.stream_from_torch(self._torch_stream)
        with self._warp_context():
            batch = {name: num_envs for name in (
                "body_ipos", "geom_friction", "body_invweight0", "dof_invweight0",
                "actuator_acc0", "actuator_gainprm", "actuator_biasprm")}
            self.model = mw.put_model(self._model, batch_sizes=batch)
            self.model.stat.meaninertia = wp.full(
                num_envs, float(self._model.stat.meaninertia), dtype=wp.float32, device=self.device)
            self.data = mw.make_data(
                self._model, nworld=num_envs,
                nconmax=self.task_cfg.warp_nconmax, njmax=self.task_cfg.warp_njmax)
            self._qpos = wp.to_torch(self.data.qpos)
            self._qvel = wp.to_torch(self.data.qvel)
            self._ctrl = wp.to_torch(self.data.ctrl)
            self._body_ipos = wp.to_torch(self.model.body_ipos)
            self._base_body_ipos = self._body_ipos.clone()
            self._geom_friction = wp.to_torch(self.model.geom_friction)
            self._sample_context(torch.arange(num_envs, device=self.device), initialize=True)
            self._apply_domain_randomization()
            self._graph = None
            if self.task_cfg.warp_use_graph:
                # Compile outside capture, then capture four physics steps.
                mw.step(self.model, self.data)
                with wp.ScopedCapture(stream=self._stream) as capture:
                    for _ in range(self.control_decimation):
                        mw.step(self.model, self.data)
                self._graph = capture.graph
        self.reset(torch.arange(num_envs, device=self.device))
        self.contract.update(
            cuda_graph=self._graph is not None, num_envs=num_envs,
            nconmax=self.task_cfg.warp_nconmax, njmax=self.task_cfg.warp_njmax,
            observation_mode=self.mdp.obs_mode,
            standalone_isaac_kit_imported=False,
        )

    @contextmanager
    def _warp_context(self):
        caller = torch.cuda.current_stream(self.device)
        self._torch_stream.wait_stream(caller)
        try:
            with (torch.cuda.stream(self._torch_stream),
                  self.wp.ScopedDevice(self.device),
                  self.wp.ScopedStream(self._stream)):
                yield
        finally:
            caller.wait_stream(self._torch_stream)

    @property
    def f(self):
        return self.motion_manager.frames

    @f.setter
    def f(self, frames):
        self.motion_manager.frames = frames

    @property
    def motion_id(self):
        return self.motion_manager.motion_ids

    @property
    def bin_returns(self):
        return self.motion_manager.bin_returns

    @property
    def bin_past_weights(self):
        return self.motion_manager.bin_past_weights

    @property
    def max_bin_return(self):
        return self.motion_manager.max_bin_return

    @property
    def reset_bin(self):
        return self.motion_manager.reset_bin

    def state(self):
        # MuJoCo free-root linear qvel is world-frame; angular qvel is local.
        p = self._qpos[:, :3].clone()
        R = quat_to_mat(self._qpos[:, 3:7].clone())
        v = self._qvel[:, :3].clone()
        w = (R @ self._qvel[:, 3:6, None]).squeeze(-1)
        q = self._qpos[:, self._qpos_ids].clone()
        qd = self._qvel[:, self._qvel_ids].clone()
        return p, R, v, w, q, qd

    def _sample_context(self, env_ids, initialize=False):
        n = len(env_ids)
        if self.eval_mode:
            self.xi[env_ids] = 0.
            self.xi[env_ids, 0] = 1.
            return
        if initialize:
            low, high = self.friction_range
            self.xi[env_ids, 0] = low + (high - low) * torch.rand(n, device=self.device)
            self.xi[env_ids, -3:] = (2 * torch.rand(n, 3, device=self.device) - 1) * self.com_offset_range
        self.xi[env_ids, 1:-3] = (
            2 * torch.rand(n, self.num_joints, device=self.device) - 1) * self.joint_offset_range

    def _apply_domain_randomization(self):
        self._geom_friction[:, :, 0] = self.xi[:, 0, None]
        self._body_ipos.copy_(self._base_body_ipos)
        self._body_ipos[:, self._com_id] += self.xi[:, -3:]
        # COM changes affect fixed inverse-inertia fields. Their arrays were
        # allocated per world; update them once, not on every episode reset.
        self.mw.set_const(self.model, self.data)

    def _sample_frames(self, env_ids):
        return self.motion_manager.sample_frames(env_ids)

    def _update_bin_returns(self, finished, terminated):
        self.motion_manager.update_returns(
            finished, terminated, self.ep_step, self.ep_return, self.ep_discounted_return)

    @torch.no_grad()
    def reset(self, env_ids=None, frame_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset(env_ids, frame_ids)

    @torch.no_grad()
    def _reset(self, env_ids, frame_ids=None, initialize_context=False):
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if initialize_context:
            raise ValueError("Warp physical context is initialized once in the constructor")
        self.motion_manager.reset(env_ids, frame_ids)
        self.ep_step[env_ids] = 0
        self.ep_return[env_ids] = 0.
        self.ep_discounted_return[env_ids] = 0.
        self.a_prev[env_ids] = 0.
        self.done_eval[env_ids] = False
        self.failed_eval[env_ids] = False
        self._sample_context(env_ids)
        p, R, v, w, q, qd = self.motion_manager.current_reference(env_ids)
        if self.reset_noise:
            def delta(scale):
                return (2 * torch.rand(len(env_ids), len(scale), device=self.device) - 1) * torch.tensor(scale, device=self.device)
            p = p + delta(self.task_cfg.reset_position)
            R = so3_exp(delta(self.task_cfg.reset_rotation)) @ R
            v = v + delta(self.task_cfg.reset_linvel)
            w = w + delta(self.task_cfg.reset_angvel)
            q = q + (2 * torch.rand_like(q) - 1) * self.task_cfg.reset_joint
        q = q.clamp(self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        self._reset_mask.zero_()
        self._reset_mask[env_ids] = True
        with self._warp_context():
            self.mw.reset_data(self.model, self.data, self.wp.from_torch(self._reset_mask))
            self._qpos[env_ids, :3] = p
            self._qpos[env_ids, 3:7] = mat_to_quat(R)
            self._qpos[env_ids[:, None], self._qpos_ids] = q
            self._qvel[env_ids, :3] = v
            self._qvel[env_ids, 3:6] = (R.transpose(-1, -2) @ w[..., None]).squeeze(-1)
            self._qvel[env_ids[:, None], self._qvel_ids] = qd
            self._ctrl[env_ids[:, None], self._actuator_ids] = q + self.xi[env_ids, 1:-3]
            self.mw.forward(self.model, self.data)

    @torch.no_grad()
    def step(self, action):
        S0 = self.state()
        s = flatten_state(S0)
        a_prev = self.a_prev.clone()
        policy_action = action.clone()
        a = action.clamp(-20., 20.)
        if self.eval_mode:
            a = torch.where(self.done_eval[:, None], torch.zeros_like(a), a)
        active = ~self.done_eval if self.eval_mode else torch.ones_like(self.done_eval)
        target = self.default_q + self.xi[:, 1:-3] + a * self.action_scale
        with self._warp_context():
            self._ctrl[:, self._actuator_ids] = target
            if self._graph is None:
                for _ in range(self.control_decimation):
                    self.mw.step(self.model, self.data)
            else:
                self.wp.capture_launch(self._graph, stream=self._stream)
        S1 = self.state()
        end = self.motion.clip_end[self.motion.clip_id[self.f]]
        f_next = self.motion_manager._motion_buffer.next_frames()
        if hasattr(self, '_collection_heads'):
            reward, reward_terms, terminated_raw = self._collection_heads(S1, f_next, a, a_prev)
        else:
            reward, reward_terms = self.mdp.reward(S1, f_next, a, a_prev, terms=True)
            terminated_raw = self.mdp.termination(S1, f_next)
        terminated = terminated_raw & active
        timeout = self.ep_step + 1 >= self.max_episode_steps
        truncated = ((self.f + 1 >= end) | timeout) & ~terminated & active
        done = terminated | truncated
        reward = torch.where(self.done_eval, torch.zeros_like(reward), reward)
        self.ep_return += reward
        self.ep_discounted_return += (self.gamma**self.ep_step.float()) * reward
        self.ep_step += (~self.done_eval).to(self.ep_step.dtype)
        self.motion_manager.compute(f_next)
        self.a_prev = a.clone()
        transition = {
            "s": s, "s1": flatten_state(S1), "a": a, "policy_action": policy_action,
            "a_prev": a_prev, "xi": self.xi.clone(), "r": reward,
            "reward_terms": reward_terms, "f": f_next - 1,
            "motion_id": self.motion_id.clone(), "ep_step": self.ep_step - 1,
            "terminated": terminated, "truncated": truncated, "done": done,
        }
        if self.eval_mode:
            self.done_eval |= done
            self.failed_eval |= terminated
        else:
            finished = done.nonzero(as_tuple=False).squeeze(-1)
            if finished.numel():
                self._update_bin_returns(finished, terminated)
                transition["ep_len"] = torch.where(done, self.ep_step.float(), torch.zeros_like(self.ep_return))
                transition["ep_ret_norm"] = torch.where(
                    done, self.ep_return / (self.ep_step.float() * MAX_STEP_REWARD).clamp_min(1e-8),
                    torch.zeros_like(self.ep_return))
                transition["term"] = terminated
                self._reset(finished)
        return transition

"""PhysX data-collection environment for SuperMimic."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import MISSING
from typing import Iterator
import traceback

import torch

from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from trackerLab.tracker_env.mdp.supermimic import MAX_STEP_REWARD, MDP, flatten_state
from trackerLab.motion_buffer.motion_lib.supermimic import CONTROL_DT, MotionLib
from trackerLab.managers.supermimic import SuperMimicMotionManager
from trackerTask.supermimic.robots import get_robot, get_task_cfg
from trackerLab.kinematics.supermimic.rotations import mat_to_quat, quat_to_mat, so3_exp


PHYSICS_DT = 0.005
CONTROL_DECIMATION = round(CONTROL_DT / PHYSICS_DT)


def add_launcher_args(parser) -> None:
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, device="cuda:0")


@contextmanager
def launch_simulation(sim_cfg: SimulationCfg, args, app_launcher=None) -> Iterator[SimulationContext]:
    if app_launcher is None:
        app_launcher = AppLauncher(args)
    sim = SimulationContext(sim_cfg)
    try:
        yield sim
    except BaseException as error:
        # Kit shutdown can exit the process before an active Python exception is printed.
        if not isinstance(error, SystemExit) or error.code not in (None, 0):
            traceback.print_exc()
        raise
    finally:
        # Unsubscribe the STOP callback before closing Kit; it otherwise renders
        # indefinitely while waiting for the timeline to resume.
        try:
            SimulationContext.clear_instance()
        finally:
            app_launcher.app.close()


@configclass
class SuperMimicSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = MISSING


def make_sim(device: str, args=None, task_cfg=None):
    physics_dt = getattr(task_cfg, "physics_dt", PHYSICS_DT)
    control_decimation = getattr(task_cfg, "control_decimation", CONTROL_DECIMATION)
    return sim_utils.SimulationCfg(
        dt=physics_dt,
        render_interval=control_decimation,
        device=device,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=getattr(task_cfg, "gpu_max_rigid_patch_count", 2**21),
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        log_dir=getattr(args, "log_dir", None),
    )


class TrackingSim:
    """Task-selected batched simulator preserving the terminal transition before reset."""

    def __init__(
        self,
        sim_cfg,
        num_envs: int,
        motion: MotionLib,
        mdp: MDP,
        gamma: float = 0.99,
        eval_mode: bool = False,
        task_cfg=None,
        env_spacing: float | None = None,
        friction_range: tuple[float, float] | None = None,
        joint_offset_range: float | None = None,
        com_offset_range: tuple[float, float, float] | None = None,
    ):
        self.sim = SimulationContext.instance()
        if self.sim is None:
            self.sim = SimulationContext(sim_cfg)
        self.motion, self.mdp = motion, mdp
        self.task_cfg = task_cfg or get_task_cfg()
        self.spec = get_robot(self.task_cfg.robot_name)
        self.num_joints = self.spec.num_joints
        if motion.num_joints != self.num_joints or mdp.num_joints != self.num_joints:
            raise ValueError("Task, motion and MDP joint dimensions differ")
        self.num_envs, self.device = num_envs, sim_cfg.device
        self.physics_dt = sim_cfg.dt
        self.control_decimation = sim_cfg.render_interval
        self.eval_mode = eval_mode
        self.reset_noise = getattr(task_cfg, "reset_noise", False) and not eval_mode
        self.gamma = gamma
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must lie between zero and one")
        self.max_episode_steps = 2**31 - 1 if eval_mode else getattr(task_cfg, "max_episode_steps", 500)
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        env_spacing = env_spacing if env_spacing is not None else getattr(task_cfg, "env_spacing", 2.0)
        self.friction_range = friction_range or getattr(task_cfg, "friction_range", (0.3, 1.2))
        self.joint_offset_range = (
            joint_offset_range if joint_offset_range is not None else getattr(task_cfg, "joint_offset_range", 0.01)
        )
        self.com_offset_range = torch.tensor(
            com_offset_range or getattr(task_cfg, "torso_com_offset_range", (0.025, 0.05, 0.05)),
            device=self.device,
        )
        scene_cfg = SuperMimicSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        scene_cfg.robot = self.spec.make_articulation().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene["robot"]
        self.sim.reset()
        self.scene.reset()
        self.scene.update(self.physics_dt)

        name_to_id = {name: i for i, name in enumerate(self.robot.joint_names)}
        missing = set(self.spec.joint_names).difference(name_to_id)
        if missing:
            raise RuntimeError(f"{self.spec.name} USD is missing expected joints: {sorted(missing)}")
        self.joint_ids = [name_to_id[name] for name in self.spec.joint_names]
        self.default_q = torch.tensor(self.spec.default_q, device=self.device)
        self.action_scale = torch.tensor(self.spec.action_scale, device=self.device)
        self.env_origins = self.scene.env_origins

        self.motion_manager = SuperMimicMotionManager(
            motion, num_envs, self.device, gamma, MAX_STEP_REWARD,
        )
        self.ep_step = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.ep_return = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.ep_discounted_return = torch.zeros_like(self.ep_return)
        self.a_prev = torch.zeros(num_envs, self.num_joints, device=self.device)
        self.xi = torch.zeros(num_envs, self.spec.context_dim, device=self.device)
        self.done_eval = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self._default_coms = None
        self._torso_body_id = self.robot.body_names.index(self.spec.com_body)
        self._reset(torch.arange(num_envs, device=self.device), initialize_context=True)

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
        data = self.robot.data
        p = data.root_link_pos_w - self.env_origins
        R = quat_to_mat(data.root_link_quat_w)
        v = data.root_link_lin_vel_w
        w = data.root_link_ang_vel_w
        q = data.joint_pos[:, self.joint_ids]
        qd = data.joint_vel[:, self.joint_ids]
        return p, R, v, w, q, qd

    def _sample_frames(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.motion_manager.sample_frames(env_ids)

    def _update_bin_returns(self, finished: torch.Tensor, terminated: torch.Tensor):
        self.motion_manager.update_returns(
            finished, terminated, self.ep_step, self.ep_return, self.ep_discounted_return,
        )

    def _sample_context(self, env_ids: torch.Tensor, initialize: bool = False):
        count = len(env_ids)
        if self.eval_mode:
            self.xi[env_ids] = 0
            self.xi[env_ids, 0] = 1.0
            return
        if initialize:
            low, high = self.friction_range
            self.xi[env_ids, 0] = low + (high - low) * torch.rand(count, device=self.device)
            self.xi[env_ids, -3:] = (
                2 * torch.rand(count, 3, device=self.device) - 1
            ) * self.com_offset_range
        self.xi[env_ids, 1:-3] = (2 * torch.rand(count, self.num_joints, device=self.device) - 1) * self.joint_offset_range

    def _set_domain_randomization(self, env_ids: torch.Tensor):
        view = self.robot.root_physx_view
        material = view.get_material_properties()
        selected = env_ids.to(material.device)
        friction = self.xi[env_ids, 0].to(material.device).unsqueeze(-1)
        material[selected, :, 0] = friction
        material[selected, :, 1] = friction
        material[selected, :, 2] = 0.0
        indices = env_ids.to(device="cpu", dtype=torch.int32)
        view.set_material_properties(material.cpu(), indices)

        if self._default_coms is None:
            self._default_coms = view.get_coms().clone()
        coms = view.get_coms().clone()
        selected = env_ids.to(coms.device)
        coms[selected] = self._default_coms[selected]
        coms[selected, self._torso_body_id, :3] += self.xi[env_ids, -3:].to(coms.device)
        view.set_coms(coms.cpu(), indices)

    @torch.no_grad()
    def _reset(
        self,
        env_ids: torch.Tensor,
        frame_ids: torch.Tensor | None = None,
        initialize_context: bool = False,
    ):
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.motion_manager.reset(env_ids, frame_ids)
        self.ep_step[env_ids] = 0
        self.ep_return[env_ids] = 0.0
        self.ep_discounted_return[env_ids] = 0.0
        self.a_prev[env_ids] = 0.0
        self.done_eval[env_ids] = False
        self.failed_eval[env_ids] = False
        self._sample_context(env_ids, initialize=initialize_context)

        root_p, root_R, root_v, root_w, q, qd = self.motion_manager.current_reference(env_ids)
        if self.reset_noise:
            # Initial-state perturbations from vibed; no external pushes during a rollout.
            def uniform_delta(scale):
                return (2 * torch.rand(len(env_ids), len(scale), device=self.device) - 1) * torch.tensor(
                    scale, device=self.device
                )

            root_p = root_p + uniform_delta(self.task_cfg.reset_position)
            root_R = so3_exp(uniform_delta(self.task_cfg.reset_rotation)) @ root_R
            root_v = root_v + uniform_delta(self.task_cfg.reset_linvel)
            root_w = root_w + uniform_delta(self.task_cfg.reset_angvel)
            q = q + (2 * torch.rand_like(q) - 1) * self.task_cfg.reset_joint
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        root_pose = torch.cat(
            (root_p + self.env_origins[env_ids], mat_to_quat(root_R)),
            dim=-1,
        )
        self.robot.write_root_link_pose_to_sim(root_pose, env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((root_v, root_w), dim=-1), env_ids)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(
            q + self.xi[env_ids, 1:-3],
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )
        # Friction and CoM are fixed per environment; joint offsets change on reset.
        if initialize_context:
            self._set_domain_randomization(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(self.physics_dt)

    def reset(self, env_ids: torch.Tensor | None = None, frame_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset(env_ids, frame_ids)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        S0 = self.state()
        s = flatten_state(S0)
        a_prev = self.a_prev.clone()
        policy_action = action.clone()
        a = torch.clamp(action, -20.0, 20.0)
        if self.eval_mode:
            a = torch.where(self.done_eval[:, None], torch.zeros_like(a), a)
        active = ~self.done_eval if self.eval_mode else torch.ones_like(self.done_eval)
        target = self.default_q + self.xi[:, 1:-3] + a * self.action_scale
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        for _ in range(self.control_decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

        S1 = self.state()
        end = self.motion.clip_end[self.motion.clip_id[self.f]]
        f_next = self.motion_manager._motion_buffer.next_frames()
        terminated = self.mdp.termination(S1, f_next) & active
        timeout = self.ep_step + 1 >= self.max_episode_steps
        truncated = ((self.f + 1 >= end) | timeout) & ~terminated & active
        done = terminated | truncated
        if self.eval_mode:
            done = done & ~self.done_eval
        reward, reward_terms = self.mdp.reward(S1, f_next, a, a_prev, terms=True)
        reward = torch.where(self.done_eval, torch.zeros_like(reward), reward)
        self.ep_return += reward
        self.ep_discounted_return += (self.gamma**self.ep_step.float()) * reward
        self.ep_step += (~self.done_eval).to(self.ep_step.dtype)
        self.motion_manager.compute(f_next)
        self.a_prev = a
        transition = {
            "s": s,
            "s1": flatten_state(S1),
            "a": a,
            "policy_action": policy_action,
            "a_prev": a_prev,
            "xi": self.xi.clone(),
            "r": reward,
            "reward_terms": reward_terms,
            "f": f_next - 1,
            "motion_id": self.motion_id.clone(),
            "ep_step": self.ep_step - 1,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
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
                    done,
                    self.ep_return / (self.ep_step.float() * MAX_STEP_REWARD).clamp_min(1e-8),
                    torch.zeros_like(self.ep_return),
                )
                transition["term"] = terminated
                self._reset(finished)
                transition["done"] = done
        return transition

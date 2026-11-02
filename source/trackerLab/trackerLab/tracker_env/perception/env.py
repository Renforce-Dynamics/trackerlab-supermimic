"""RSL-RL vector environment with MuJoCo-Warp physics and actual terrain rays."""

from contextlib import contextmanager

import numpy as np
import torch

from trackerLab.kinematics.supermimic.rotations import quat_to_mat
from trackerLab.tracker_env.perception.model import build_model
from trackerLab.tracker_env.perception.terrain import generate
from trackerTask.native.locomotion.perception.registry import get_task, robot_parameters


class PerceptionEnv:
    step_dt = .02

    def __init__(self, task, num_envs=4096, device="cuda:0", seed=0, eval_mode=False,
                 level=None, solver_iterations=50):
        import mujoco_warp as mw
        import warp as wp
        from mujoco_warp._src.types import vec6

        if num_envs < 1 or not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("Perception locomotion requires a positive world count and CUDA")
        if level is not None and level not in range(6):
            raise ValueError("Terrain level must be between zero and five")
        self.task, self.num_envs, self.device = get_task(task), num_envs, device
        self.eval_mode, self.fixed_level = eval_mode, level
        self.mw, self.wp = mw, wp
        wp.init()
        spec, height, _ = robot_parameters(self.task.robot)
        self.height, self.scale = height, height / .76
        self.terrain = generate(self.task.profile, height, seed)
        self.cpu_model, self.spec, _, qids, vids, self.root_id, self.contract = build_model(
            self.task.robot, self.terrain, solver_iterations,
        )
        self.num_actions = spec.num_joints
        self.cfg = {"task": self.task.describe(), "seed": seed, "num_envs": num_envs}
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.levels = torch.randint(0, 2, (num_envs,), device=device)
        if level is not None:
            self.levels.fill_(level)
        self.centers = torch.tensor(self.terrain.centers, device=device, dtype=torch.float32)
        self.qids = torch.tensor(qids, device=device)
        self.vids = torch.tensor(vids, device=device)
        self.default_q = torch.tensor(spec.default_q, device=device, dtype=torch.float32)
        self.action_scale = torch.tensor(spec.action_scale, device=device, dtype=torch.float32)
        self.qlo = torch.tensor(self.cpu_model.jnt_range[:, 0], device=device, dtype=torch.float32)[
            torch.tensor([self.cpu_model.joint(n).id for n in spec.joint_names], device=device)]
        self.qhi = torch.tensor(self.cpu_model.jnt_range[:, 1], device=device, dtype=torch.float32)[
            torch.tensor([self.cpu_model.joint(n).id for n in spec.joint_names], device=device)]
        self.actions = torch.zeros(num_envs, self.num_actions, device=device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.command = torch.ones(num_envs, device=device) * (.6 * self.scale)
        self.episode_return = torch.zeros(num_envs, device=device)
        self.completed = self.successes = self.falls = 0
        self.perception_mode = "normal"
        self._stream_torch = torch.cuda.Stream(device=device)
        self._stream = wp.stream_from_torch(self._stream_torch)
        xx, yy = torch.meshgrid(torch.linspace(-.3, 1.6, 17, device=device),
                                torch.linspace(-.55, .55, 11, device=device), indexing="ij")
        self.scan_offsets = torch.cat((torch.zeros(1, 2, device=device),
                                       torch.stack((xx.flatten(), yy.flatten()), -1)), 0) * self.scale
        self.num_rays = len(self.scan_offsets)
        self.ray_origin = torch.zeros(num_envs, self.num_rays, 3, device=device)
        self.ray_direction = torch.zeros_like(self.ray_origin)
        self.ray_direction[..., 2] = -1
        self.ray_dist = torch.zeros(num_envs, self.num_rays, device=device)
        self.ray_geom = torch.zeros(num_envs, self.num_rays, dtype=torch.int32, device=device)
        self.ray_normal = torch.zeros_like(self.ray_origin)
        self._ray_group = vec6(0, 0, 0, 0, 0, 1)
        with self._warp_context():
            self.model = mw.put_model(self.cpu_model)
            self.data = mw.make_data(self.cpu_model, nworld=num_envs, nconmax=96, njmax=384)
            self.qpos, self.qvel = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel)
            self.ctrl = wp.to_torch(self.data.ctrl)
            self._qacc_warmstart = wp.to_torch(self.data.qacc_warmstart)
            self._ray_p = wp.from_torch(self.ray_origin, dtype=wp.vec3)
            self._ray_v = wp.from_torch(self.ray_direction, dtype=wp.vec3)
            self._ray_d = wp.from_torch(self.ray_dist)
            self._ray_g = wp.from_torch(self.ray_geom)
            self._ray_n = wp.from_torch(self.ray_normal, dtype=wp.vec3)
            self._ray_exclude = wp.full(self.num_rays, -1, dtype=wp.int32, device=device)
            self._set_reset_state(torch.arange(num_envs, device=device))
            mw.forward(self.model, self.data)
            mw.step(self.model, self.data)
            with wp.ScopedCapture(stream=self._stream) as capture:
                for _ in range(4):
                    mw.step(self.model, self.data)
            self.physics_graph = capture.graph
            self._cast_rays()
            with wp.ScopedCapture(stream=self._stream) as capture:
                self._cast_rays()
            self.ray_graph = capture.graph
        self.reset()
        self.contract["sensor"] = {
            "type": "MuJoCo-Warp vertical terrain ray casting", "grid": [17, 11],
            "rays_including_center": self.num_rays, "terrain_only_group": 5,
            "scan_offsets_m": self.scan_offsets.cpu().tolist(),
            "actor_input": "root-relative measured heights; 0.01m uniform range noise in training",
            "limitations": "Idealized height scanner; no RGB/depth image encoder or real sensor occlusion model.",
        }
        self.contract["training"] = {
            "algorithm": "RSL-RL PPO 2.3.3", "terrain_curriculum_levels": 6,
            "num_envs": num_envs, "max_episode_steps": self.max_episode_length,
            "observation_dim": self.get_observations()[0].shape[-1],
            "actor_has_terrain_id_or_future_waypoints": False,
        }

    @property
    def unwrapped(self):
        return self

    @contextmanager
    def _warp_context(self):
        caller = torch.cuda.current_stream(self.device)
        self._stream_torch.wait_stream(caller)
        try:
            with (torch.cuda.stream(self._stream_torch), self.wp.ScopedDevice(self.device),
                  self.wp.ScopedStream(self._stream)):
                yield
        finally:
            caller.wait_stream(self._stream_torch)

    def _cast_rays(self):
        self.mw.rays(self.model, self.data, self._ray_p, self._ray_v, self._ray_group,
                     True, self._ray_exclude, self._ray_d, self._ray_g, self._ray_n)

    def _set_reset_state(self, ids):
        count = len(ids)
        self.qpos[ids] = 0
        self.qvel[ids] = 0
        self._qacc_warmstart[ids] = 0
        self.qpos[ids, 0] = 0
        self.qpos[ids, 1] = self.centers[self.levels[ids]]
        self.qpos[ids, 2] = self.height + .02 * self.scale
        yaw = torch.zeros(count, device=self.device)
        if not self.eval_mode:
            self.qpos[ids, 1] += (torch.rand(count, device=self.device) - .5) * .15 * self.scale
            yaw = (torch.rand(count, device=self.device) - .5) * .12
        self.qpos[ids, 3], self.qpos[ids, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        q = self.default_q.expand(count, -1).clone()
        if not self.eval_mode:
            q += (torch.rand_like(q) - .5) * .04
        q = q.clamp(self.qlo, self.qhi)
        self.qpos[ids[:, None], self.qids] = q
        self.ctrl[ids] = q
        self.actions[ids], self.previous_actions[ids] = 0, 0
        self.episode_length_buf[ids], self.episode_return[ids] = 0, 0
        self.command[ids] = .6 * self.scale
        if not self.eval_mode:
            self.command[ids] = (.3 + .6 * torch.rand(count, device=self.device)) * self.scale

    @torch.no_grad()
    def reset(self):
        with self._warp_context():
            self._set_reset_state(torch.arange(self.num_envs, device=self.device))
            self.mw.forward(self.model, self.data)
        self._update_scan()
        return self.get_observations()

    def _update_scan(self):
        R = quat_to_mat(self.qpos[:, 3:7])
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        x, y = self.scan_offsets[:, 0], self.scan_offsets[:, 1]
        self.ray_origin[..., 0] = self.qpos[:, 0:1] + c * x - s * y
        self.ray_origin[..., 1] = self.qpos[:, 1:2] + s * x + c * y
        self.ray_origin[..., 2] = self.qpos[:, 2:3] + 2 * self.height
        with self._warp_context():
            self.wp.capture_launch(self.ray_graph)

    def measured_heights(self):
        return self.ray_origin[..., 2] - self.ray_dist

    def get_observations(self):
        R = quat_to_mat(self.qpos[:, 3:7])
        angular = self.qvel[:, 3:6]
        gravity = -R[:, 2, :]
        velocity = (R.transpose(-1, -2) @ self.qvel[:, :3, None]).squeeze(-1)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        command = torch.stack((self.command / self.scale, torch.zeros_like(yaw), -yaw), -1)
        q, qd = self.qpos[:, self.qids], self.qvel[:, self.vids]
        scan = (self.qpos[:, 2:3] - self.measured_heights()[:, 1:] - self.height) / self.scale
        scan = torch.where(self.ray_geom[:, 1:] >= 0, scan, torch.ones_like(scan))
        if not self.eval_mode:
            scan = scan + (torch.rand_like(scan) * 2 - 1) * .01 / self.scale
        if self.perception_mode == "zero":
            scan = torch.zeros_like(scan)
        elif self.perception_mode == "shuffle":
            scan = scan.flip(-1)
        elif self.perception_mode != "normal":
            raise ValueError("Unknown perception mode")
        proprio = torch.cat((velocity / self.scale, angular * .25, gravity, command,
                             q - self.default_q, qd * .05, self.actions), -1)
        obs = torch.cat((proprio, scan.clamp(-1, 1) * 2), -1).clamp(-20, 20)
        return obs, {"observations": {}}

    @torch.no_grad()
    def step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-3, 3))
        self.ctrl.copy_((self.default_q + self.actions * self.action_scale).clamp(self.qlo, self.qhi))
        with self._warp_context():
            self.wp.capture_launch(self.physics_graph)
        self.episode_length_buf += 1
        self._update_scan()
        R = quat_to_mat(self.qpos[:, 3:7])
        clear = self.qpos[:, 2] - self.measured_heights()[:, 0]
        lane_error = (self.qpos[:, 1] - self.centers[self.levels]) / self.scale
        velocity = self.qvel[:, :3] / self.scale
        tracking = torch.exp(-((velocity[:, 0] - self.command / self.scale).square()
                               + velocity[:, 1].square()) / .25)
        progress = velocity[:, 0].clamp(-1, 1.5)
        pose = (self.qpos[:, self.qids] - self.default_q).square().mean(-1)
        rate = (self.actions - self.previous_actions).square().mean(-1)
        reward = (1.5 * tracking + .5 * progress + .25
                  - .5 * (1 - R[:, 2, 2]) - .05 * lane_error.square()
                  - .1 * ((clear / self.height - 1).square())
                  - .03 * pose - .04 * rate
                  - .001 * self.qvel[:, self.vids].square().mean(-1)) * self.step_dt
        success = self.qpos[:, 0] >= self.terrain.goal_x
        invalid = ~(torch.isfinite(self.qpos).all(-1) & torch.isfinite(self.qvel).all(-1))
        fallen = ((clear < .42 * self.height) | (R[:, 2, 2] < .35)
                  | (lane_error.abs() > 1.75) | (self.ray_geom[:, 0] < 0) | invalid)
        timeout = self.episode_length_buf >= self.max_episode_length
        done = fallen | success | timeout
        reward = reward + .5 * success - .1 * fallen
        reward = torch.nan_to_num(reward, nan=-.1, posinf=-.1, neginf=-.1)
        self.episode_return += reward
        ids = done.nonzero(as_tuple=False).flatten()
        # Task success is terminal; RSL-RL bootstraps only pure time-limit truncations.
        extras = {"observations": {}, "time_outs": timeout & ~fallen & ~success}
        if self.eval_mode:
            extras["terminal"] = {
                "success": success.clone(), "fall": fallen.clone(),
                "progress": self.qpos[:, 0].clone(), "qpos": self.qpos.clone(),
            }
        if len(ids):
            extras["log"] = {
                "Episode/return": self.episode_return[ids].mean(),
                "Episode/seconds": self.episode_length_buf[ids].float().mean() * self.step_dt,
                "Terrain/level": self.levels.float().mean(),
                "Terrain/success_fraction": success[ids].float().mean(),
                "Terrain/fall_fraction": fallen[ids].float().mean(),
                "Terrain/progress_m": self.qpos[ids, 0].mean(),
            }
            if not self.eval_mode and self.fixed_level is None:
                promote = success[ids]
                demote = fallen[ids] & (self.qpos[ids, 0] < 1. * self.scale)
                self.levels[ids] = (self.levels[ids] + promote.long() - demote.long()).clamp(0, 5)
            with self._warp_context():
                self._set_reset_state(ids)
                self.mw.forward(self.model, self.data)
            self._update_scan()
        obs, _ = self.get_observations()
        if invalid.any() or not torch.isfinite(obs).all():
            raise FloatingPointError("Nonfinite simulator state or observation; inspect the saved run")
        return obs, reward, done.long(), extras

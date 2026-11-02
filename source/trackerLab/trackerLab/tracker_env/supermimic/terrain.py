"""MuJoCo-Warp vector simulation for SuperMimic terrain objectives."""

from contextlib import contextmanager

import torch

from trackerLab.kinematics.supermimic.rotations import quat_to_mat
from trackerLab.tracker_env.mdp.supermimic import flatten_state
from trackerLab.tracker_env.mdp.supermimic_terrain import terminal_masks
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerTask.supermimic.terrain import ROBOT_HEIGHTS, generate
from trackerTask.supermimic.robots import get_robot


class TerrainSim:
    def __init__(self, task, mdp, num_envs=4096, device="cuda:0", seed=42,
                 solver_iterations=50, max_episode_steps=1000):
        import mujoco
        import mujoco_warp as mw
        import warp as wp

        if num_envs < 1 or not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("SuperMimic terrain training requires CUDA and a positive world count")
        self.task, self.mdp = task, mdp
        self.robot = task.robot
        self.num_envs, self.device = num_envs, str(torch.device(device))
        self.mw, self.wp, self.max_episode_steps = mw, wp, max_episode_steps
        self.curriculum = True
        self.spec = get_robot(self.robot)
        self.height = ROBOT_HEIGHTS[self.robot]
        self.scale = self.height / .76
        self.terrain = generate(task.profile, self.height, seed)
        self.max_episode_steps = max_episode_steps
        self.ep_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.ep_return = torch.zeros(num_envs, device=device)
        self.a_prev = torch.zeros(num_envs, self.spec.num_joints, device=device)
        self.xi = torch.zeros(num_envs, self.spec.context_dim, device=device)
        self.c = torch.zeros(num_envs, 2, device=device)
        self.levels = torch.randint(0, 2, (num_envs,), device=device)
        self.centers = torch.tensor(self.terrain.centers, device=device, dtype=torch.float32)
        self.default_q = torch.tensor(self.spec.default_q, device=device)
        self.action_scale = torch.tensor(self.spec.action_scale, device=device)
        model_result = build_model(self.robot, self.terrain, solver_iterations)
        (self.cpu_model, _, _, self._qpos_ids_np, self._qvel_ids_np, self.root_id,
         self.com_id, self._actuator_ids_np, self.contract) = model_result
        self._qpos_ids = torch.tensor(self._qpos_ids_np, device=device)
        self._qvel_ids = torch.tensor(self._qvel_ids_np, device=device)
        self._actuator_ids = torch.tensor(self._actuator_ids_np, device=device)
        self._free_joint = next(
            index for index in range(self.cpu_model.njnt)
            if self.cpu_model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
            and self.cpu_model.jnt_bodyid[index] == self.root_id
        )
        self._root_qpos = int(self.cpu_model.jnt_qposadr[self._free_joint])
        self._root_qvel = int(self.cpu_model.jnt_dofadr[self._free_joint])
        joint_ids = torch.tensor([self.cpu_model.joint(name).id for name in self.spec.joint_names], device=device)
        self.qlo = torch.tensor(self.cpu_model.jnt_range[:, 0], device=device, dtype=torch.float32)[joint_ids]
        self.qhi = torch.tensor(self.cpu_model.jnt_range[:, 1], device=device, dtype=torch.float32)[joint_ids]
        if int((self.cpu_model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE).sum()) != self.spec.num_joints:
            raise ValueError("Terrain model hinge count differs from the robot policy contract")

        wp.init()
        self._torch_stream = torch.cuda.Stream(device=device)
        self._stream = wp.stream_from_torch(self._torch_stream)
        with self._warp_context():
            batch = {name: num_envs for name in (
                "body_ipos", "geom_friction", "body_invweight0", "dof_invweight0",
                "actuator_acc0", "actuator_gainprm", "actuator_biasprm")}
            self.model = mw.put_model(self.cpu_model, batch_sizes=batch)
            self.model.stat.meaninertia = wp.full(
                num_envs, float(self.cpu_model.stat.meaninertia), dtype=wp.float32, device=device)
            self.data = mw.make_data(self.cpu_model, nworld=num_envs, nconmax=96, njmax=384)
            self.qpos, self.qvel = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel)
            self.ctrl = wp.to_torch(self.data.ctrl)
            self.qacc_warmstart = wp.to_torch(self.data.qacc_warmstart)
            self.body_ipos = wp.to_torch(self.model.body_ipos)
            self.base_body_ipos = self.body_ipos.clone()
            self.geom_friction = wp.to_torch(self.model.geom_friction)
            self.base_geom_friction = self.geom_friction.clone()
            self._randomize(torch.arange(num_envs, device=device), initialize=True)
            self.mw.set_const(self.model, self.data)
            self._reset_state(torch.arange(num_envs, device=device))
            self.mw.forward(self.model, self.data)
            mw.step(self.model, self.data)
            with wp.ScopedCapture(stream=self._stream) as capture:
                for _ in range(4):
                    mw.step(self.model, self.data)
            self.physics_graph = capture.graph
            self._setup_rays()
            self._cast_rays()
            with wp.ScopedCapture(stream=self._stream) as capture:
                self._cast_rays()
            self.ray_graph = capture.graph
        self.reset(torch.arange(num_envs, device=device))
        self.contract.update(
            num_envs=num_envs, physics="four 5ms MuJoCo-Warp steps per 20ms action",
            terrain_curriculum_levels=6, max_episode_steps=max_episode_steps,
            perception=mdp.field.offsets.shape[0], algorithm="SuperMimic world-model imagination + GAE critic",
        )

    @contextmanager
    def _warp_context(self):
        caller = torch.cuda.current_stream(self.device)
        self._torch_stream.wait_stream(caller)
        try:
            with (torch.cuda.stream(self._torch_stream), self.wp.ScopedDevice(self.device),
                  self.wp.ScopedStream(self._stream)):
                yield
        finally:
            caller.wait_stream(self._torch_stream)

    def _setup_rays(self):
        from mujoco_warp._src.types import vec6

        self.scan_offsets = self.mdp.field.offsets
        self.num_rays = len(self.scan_offsets) + 1
        self.ray_origin = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        self.ray_direction = torch.zeros_like(self.ray_origin)
        self.ray_direction[..., 2] = -1
        self.ray_dist = torch.zeros(self.num_envs, self.num_rays, device=self.device)
        self.ray_geom = torch.zeros(self.num_envs, self.num_rays, dtype=torch.int32, device=self.device)
        self.ray_normal = torch.zeros_like(self.ray_origin)
        self._ray_group = vec6(0, 0, 0, 0, 0, 1)
        self._ray_p = self.wp.from_torch(self.ray_origin, dtype=self.wp.vec3)
        self._ray_v = self.wp.from_torch(self.ray_direction, dtype=self.wp.vec3)
        self._ray_d = self.wp.from_torch(self.ray_dist)
        self._ray_g = self.wp.from_torch(self.ray_geom)
        self._ray_n = self.wp.from_torch(self.ray_normal, dtype=self.wp.vec3)
        self._ray_exclude = self.wp.full(self.num_rays, -1, dtype=self.wp.int32, device=self.device)

    def _cast_rays(self):
        self.mw.rays(self.model, self.data, self._ray_p, self._ray_v, self._ray_group,
                     True, self._ray_exclude, self._ray_d, self._ray_g, self._ray_n)

    def _randomize(self, ids, initialize=False):
        count = len(ids)
        self.xi[ids, 1:-3] = (2 * torch.rand(count, self.spec.num_joints, device=self.device) - 1) * .01
        if initialize:
            self.xi[ids, 0] = .3 + .9 * torch.rand(count, device=self.device)
            self.xi[ids, -3:] = (2 * torch.rand(count, 3, device=self.device) - 1) * (.025 * self.scale)
            self.geom_friction[ids] = self.base_geom_friction[ids]
            self.geom_friction[ids, :, 0] = self.xi[ids, 0:1]
            self.body_ipos[ids] = self.base_body_ipos[ids]
            self.body_ipos[ids, self.com_id, :3] += self.xi[ids, -3:]

    def _reset_state(self, ids):
        self.mw.reset_data(self.model, self.data, self.wp.from_torch(
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device).scatter_(0, ids, True)))
        count = len(ids)
        center = self.centers[self.levels[ids]]
        x = torch.zeros(count, device=self.device)
        y = center + (torch.rand(count, device=self.device) - .5) * .15 * self.scale
        xy = torch.stack((x, y), -1)
        ground = self.mdp.field.sample(xy)
        yaw = (torch.rand(count, device=self.device) - .5) * .12
        quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw),
                            torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
        self.qpos[ids, self._root_qpos:self._root_qpos + 3] = torch.stack(
            (x, y, ground + self.height + .02 * self.scale), -1)
        self.qpos[ids, self._root_qpos + 3:self._root_qpos + 7] = quat
        q = (self.default_q + self.xi[ids, 1:-3]).clamp(self.qlo, self.qhi)
        self.qpos[ids[:, None], self._qpos_ids] = q
        self.ctrl[ids[:, None], self._actuator_ids] = q
        self.ep_step[ids], self.ep_return[ids] = 0, 0
        self.a_prev[ids] = 0
        self.c[ids, 0] = center
        self.c[ids, 1] = (.5 + .6 * torch.rand(count, device=self.device)) * self.scale

    @torch.no_grad()
    def reset(self, ids=None, levels=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        ids = ids.to(device=self.device, dtype=torch.long)
        self._randomize(ids)
        self.levels[ids] = (torch.randint(0, 2, (len(ids),), device=self.device)
                            if levels is None else levels.to(device=self.device, dtype=torch.long))
        with self._warp_context():
            self._reset_state(ids)
            self.mw.forward(self.model, self.data)
        self._update_rays()
        return self.observe()

    def state(self):
        p = self.qpos[:, self._root_qpos:self._root_qpos + 3].clone()
        R = quat_to_mat(self.qpos[:, self._root_qpos + 3:self._root_qpos + 7].clone())
        v = self.qvel[:, self._root_qvel:self._root_qvel + 3].clone()
        w_local = self.qvel[:, self._root_qvel + 3:self._root_qvel + 6].clone()
        w = (R @ w_local.unsqueeze(-1)).squeeze(-1)
        q = self.qpos[:, self._qpos_ids].clone()
        qd = self.qvel[:, self._qvel_ids].clone()
        return p, R, v, w, q, qd

    def _update_rays(self):
        S = self.state()
        R = S[1]
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        offsets = torch.cat((torch.zeros(1, 2, device=self.device), self.scan_offsets), 0)
        x, y = offsets[:, 0], offsets[:, 1]
        self.ray_origin[..., 0] = S[0][:, 0:1] + c * x - s * y
        self.ray_origin[..., 1] = S[0][:, 1:2] + s * x + c * y
        self.ray_origin[..., 2] = S[0][:, 2:3] + 2 * self.height
        with self._warp_context():
            self.wp.capture_launch(self.ray_graph, stream=self._stream)

    def observe(self):
        S = self.state()
        obs = self.mdp.obs(S, self.a_prev, self.xi, self.c)
        measured = self.ray_origin[:, 1:, 2] - self.ray_dist[:, 1:]
        scan = (S[0][:, 2:3] - measured - self.height) / self.scale
        scan = torch.where(self.ray_geom[:, 1:] >= 0, scan, torch.ones_like(scan)).clamp(-1, 1) * 2
        obs[..., -self.mdp.field.offsets.shape[0]:] = scan
        return obs

    @torch.no_grad()
    def step(self, action):
        S0 = self.state()
        a_prev = self.a_prev.clone()
        action = action.clone()
        a = action.clamp(-20, 20)
        target = (self.default_q + self.xi[:, 1:-3] + a * self.action_scale).clamp(self.qlo, self.qhi)
        with self._warp_context():
            self.ctrl[:, self._actuator_ids] = target
            self.wp.capture_launch(self.physics_graph, stream=self._stream)
        S1 = self.state()
        self.ep_step += 1
        reached = S1[0][:, 0] >= self.terrain.goal_x
        fallen = self.mdp.termination(S1, self.c)
        timeout = self.ep_step >= self.max_episode_steps
        success, done, truncated = terminal_masks(reached, fallen, timeout)
        reward, reward_terms = self.mdp.reward(S1, a, a_prev, self.c, terms=True)
        reward = reward + .5 * success - .1 * fallen
        reward = torch.nan_to_num(reward, nan=-.1, posinf=-.1, neginf=-.1)
        self.ep_return += reward
        ids = done.nonzero(as_tuple=False).flatten()
        transition = {
            "s": flatten_state(S0), "s1": flatten_state(S1), "a": a,
            "policy_action": action, "a_prev": a_prev, "xi": self.xi.clone(),
            "c": self.c.clone(), "r": reward, "reward_terms": reward_terms,
            "f": self.ep_step - 1, "motion_id": self.levels.clone(),
            "ep_step": self.ep_step - 1, "terminated": fallen | success,
            "truncated": truncated, "done": done,
            "episode_success": success, "episode_fall": fallen & done,
            "episode_timeout": truncated,
            "episode_return": torch.where(done, self.ep_return, torch.zeros_like(self.ep_return)),
        }
        self.a_prev.copy_(a)
        if len(ids):
            transition["ep_len"] = torch.where(done, self.ep_step.float(), torch.zeros_like(self.ep_return))
            transition["ep_ret_norm"] = torch.where(done, self.ep_return / self.ep_step.clamp_min(1),
                                                     torch.zeros_like(self.ep_return))
            transition["term"] = fallen
            if self.curriculum:
                advance = success[ids].long()
                retreat = (fallen[ids] & (S1[0][ids, 0] < self.scale)).long()
                self.levels[ids] = (self.levels[ids] + advance - retreat).clamp(0, 5)
            self.c[ids, 0] = self.centers[self.levels[ids]]
            self._randomize(ids)
            with self._warp_context():
                self._reset_state(ids)
                self.mw.forward(self.model, self.data)
        self._update_rays()
        return transition

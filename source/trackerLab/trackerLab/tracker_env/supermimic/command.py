"""MuJoCo-Warp command locomotion with opt-in training-only pose resets."""

import math

import torch

from trackerLab.kinematics.supermimic.rotations import mat_to_quat, quat_to_mat, yaw_mat
from trackerLab.tracker_env.mdp.supermimic import flatten_state
from trackerLab.tracker_env.mdp.supermimic_command import (
    ACTION_LIMIT, COMMAND_FIELDS, COMMAND_VERSION, CONTROL_DT, SCENARIO_DIM,
    STEALTH_KEYBOARD_MODES,
    advance_command,
)
from trackerLab.tracker_env.mdp.supermimic_terrain import (
    FALL_PENALTY, terminal_masks, update_terrain_levels,
)
from trackerLab.tracker_env.supermimic.terrain import TerrainSim
from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS


class CommandSim(TerrainSim):
    scenario_dim = SCENARIO_DIM

    def __init__(self, task, mdp, num_envs=4096, device="cuda:0", seed=42,
                 solver_iterations=50, max_episode_steps=1000,
                 physical_ray_validation=False, *, target_clearance_m=None,
                 jump_height=0., action_limit=ACTION_LIMIT,
                 command_resample_steps=(100, 250), jump_probability=.15,
                 stealth_keyboard_mode=False, self_collisions=False,
                 physical_domain_randomization=False):
        if task.profile != mdp.profile:
            raise ValueError("Command task and MDP profiles must match")
        clearance = mdp.target_clearance_m if target_clearance_m is None else target_clearance_m
        if not math.isfinite(clearance) or clearance <= 0:
            raise ValueError("Physical target clearance must be finite and positive")
        if not math.isfinite(jump_height) or jump_height < 0:
            raise ValueError("Jump height must be finite and nonnegative")
        if not math.isfinite(jump_probability) or not 0 <= jump_probability <= 1:
            raise ValueError("Jump probability must lie in [0, 1]")
        if not math.isfinite(action_limit) or action_limit <= 0:
            raise ValueError("Action limit must be finite and positive")
        if max_episode_steps < 1:
            raise ValueError("Episode length must be positive")
        if stealth_keyboard_mode and task.profile != "stealth":
            raise ValueError("Keyboard command modes require a stealth task")
        if (len(command_resample_steps) != 2
                or any(type(value) is not int for value in command_resample_steps)
                or not 100 <= command_resample_steps[0] <= command_resample_steps[1]
                or command_resample_steps[1] > max_episode_steps):
            raise ValueError("Command resampling interval must lie within the episode")
        self.target_clearance_m = float(clearance)
        self.jump_height, self.action_limit = float(jump_height), float(action_limit)
        self.jump_probability = float(jump_probability)
        self.stealth_keyboard_mode = bool(stealth_keyboard_mode)
        self.self_collisions = bool(self_collisions)
        self._stealth_mode_ids = (
            torch.zeros(num_envs, dtype=torch.long, device=device)
            if self.stealth_keyboard_mode else None)
        self.min_success_steps = math.ceil(10. / CONTROL_DT)
        self.command_resample_steps = tuple(command_resample_steps)
        super().__init__(task, mdp, num_envs, device, seed, solver_iterations,
                         max_episode_steps, physical_ray_validation,
                         scenario_dim=SCENARIO_DIM, self_collisions=self_collisions,
                         physical_domain_randomization=physical_domain_randomization)
        self.curriculum = mdp.terrain_navigation
        self.contract.update(
            objective="command-driven-locomotion", objective_version=COMMAND_VERSION,
            scenario_dim=SCENARIO_DIM, scenario_fields=COMMAND_FIELDS,
            obs_dim=mdp.obs_dim, goal_dim=mdp.goal_dim, action_limit=self.action_limit,
            target_clearance_m=self.target_clearance_m, jump_height_m=self.jump_height,
            jump_probability=self.jump_probability, min_success_steps=self.min_success_steps,
            runtime_reference_motion=False, terrain_curriculum_levels=6 if self.curriculum else 0,
            contact_constraints={
                "ground": "finite MuJoCo heightfield",
                "robot_geom_contype": 2,
                "robot_geom_conaffinity": 3 if self.self_collisions else 1,
                "self_collisions": self.self_collisions,
                "actuation": "joint-limited PD targets with source robot effort caps",
                "root_motion": "MuJoCo-Warp physics only",
                "foot_contact_classifier": False,
            },
            command_resample_steps=self.command_resample_steps,
            stealth_keyboard_modes=(
                [{"name": name, "command": [vx, vy, yaw], "probability": probability}
                 for name, vx, vy, yaw, probability in STEALTH_KEYBOARD_MODES]
                if self.stealth_keyboard_mode else None),
            flat_success="survive at least 10 seconds through timeout and track commands on at least 80% of actions",
        )

    def _reset_state(self, ids):
        # The parent invokes this override during construction, before reset().
        if self.c.shape[-1] != SCENARIO_DIM:
            self.c = self.c.new_zeros((self.num_envs, SCENARIO_DIM))
        if not hasattr(self, "_command_age"):
            self._command_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._command_interval = torch.zeros_like(self._command_age)
            self._episode_tracking = torch.zeros(self.num_envs, device=self.device)
            self._episode_velocity_error = torch.zeros_like(self._episode_tracking)
            self._episode_yaw_error = torch.zeros_like(self._episode_tracking)
            self._episode_clearance_error = torch.zeros_like(self._episode_tracking)
        super()._reset_state(ids)
        self.c[ids] = 0
        if self.mdp.terrain_navigation:
            self.c[ids, 0] = self.centers[self.levels[ids]]
        self.sample_commands(ids)
        for value in (self._episode_tracking, self._episode_velocity_error,
                      self._episode_yaw_error, self._episode_clearance_error):
            value[ids] = 0
        if hasattr(self, "_reference_reset_states"):
            self._apply_reference_resets(ids)

    def set_stealth_keyboard_mode(self):
        if self.mdp.profile != "stealth":
            raise ValueError("Keyboard command modes are only available for stealth locomotion")
        self.stealth_keyboard_mode = True
        self.contract["stealth_keyboard_modes"] = [
            {"name": name, "command": [vx, vy, yaw], "probability": probability}
            for name, vx, vy, yaw, probability in STEALTH_KEYBOARD_MODES
        ]
        self.sample_commands()

    @torch.no_grad()
    def set_reference_resets(self, motion, probability=.7, grounded_mask=None):
        """Cache safe poses from floor-aligned training data, outside actor inputs.

        Call before ``reset()``. Pass ``None`` or probability zero to discard all
        cached reference state. Inference environments need no motion argument.
        Foot-height checks, when FK is available, are support heuristics rather
        than a physical contact certificate. A caller-verified boolean mask of
        shape (frames,) replaces that heuristic. The caller checks physical
        foot-collider distances between -.005*scale and .04*scale.
        """
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Reference reset probability must lie in [0, 1]")
        if motion is None or probability == 0:
            for name in ("_reference_reset_states", "_reference_reset_probability"):
                if hasattr(self, name):
                    delattr(self, name)
            receipt = {"enabled": False, "training_only": True}
            self.contract["reference_resets"] = receipt
            return receipt
        names = ("root_p", "root_R", "root_v", "root_w", "q", "qd")
        source = tuple(getattr(motion, name, None) for name in names)
        if not all(isinstance(value, torch.Tensor) and value.is_floating_point() for value in source):
            raise ValueError("Reference resets require floating-point motion state tensors")
        frames = source[0].shape[0] if source[0].ndim else 0
        n = self.spec.num_joints
        tails = ((3,), (3, 3), (3,), (3,), (n,), (n,))
        if frames == 0 or any(value.shape != (frames, *tail) for value, tail in zip(source, tails)):
            raise ValueError("Reference reset states must match frame and robot joint dimensions")
        if motion is not None and probability and self.self_collisions:
            import mujoco
            import numpy as np

            positions = motion.root_p.detach().cpu().numpy()
            rotations = motion.root_R.detach().cpu().numpy()
            joints = motion.q.detach().cpu().numpy()
            if len(positions) != len(rotations) or len(positions) != len(joints):
                raise ValueError("Reference reset pose arrays have different frame counts")
            data = mujoco.MjData(self.cpu_model)
            robot_bodies = self.cpu_model.geom_bodyid
            quaternion = np.empty(4, dtype=np.float64)
            collision_free = np.ones(len(positions), dtype=bool)
            for frame in range(len(positions)):
                data.qpos[self._root_qpos:self._root_qpos + 3] = positions[frame]
                mujoco.mju_mat2Quat(quaternion, rotations[frame].reshape(-1))
                data.qpos[self._root_qpos + 3:self._root_qpos + 7] = quaternion
                data.qpos[self._qpos_ids_np] = joints[frame]
                mujoco.mj_forward(self.cpu_model, data)
                for contact in data.contact[:data.ncon]:
                    body1 = robot_bodies[contact.geom1]
                    body2 = robot_bodies[contact.geom2]
                    if (body1 and body2 and body1 != body2
                            and contact.dist < -1e-3):
                        collision_free[frame] = False
                        break
            contact_free = torch.as_tensor(collision_free, dtype=torch.bool)
            grounded_mask = (
                contact_free if grounded_mask is None
                else torch.as_tensor(grounded_mask, dtype=torch.bool).cpu() & contact_free)
            self.contract["reference_self_collision_filter"] = {
                "enabled": True,
                "collision_free_frames": int(collision_free.sum()),
                "total_frames": int(len(collision_free)),
            }

        supplied_grounding = grounded_mask is not None
        if supplied_grounding:
            grounded_mask = torch.as_tensor(grounded_mask)
            if grounded_mask.dtype != torch.bool or grounded_mask.shape != (frames,):
                raise ValueError("Grounded mask must be boolean with shape (frames,)")
            grounded_mask = grounded_mask.detach().to(device=self.device)
        body_p, body_R = getattr(motion, "body_p", None), getattr(motion, "body_R", None)
        tracked = getattr(self.spec, "tracked_bodies", ())
        feet = [tracked.index(name) for name in FOOT_BODIES[self.robot] if name in tracked]
        use_feet = not supplied_grounding and body_p is not None and bool(feet)
        offset = self.qpos.new_tensor(FOOT_OFFSETS.get(self.robot, (0., 0., 0.)))
        if use_feet:
            if (not isinstance(body_p, torch.Tensor) or not body_p.is_floating_point()
                    or body_p.shape != (frames, len(tracked), 3)):
                raise ValueError("Reference tracked-body positions must match the robot")
            if bool(offset.any()) and (
                    not isinstance(body_R, torch.Tensor) or not body_R.is_floating_point()
                    or body_R.shape != (frames, len(tracked), 3, 3)):
                raise ValueError("Reference foot offsets require tracked-body rotations")
        minimum = max(.5 * self.height, .65 * self.target_clearance_m)
        maximum = (min(1.15 * self.height, 1.25 * self.target_clearance_m)
                   if self.mdp.profile == "stealth" else 1.15 * self.height)
        vertical_limit = (.5 if self.mdp.terrain_navigation else .35) * self.scale
        identity = torch.eye(3, device=self.device, dtype=self.qpos.dtype)
        action_span = self.action_limit * self.action_scale.abs()
        chunks = [[] for _ in source]
        for start in range(0, frames, 8192):
            stop = min(start + 8192, frames)
            state = tuple(value[start:stop].detach().to(device=self.device, dtype=self.qpos.dtype)
                          for value in source)
            p, R, v, w, q, qd = state
            safe = torch.ones(stop - start, device=self.device, dtype=torch.bool)
            for value in state:
                safe &= torch.isfinite(value).flatten(1).all(-1)
            safe &= ((R[:, 2, 2] >= .8) & (p[:, 2] >= minimum) & (p[:, 2] <= maximum)
                     & (v[:, 2].abs() <= vertical_limit) & (v.norm(dim=-1) <= 3 * self.scale)
                     & (w.norm(dim=-1) <= 3) & (qd.abs() <= 20).all(-1)
                     & (q >= self.qlo).all(-1) & (q <= self.qhi).all(-1)
                     & ((q - self.default_q).abs() <= action_span).all(-1))
            safe &= ((R.transpose(-1, -2) @ R - identity).abs().flatten(1).amax(-1) <= 1e-3)
            safe &= (torch.linalg.det(torch.nan_to_num(R)) - 1).abs() <= 1e-3
            if supplied_grounding:
                safe &= grounded_mask[start:stop]
            elif use_feet:
                foot = body_p[start:stop, feet].detach().to(device=self.device, dtype=self.qpos.dtype)
                if bool(offset.any()):
                    rotation = body_R[start:stop, feet].detach().to(device=self.device, dtype=self.qpos.dtype)
                    foot = foot + (rotation @ offset[:, None]).squeeze(-1)
                support_z = foot[..., 2].amin(-1)
                safe &= (torch.isfinite(foot).all((-1, -2))
                         & (support_z >= -.04 * self.scale) & (support_z <= .10 * self.scale))
            for values, value in zip(chunks, state):
                values.append(value[safe].clone())
        states = tuple(torch.cat(values) for values in chunks)
        eligible = len(states[0])
        if not eligible:
            raise ValueError("Reference motion contains no safe upright grounded reset frames")
        self._reference_reset_states = states
        self._reference_reset_probability = float(probability)
        receipt = {
            "enabled": True, "training_only": True, "probability": float(probability),
            "eligible_frames": eligible, "total_frames": frames,
            "foot_support_filter": supplied_grounding or use_feet,
            "support_filter": ("provided_physical_grounded_mask" if supplied_grounding else
                               "foot_origin_height_heuristic" if use_feet else "root_state_only"),
            "grounded_mask_provided": supplied_grounding,
            "grounded_mask_true_frames": int(grounded_mask.sum().item()) if supplied_grounding else None,
            "source_ground": "floor-aligned reference Z, reanchored to physical terrain",
        }
        self.contract["reference_resets"] = receipt
        return receipt

    def _apply_reference_resets(self, ids):
        selected = torch.rand(len(ids), device=self.device) < self._reference_reset_probability
        ids = ids[selected]
        if not len(ids):
            return
        states = self._reference_reset_states
        frames = torch.randint(len(states[0]), (len(ids),), device=self.device)
        p, R, v, w, q, qd = (value[frames] for value in states)
        spawn_R = quat_to_mat(self.qpos[ids, self._root_qpos + 3:self._root_qpos + 7])
        alignment = yaw_mat(spawn_R) @ yaw_mat(R).transpose(-1, -2)
        root_R = alignment @ R
        xy = self.qpos[ids, self._root_qpos:self._root_qpos + 2]
        self.qpos[ids, self._root_qpos + 2] = self.mdp.field.sample(xy) + p[:, 2]
        self.qpos[ids, self._root_qpos + 3:self._root_qpos + 7] = mat_to_quat(root_R)
        self.qvel[ids, self._root_qvel:self._root_qvel + 3] = (alignment @ v[..., None]).squeeze(-1)
        # MuJoCo free-root angular velocity is body-local; the shared state is world-frame.
        world_w = (alignment @ w[..., None]).squeeze(-1)
        self.qvel[ids, self._root_qvel + 3:self._root_qvel + 6] = (
            root_R.transpose(-1, -2) @ world_w[..., None]).squeeze(-1)
        offsets = self.xi[ids, 1:-3]
        target = (q + offsets).clamp(self.qlo, self.qhi)
        self.qpos[ids[:, None], self._qpos_ids] = target
        self.qvel[ids[:, None], self._qvel_ids] = qd
        self.ctrl[ids[:, None], self._actuator_ids] = target
        self.a_prev[ids] = ((target - self.default_q - offsets) / self.action_scale).clamp(
            -self.action_limit, self.action_limit)
        if self.stealth_keyboard_mode:
            pass
        elif hasattr(self, "_command_ranges"):
            body_v = (R.transpose(-1, -2) @ v[..., None]).squeeze(-1)
            body_w = (R.transpose(-1, -2) @ w[..., None]).squeeze(-1)
            twist = torch.cat((body_v[:, :2], body_w[:, 2:3]), -1)
            if self.mdp.terrain_navigation:
                twist[:, 1:] = 0.
            self.c[ids, 1:4] = twist.clamp(self._command_ranges[:, 0], self._command_ranges[:, 1])

    def set_command_ranges(self, ranges):
        ranges = torch.as_tensor(ranges, device=self.device, dtype=self.c.dtype)
        if (ranges.shape != (3, 2) or not bool(torch.isfinite(ranges).all())
                or bool((ranges[:, 0] > ranges[:, 1]).any())):
            raise ValueError("Command ranges must be finite ordered vx/vy/wz pairs")
        self._command_ranges = ranges.clone()
        self.contract["command_ranges"] = ranges.cpu().tolist()

    @torch.no_grad()
    def sample_commands(self, ids=None, *, jump_height=None):
        """Resample real-data twists and optional jumps; keep lane context.

        ``jump_height=0`` disables jumps for this sample. Configuring a positive
        height and ``jump_probability`` enables periodic optional jump skills.
        This method is never used by the imagined state transition.
        """
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        ids = ids.to(device=self.device, dtype=torch.long)
        height = self.jump_height if jump_height is None else jump_height
        if not math.isfinite(height) or height < 0:
            raise ValueError("Jump height must be finite and nonnegative")
        count = len(ids)
        if not count:
            return
        u = torch.rand(count, 4, device=self.device, dtype=self.c.dtype)
        if self.stealth_keyboard_mode:
            if not hasattr(self, "_stealth_mode_ids") or self._stealth_mode_ids is None:
                self._stealth_mode_ids = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device)
            probabilities = torch.tensor(
                [row[4] for row in STEALTH_KEYBOARD_MODES],
                device=self.device, dtype=self.c.dtype)
            indices = torch.multinomial(probabilities.expand(count, -1), 1).squeeze(-1)
            mode_commands = torch.tensor(
                [row[1:4] for row in STEALTH_KEYBOARD_MODES],
                device=self.device, dtype=self.c.dtype)
            commands = mode_commands[indices]
            self._stealth_mode_ids[ids] = indices
        elif hasattr(self, "_command_ranges"):
            commands = self._command_ranges[:, 0] + u[:, :3] * (
                self._command_ranges[:, 1] - self._command_ranges[:, 0])
            if self.mdp.terrain_navigation:
                commands[:, 0].clamp_min_(.1 * self.scale)
                commands[:, 1:] = 0.
        elif self.mdp.terrain_navigation:
            commands = torch.zeros(count, 3, device=self.device, dtype=self.c.dtype)
            commands[:, 0] = (.5 + .6 * u[:, 0]) * self.scale
        else:
            # Include exact stops and single-axis commands alongside mixed twists.
            mode = torch.randint(0, 6, (count,), device=self.device)
            commands = torch.stack(((2 * u[:, 0] - 1) * 1.1 * self.scale,
                                    (2 * u[:, 1] - 1) * .65 * self.scale,
                                    (2 * u[:, 2] - 1) * 1.2), -1)
            commands[mode == 0] = 0
            for selected in (mode == 1, mode == 2, mode == 3, mode == 4):
                commands[selected] = 0
            forward, reverse, side, turn = (mode == index for index in range(1, 5))
            commands[forward, 0] = (.3 + .8 * u[forward, 0]) * self.scale
            commands[reverse, 0] = -(.3 + .7 * u[reverse, 0]) * self.scale
            side_sign = torch.where(u[side, 1] < .5, -1., 1.)
            commands[side, 1] = side_sign * (.2 + .5 * u[side, 0]) * self.scale
            turn_sign = torch.where(u[turn, 2] < .5, -1., 1.)
            commands[turn, 2] = turn_sign * (.35 + .85 * u[turn, 0])
            if self.mdp.profile == "stealth":
                commands *= .6
        self.c[ids, 1:4] = commands
        self.c[ids, 4] = self.target_clearance_m
        self.c[ids, 5] = torch.where(u[:, 3] < self.jump_probability, height, 0.)
        self.c[ids, 6] = 0
        self._command_age[ids] = 0
        lo, hi = self.command_resample_steps
        self._command_interval[ids] = torch.randint(lo, hi + 1, (count,), device=self.device)

    @torch.no_grad()
    def step(self, action):
        S0 = self.state()
        c0, a_prev = self.c.clone(), self.a_prev.clone()
        policy_action = action.clone()
        a = policy_action.clamp(-self.action_limit, self.action_limit)
        target = (self.default_q + self.xi[:, 1:-3] + a * self.action_scale).clamp(self.qlo, self.qhi)
        with self._warp_context():
            self.ctrl[:, self._actuator_ids] = target
            self.wp.capture_launch(self.physics_graph, stream=self._stream)
        S1 = self.state()
        c1 = advance_command(c0, CONTROL_DT)
        self.c.copy_(c1)
        self.ep_step += 1
        self._command_age += 1
        output_fn = getattr(self, "command_outputs_fn", self.mdp.transition_outputs)
        reward, reward_terms, metrics, fallen = output_fn(S1, a, a_prev, c1)
        self._episode_tracking += metrics["command_tracking"]
        self._episode_velocity_error += metrics["velocity_error_mps"]
        self._episode_yaw_error += metrics["yaw_rate_error_rps"]
        self._episode_clearance_error += metrics["clearance_error_m"]
        fraction = self._episode_tracking / self.ep_step
        timeout = self.ep_step >= self.max_episode_steps
        reached = (S1[0][:, 0] >= self.terrain.goal_x if self.mdp.terrain_navigation
                   else torch.zeros_like(fallen))
        success, done, truncated = terminal_masks(reached, fallen, timeout)
        terminated = fallen | success
        if not self.mdp.terrain_navigation:
            success = timeout & ~fallen & (fraction >= .8) & (self.ep_step >= self.min_success_steps)
        if hasattr(self, "locomotion_reward_fn"):
            penalty = self.locomotion_reward_fn(S0, S1, a, self.xi)
            reward = reward + penalty
            reward_terms["locomotion_regularization"] = penalty
        terminal_reward = self.mdp.terminal_reward(success, fallen)
        reward_terms["terminal"] = terminal_reward
        reward = torch.nan_to_num(reward + terminal_reward, nan=-FALL_PENALTY,
                                  posinf=-FALL_PENALTY, neginf=-FALL_PENALTY)
        self.ep_return += reward
        ids = done.nonzero(as_tuple=False).flatten()
        zero = torch.zeros_like(self.ep_return)
        transition = {
            "s": flatten_state(S0), "s1": flatten_state(S1), "a": a,
            "policy_action": policy_action, "a_prev": a_prev, "xi": self.xi.clone(),
            "c": c0, "c1": c1, "r": reward, "reward_terms": reward_terms,
            "f": self.ep_step - 1, "motion_id": self.levels.clone(),
            "ep_step": self.ep_step - 1, "terminated": terminated,
            "truncated": truncated, "done": done, "command_metrics": metrics,
            "episode_success": success, "episode_fall": fallen & done,
            "episode_timeout": truncated, "episode_survived": timeout & ~fallen,
            "episode_return": torch.where(done, self.ep_return, zero),
            "episode_command_tracking_fraction": torch.where(done, fraction, zero),
            "episode_velocity_error_mps": torch.where(done, self._episode_velocity_error / self.ep_step, zero),
            "episode_yaw_rate_error_rps": torch.where(done, self._episode_yaw_error / self.ep_step, zero),
            "episode_clearance_error_m": torch.where(done, self._episode_clearance_error / self.ep_step, zero),
            "ep_len": torch.where(done, self.ep_step.float(), zero),
            "ep_ret_norm": torch.where(done, self.ep_return / self.ep_step, zero),
            "term": fallen,
        }
        if self.stealth_keyboard_mode:
            transition["stealth_mode_id"] = self._stealth_mode_ids.clone()
        self.a_prev.copy_(a)
        if len(ids):
            if self.curriculum:
                self.levels[ids] = update_terrain_levels(self.levels[ids], success[ids], fallen[ids])
            with self._warp_context():
                self._randomize(ids)
                self._reset_state(ids)
                self.mw.forward(self.model, self.data)
        resample = (self._command_age >= self._command_interval) & ~done
        transition["command_resampled"] = resample
        self.sample_commands(resample.nonzero(as_tuple=False).flatten())
        if self.physical_ray_validation:
            self._update_rays()
        return transition

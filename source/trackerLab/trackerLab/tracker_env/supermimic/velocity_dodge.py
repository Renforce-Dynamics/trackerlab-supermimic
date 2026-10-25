"""PhysX command tracking with repeated kinematic cuboids at arbitrary yaw."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from trackerLab.tracker_env.supermimic.env import SuperMimicSceneCfg, TrackingSim
from trackerLab.tracker_env.mdp.supermimic import flatten_state
from trackerLab.tracker_env.mdp.supermimic_velocity_dodge import (
    advance_scenario, bar_coordinates, bar_pose, make_scenario, next_gait_frame, offset_scenario,
    scenario_state,
)
from trackerTask.supermimic.go2_velocity_dodge_cfg import VelocityDodgeCfg


@configclass
class VelocityDodgeSceneCfg(SuperMimicSceneCfg):
    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/VelocityBar",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 6.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.18, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 3.0)),
    )
    bar_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/VelocityBar", update_period=0.0, history_length=1,
    )


class VelocityDodgeSim(TrackingSim):
    def __init__(self, sim_cfg, num_envs, motion, mdp, task_cfg, eval_mode=False, **kwargs):
        self.task = mdp.task_cfg
        if abs(sim_cfg.dt * sim_cfg.render_interval - self.task.control_dt) > 1e-8:
            raise ValueError("Physics/control duration differs from the transferred world model")
        self.difficulty = 0.0
        device = sim_cfg.device
        self.scenario = torch.zeros(num_envs, self.task.scenario_dim, device=device)
        self.encounter_collided = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.encounter_passed = torch.zeros_like(self.encounter_collided)
        self.collided = torch.zeros_like(self.encounter_collided)
        self.ever_fallen = torch.zeros_like(self.encounter_collided)
        self.encounters = torch.zeros(num_envs, device=device)
        self.successes = torch.zeros_like(self.encounters)
        self.last_crossing_time = torch.zeros_like(self.encounters)
        scene = VelocityDodgeSceneCfg(num_envs=num_envs, env_spacing=task_cfg.env_spacing)
        scene.bar = scene.bar.replace(spawn=scene.bar.spawn.replace(
            size=(2 * self.task.bar_half_thickness, 2 * self.task.bar_half_width,
                  2 * self.task.bar_half_thickness)))
        super().__init__(sim_cfg, num_envs, motion, mdp, task_cfg=task_cfg, eval_mode=eval_mode,
                         scene_cfg=scene, **kwargs)
        self.max_episode_steps = self.task.episode_steps

    def _world_bar_pose(self, scenario):
        pose = bar_pose(scenario)
        pose[:, :3] += self.env_origins
        return pose

    def _teleport_bar(self, scenario, env_ids):
        if not env_ids.numel():
            return
        self.scene["bar"].write_root_pose_to_sim(self._world_bar_pose(scenario)[env_ids], env_ids)
        self.scene["bar_contact"].reset(env_ids)

    def _target_bar(self, scenario):
        pose = self._world_bar_pose(scenario)
        # Isaac Sim 5.1's GPU tensor backend lacks set_kinematic_targets.
        # Drive the collidable body with explicit pose and world velocity.
        self.scene["bar"].write_root_pose_to_sim(pose)
        velocity = torch.zeros(self.num_envs, 6, device=self.device)
        velocity[:, :3] = scenario_state(scenario)[1]
        self.scene["bar"].write_root_velocity_to_sim(velocity)

    @torch.no_grad()
    def _reset(self, env_ids, frame_ids=None, initialize_context=False):
        super()._reset(env_ids, frame_ids, initialize_context)
        if not env_ids.numel():
            return
        states = tuple(s[env_ids] for s in self.state())
        seed = torch.randint(1, 2**20, (len(env_ids),), device=self.device)
        self.scenario[env_ids] = make_scenario(states, self.difficulty, seed, self.task)
        for values in (self.encounter_collided, self.encounter_passed, self.collided,
                       self.ever_fallen, self.encounters, self.successes, self.last_crossing_time):
            values[env_ids] = 0
        self._teleport_bar(self.scenario, env_ids)

    @torch.no_grad()
    def step(self, action):
        s = flatten_state(self.state())
        c = self.scenario.clone()
        f, a_prev, xi = self.f.clone(), self.a_prev.clone(), self.xi.clone()
        active = ~self.done_eval if self.eval_mode else torch.ones_like(self.done_eval)
        a = torch.where(active[:, None], action.clamp(-4, 4), torch.zeros_like(action))
        self.robot.set_joint_position_target(
            self.default_q + xi[:, 1:-3] + a * self.action_scale, joint_ids=self.joint_ids)
        contact = torch.zeros(self.num_envs, device=self.device)
        pose_error = torch.zeros_like(contact)
        rotation_error = torch.zeros_like(contact)
        for substep in range(self.control_decimation):
            target_c = offset_scenario(c, (substep + 1) * self.physics_dt)
            self._target_bar(target_c)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            force = self.scene["bar_contact"].data.net_forces_w.norm(dim=-1).amax(dim=-1)
            contact = torch.maximum(contact, force)
            expected = self._world_bar_pose(target_c)
            actual = self.scene["bar"].data
            pose_error = torch.maximum(pose_error, (actual.root_link_pos_w - expected[:, :3]).norm(dim=-1))
            dot = (actual.root_link_quat_w * expected[:, 3:]).sum(-1).abs().clamp_max(1)
            rotation_error = torch.maximum(rotation_error, 1 - dot)
        S1 = self.state()
        spawn_difficulty = (self.difficulty if
                            (self.task.jump_only or self.task.autonomous_passage) else None)
        reward_c, c1, encounter_end = advance_scenario(
            c, S1, self.task, spawn_difficulty=spawn_difficulty)
        f1 = next_gait_frame(f, self.motion)
        reward, reward_terms = self.mdp.task_reward(S1, f1, a, a_prev, reward_c, terms=True)
        collision = (contact > 1.0) & (c[:, 8] > 0) & active
        fallen = self.mdp.task_termination(S1, reward_c) & active
        timed_out = (self.ep_step + 1 >= self.max_episode_steps) & active & ~fallen
        done = fallen | timed_out
        local = bar_coordinates(S1[0].unsqueeze(-2), reward_c).squeeze(-2)
        passed = ((local[:, 0] > 0.45) & (local[:, 1].abs() < self.task.bar_half_width) &
                  (c[:, 8] > 0) & active)
        crossing = passed & ~self.encounter_passed
        crossing_time = (self.ep_step + 1) * self.task.control_dt
        interval_valid = crossing & (self.last_crossing_time > 0)
        crossing_interval = torch.where(
            interval_valid, crossing_time - self.last_crossing_time, 0.)
        self.last_crossing_time = torch.where(crossing, crossing_time, self.last_crossing_time)
        self.encounter_collided |= collision
        self.encounter_passed |= passed
        self.collided |= collision
        self.ever_fallen |= fallen
        natural_end = encounter_end & active & (c[:, 8] > 0)
        finished = (encounter_end | done) & active & (c[:, 8] > 0)
        success = natural_end & self.encounter_passed & ~self.encounter_collided & ~fallen
        failed = finished & ~success
        encounter_timeout = finished & timed_out & ~natural_end
        self.encounters += finished
        self.successes += success
        reward = torch.where(active, reward, 0)
        self.ep_return += reward
        self.ep_step += active.int()
        self.f, self.scenario, self.a_prev = f1, c1, a.clone()
        ev, ew = self.mdp.command_error(S1, reward_c)
        out = {
            "s": s, "s1": flatten_state(S1), "a": a, "policy_action": action.clone(),
            "a_prev": a_prev, "xi": xi, "r": reward, "reward_terms": reward_terms,
            "f": f, "motion_id": self.motion_id.clone(), "ep_step": self.ep_step - 1,
            "terminated": fallen, "truncated": timed_out, "done": done,
            "scenario": c, "scenario1": c1.clone(), "reward_scenario": reward_c,
            "collision": collision, "bar_contact_force": contact,
            "bar_pose_error": pose_error, "bar_rotation_error": rotation_error,
            "encounter_end": finished, "encounter_success": success,
            "encounter_natural_end": natural_end, "encounter_failure": failed,
            "encounter_timeout": encounter_timeout, "encounter_fall": finished & fallen,
            "encounter_elapsed_s": reward_c[:, 9].clone(),
            "encounter_period_s": c[:, 10].clone(),
            "encounter_clearance_delay_s": (reward_c[:, 9] - c[:, 10]).clamp_min(0),
            "encounter_deadline_miss": finished & (
                reward_c[:, 9] > c[:, 10] + self.task.control_dt + 1e-5),
            "encounter_expired_ahead": natural_end & ~self.encounter_passed & (local[:, 0] < 0),
            "encounter_lateral_bypass": natural_end & ~self.encounter_passed & (
                local[:, 1].abs() >= self.task.bar_half_width),
            "crossing": crossing, "crossing_interval_s": crossing_interval,
            "crossing_interval_valid": interval_valid,
            "encounter_index": c[:, 14].clone(),
            "next_encounter_index": c1[:, 14].clone(),
            "encounters": self.encounters.clone(), "successes": self.successes.clone(),
            "velocity_error_sq": ev.square().sum(-1), "yaw_error_sq": ew.square(),
            "ep_ret_norm": self.ep_return / (self.ep_step * self.task.max_step_reward).clamp_min(1e-8),
        }
        # Respawn the obstacle only. Observation history and reference phase continue.
        changed_ids = (encounter_end & active & ~done).nonzero(as_tuple=False).squeeze(-1)
        self._teleport_bar(c1, changed_ids)
        self.encounter_collided[changed_ids] = False
        self.encounter_passed[changed_ids] = False
        if self.eval_mode:
            self.done_eval |= done
            self.failed_eval |= fallen
        else:
            self._reset(done.nonzero(as_tuple=False).squeeze(-1))
        return out

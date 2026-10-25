"""PhysX Go2 task with a moving, collidable bar in every environment."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from trackerLab.tracker_env.supermimic.env import SuperMimicSceneCfg, TrackingSim
from trackerLab.tracker_env.mdp.supermimic import flatten_state
from trackerLab.tracker_env.mdp.supermimic_chase_dodge import advance_scenario, scenario_state
from trackerTask.supermimic.go2_chase_dodge_cfg import ChaseDodgeCfg


@configclass
class ChaseDodgeSceneCfg(SuperMimicSceneCfg):
    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 6.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.18, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 1.0)),
    )
    bar_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Bar", update_period=0.0, history_length=1,
    )


class ChaseDodgeSim(TrackingSim):
    def __init__(self, sim_cfg, num_envs, motion, mdp, task_cfg, eval_mode=False, **kwargs):
        self.difficulty = 0.0
        self.task = ChaseDodgeCfg()
        self.scenario = torch.zeros(num_envs, self.task.scenario_dim, device=sim_cfg.device)
        self.collided = torch.zeros(num_envs, dtype=torch.bool, device=sim_cfg.device)
        self.passed = torch.zeros_like(self.collided)
        self.ever_fallen = torch.zeros_like(self.collided)
        self.episode_mode = torch.zeros(num_envs, dtype=torch.long, device=sim_cfg.device)
        super().__init__(
            sim_cfg, num_envs, motion, mdp, task_cfg=task_cfg, eval_mode=eval_mode,
            scene_cfg=ChaseDodgeSceneCfg(num_envs=num_envs, env_spacing=task_cfg.env_spacing),
            **kwargs,
        )
        self.max_episode_steps = self.task.episode_steps

    def _write_bar(self, scenario):
        bar = scenario_state(scenario)[2].clone()
        bar[:, 2] = torch.where(scenario[:, 8] > 0, bar[:, 2], 3.0)
        pose = torch.zeros(self.num_envs, 7, device=self.device)
        pose[:, :3] = bar + self.env_origins
        pose[:, 3] = 1.0
        self.scene["bar"].write_root_pose_to_sim(pose)
        velocity = torch.zeros(self.num_envs, 6, device=self.device)
        velocity[:, 0] = scenario[:, 6]
        self.scene["bar"].write_root_velocity_to_sim(velocity)

    @torch.no_grad()
    def _reset(self, env_ids, frame_ids=None, initialize_context=False):
        super()._reset(env_ids, frame_ids, initialize_context)
        if not env_ids.numel():
            return
        n, d = len(env_ids), self.difficulty
        p = self.state()[0][env_ids]
        c = self.scenario[env_ids].clone()
        c.zero_()
        c[:, :2] = p[:, :2]
        c[:, 0] += self.task.target_distance
        c[:, 2] = 0.83 + d * (torch.rand(n, device=self.device) * 0.85 - 0.18)
        c[:, 3] = d * (torch.rand(n, device=self.device) * 0.5 - 0.25)
        c[:, 6] = -d * (0.3 + 0.7 * torch.rand(n, device=self.device))
        c[:, 4] = p[:, 0] + (c[:, 2] - c[:, 6]) * (2.0 + 2.0 * torch.rand(n, device=self.device))
        c[:, 5] = p[:, 1]
        modes = torch.randint(0, 3, (n,), device=self.device) if d > 0.05 else torch.zeros(n, dtype=torch.long, device=self.device)
        c[:, 8] = modes
        c[:, 7] = torch.where(modes == 1, 0.65 - 0.31 * d, 0.025 + 0.095 * d)
        c[:, 10] = d
        c[:, 11] = self.task.normal_height
        self.scenario[env_ids] = c
        self.collided[env_ids] = False
        self.passed[env_ids] = False
        self.ever_fallen[env_ids] = False
        self.episode_mode[env_ids] = modes
        self._write_bar(self.scenario)

    @torch.no_grad()
    def step(self, action):
        S0 = self.state()
        scenario = self.scenario.clone()
        f = self.f.clone()
        a_prev = self.a_prev.clone()
        active = ~self.done_eval if self.eval_mode else torch.ones_like(self.done_eval)
        a = action.clamp(-4, 4)
        a = torch.where(active[:, None], a, torch.zeros_like(a))
        target = self.default_q + self.xi[:, 1:-3] + a * self.action_scale
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        contact = torch.zeros(self.num_envs, device=self.device)
        for substep in range(self.control_decimation):
            self._write_bar(advance_scenario(scenario, (substep + 1) * self.physics_dt))
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            force = self.scene["bar_contact"].data.net_forces_w.norm(dim=-1).amax(dim=-1)
            contact = torch.maximum(contact, force)
        S1 = self.state()
        c1 = advance_scenario(scenario)
        f1 = self.next_frames(f)
        reward, reward_terms = self.mdp.task_reward(S1, f1, a, a_prev, c1, terms=True)
        collision = (contact > 1.0) & (scenario[:, 8] > 0) & active
        # Physical contact is measured separately; the differentiable geometric
        # collision reward remains identical in collection and imagination.
        fallen = self.mdp.task_termination(S1, c1) & active
        timed_out = (self.ep_step + 1 >= self.max_episode_steps) & active & ~fallen
        done = fallen | timed_out
        bar_x = scenario_state(c1)[2][:, 0]
        passed = (S1[0][:, 0] > bar_x + 0.40) & (c1[:, 8] > 0) & active
        self.collided |= collision
        self.passed |= passed
        self.ever_fallen |= fallen
        reward = torch.where(active, reward, 0)
        self.ep_return += reward
        self.ep_step += active.int()
        self.f = f1
        self.scenario = c1
        self.a_prev = a
        out = {
            "s": flatten_state(S0), "s1": flatten_state(S1), "a": a, "policy_action": action,
            "a_prev": a_prev, "xi": self.xi.clone(), "r": reward, "reward_terms": reward_terms,
            "f": f, "motion_id": self.motion_id.clone(), "ep_step": self.ep_step - 1,
            "terminated": fallen, "truncated": timed_out, "done": done,
            "scenario": scenario, "scenario1": c1.clone(),
            "collision": collision, "bar_contact_force": contact, "passed": passed,
            "episode_success": self.passed & ~self.collided & ~self.ever_fallen,
            "episode_collision": self.collided.clone(), "episode_mode": self.episode_mode.clone(),
            "ep_ret_norm": self.ep_return / (self.ep_step * 0.1).clamp_min(0.1),
        }
        if self.eval_mode:
            self.done_eval |= done
            self.failed_eval |= fallen
        else:
            self._reset(done.nonzero(as_tuple=False).squeeze(-1))
        return out

    def next_frames(self, f):
        clip = self.motion.clip_id[f]
        return torch.where(f + 1 >= self.motion.clip_end[clip], self.motion.clip_start[clip], f + 1)

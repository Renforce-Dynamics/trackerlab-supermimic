"""Differentiable task signals, shared by PhysX and imagination.

Scenario columns: target initial XY, target velocity XY, bar initial X/Y,
bar X velocity, bar Z, mode (0 flat, 1 duck, 2 jump), elapsed seconds,
curriculum difficulty, nominal root height.
"""

import torch

from trackerLab.kinematics.supermimic.rotations import mat_to_6d, yaw_mat
from trackerLab.tracker_env.mdp.supermimic import MDP, invariant_features
from trackerTask.supermimic.go2_chase_dodge_cfg import ChaseDodgeCfg


TERMS = (
    "velocity", "target_distance", "upright", "height", "gait_style", "alive",
    "action_smoothness", "joint_limit", "bar_collision",
)


def advance_scenario(scenario, dt=0.02):
    delta = torch.zeros_like(scenario)
    delta[..., 9] = dt
    return scenario + delta


def scenario_state(c):
    t = c[..., 9]
    wave = 0.3 * c[..., 10]
    target = c[..., :2] + c[..., 2:4] * t.unsqueeze(-1)
    target = target + torch.stack((torch.zeros_like(t), wave * torch.sin(0.7 * t)), -1)
    velocity = c[..., 2:4] + torch.stack((torch.zeros_like(t), 0.7 * wave * torch.cos(0.7 * t)), -1)
    bar = torch.stack((c[..., 4] + c[..., 6] * t, c[..., 5], c[..., 7]), -1)
    return target, velocity, bar


def desired_rotation(c):
    velocity = scenario_state(c)[1]
    heading = torch.atan2(velocity[..., 1], velocity[..., 0])
    cs, sn, zero, one = heading.cos(), heading.sin(), torch.zeros_like(heading), torch.ones_like(heading)
    return torch.stack((cs, -sn, zero, sn, cs, zero, zero, zero, one), -1).reshape(*heading.shape, 3, 3)


class ChaseDodgeMDP(MDP):
    def __init__(self, *args, task_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_cfg = task_cfg or ChaseDodgeCfg()

    def obs(self, S, a_prev, xi, f, scenario=None):
        o = super().obs(S, a_prev, xi, f)
        if scenario is not None:
            relative = mat_to_6d(S[1].transpose(-1, -2) @ desired_rotation(scenario))
            o = torch.cat((o[..., :-6], relative), -1)
        return o

    def task_features(self, S, c):
        p, R, v, w = S[:4]
        target, tv, bar = scenario_state(c)
        Ht = yaw_mat(R).transpose(-1, -2)
        target3 = torch.cat((target - p[..., :2], torch.zeros_like(p[..., 2:3])), -1)
        tv3 = torch.cat((tv, torch.zeros_like(p[..., 2:3])), -1)
        target_local = (Ht @ target3.unsqueeze(-1)).squeeze(-1)[..., :2]
        velocity_local = (Ht @ tv3.unsqueeze(-1)).squeeze(-1)[..., :2]
        bar_local = (Ht @ (bar - p).unsqueeze(-1)).squeeze(-1)
        bar_v = torch.stack((c[..., 6], torch.zeros_like(c[..., 6]), torch.zeros_like(c[..., 6])), -1)
        relative_bar_v = (Ht @ (bar_v - v).unsqueeze(-1)).squeeze(-1)[..., :2]
        own_v = (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
        mode = torch.stack(((c[..., 8] == 1).float(), (c[..., 8] == 2).float()), -1)
        active = (c[..., 8] > 0).unsqueeze(-1)
        return torch.cat((
            target_local.clamp(-5, 5) / 3, velocity_local / 2,
            torch.where(active, bar_local.clamp(-5, 5), torch.zeros_like(bar_local)) / 3,
            torch.where(active, relative_bar_v, torch.zeros_like(relative_bar_v)) / 3,
            mode, own_v / 2, p[..., 2:3], w[..., 2:3],
            self.desired_height(S, c).unsqueeze(-1), c[..., 10:11],
        ), -1)

    def desired_height(self, S, c):
        target, tv, bar = scenario_state(c)
        crossing_time = (bar[..., 0] - S[0][..., 0]) / (tv[..., 0] - c[..., 6]).clamp_min(0.3)
        enabled = (c[..., 8] > 0).float()
        duck = (c[..., 8] == 1).float()
        jump = (c[..., 8] == 2).float()
        # Anticipate the bar using its relative approach time.
        duck_gate = torch.exp(-((crossing_time - 0.05) / 0.70) ** 4)
        jump_gate = torch.exp(-((crossing_time - 0.06) / 0.33) ** 2)
        crouch = (bar[..., 2] - self.task_cfg.bar_half_thickness - 0.10).clamp(0.17, 0.29)
        lift = (bar[..., 2] + self.task_cfg.bar_half_thickness + 0.08) * jump_gate
        return c[..., 11] + enabled * (duck * duck_gate * (crouch - c[..., 11]) + jump * lift)

    def collision_probability(self, S, c):
        full = self.kin.forward(*S)
        bp = full[0]
        bar = scenario_state(c)[2]
        # Smooth clearance around tracked body centers and four feet.
        from trackerTask.supermimic.go2_robot import FOOT_OFFSET
        calf_ids = [list(self.kin.robot.tracked_bodies).index(f"{leg}_calf") for leg in ("FL", "FR", "RL", "RR")]
        foot = full[0][..., calf_ids, :] + (
            full[1][..., calf_ids, :, :] @ bp.new_tensor(FOOT_OFFSET).view(3, 1)
        ).squeeze(-1)
        points = torch.cat((bp, foot), -2)
        radii = torch.full((points.shape[-2],), 0.035, device=bp.device)
        radii[0] = 0.10
        radii[-4:] = 0.022
        # The base's long body is sampled along its forward axis as well.
        offsets = bp.new_tensor([[-0.18, 0, 0], [0.18, 0, 0]])
        torso = S[0].unsqueeze(-2) + (S[1].unsqueeze(-3) @ offsets.unsqueeze(-1)).squeeze(-1)
        points = torch.cat((points, torso), -2)
        radii = torch.cat((radii, bp.new_full((2,), 0.09)))
        dx = (points[..., 0] - bar[..., None, 0]).abs()
        dz = (points[..., 2] - bar[..., None, 2]).abs()
        dy = (points[..., 1] - bar[..., None, 1]).abs()
        proximity = torch.sigmoid((radii + 0.025 - dx) / 0.018)
        proximity = proximity * torch.sigmoid((radii + 0.025 - dz) / 0.018)
        proximity = proximity * torch.sigmoid((self.task_cfg.bar_half_width - dy) / 0.04)
        return proximity.amax(-1) * (c[..., 8] > 0).float()

    def task_reward(self, S, f, a, a_prev, c, terms=False):
        p, R, v, w, q, qd = S
        target, tv, bar = scenario_state(c)
        direction = tv / tv.norm(dim=-1, keepdim=True).clamp_min(0.1)
        desired_p = target - self.task_cfg.target_distance * direction
        height_error = p[..., 2] - self.desired_height(S, c)
        encounter = torch.exp(-((bar[..., 0] - p[..., 0]) / 0.8) ** 2) * (c[..., 8] > 0)
        style = torch.exp(-((q - self.motion.q[f]) ** 2).mean(-1) / 0.5**2)
        style = style * (1 - 0.9 * encounter) + 0.9 * encounter
        values = (
            0.035 * torch.exp(-((v[..., :2] - tv) ** 2).sum(-1) / 0.6**2),
            0.015 * torch.exp(-((p[..., :2] - desired_p) ** 2).sum(-1) / 1.0**2),
            0.01 * torch.exp(-((1 - R[..., 2, 2]) / 0.3) ** 2),
            0.02 * torch.exp(-(height_error / 0.07) ** 2),
            0.01 * style,
            torch.full_like(height_error, 0.01),
            -0.001 * ((a - a_prev) ** 2).sum(-1),
            -0.04 * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1),
            -0.10 * self.collision_probability(S, c),
        )
        reward = sum(values)
        return (reward, dict(zip(TERMS, values))) if terms else reward

    def task_termination(self, S, c):
        p, R = S[:2]
        return (p[..., 2] < self.task_cfg.termination_height) | (R[..., 2, 2] < self.task_cfg.termination_upright)

    def task_soft_termination(self, S, c):
        p, R = S[:2]
        fall = torch.sigmoid((self.task_cfg.termination_height - p[..., 2]) / 0.015)
        tilt = torch.sigmoid((self.task_cfg.termination_upright - R[..., 2, 2]) / 0.04)
        return 1 - (1 - fall) * (1 - tilt)

    def task_critic_features(self, S, xi, c):
        return torch.cat((invariant_features(S), xi, self.task_features(S, c)), -1)

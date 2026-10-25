"""Shared commanded-velocity objective and deterministic encounter transitions.

Scenario layout (15 floats):
0:3 body vx/vy/yaw commands; 3:6 bar initial XYZ; 6 bar normal yaw;
7 bar speed along minus-normal; 8 mode (0 flat, 1 duck, 2 jump);
9 encounter age; 10 duration; 11 difficulty; 12 nominal height;
13 episode seed; 14 encounter index.
"""

import math
import torch

from trackerLab.kinematics.supermimic.rotations import mat_to_6d, yaw_mat
from trackerLab.tracker_env.mdp.supermimic import MDP, invariant_features
from trackerTask.supermimic.go2_velocity_dodge_cfg import VelocityDodgeCfg


TERMS = (
    "body_velocity", "yaw_rate", "upright", "height", "jump_clearance",
    "gait_style", "alive", "action_smoothness", "joint_limit", "bar_collision",
)


def reward_term_names(cfg):
    return tuple("passage_clearance" if cfg.autonomous_passage and name == "jump_clearance"
                 else name for name in TERMS)


def yaw_rotation(angle):
    cs, sn = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack((cs, -sn, zero, sn, cs, zero, zero, zero, one), -1).reshape(
        *angle.shape, 3, 3)


def offset_scenario(c, dt):
    delta = torch.zeros_like(c)
    delta[..., 9] = dt
    return c + delta


def scenario_state(c):
    """Return bar center, world velocity and rotation; local X is its normal."""
    rotation = yaw_rotation(c[..., 6])
    velocity = -rotation[..., :, 0] * c[..., 7:8]
    center = c[..., 3:6] + velocity * c[..., 9:10]
    center = torch.cat((center[..., :2], torch.where(
        c[..., 8:9] > 0, center[..., 2:3], torch.full_like(center[..., 2:3], 3.0))), -1)
    return center, velocity, rotation


def bar_pose(c):
    """The exact wxyz pose used by the physical cuboid and geometric reward."""
    center = scenario_state(c)[0]
    angle = c[..., 6] * 0.5
    zero = torch.zeros_like(angle)
    quaternion = torch.stack((angle.cos(), zero, zero, angle.sin()), -1)
    return torch.cat((center, quaternion), -1)


def bar_coordinates(points, c):
    center, _, rotation = scenario_state(c)
    return (rotation.transpose(-1, -2).unsqueeze(-3) @
            (points - center.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)


def crossing_state(S, c):
    center, velocity, rotation = scenario_state(c)
    normal = rotation[..., :, 0]
    distance = ((center - S[0]) * normal).sum(-1)
    closing = ((S[2] - velocity) * normal).sum(-1)
    denominator = torch.where(closing >= 0, closing.clamp_min(0.15), closing.clamp_max(-0.15))
    return distance, closing, distance / denominator


def _encounter_random(c, encounter_index=None):
    # Integer hashing makes the next draw reproducible from replay, on CPU or GPU.
    salt = torch.arange(8, device=c.device, dtype=torch.int64)
    index = c[..., 14:15] if encounter_index is None else encounter_index.unsqueeze(-1)
    value = c[..., 13:14].long() + 104729 * index.long() + 15485863 * salt
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & 0x7FFFFFFF
    value = ((value ^ (value >> 16)) * 0x45D9F3B) & 0x7FFFFFFF
    value = value ^ (value >> 16)
    return value.to(c.dtype) / 2147483648.0


def _spawn_encounter(S, c, cfg):
    u = _encounter_random(c)
    command_u = u
    if cfg.command_hold_encounters > 1:
        command_index = torch.floor(c[..., 14] / cfg.command_hold_encounters)
        command_u = _encounter_random(c, command_index)
    d = c[..., 11]
    vx = cfg.initial_vx + d * (
        cfg.vx_range[0] + (cfg.vx_range[1] - cfg.vx_range[0]) * command_u[..., 0] - cfg.initial_vx)
    vy = d * cfg.vy_max * (2 * command_u[..., 1] - 1)
    wz = d * cfg.yaw_rate_max * (2 * command_u[..., 2] - 1)
    command = torch.stack((vx, vy, wz), -1)
    H = yaw_mat(S[1])
    heading = torch.atan2(H[..., 1, 0], H[..., 0, 0])
    sector = torch.remainder(c[..., 13] + c[..., 14], 8)
    if cfg.front_only:
        offset = d * ((sector - 3.5) * (math.pi / 4) +
                      (u[..., 3] - 0.5) * (math.pi / 4))
    else:
        offset = (sector - 3.5) * (math.pi / 4) + (u[..., 3] - 0.5) * (math.pi / 4)
    direction_scale = 0.10 + 0.90 * d if cfg.autonomous_passage else d
    angle = heading + direction_scale * offset
    normal = yaw_rotation(angle)[..., :, 0]
    initial_speed = cfg.initial_bar_speed[0] + u[..., 4] * (
        cfg.initial_bar_speed[1] - cfg.initial_bar_speed[0])
    final_speed = cfg.bar_speed_range[0] + u[..., 4] * (
        cfg.bar_speed_range[1] - cfg.bar_speed_range[0])
    speed = initial_speed + d * (final_speed - initial_speed)
    world_command = (H @ torch.stack((vx, vy, torch.zeros_like(vx)), -1).unsqueeze(-1)).squeeze(-1)
    speed = torch.maximum(speed, cfg.minimum_approach_speed - (world_command * normal).sum(-1))
    initial_period = cfg.initial_period[0] + u[..., 5] * (cfg.initial_period[1] - cfg.initial_period[0])
    final_period = cfg.period_range[0] + u[..., 5] * (cfg.period_range[1] - cfg.period_range[0])
    period = torch.round((initial_period + d * (final_period - initial_period)) / cfg.control_dt) * cfg.control_dt
    lead = cfg.lead_fraction * period
    # Integrate a constant body twist to aim the bar at the future commanded path.
    turn = wz * lead
    sinc = torch.sinc(turn / math.pi)
    cosc = 0.5 * turn * torch.sinc(turn / (2 * math.pi)).square()
    displacement = torch.stack((lead * (sinc * vx - cosc * vy),
                                lead * (cosc * vx + sinc * vy), torch.zeros_like(vx)), -1)
    if cfg.enforce_period_clearance:
        lead_projection = ((H @ displacement.unsqueeze(-1)).squeeze(-1) * normal).sum(-1)
        full_turn = wz * period
        full_sinc = torch.sinc(full_turn / math.pi)
        full_cosc = 0.5 * full_turn * torch.sinc(full_turn / (2 * math.pi)).square()
        full_displacement = torch.stack((
            period * (full_sinc * vx - full_cosc * vy),
            period * (full_cosc * vx + full_sinc * vy), torch.zeros_like(vx),
        ), -1)
        full_projection = ((H @ full_displacement.unsqueeze(-1)).squeeze(-1) * normal).sum(-1)
        # D(0) = projected_motion(lead) + speed*lead >= spawn clearance.
        # -D(T) = projected_motion(T)-projected_motion(lead) + speed*(T-lead).
        spawn_speed = (cfg.respawn_clearance + 0.3 - lead_projection) / lead
        exit_speed = (cfg.respawn_clearance + 0.1 - full_projection + lead_projection) / (period - lead)
        speed = torch.maximum(speed, torch.maximum(spawn_speed, exit_speed))
    center = S[0] + (H @ displacement.unsqueeze(-1)).squeeze(-1) + normal * (speed * lead).unsqueeze(-1)
    # A fresh bar is always outside the robot's normal-axis collision extent.
    separation = ((center - S[0]) * normal).sum(-1)
    center = center + normal * (cfg.respawn_clearance + 0.3 - separation).clamp_min(0).unsqueeze(-1)
    mode = torch.where(u[..., 6] < cfg.flat_fraction, 0.0,
                       torch.where(u[..., 6] < (1 + cfg.flat_fraction) / 2, 1.0, 2.0))
    if cfg.jump_only:
        mode = torch.full_like(mode, 2.0)
    if cfg.flat_warmup:
        mode = torch.where(d <= 0.03, torch.zeros_like(mode), mode)
    high = cfg.initial_high_bar_height + d * (
        cfg.high_bar_range[0] + u[..., 7] * (
            cfg.high_bar_range[1] - cfg.high_bar_range[0]) -
        cfg.initial_high_bar_height)
    low = cfg.initial_low_bar_height + d * (
        cfg.low_bar_range[0] + u[..., 7] * (
            cfg.low_bar_range[1] - cfg.low_bar_range[0]) -
        cfg.initial_low_bar_height)
    if cfg.autonomous_passage:
        # Height varies from the first rollout; mode is a reporting bucket only.
        high = high + (1 - d) * 0.04 * (u[..., 7] - 0.5)
        low = low + (1 - d) * 0.02 * (u[..., 7] - 0.5)
    height = torch.where(mode == 1, high, low)
    center = torch.cat((center[..., :2], height.unsqueeze(-1)), -1)
    return torch.cat((command, center, angle.unsqueeze(-1), speed.unsqueeze(-1),
                      mode.unsqueeze(-1), torch.zeros_like(mode).unsqueeze(-1),
                      period.unsqueeze(-1), c[..., 11:15]), -1)


def make_scenario(S, difficulty, seed, cfg=None):
    cfg = cfg or VelocityDodgeCfg()
    if not 0 <= difficulty <= 1:
        raise ValueError("difficulty must lie in [0, 1]")
    c = S[0].new_zeros((*S[0].shape[:-1], cfg.scenario_dim))
    c[..., 11], c[..., 12], c[..., 13] = difficulty, cfg.normal_height, seed
    return _spawn_encounter(S, c, cfg)


def advance_scenario(c, S1, cfg=None, spawn_difficulty=None):
    """Return (reward scenario, next scenario, encounter-ended mask).

    The old bar moves for the entire control step. Reward uses that end pose.
    An expired bar respawns only after this step, clear of the robot. The new
    encounter uses a deterministic draw; it never resets robot state. Its
    state-relative placement retains derivatives through position and heading.
    """
    cfg = cfg or VelocityDodgeCfg()
    end = offset_scenario(c, cfg.control_dt)
    local = bar_coordinates(S1[0].unsqueeze(-2), end).squeeze(-2)
    clear = ((local[..., 0].abs() > cfg.respawn_clearance) |
             (local[..., 1].abs() > cfg.bar_half_width + cfg.respawn_clearance) |
             (end[..., 8] == 0))
    changed = (end[..., 9] >= end[..., 10] - 1e-5) & clear
    next_seed = end.clone()
    next_seed[..., 14] = next_seed[..., 14] + 1
    if spawn_difficulty is not None:
        next_seed[..., 11] = spawn_difficulty
    spawned = _spawn_encounter(S1, next_seed.detach(), cfg)
    return end, torch.where(changed.unsqueeze(-1), spawned, end), changed


def next_gait_frame(f, motion):
    clip = motion.clip_id[f]
    return torch.where(f + 1 > motion.clip_end[clip], motion.clip_start[clip], f + 1)


class VelocityDodgeMDP(MDP):
    def __init__(self, *args, task_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_cfg = task_cfg or VelocityDodgeCfg()

    def obs(self, S, a_prev, xi, f, scenario=None):
        o = super().obs(S, a_prev, xi, f)
        if scenario is not None:
            desired = yaw_mat(S[1]) @ yaw_rotation(0.3 * scenario[..., 2])
            relative = mat_to_6d(S[1].transpose(-1, -2) @ desired)
            o = torch.cat((o[..., :-6], relative), -1)
        return o

    def desired_height(self, S, c):
        if self.task_cfg.autonomous_passage:
            return c[..., 12]
        center, _, _ = scenario_state(c)
        _, closing, ttc = crossing_state(S, c)
        lateral = bar_coordinates(S[0].unsqueeze(-2), c)[..., 0, 1].abs()
        active = ((c[..., 8] > 0) & (closing > 0.15) &
                  (lateral < self.task_cfg.bar_half_width + 0.3)).to(c.dtype)
        duck = (c[..., 8] == 1).to(c.dtype)
        jump = (c[..., 8] == 2).to(c.dtype)
        duck_gate = torch.exp(-((ttc - 0.05) / 0.65) ** 4)
        if self.task_cfg.jump_clearance_weight > 0:
            jump_gate = torch.sigmoid((1.0 - ttc) / 0.25) * torch.sigmoid((ttc + 0.15) / 0.12)
        else:
            jump_gate = torch.exp(-((ttc - 0.06) / 0.33) ** 2)
        crouch = (center[..., 2] - self.task_cfg.bar_half_thickness - 0.10).clamp(0.17, 0.29)
        if self.task_cfg.jump_clearance_weight > 0:
            target_height = (center[..., 2] + self.task_cfg.normal_height +
                             self.task_cfg.bar_half_thickness + 0.08)
            lift = target_height - c[..., 12]
        else:
            lift = center[..., 2] + self.task_cfg.bar_half_thickness + 0.08
        return c[..., 12] + active * (
            duck * duck_gate * (crouch - c[..., 12]) + jump * jump_gate * lift)

    def task_features(self, S, c):
        p, R, v, w = S[:4]
        Ht = yaw_mat(R).transpose(-1, -2)
        center, velocity, rotation = scenario_state(c)
        relative = (Ht @ (center - p).unsqueeze(-1)).squeeze(-1)
        relative_v = (Ht @ (velocity - v).unsqueeze(-1)).squeeze(-1)
        normal = (Ht @ rotation[..., :, 0].unsqueeze(-1)).squeeze(-1)[..., :2]
        own_v = (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
        virtual_target = 0.8 * c[..., :2] / c[..., :2].norm(dim=-1, keepdim=True).clamp_min(0.1)
        mode = torch.stack((c[..., 8] == 1, c[..., 8] == 2), -1).to(p.dtype)
        if self.task_cfg.autonomous_passage:
            mode = torch.zeros_like(mode)
        active = (c[..., 8:9] > 0).to(p.dtype)
        # First 18 columns retain the chase head's feature meanings and scales.
        legacy = torch.cat((
            virtual_target / 3, c[..., :2] / 2,
            active * relative.clamp(-5, 5) / 3, active * relative_v[..., :2] / 3,
            mode, own_v / 2, p[..., 2:3], w[..., 2:3],
            self.desired_height(S, c).unsqueeze(-1), c[..., 11:12],
        ), -1)
        extra = torch.cat((c[..., 2:3] / self.task_cfg.yaw_rate_max, active * normal,
                           (c[..., 9:10] / c[..., 10:11]).clamp(0, 2),
                           c[..., 10:11] / 5, c[..., 7:8] / 4), -1)
        return torch.cat((legacy, extra), -1)

    def collision_probability(self, S, c, geometry=None):
        from trackerTask.supermimic.go2_robot import FOOT_OFFSET

        bp, bR = geometry if geometry is not None else self.kin.forward(*S)[:2]
        calf_ids = [list(self.kin.robot.tracked_bodies).index(f"{leg}_calf")
                    for leg in ("FL", "FR", "RL", "RR")]
        feet = bp[..., calf_ids, :] + (
            bR[..., calf_ids, :, :] @ bp.new_tensor(FOOT_OFFSET).view(3, 1)).squeeze(-1)
        offsets = bp.new_tensor([[-0.18, 0, 0], [0.18, 0, 0]])
        torso = S[0].unsqueeze(-2) + (S[1].unsqueeze(-3) @ offsets.unsqueeze(-1)).squeeze(-1)
        points = torch.cat((bp, feet, torso), -2)
        radii = bp.new_full((points.shape[-2],), 0.035)
        radii[0], radii[-6:-2], radii[-2:] = 0.10, 0.022, 0.09
        cfg = self.task_cfg
        half = bp.new_tensor((cfg.bar_half_thickness, cfg.bar_half_width, cfg.bar_half_thickness))
        distances = bar_coordinates(points, c).abs() - half
        outside = (distances.clamp_min(0).square().sum(-1) + 1e-12).sqrt()
        signed_distance = outside + distances.amax(-1).clamp_max(0)
        proximity = torch.sigmoid((radii - signed_distance) / 0.015)
        return proximity.amax(-1) * (c[..., 8] > 0).to(bp.dtype)

    def passage_geometry(self, S, geometry=None):
        """Body samples for either over-bar or under-bar passage."""
        from trackerTask.supermimic.go2_robot import FOOT_OFFSET

        bp, bR = geometry if geometry is not None else self.kin.forward(*S)[:2]
        calf_ids = [list(self.kin.robot.tracked_bodies).index(f"{leg}_calf")
                    for leg in ("FL", "FR", "RL", "RR")]
        calves = bp[..., calf_ids, :]
        feet = calves + (bR[..., calf_ids, :, :] @
                         bp.new_tensor(FOOT_OFFSET).view(3, 1)).squeeze(-1)
        shin = (calves + feet) * 0.5
        offsets = bp.new_tensor([[-0.18, 0, 0], [0.18, 0, 0], [0.26, 0, 0.015]])
        torso = S[0].unsqueeze(-2) + (S[1].unsqueeze(-3) @ offsets.unsqueeze(-1)).squeeze(-1)
        points = torch.cat((bp, shin, feet, torso), -2)
        radii = bp.new_full((points.shape[-2],), 0.035)
        radii[0], radii[-7:-3], radii[-3:] = 0.10, 0.022, 0.09
        return points, radii, feet

    def passage_clearance(self, S, c, geometry=None):
        """Smooth union of above/below clearance, without a prescribed action."""
        points, radii, _ = self.passage_geometry(S, geometry)
        local = bar_coordinates(points, c)
        radius = radii + self.task_cfg.bar_half_thickness + 0.01
        # Clearance is positive on either side; broad gradients also reach
        # grounded feet. Each limb can cross at its own time.
        above = (local[..., 2] - radius) / 0.06
        below = (-local[..., 2] - radius) / 0.06
        approaching = torch.exp(-(local[..., 0] / 0.65).square())
        in_span = torch.sigmoid((self.task_cfg.bar_half_width - local[..., 1].abs()) / 0.05)
        active = (c[..., 8:9] > 0).to(points.dtype)
        risk = approaching * in_span * active
        # A mean over the worst samples prevents safe torso samples from hiding
        # a colliding foot, while retaining gradients to several body points.
        routes = []
        for vertical_safe in (torch.sigmoid(above), torch.sigmoid(below)):
            safety = 1 - risk * (1 - vertical_safe)
            routes.append(-0.08 * (torch.logsumexp(-safety / 0.08, dim=-1) -
                                   math.log(safety.shape[-1])))
        # Select a coherent whole-body route: allowing each point to choose
        # independently would reward feet below a low bar and torso above it.
        return 0.04 * (torch.logsumexp(torch.stack(routes, -1) / 0.04, dim=-1) -
                       math.log(2))

    def jump_clearance(self, S, c, geometry=None):
        """Reward predicted foot/calf clearance over the bar during crossing."""
        from trackerTask.supermimic.go2_robot import FOOT_OFFSET

        bp, bR = geometry if geometry is not None else self.kin.forward(*S)[:2]
        calf_ids = [list(self.kin.robot.tracked_bodies).index(f"{leg}_calf")
                    for leg in ("FL", "FR", "RL", "RR")]
        feet = bp[..., calf_ids, :] + (
            bR[..., calf_ids, :, :] @ bp.new_tensor(FOOT_OFFSET).view(3, 1)).squeeze(-1)
        points = torch.cat((bp[..., calf_ids, :], feet), -2)
        local = bar_coordinates(points, c)
        # A positive margin means the body point has passed above the top face.
        radii = bp.new_tensor([0.035] * 4 + [0.022] * 4)
        vertical_margin = local[..., 2] - self.task_cfg.bar_half_thickness - radii - 0.01
        vertical_clear = torch.sigmoid(vertical_margin / 0.025)
        within_span = (local[..., 1].abs() < self.task_cfg.bar_half_width).to(points.dtype)
        near_crossing = torch.exp(-(local[..., 0] / 0.45).square())
        jump = (c[..., 8] == 2).to(points.dtype).unsqueeze(-1)
        return (vertical_clear * within_span * near_crossing * jump).mean(-1)

    def jump_clearance_metric(self, S, c):
        """Return minimum signed foot/calf clearance and whether crossing was sampled."""
        from trackerTask.supermimic.go2_robot import FOOT_OFFSET

        bp, bR = self.kin.forward(*S)[:2]
        calf_ids = [list(self.kin.robot.tracked_bodies).index(f"{leg}_calf")
                    for leg in ("FL", "FR", "RL", "RR")]
        calves = bp[..., calf_ids, :]
        feet = calves + (bR[..., calf_ids, :, :] @
                         bp.new_tensor(FOOT_OFFSET).view(3, 1)).squeeze(-1)
        local = bar_coordinates(torch.cat((calves, feet), -2), c)
        radii = bp.new_tensor([0.035] * 4 + [0.022] * 4)
        margin = local[..., 2] - self.task_cfg.bar_half_thickness - radii
        near = ((local[..., 0].abs() <= 0.25) &
                (local[..., 1].abs() <= self.task_cfg.bar_half_width) &
                (c[..., 8:9] == 2))
        valid = near.any(-1)
        margin = margin.masked_fill(~near, torch.inf).amin(-1)
        return torch.where(valid, margin, torch.ones_like(margin)), valid

    def command_error(self, S, c):
        Rt = S[1].transpose(-1, -2)
        body_v = (Rt @ S[2].unsqueeze(-1)).squeeze(-1)
        body_w = (Rt @ S[3].unsqueeze(-1)).squeeze(-1)
        return body_v[..., :2] - c[..., :2], body_w[..., 2] - c[..., 2]

    def task_reward(self, S, f, a, a_prev, c, terms=False):
        cfg = self.task_cfg
        p, R, _, _, q, _ = S
        ev, ew = self.command_error(S, c)
        distance = crossing_state(S, c)[0]
        encounter = torch.exp(-(distance / 0.8) ** 2) * (c[..., 8] > 0)
        style = torch.exp(-((q - self.motion.q[f]) ** 2).mean(-1) / 0.5**2)
        style = style * (1 - 0.9 * encounter) + 0.9 * encounter
        geometry = self.kin.forward(*S)[:2] if cfg.jump_clearance_weight > 0 else None
        clearance = (self.jump_clearance(S, c, geometry) if geometry is not None
                     else torch.zeros_like(ew))
        height = torch.exp(-((p[..., 2] - self.desired_height(S, c)) / 0.07) ** 2)
        velocity = torch.exp(-ev.square().sum(-1) / 0.6**2)
        smoothness, collision_weight = 0.001, 0.10
        if cfg.autonomous_passage:
            clearance = self.passage_clearance(S, c, geometry)
            # Relax trot, root height and horizontal speed near a bar to allow
            # takeoff, crouching and landing within the same policy.
            maneuver = torch.exp(-(distance / 1.0).square()) * (c[..., 8] > 0)
            style = style * (1 - maneuver) + maneuver
            height = height * (1 - maneuver) + maneuver
            velocity = velocity * (1 - 0.7 * maneuver) + 0.7 * maneuver
            smoothness, collision_weight = 0.0005, 0.20
        values = (
            cfg.velocity_weight * velocity,
            cfg.yaw_rate_weight * torch.exp(-ew.square() / 0.6**2),
            cfg.upright_weight * torch.exp(-((1 - R[..., 2, 2]) / 0.3) ** 2),
            cfg.height_weight * height,
            cfg.jump_clearance_weight * clearance,
            cfg.gait_style_weight * style,
            torch.full_like(ew, cfg.alive_weight),
            -smoothness * (a - a_prev).square().sum(-1),
            -0.04 * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1),
            -collision_weight * self.collision_probability(S, c, geometry),
        )
        reward = sum(values)
        return (reward, dict(zip(reward_term_names(cfg), values))) if terms else reward

    def task_termination(self, S, c):
        return ((S[0][..., 2] < self.task_cfg.termination_height) |
                (S[1][..., 2, 2] < self.task_cfg.termination_upright))

    def task_soft_termination(self, S, c):
        fall = torch.sigmoid((self.task_cfg.termination_height - S[0][..., 2]) / 0.015)
        tilt = torch.sigmoid((self.task_cfg.termination_upright - S[1][..., 2, 2]) / 0.04)
        return 1 - (1 - fall) * (1 - tilt)

    def task_critic_features(self, S, xi, c):
        return torch.cat((invariant_features(S), xi, self.task_features(S, c)), -1)

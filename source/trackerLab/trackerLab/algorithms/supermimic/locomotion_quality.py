"""Reference-free foot and actuator regularization for command locomotion."""

import torch

from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS
from trackerLab.tracker_env.mdp.supermimic_command import CONTROL_DT


class LocomotionQuality:
    def __init__(self, kin, field, motors):
        self.kin, self.field = kin, field
        spec = kin.robot
        tracked = [kin.body_names[index] for index in kin.tracked_idx]
        self.feet = [tracked.index(name) for name in FOOT_BODIES[spec.name]]
        device = kin.device
        self.offset = torch.tensor(FOOT_OFFSETS.get(spec.name, (0., 0., 0.)), device=device)
        self.default_q = torch.tensor(spec.default_q, device=device)
        self.action_scale = torch.tensor(spec.action_scale, device=device)
        motor = torch.tensor(motors, device=device)
        self.effort, self.velocity_limit, self.kp, self.kd = motor[:, :4].unbind(1)
        self.qlo, self.qhi = kin.q_lo, kin.q_hi
        self.biped = len(self.feet) == 2
        self.sole_height = torch.zeros(len(self.feet), device=device)

    @torch.no_grad()
    def calibrate(self, motion):
        positions, *_ = self.foot_state(
            tuple(getattr(motion, name) for name in
                  ("root_p", "root_R", "root_v", "root_w", "q", "qd")))
        self.sole_height = positions[..., 2].quantile(.1, dim=0).clamp(
            -.03 * self.field.scale, .08 * self.field.scale)

    def foot_state(self, state):
        p, R, v, w = self.kin.forward(*state)
        arm = (R[..., self.feet, :, :] @ self.offset[..., None]).squeeze(-1)
        return (p[..., self.feet, :] + arm, R[..., self.feet, :, :],
                v[..., self.feet, :] + torch.cross(w[..., self.feet, :], arm, dim=-1))

    def measurements(self, state, previous=None):
        position, rotation, velocity = self.foot_state(state)
        scale = self.field.scale
        height = position[..., 2] - self.field.sample(position[..., :2]) - self.sole_height
        support = torch.sigmoid((.02 * scale - height) / (.01 * scale))
        speed = torch.sqrt(velocity[..., :2].square().sum(-1) + 1e-8) / scale
        slip = (speed * support).sum(-1) / support.sum(-1).clamp_min(1)
        flat = (rotation[..., 2, :2].square().sum(-1) * support).mean(-1)
        separation = torch.zeros_like(slip)
        if self.biped:
            delta = position[..., 0, :2] - position[..., 1, :2]
            heading = state[1][..., :2, 0]
            lateral = torch.stack((-heading[..., 1], heading[..., 0]), -1)
            separation = (delta * lateral).sum(-1).abs() / scale
        acceleration = torch.zeros_like(slip)
        if previous is not None:
            acceleration = ((state[5] - previous[5]) / CONTROL_DT).square().mean(-1)
        return {"support_foot_slip_scale_m_s": slip,
                "support_foot_tilt_squared": flat,
                "foot_separation_scale_m": separation,
                "joint_acceleration_squared": acceleration,
                "support_heuristic": support}

    def reward(self, previous_state, state, action, xi):
        metrics = self.measurements(state, previous_state)
        target = (self.default_q + xi[..., 1:-3] + action * self.action_scale).clamp(
            self.qlo, self.qhi)
        torque = (self.kp * (target - state[4]) - self.kd * state[5]).clamp(
            -self.effort, self.effort)
        terms = {
            "foot_slide": -.008 * metrics["support_foot_slip_scale_m_s"],
            "foot_tilt": -.008 * metrics["support_foot_tilt_squared"] if self.biped else
                         torch.zeros_like(metrics["support_foot_tilt_squared"]),
            "feet_close": -.008 * (torch.relu(.1 - metrics["foot_separation_scale_m"]) / .1).square()
                          if self.biped else torch.zeros_like(metrics["foot_separation_scale_m"]),
            "joint_acceleration": -2.5e-9 * metrics["joint_acceleration_squared"],
            "motor_torque": -3e-9 * torque.square().sum(-1),
            "joint_speed_limit": -.02 * torch.relu(
                state[5].abs() - .9 * self.velocity_limit).square().mean(-1),
            "body_roll_pitch_rate": -.001 * (
                state[1].transpose(-1, -2) @ state[3][..., None]
            ).squeeze(-1)[..., :2].square().sum(-1),
        }
        return sum(terms.values())

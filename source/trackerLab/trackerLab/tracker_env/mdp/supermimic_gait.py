"""Reference gait imitation with explicit foot and joint posture constraints."""

import torch

from trackerLab.kinematics.supermimic.rotations import yaw_mat
from trackerLab.tracker_env.mdp.supermimic import MDP
from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS
from trackerTask.supermimic.terrain import ROBOT_HEIGHTS


class GaitMDP(MDP):
    def __init__(self, kin, motion, default_q, **kwargs):
        super().__init__(kin, motion, default_q, **kwargs)
        robot = kin.robot.name
        self.scale = ROBOT_HEIGHTS[robot] / .76
        tracked = [kin.body_names[index] for index in kin.tracked_idx]
        self.feet = torch.tensor([tracked.index(name) for name in FOOT_BODIES[robot]],
                                 device=default_q.device)
        self.foot_offset = torch.tensor(FOOT_OFFSETS.get(robot, (0., 0., 0.)),
                                        device=default_q.device)
        self.reference_feet, reference_velocity = self.foot_state(
            (motion.body_p, motion.body_R, motion.body_v, motion.body_w))
        self.reference_stance = torch.zeros(
            motion.num_frames, len(self.feet), dtype=torch.bool, device=default_q.device)
        for start, end in zip(motion.clip_start.tolist(), motion.clip_end.tolist()):
            foot = self.reference_feet[start:end + 1]
            speed = reference_velocity[start:end + 1, ..., :2].norm(dim=-1)
            floor = foot[..., 2].quantile(.10, dim=0)
            self.reference_stance[start:end + 1] = (
                (foot[..., 2] <= floor + .035 * self.scale) & (speed < .25 * self.scale))

    def foot_state(self, body_state):
        p, R, v, w = body_state
        arm = (R[..., self.feet, :, :] @ self.foot_offset[..., None]).squeeze(-1)
        return (p[..., self.feet, :] + arm,
                v[..., self.feet, :] + torch.cross(w[..., self.feet, :], arm, dim=-1))

    def gait_errors(self, S, f, body_state=None):
        p, R, v, w, q, qd = S
        body_state = self.kin.forward(p, R, v, w, q, qd) if body_state is None else body_state
        foot_position, foot_velocity = self.foot_state(body_state)
        ref_root = self.motion.root_p[f]
        alignment = yaw_mat(R @ self.motion.root_R[f].transpose(-1, -2))
        anchor = torch.cat((p[..., :2], ref_root[..., 2:3]), -1)
        reference = anchor.unsqueeze(-2) + (
            alignment.unsqueeze(-3)
            @ (self.reference_feet[f] - ref_root.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        foot_error = (foot_position - reference) / self.scale
        stance = self.reference_stance[f]
        foot_speed = foot_velocity[..., :2] / self.scale
        return {
            "foot_tracking": foot_error.square().sum(-1).mean(-1),
            "stance_slip": (foot_speed.square().sum(-1) * stance).sum(-1) / stance.sum(-1).clamp_min(1),
            "joint_tracking": (q - self.motion.q[f]).square().mean(-1),
        }

    def reward(self, S, f, a, a_prev, terms=False):
        body_state = self.kin.forward(*S)
        reward, contributions = super().reward(S, f, a, a_prev, terms=True, body_state=body_state)
        errors = self.gait_errors(S, f, body_state)
        additions = {
            "gait_foot_tracking": -.004 * errors["foot_tracking"] / .06 ** 2,
            "gait_stance_slip": -.002 * errors["stance_slip"],
            "gait_joint_tracking": -.002 * errors["joint_tracking"] / .3 ** 2,
        }
        reward = reward + sum(additions.values())
        return (reward, contributions | additions) if terms else reward

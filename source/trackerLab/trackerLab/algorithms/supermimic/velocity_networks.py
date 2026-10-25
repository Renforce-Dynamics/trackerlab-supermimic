"""Chase-weight-preserving adapters for body commands and oriented bars."""

import torch
from torch import nn

from trackerLab.kinematics.supermimic.rotations import so3_exp, yaw_mat
from trackerLab.tracker_env.mdp.supermimic_velocity_dodge import scenario_state
from .chase_networks import TaskPolicy, TaskWorldModel
from .networks import RunningNorm, mlp, run_mlp


def _load_chase(module, state_dict, added):
    result = module.load_state_dict(state_dict, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith(added)]
    if missing or result.unexpected_keys:
        raise ValueError(f"Incompatible chase checkpoint: missing={missing}, unexpected={result.unexpected_keys}")
    return result


class VelocityPolicy(TaskPolicy):
    """Retain all chase tensors; add a zero-output command/orientation branch."""

    def __init__(self, obs_dim=48, goal_dim=390, hist=10, num_joints=12, action_limit=4.0):
        super().__init__(obs_dim, goal_dim, hist=hist, num_joints=num_joints,
                         action_limit=action_limit, task_dim=18)
        self.command_norm = RunningNorm(6)
        self.command_head = mlp(6 + obs_dim, [256, 128], 32)
        nn.init.zeros_(self.command_head[-1].weight)
        nn.init.zeros_(self.command_head[-1].bias)

    def load_chase(self, state_dict):
        return _load_chase(self, state_dict, ("command_norm.", "command_head."))

    @torch.no_grad()
    def update_task_norms(self, features):
        self.task_norm.update(features[..., :18])
        self.command_norm.update(features[..., 18:])

    def forward(self, obs_hist, goal, task_features):
        o = self.obs_norm(obs_hist)
        legacy = torch.cat((self.task_norm(task_features[..., :18]), o[..., -1, :]), -1)
        command = torch.cat((self.command_norm(task_features[..., 18:]), o[..., -1, :]), -1)
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)) +
                       run_mlp(self.task_head, legacy) + run_mlp(self.command_head, command))
        action = run_mlp(self.dec, torch.cat((z, o.flatten(-2)), -1))
        if self.action_limit is not None:
            action = self.action_limit * torch.tanh(action / self.action_limit)
        return action


class VelocityWorldModel(TaskWorldModel):
    """Keep the chase dynamics and add the missing horizontal bar orientation."""

    def __init__(self, hidden=(2048, 2048, 2048), num_joints=12, geometry_only=False):
        super().__init__(hidden=hidden, num_joints=num_joints)
        self.geometry_only = geometry_only
        self.orientation_norm = RunningNorm(2)
        self.orientation_adapter = mlp(
            self.in_dim + 8 + 2, [256, 128], 12 + 2 * num_joints, act=nn.SiLU)
        nn.init.zeros_(self.orientation_adapter[-1].weight)
        nn.init.zeros_(self.orientation_adapter[-1].bias)

    def load_chase(self, state_dict):
        return _load_chase(self, state_dict, ("orientation_norm.", "orientation_adapter."))

    def task_inputs(self, S, c):
        center, velocity, _ = scenario_state(c)
        Ht = yaw_mat(S[1]).transpose(-1, -2)
        relative_p = (Ht @ (center - S[0]).unsqueeze(-1)).squeeze(-1)
        relative_v = (Ht @ (velocity - S[2]).unsqueeze(-1)).squeeze(-1)
        active = (c[..., 8:9] > 0).to(S[0].dtype)
        mode = torch.stack((c[..., 8] == 1, c[..., 8] == 2), -1).to(S[0].dtype)
        if self.geometry_only:
            mode = torch.zeros_like(mode)
        return torch.cat((active * relative_p, active * relative_v, mode), -1)

    def orientation_inputs(self, S, c):
        normal = scenario_state(c)[2][..., :, 0]
        local = (yaw_mat(S[1]).transpose(-1, -2) @ normal.unsqueeze(-1)).squeeze(-1)
        return local[..., :2] * (c[..., 8:9] > 0).to(local.dtype)

    def step(self, S, a, xi_n, scenario):
        p, R, v, w, q, qd = S
        x = self.in_norm(self.inputs(S, a, xi_n))
        bar = self.bar_norm(self.task_inputs(S, scenario))
        normal = self.orientation_norm(self.orientation_inputs(S, scenario))
        legacy = torch.cat((x, bar), -1)
        y = (run_mlp(self.net, x) + run_mlp(self.bar_adapter, legacy) +
             run_mlp(self.orientation_adapter, torch.cat((legacy, normal), -1)))
        d = y * self.out_norm.std() + self.out_norm.mean
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        p1 = p + (yaw_mat(R) @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return p1, R1, v1, w1, q + dq, qd + dqd

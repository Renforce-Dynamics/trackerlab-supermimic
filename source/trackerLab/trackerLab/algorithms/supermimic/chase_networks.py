"""Go2 task adapters that preserve gait checkpoints at initialization."""

from __future__ import annotations

import torch
import torch.nn as nn

from trackerLab.kinematics.supermimic.rotations import so3_exp, yaw_mat
from trackerLab.tracker_env.mdp.supermimic_chase_dodge import scenario_state

from .networks import Policy, RunningNorm, WorldModel, mlp, run_mlp


class TaskPolicy(Policy):
    """Add a task residual before the gait encoder's latent tanh."""

    def __init__(
        self, obs_dim: int = 48, goal_dim: int = 390, hist: int = 10,
        z_dim: int = 32, num_joints: int = 12,
        action_limit: float | None = 4.0, task_dim: int = 18,
    ):
        super().__init__(
            obs_dim, goal_dim, hist=hist, z_dim=z_dim,
            num_joints=num_joints, action_limit=action_limit,
        )
        self.task_norm = RunningNorm(task_dim)
        self.task_head = mlp(task_dim + obs_dim, [256, 128], z_dim)
        nn.init.zeros_(self.task_head[-1].weight)
        nn.init.zeros_(self.task_head[-1].bias)

    def load_gait(self, state_dict):
        """Load the original Policy keys, leaving the task branch untouched."""
        return self.load_state_dict(state_dict, strict=False)

    def forward(
        self, obs_hist: torch.Tensor, goal: torch.Tensor,
        task_features: torch.Tensor,
    ) -> torch.Tensor:
        o = self.obs_norm(obs_hist)
        task_input = torch.cat((self.task_norm(task_features), o[..., -1, :]), -1)
        delta = run_mlp(self.task_head, task_input)
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)) + delta)
        action = run_mlp(self.dec, torch.cat((z, o.flatten(-2)), -1))
        if self.action_limit is not None:
            action = self.action_limit * torch.tanh(action / self.action_limit)
        return action


class TaskWorldModel(WorldModel):
    """Condition normalized gait dynamics increments on a moving bar."""

    def __init__(self, hidden=(2048, 2048, 2048), num_joints: int = 12):
        super().__init__(hidden=hidden, num_joints=num_joints)
        self.bar_norm = RunningNorm(8)
        self.bar_adapter = mlp(
            self.in_dim + 8, [256, 128], 12 + 2 * num_joints, act=nn.SiLU,
        )
        nn.init.zeros_(self.bar_adapter[-1].weight)
        nn.init.zeros_(self.bar_adapter[-1].bias)

    def load_gait(self, state_dict):
        """Keep the original dynamics weights and normalization statistics."""
        return self.load_state_dict(state_dict, strict=False)

    def task_inputs(self, S, c):
        """Yaw-frame relative position/velocity, then duck/jump indicators.

        Units are meters and meters per second before ``bar_norm``. Flat
        scenarios have zero bar features. Callers update ``bar_norm`` from
        collected data; forward rollouts leave all normalization buffers alone.
        """
        p, R, v = S[:3]
        bar = scenario_state(c)[2]
        zero = torch.zeros_like(c[..., 6])
        bar_velocity = torch.stack((c[..., 6], zero, zero), -1)
        Ht = yaw_mat(R).transpose(-1, -2)
        relative_p = (Ht @ (bar - p).unsqueeze(-1)).squeeze(-1)
        relative_v = (Ht @ (bar_velocity - v).unsqueeze(-1)).squeeze(-1)
        geometry = torch.cat((relative_p, relative_v), -1)
        active = (c[..., 8] > 0).unsqueeze(-1)
        geometry = torch.where(active, geometry, torch.zeros_like(geometry))
        mode = torch.stack((c[..., 8] == 1, c[..., 8] == 2), -1).to(p.dtype)
        return torch.cat((geometry, mode), -1)

    def step(self, S, a, xi_n, scenario):
        p, R, v, w, q, qd = S
        x = self.in_norm(self.inputs(S, a, xi_n))
        bar = self.bar_norm(self.task_inputs(S, scenario))
        y = run_mlp(self.net, x) + run_mlp(self.bar_adapter, torch.cat((x, bar), -1))
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), -1,
        )
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return p1, R1, v1, w1, q + dq, qd + dqd

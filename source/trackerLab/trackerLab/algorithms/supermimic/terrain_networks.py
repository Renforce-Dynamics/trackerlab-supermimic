"""Terrain-conditioned branches on the shared SuperMimic learner networks."""

import torch
from torch import nn

from trackerLab.algorithms.supermimic.networks import (
    Policy, RunningNorm, WorldModel, mlp, run_mlp,
)
from trackerLab.kinematics.supermimic.rotations import so3_exp, yaw_mat


class TerrainPolicy(Policy):
    """Encode each height scan before the existing history decoder."""

    def __init__(self, obs_dim, goal_dim, hist=10, num_joints=29, scan_dim=187):
        self.scan_dim = scan_dim
        proprio_dim = obs_dim - scan_dim
        super().__init__(proprio_dim + 32, goal_dim, hist=hist, num_joints=num_joints)
        self.obs_norm = RunningNorm(obs_dim)
        self.scan_encoder = mlp(scan_dim, [64], 32, act=nn.SiLU)

    def forward(self, obs_hist, goal):
        normalized = self.obs_norm(obs_hist)
        scan = run_mlp(self.scan_encoder, normalized[..., -self.scan_dim:])
        history = torch.cat((normalized[..., :-self.scan_dim], scan), -1).flatten(-2)
        latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        return run_mlp(self.dec, torch.cat((latent, history), -1))


class TerrainWorldModel(WorldModel):
    def __init__(self, field, num_joints, hidden=(2048, 2048, 2048)):
        super().__init__(hidden=hidden, num_joints=num_joints)
        self.field = field
        self.terrain_norm = RunningNorm(len(field.model_offsets))
        output_dim = 12 + 2 * num_joints
        self.terrain_adapter = mlp(self.in_dim + len(field.model_offsets), [256, 128],
                                   output_dim, act=nn.SiLU)
        nn.init.zeros_(self.terrain_adapter[-1].weight)
        nn.init.zeros_(self.terrain_adapter[-1].bias)

    @torch.no_grad()
    def update_terrain_norm(self, S):
        self.terrain_norm.update(self.field.model_features(S))

    def step(self, S, a, xi_n):
        p, R, v, w, q, qd = S
        x = self.inputs(S, a, xi_n)
        x_normalized = self.in_norm(x)
        terrain = self.terrain_norm(self.field.model_features(S))
        y = run_mlp(self.net, x_normalized)
        y = y + run_mlp(self.terrain_adapter, torch.cat((x_normalized, terrain), -1))
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

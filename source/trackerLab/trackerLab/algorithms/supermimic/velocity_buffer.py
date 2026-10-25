"""The chase replay sampling contract with the new task's scenario layout."""

import torch

from .chase_buffer import ChaseReplayBuffer
from trackerTask.supermimic.go2_velocity_dodge_cfg import VelocityDodgeCfg


class VelocityReplayBuffer(ChaseReplayBuffer):
    def __init__(self, T, N, device):
        super().__init__(T, N, device)
        for name in ("scenario", "scenario1"):
            self.fields[name] = (VelocityDodgeCfg().scenario_dim, torch.float32)
            self.data[name] = torch.zeros(T, N, self.fields[name][0], device=device)

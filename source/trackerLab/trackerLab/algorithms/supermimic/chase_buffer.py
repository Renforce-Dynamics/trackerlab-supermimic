"""Scenario-conditioned replay using the original temporal sampling rules."""

import torch

from trackerLab.algorithms.supermimic.buffer import ReplayBuffer


class ChaseReplayBuffer(ReplayBuffer):
    def __init__(self, T, N, device):
        super().__init__(T, N, device, num_joints=12)
        for name in ("scenario", "scenario1"):
            self.fields[name] = (12, torch.float32)
            self.data[name] = torch.zeros(T, N, 12, device=device)

    def sample_windows(self, B, k):
        age = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        rows = self._row(age[:, None] + torch.arange(k, device=self.device))
        out = {key: self.data[key][rows, env[:, None]] for key in (
            "s", "s1", "a", "xi", "done", "scenario", "scenario1",
        )}
        out["mask"] = torch.cumprod(torch.cat((
            torch.ones(B, 1, device=self.device), 1 - out["done"][:, :-1].float(),
        ), -1), -1)
        return out

    def sample_starts(self, B, hist):
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.minimum(torch.arange(hist - 1, -1, -1, device=self.device)[None], ep_step[:, None])
        back = torch.minimum(back, age[:, None])
        hrows = self._row(age[:, None] - back)
        keys = ("s", "a_prev", "xi", "f", "scenario")
        return (
            {key: self.data[key][row, env] for key in keys},
            {key: self.data[key][hrows, env[:, None]] for key in keys},
        )

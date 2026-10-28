"""Unmodified pre-repair statistics formula, used only by numerical admission.

The original network source SHA256 was
2a91ab8d6c60c1d85525d60fdcaa6ad4a6635e8cf4a0590f9a485734a9de012a.
Training uses the canonical RunningNorm implementation in networks.py.
"""

import torch
from torch import nn


class LegacyRunningNorm(nn.Module):
    def __init__(self, dim, max_count=2e8, eps=1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)

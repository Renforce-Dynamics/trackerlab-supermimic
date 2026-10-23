"""Per-environment reference state; simulator transitions live in ReplayBuffer."""

from __future__ import annotations

import torch

from .motion_lib.supermimic import CONTROL_DT, MotionLib


class SuperMimicMotionBuffer:
    """TrackerLab-style motion IDs and times, using exact 50 Hz frame indices."""

    def __init__(self, motion: MotionLib, num_envs: int, device: str):
        self._motion_lib = motion
        self.num_envs, self.device = num_envs, device
        self._motion_dt = CONTROL_DT
        self.frames = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._motion_ids = torch.zeros_like(self.frames)

    @property
    def _motion_times(self):
        return (self.frames - self._motion_lib.clip_start[self._motion_ids]) * self._motion_dt

    @property
    def _motion_lengths(self):
        return (self._motion_lib.clip_len[self._motion_ids] - 1) * self._motion_dt

    def update_motion_ids(self, env_ids: torch.Tensor, frame_ids: torch.Tensor):
        self._motion_ids[env_ids] = self._motion_lib.clip_id[frame_ids]
        self.frames[env_ids] = frame_ids

    def next_frames(self):
        return torch.minimum(self.frames + 1, self._motion_lib.clip_end[self._motion_ids])

    def update_motion_times(self, frames: torch.Tensor | None = None):
        self.frames = self.next_frames() if frames is None else frames

    def ref_state(self, env_ids=None):
        frames = self.frames if env_ids is None else self.frames[env_ids]
        return self._motion_lib.ref_state(frames)

    def goal(self, future_steps: int):
        return self._motion_lib.goal(self.frames, future_steps)

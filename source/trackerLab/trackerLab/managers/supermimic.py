"""Reference lifecycle and return-based reset sampling for SuperMimic."""

from __future__ import annotations

import torch

from trackerLab.motion_buffer.supermimic import SuperMimicMotionBuffer


class SuperMimicMotionManager:
    """Own a reference buffer and choose reset frames without changing the MDP."""

    def __init__(self, motion, num_envs, device, gamma, max_step_reward):
        if not 0 < gamma < 1:
            raise ValueError("gamma must lie between zero and one")
        self.motion_lib, self.device, self.gamma = motion, device, gamma
        self._motion_buffer = SuperMimicMotionBuffer(motion, num_envs, device)
        self.bin_returns = torch.zeros(motion.num_bins, device=device)
        self.max_bin_return = max_step_reward / (1.0 - gamma)
        self.bin_past_weights = torch.full_like(self.bin_returns, 1.0 / motion.num_bins)
        self.reset_bin = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def frames(self):
        return self._motion_buffer.frames

    @frames.setter
    def frames(self, value):
        self._motion_buffer.frames = value

    @property
    def motion_ids(self):
        return self._motion_buffer._motion_ids

    def sample_frames(self, env_ids):
        motion = self.motion_lib
        count = len(env_ids)
        soft_probs = torch.softmax(-25.0 * self.bin_returns / self.max_bin_return, dim=0)
        self.bin_past_weights.mul_(0.999).add_(soft_probs, alpha=0.001)
        past_probs = self.bin_past_weights / self.bin_past_weights.sum().clamp_min(1e-8)
        probabilities = 0.5 / motion.num_bins + 0.4 * soft_probs + 0.1 * past_probs
        bins = torch.multinomial(probabilities, count, replacement=True)
        starts = motion.bin_start[bins]
        clip_ids = motion.bin_clip[bins]
        clip_ends = motion.clip_end[clip_ids]
        starts = torch.minimum(starts, clip_ends - 1)
        end_exclusive = torch.minimum(motion.bin_start[bins] + 50, clip_ends)
        spans = (end_exclusive - starts).clamp_min(1)
        offsets = torch.floor(torch.rand(count, device=self.device) * spans).long()
        return clip_ids, starts + offsets

    def reset(self, env_ids, frame_ids=None):
        if frame_ids is None:
            _, frame_ids = self.sample_frames(env_ids)
        frame_ids = frame_ids.to(device=self.device, dtype=torch.long)
        self._motion_buffer.update_motion_ids(env_ids, frame_ids)
        self.reset_bin[env_ids] = self.motion_lib.frame_bin[frame_ids]
        return frame_ids

    def compute(self, frames=None):
        self._motion_buffer.update_motion_times(frames)

    def current_reference(self, env_ids=None):
        return self._motion_buffer.ref_state(env_ids)

    def goal(self, future_steps):
        return self._motion_buffer.goal(future_steps)

    def update_returns(self, finished, terminated, ep_step, ep_return, discounted_return):
        lengths = ep_step[finished].float()
        bootstrap = torch.where(
            terminated[finished],
            0.0,
            (self.gamma**lengths)
            * (ep_return[finished] / lengths.clamp_min(1))
            / (1.0 - self.gamma),
        )
        returns = discounted_return[finished] + bootstrap
        bins = self.reset_bin[finished]
        sums = torch.zeros_like(self.bin_returns).index_add_(0, bins, returns)
        counts = torch.bincount(bins, minlength=self.motion_lib.num_bins).to(self.bin_returns.dtype)
        observed = counts > 0
        means = sums / counts.clamp_min(1)
        self.bin_returns[observed] = 0.8 * self.bin_returns[observed] + 0.2 * means[observed]

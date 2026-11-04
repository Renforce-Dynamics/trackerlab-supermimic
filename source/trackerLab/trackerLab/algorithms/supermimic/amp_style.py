"""Adversarial motion-prior rewards for differentiable SuperMimic rollouts.

Only ``observe`` supplies policy training data, from detached, real simulator
transitions. Call it with ``valid=~done`` to exclude terminal/reset transitions.
``reward`` neither collects data nor updates normalization, and returns a
bounded reward rate for the task to scale. Freeze the discriminator during
actor updates with ``requires_grad_(False)`` or ``freeze()``.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from functools import partial
import hashlib
import json
import math
from numbers import Integral
from typing import Callable

import torch
from torch import nn

from .networks import RunningNorm, mlp
from trackerLab.kinematics.supermimic.rotations import mat_to_6d, yaw_mat


State = tuple[torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, torch.Tensor, torch.Tensor]
_TAIL_DIMS = (1, 2, 1, 1, 1, 1)
_STATE_NAMES = ("root_p", "root_R", "root_v", "root_w", "q", "qd")


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class AMPStyle(nn.Module):
    """Least-squares discriminator over consecutive invariant state frames.

    ``features`` / ``pair_features`` and ``frame_features`` are tensor-only
    feature paths suitable for caller-owned ``torch.compile``. Pair features
    have shape ``(*batch, feature_dim)``; both endpoints share one frame norm.
    Dependencies (kinematics and terrain) remain owned by the caller, on the
    same device as the motion library and imagined states.

    ``checkpoint()`` includes discriminator, normalization and Adam state.
    ``restore(payload)`` verifies expert identity and empties transient replay;
    collect real transitions before the next discriminator update.
    """

    RAW_LIMIT = 1e4
    NORM_LIMIT = 10.0
    MIN_STD = 0.1
    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        kin,
        motion,
        robot_height: float,
        ground_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
        hidden=(256, 128),
        replay_capacity: int = 65536,
        *,
        lr: float = 1e-4,
        gradient_penalty_weight: float = 10.0,
        feature_mode: str = "full",
        history_steps: int = 2,
        gradient_penalty_scope: str = "expert",
        activation: str = "ELU",
        optimizer_name: str = "Adam",
        weight_decay_coefficient: float = 0.,
        logit_weight_decay_coefficient: float = 0.,
        minimum_forward_speed: float = 0.,
    ):
        super().__init__()
        if feature_mode not in ("full", "proprioceptive"):
            raise ValueError("feature_mode must be 'full' or 'proprioceptive'")
        self.feature_mode = feature_mode
        self.history_steps = _positive_int(history_steps, "history_steps")
        if self.history_steps < 2:
            raise ValueError("history_steps must be at least two")
        if gradient_penalty_scope not in ("expert", "both"):
            raise ValueError("gradient_penalty_scope must be expert or both")
        self.gradient_penalty_scope = gradient_penalty_scope
        activations = {"ELU": nn.ELU, "ReLU": nn.ReLU}
        optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
        if activation not in activations or optimizer_name not in optimizers:
            raise ValueError("Unsupported AMP activation or optimizer")
        regularizers = (weight_decay_coefficient, logit_weight_decay_coefficient)
        if any(not math.isfinite(value) or value < 0 for value in regularizers):
            raise ValueError("Discriminator regularization must be finite and nonnegative")
        self.activation, self.optimizer_name = activation, optimizer_name
        self.weight_decay_coefficient = float(weight_decay_coefficient)
        self.logit_weight_decay_coefficient = float(logit_weight_decay_coefficient)
        if not math.isfinite(minimum_forward_speed) or minimum_forward_speed < 0:
            raise ValueError("minimum_forward_speed must be finite and nonnegative")
        self.minimum_forward_speed = float(minimum_forward_speed)
        self._scoring_only = False
        if not math.isfinite(robot_height) or robot_height <= 0:
            raise ValueError("robot_height must be finite and positive")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(gradient_penalty_weight) or gradient_penalty_weight < 0:
            raise ValueError("gradient_penalty_weight must be finite and nonnegative")
        if ground_sampler is not None and not callable(ground_sampler):
            raise ValueError("ground_sampler must be callable")
        self.robot_height = float(robot_height)
        self.replay_capacity = _positive_int(replay_capacity, "replay_capacity")
        hidden = tuple(_positive_int(width, "hidden width") for width in hidden)
        self.gradient_penalty_weight = float(gradient_penalty_weight)
        # These dependencies may themselves be modules; they are not learner parameters.
        object.__setattr__(self, "kin", kin)
        object.__setattr__(self, "ground_sampler", partial(ground_sampler) if ground_sampler else None)

        states = tuple(getattr(motion, name) for name in _STATE_NAMES)
        if not isinstance(states[4], torch.Tensor) or states[4].ndim != 2:
            raise ValueError("Expert q must have shape (frames, joints)")
        self.num_joints = states[4].shape[-1]
        if self.num_joints == 0:
            raise ValueError("Experts must contain joints")
        self._validate_state(states)
        num_frames = states[0].shape[0]
        if states[0].ndim != 2 or num_frames == 0:
            raise ValueError("Empty experts: expected recorded motion frames")
        if getattr(kin, "num_joints", self.num_joints) != self.num_joints:
            raise ValueError("Expert joints must match kinematics")
        device = states[4].device
        starts, ends, pair_starts = self._clip_indices(motion, num_frames, device)
        if self.history_steps > 2:
            ids = motion.clip_id.to(device)
            span = self.history_steps - 1
            pair_starts = torch.arange(max(0, num_frames - span), device=device)
            pair_starts = pair_starts[ids[pair_starts] == ids[pair_starts + span]]
            if not len(pair_starts):
                raise ValueError("No complete within-clip expert history windows")
        if self.minimum_forward_speed:
            R = states[1]
            body_v = (R.transpose(-1, -2) @ states[2][..., None]).squeeze(-1)
            starts_window = torch.arange(max(0, num_frames - self.history_steps + 1), device=device)
            offsets = torch.arange(self.history_steps, device=device)
            forward_windows = body_v[starts_window[:, None] + offsets, 0]
            moving_count = (forward_windows >= self.minimum_forward_speed).sum(-1)
            valid_window = ((moving_count >= self.history_steps - 2)
                            & (forward_windows.mean(-1) >= self.minimum_forward_speed))
            ids = motion.clip_id.to(device)
            same_clip = ids[starts_window] == ids[starts_window + self.history_steps - 1]
            allowed = torch.zeros(num_frames, dtype=torch.bool, device=device)
            allowed[starts_window] = valid_window & same_clip
            pair_starts = pair_starts[allowed[pair_starts]]
            if not len(pair_starts):
                raise ValueError("No sustained forward-moving expert history windows")
        self._set_default_q(device)
        self.num_tracked_bodies = 0
        if self.feature_mode == "full":
            with torch.no_grad():
                sample = tuple(value[:1].detach().float() for value in states)
                if not all(bool(torch.isfinite(value).all()) for value in sample):
                    raise ValueError("Nonfinite expert state")
                local_bodies = self._local_bodies(sample)
            if local_bodies.ndim != 3 or local_bodies.shape[-1] != 3 or local_bodies.shape[-2] == 0:
                raise ValueError("Kinematics must provide tracked-body positions")
            self.num_tracked_bodies = local_bodies.shape[-2]
        self.frame_dim = (9 + 2 * self.num_joints if self.feature_mode == "proprioceptive"
                          else 13 + 2 * self.num_joints + 3 * self.num_tracked_bodies)
        self.feature_dim = self.history_steps * self.frame_dim
        self.norm = RunningNorm(self.frame_dim, eps=self.MIN_STD).to(device)
        self.hidden = hidden
        self.discriminator = mlp(self.feature_dim, list(hidden), 1, act=activations[activation]).to(device)
        with torch.no_grad():
            self.discriminator[-1].weight.mul_(0.1)
            self.discriminator[-1].bias.zero_()
        self.optimizer = optimizers[optimizer_name](
            self.discriminator.parameters(), lr=lr,
            **({"betas": (.9, .999)} if optimizer_name == "AdamW" else {}),
        )

        self.feature_layout = self._feature_layout()
        if self.history_steps > 2:
            self.feature_layout["history_steps"] = self.history_steps
            self.feature_layout["endpoints"] = "oldest-to-newest full history"
        with torch.no_grad():
            expert_frames = torch.empty(num_frames, self.frame_dim, device=device)
            for start in range(0, num_frames, 8192):
                end = min(start + 8192, num_frames)
                chunk = tuple(value[start:end].detach() for value in states)
                if not all(bool(torch.isfinite(value).all()) for value in chunk):
                    raise ValueError("Nonfinite expert state")
                features = self.frame_features(chunk, reference=True)
                if not bool(torch.isfinite(features).all()):
                    raise ValueError("Nonfinite expert features")
                expert_frames[start:end].copy_(features)
                self.norm.update(features.clamp(-self.RAW_LIMIT, self.RAW_LIMIT))
        self.register_buffer("_expert_frames", expert_frames, persistent=False)
        self.register_buffer("_expert_starts", pair_starts, persistent=False)
        self.register_buffer(
            "_replay", torch.empty(self.replay_capacity, self.feature_dim, device=device),
            persistent=False,
        )
        self._replay_size = self._replay_cursor = self._observed = self._updates = 0
        digest = hashlib.sha256()
        digest.update(json.dumps(self.feature_layout, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(json.dumps([starts.tolist(), ends.tolist()], separators=(",", ":")).encode("ascii"))
        # Hash canonical FP32 bytes in chunks, without a second full host copy.
        for chunk in expert_frames.split(8192):
            digest.update(chunk.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes())
        self.expert_hash = digest.hexdigest()
        self.expert_provenance = {
            "provenance": copy.deepcopy(getattr(motion, "provenance", {})),
            "names": list(getattr(motion, "names", ())),
            "clip_lengths": (ends - starts + 1).tolist(),
            "num_frames": num_frames,
            "num_transitions": len(pair_starts),
            "feature_dim": self.feature_dim,
            "feature_mode": self.feature_mode,
            "feature_layout": copy.deepcopy(self.feature_layout),
            "expert_hash": self.expert_hash,
        }

    def _set_default_q(self, device):
        source = getattr(getattr(self.kin, "robot", None), "default_q",
                         getattr(self.kin, "default_q", (0.,) * self.num_joints))
        default = torch.as_tensor(source, dtype=torch.float32, device=device).detach().clone()
        if default.shape != (self.num_joints,) or not bool(torch.isfinite(default).all()):
            raise ValueError("Kinematics default_q must contain finite robot joint defaults")
        self.register_buffer("_default_q", default, persistent=False)

    def _feature_layout(self):
        if self.feature_mode == "proprioceptive":
            return {
                "version": 2,
                "feature_mode": "proprioceptive",
                "endpoints": ["S0", "S1"],
                "frame": [
                    ["projected_gravity", 3],
                    ["joint_position-default_q", self.num_joints],
                    ["joint_velocity*0.05", self.num_joints],
                    ["root_local_linear_velocity", 3],
                    ["root_local_angular_velocity", 3],
                ],
                "robot_height": self.robot_height,
                "minimum_forward_speed_m_s": self.minimum_forward_speed,
                "joint_names": list(getattr(self.kin, "joint_names", ())),
                "default_q": self._default_q.cpu().tolist(),
            }
        return {
            "version": 1,
            "endpoints": ["S0", "S1"],
            "frame": [
                ["terrain_relative_height/robot_height", 1],
                ["yaw_removed_root_rotation_6d", 6],
                ["root_local_linear_velocity/robot_height", 3],
                ["root_local_angular_velocity", 3],
                ["joint_position", self.num_joints],
                ["joint_velocity*0.05", self.num_joints],
                ["root_local_tracked_position/robot_height", 3 * self.num_tracked_bodies],
            ],
            "robot_height": self.robot_height,
            "minimum_forward_speed_m_s": self.minimum_forward_speed,
            "joint_names": list(getattr(self.kin, "joint_names", ())),
            "tracked_indices": list(getattr(self.kin, "tracked_idx", ())),
            "tracked_names": [
                self.kin.body_names[index] for index in getattr(self.kin, "tracked_idx", ())
            ] if hasattr(self.kin, "body_names") else [],
        }

    def _validate_state(self, state, batch_shape=None):
        if not isinstance(state, (tuple, list)) or len(state) != 6:
            raise ValueError("State must be (p, R, v, w, q, qd)")
        if not all(isinstance(value, torch.Tensor) and value.is_floating_point() for value in state):
            raise ValueError("State components must be floating-point tensors")
        if batch_shape is None:
            batch_shape = state[0].shape[:-1]
        tails = ((3,), (3, 3), (3,), (3,), (self.num_joints,), (self.num_joints,))
        if any(value.shape != (*batch_shape, *tail) for value, tail in zip(state, tails)):
            raise ValueError("State components have incompatible batch or joint dimensions")
        if any(value.device != state[0].device for value in state):
            raise ValueError("State components must share a device")
        return batch_shape

    @staticmethod
    def _clip_indices(motion, num_frames, device):
        starts, ends, ids = (getattr(motion, name) for name in ("clip_start", "clip_end", "clip_id"))
        integer_types = (torch.int32, torch.int64)
        if any(not isinstance(value, torch.Tensor) or value.dtype not in integer_types
               for value in (starts, ends, ids)):
            raise ValueError("Expert clip indices must be integer tensors")
        starts, ends, ids = (value.detach().to(device=device, dtype=torch.long) for value in (starts, ends, ids))
        if starts.ndim != 1 or ends.shape != starts.shape or len(starts) == 0 or ids.shape != (num_frames,):
            raise ValueError("Invalid expert clip index shapes")
        if (starts[0].item() != 0 or ends[-1].item() != num_frames - 1
                or bool((ends < starts).any()) or not torch.equal(starts[1:], ends[:-1] + 1)):
            raise ValueError("Expert clip bounds must partition the recorded frames")
        lengths = ends - starts + 1
        expected_ids = torch.repeat_interleave(torch.arange(len(starts), device=device), lengths)
        if not torch.equal(ids, expected_ids):
            raise ValueError("Expert clip IDs disagree with clip bounds")
        if hasattr(motion, "clip_len") and not torch.equal(motion.clip_len.to(device), lengths):
            raise ValueError("Expert clip lengths disagree with clip bounds")
        pair_starts = torch.arange(num_frames - 1, device=device)
        pair_starts = pair_starts[ids[:-1] == ids[1:]]
        if pair_starts.numel() == 0:
            raise ValueError("Empty experts: no within-clip expert transitions")
        return starts, ends, pair_starts

    def _local_bodies(self, state):
        p, R, v, w, q, qd = state
        # Canonical-root FK avoids subtracting large world coordinates and
        # leaves the q -> tracked-position path differentiable.
        identity = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
        return self.kin.forward(
            torch.zeros_like(p), identity, torch.zeros_like(v), torch.zeros_like(w),
            q, torch.zeros_like(qd),
        )[0]

    def frame_features(self, state: State, reference: bool = False) -> torch.Tensor:
        """Raw invariant frame features; references always use aligned ground=0."""
        state = tuple(value.to(dtype=self.norm.mean.dtype) for value in state)
        p, R, v, w, q, qd = state
        with torch.autocast(device_type=p.device.type, enabled=False):
            Rt = R.transpose(-1, -2)
            if self.feature_mode == "proprioceptive":
                return torch.cat((
                    -R[..., 2, :], q - self._default_q, qd * 0.05,
                    (Rt @ v.unsqueeze(-1)).squeeze(-1),
                    (Rt @ w.unsqueeze(-1)).squeeze(-1),
                ), -1)
            ground = torch.zeros_like(p[..., 2])
            if not reference and self.ground_sampler is not None:
                ground = self.ground_sampler(p[..., :2]).to(p).reshape(p.shape[:-1])
            return torch.cat((
                ((p[..., 2] - ground) / self.robot_height).unsqueeze(-1),
                mat_to_6d(yaw_mat(R).transpose(-1, -2) @ R),
                (Rt @ v.unsqueeze(-1)).squeeze(-1) / self.robot_height,
                (Rt @ w.unsqueeze(-1)).squeeze(-1),
                q, qd * 0.05,
                (self._local_bodies(state) / self.robot_height).flatten(-2),
            ), -1)

    def features(self, S0: State, S1: State, reference: bool = False) -> torch.Tensor:
        """Raw endpoint concatenation with one batched FK; safe to compile."""
        if self.history_steps != 2:
            raise ValueError("Use complete history windows for this discriminator")
        batch_shape = S0[0].shape[:-1]
        combined = tuple(
            torch.stack((a, b), 0).reshape(-1, *a.shape[-tail:])
            for a, b, tail in zip(S0, S1, _TAIL_DIMS)
        )
        frames = self.frame_features(combined, reference).reshape(2, *batch_shape, self.frame_dim)
        return torch.cat((frames[0], frames[1]), -1)

    pair_features = features

    def _normalize(self, features):
        frames = features.reshape(*features.shape[:-1], self.history_steps, self.frame_dim)
        normalized = self.norm(frames.clamp(-self.RAW_LIMIT, self.RAW_LIMIT))
        return normalized.clamp(-self.NORM_LIMIT, self.NORM_LIMIT).reshape(features.shape)

    def _score(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.discriminator(features).squeeze(-1)

    def reward(self, S0: State, S1: State) -> torch.Tensor:
        """Differentiable reward rate in [0, 1], with fixed normalization."""
        return self.reward_from_features(self.features(S0, S1))

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Discriminator scores from RAW pair features; normalization is read-only."""
        return self._score(self._normalize(features.to(dtype=self.norm.mean.dtype)))

    def reward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reward rate from RAW pair features, preserving all feature gradients."""
        score = self.score(features)
        return (1 - 0.25 * (score - 1).square()).clamp_min(0)

    def window_features(self, frames):
        """Flatten oldest-first frame histories without changing their gradients."""
        if frames.shape[-2:] != (self.history_steps, self.frame_dim):
            raise ValueError("History window has the wrong frame count or feature width")
        return frames.flatten(-2)

    def rollout_features(self, history, following):
        """Windows ending at each imagined state; history includes the start state."""
        if history.shape[-2:] != (self.history_steps, self.frame_dim):
            raise ValueError("Initial style history is incomplete")
        sequence = torch.cat((history[:, 1:], following.transpose(0, 1)), 1)
        windows = sequence.unfold(1, self.history_steps, 1).transpose(-1, -2)
        return windows.transpose(0, 1).flatten(-2)

    @contextmanager
    def freeze(self):
        """Freeze discriminator weights while preserving input-state gradients."""
        parameters = tuple(self.discriminator.parameters())
        flags = tuple(parameter.requires_grad for parameter in parameters)
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield self
        finally:
            for parameter, flag in zip(parameters, flags):
                parameter.requires_grad_(flag)

    @property
    def replay_size(self) -> int:
        return self._replay_size

    def _require_training(self):
        if self._scoring_only:
            raise RuntimeError("Scoring-only AMP instances cannot collect, train or create training checkpoints")

    @torch.no_grad()
    def observe(self, S0: State, S1: State, valid=None) -> int:
        """Store finite, detached REAL pairs; ``valid`` excludes done/reset pairs.

        Autograd-connected inputs are rejected to catch accidental imagined
        collection. Provenance of explicitly detached states is the caller's
        responsibility. Returns the accepted count (before replay eviction).
        """
        self._require_training()
        batch_shape = self._validate_state(S0)
        self._validate_state(S1, batch_shape)
        if any(value.requires_grad or value.grad_fn is not None for value in (*S0, *S1)):
            raise ValueError("observe requires detached REAL simulator states, never imagined states")
        if S0[0].device != self._replay.device or S1[0].device != self._replay.device:
            raise ValueError("Observed states must share the learner device")
        flat0 = tuple(value.reshape(-1, *value.shape[-tail:]) for value, tail in zip(S0, _TAIL_DIMS))
        flat1 = tuple(value.reshape(-1, *value.shape[-tail:]) for value, tail in zip(S1, _TAIL_DIMS))
        finite = torch.ones(flat0[0].shape[0], dtype=torch.bool, device=self._replay.device)
        for value in (*flat0, *flat1):
            finite &= torch.isfinite(value).flatten(1).all(-1)
        if valid is not None:
            valid = torch.as_tensor(valid, device=finite.device)
            if valid.dtype != torch.bool or valid.shape not in (batch_shape, (*batch_shape, 1)):
                raise ValueError("valid must be a boolean episode-interior mask matching the state batch")
            finite &= valid.reshape(-1)
        indices = finite.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return 0
        features = self.features(
            tuple(value[indices] for value in flat0), tuple(value[indices] for value in flat1),
        ).detach()
        features = features[torch.isfinite(features).all(-1)]
        return self._store_features(features)

    @torch.no_grad()
    def observe_window(self, frames, valid):
        """Collect detached real histories, excluding terminal and partial windows."""
        self._require_training()
        if not isinstance(frames, torch.Tensor) or not isinstance(valid, torch.Tensor):
            raise ValueError("History features and validity mask must be tensors")
        if frames.requires_grad or frames.grad_fn is not None:
            raise ValueError("observe_window requires detached REAL simulator histories")
        features = self.window_features(frames)
        if (features.ndim != 2 or valid.dtype != torch.bool or valid.shape != features.shape[:1]
                or valid.device != features.device):
            raise ValueError("Expected a boolean validity flag per real history")
        features = features[valid & torch.isfinite(features).all(-1)]
        return self._store_features(features)

    def _store_features(self, features):
        count = len(features)
        if count == 0:
            return 0
        self.norm.update(features.reshape(-1, self.frame_dim).clamp(-self.RAW_LIMIT, self.RAW_LIMIT))
        self._observed += count
        if count >= self.replay_capacity:
            self._replay.copy_(features[-self.replay_capacity:])
            self._replay_size, self._replay_cursor = self.replay_capacity, 0
        else:
            first = min(count, self.replay_capacity - self._replay_cursor)
            self._replay[self._replay_cursor:self._replay_cursor + first].copy_(features[:first])
            self._replay[:count - first].copy_(features[first:])
            self._replay_cursor = (self._replay_cursor + count) % self.replay_capacity
            self._replay_size = min(self._replay_size + count, self.replay_capacity)
        return count

    @torch.no_grad()
    def sample_expert(self, batch_size: int, *, return_indices: bool = False):
        """Sample recorded adjacent pairs, optionally returning (features, f0, f1)."""
        self._require_training()
        batch_size = _positive_int(batch_size, "batch_size")
        indices = torch.randint(len(self._expert_starts), (batch_size,), device=self._expert_starts.device)
        f0 = self._expert_starts[indices]
        frame_ids = f0[:, None] + torch.arange(self.history_steps, device=f0.device)
        features = self._expert_frames[frame_ids].flatten(-2)
        return (features, f0, f0 + self.history_steps - 1) if return_indices else features

    @torch.enable_grad()
    def update(self, batch_size: int = 1024, updates: int = 1) -> dict[str, float | int]:
        """Fit only cached experts and REAL replay with LSGAN and expert R1 GP."""
        self._require_training()
        batch_size = _positive_int(batch_size, "batch_size")
        if isinstance(updates, bool) or not isinstance(updates, Integral) or updates < 0:
            raise ValueError("updates must be a nonnegative integer")
        names = ("loss", "expert_score", "policy_score", "gradient_penalty")
        metrics = dict.fromkeys(names, 0.0)
        metrics.update(updates=0, replay_size=self.replay_size)
        if updates == 0 or self.replay_size == 0:
            return metrics
        if not all(parameter.requires_grad for parameter in self.discriminator.parameters()):
            raise RuntimeError("Unfreeze the discriminator before update")
        sums = self._replay.new_zeros(4)
        for _ in range(updates):
            expert = self._normalize(self.sample_expert(batch_size)).detach().requires_grad_(True)
            indices = torch.randint(self.replay_size, (batch_size,), device=self._replay.device)
            policy = self._normalize(self._replay[indices].detach()).detach()
            expert_score, policy_score = self._score(expert), self._score(policy)
            gp_input = (torch.cat((policy, expert), 0).detach().requires_grad_(True)
                        if self.gradient_penalty_scope == "both" else expert)
            gp_score = self._score(gp_input) if self.gradient_penalty_scope == "both" else expert_score
            gradient = torch.autograd.grad(
                gp_score.sum(), gp_input, create_graph=True, only_inputs=True,
            )[0]
            # Squared L2 avoids the undefined norm derivative at zero during GP backprop.
            penalty = gradient.square().sum(-1).mean()
            loss = 0.5 * ((expert_score - 1).square().mean()
                          + (policy_score + 1).square().mean())
            loss = loss + self.gradient_penalty_weight * penalty
            if self.weight_decay_coefficient:
                loss = loss + self.weight_decay_coefficient * sum(
                    parameter.square().sum() for parameter in self.discriminator.parameters())
            if self.logit_weight_decay_coefficient:
                loss = loss + self.logit_weight_decay_coefficient * self.discriminator[-1].weight.square().sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite AMP discriminator loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            sums += torch.stack((loss.detach(), expert_score.detach().mean(),
                                 policy_score.detach().mean(), penalty.detach()))
            self._updates += 1
        metrics.update(zip(names, (sums / updates).cpu().tolist()))
        metrics["updates"] = int(updates)
        return metrics

    def checkpoint(self) -> dict:
        """Snapshot persistent training state; expert cache and replay are omitted."""
        self._require_training()
        return {
            "version": self.CHECKPOINT_VERSION,
            "expert_hash": self.expert_hash,
            "feature_dim": self.feature_dim,
            "feature_mode": self.feature_mode,
            "feature_layout": copy.deepcopy(self.feature_layout),
            "expert_provenance": copy.deepcopy(self.expert_provenance),
            "discriminator": copy.deepcopy(self.discriminator.state_dict()),
            "norm": copy.deepcopy(self.norm.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "gradient_penalty_scope": self.gradient_penalty_scope,
            "history_steps": self.history_steps,
            "activation": self.activation,
            "hidden": list(self.hidden),
            "optimizer_name": self.optimizer_name,
            "weight_decay_coefficient": self.weight_decay_coefficient,
            "logit_weight_decay_coefficient": self.logit_weight_decay_coefficient,
            "minimum_forward_speed": self.minimum_forward_speed,
            "updates": self._updates,
            "observed": self._observed,
            "replay_size_at_save": self.replay_size,
            "replay_restored": False,
            "replay_refill_required": True,
        }

    def restore(self, checkpoint: dict) -> dict[str, bool]:
        """Restore compatible training state and require real-data replay refill."""
        self._require_training()
        if checkpoint.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError("Unsupported AMP style checkpoint version")
        self._validate_metadata(checkpoint)
        if (checkpoint.get("expert_hash") != self.expert_hash
                or checkpoint.get("feature_dim") != self.feature_dim
                or checkpoint.get("feature_mode", "full") != self.feature_mode
                or checkpoint.get("feature_layout") != self.feature_layout
                or checkpoint.get("history_steps", 2) != self.history_steps
                or checkpoint.get("activation", "ELU") != self.activation
                or tuple(checkpoint.get("hidden", (256, 128))) != self.hidden
                or checkpoint.get("optimizer_name", "Adam") != self.optimizer_name
                or checkpoint.get("minimum_forward_speed", 0.) != self.minimum_forward_speed):
            raise ValueError("AMP checkpoint expert identity or feature layout differs")
        self._validate_weights(checkpoint)
        penalty_weight = checkpoint["gradient_penalty_weight"]
        regularizers = (penalty_weight, checkpoint.get("weight_decay_coefficient", 0.),
                        checkpoint.get("logit_weight_decay_coefficient", 0.))
        if any(not math.isfinite(value) or value < 0 for value in regularizers):
            raise ValueError("Invalid AMP regularization weights")
        for state in checkpoint["optimizer"]["state"].values():
            if any(isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
                   for value in state.values()):
                raise ValueError("Nonfinite AMP optimizer checkpoint state")
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.norm.load_state_dict(checkpoint["norm"])
        # Adam maps moment tensors to each parameter's device/dtype and keeps
        # eager, noncapturable step counters on the CPU.
        self.optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        self.optimizer.zero_grad(set_to_none=True)
        self.gradient_penalty_weight = float(penalty_weight)
        self.gradient_penalty_scope = checkpoint.get("gradient_penalty_scope", "expert")
        self.weight_decay_coefficient = float(regularizers[1])
        self.logit_weight_decay_coefficient = float(regularizers[2])
        self._updates, self._observed = int(checkpoint["updates"]), int(checkpoint["observed"])
        self._replay_size = self._replay_cursor = 0
        return {"replay_restored": False, "replay_refill_required": True}

    @staticmethod
    def _validate_metadata(checkpoint):
        if checkpoint.get("activation") not in ("ELU", "ReLU"):
            raise ValueError("Missing or invalid AMP activation metadata")
        hidden = checkpoint.get("hidden")
        if (not isinstance(hidden, (tuple, list))
                or any(type(width) is not int or width <= 0 for width in hidden)):
            raise ValueError("Missing or invalid AMP hidden metadata")
        history = checkpoint.get("history_steps")
        if type(history) is not int or history < 2:
            raise ValueError("Missing or invalid AMP history length")
        if checkpoint.get("optimizer_name") not in ("Adam", "AdamW"):
            raise ValueError("Missing or invalid AMP optimizer metadata")
        if checkpoint.get("gradient_penalty_scope") not in ("expert", "both"):
            raise ValueError("Missing or invalid AMP gradient penalty scope")
        for name in ("weight_decay_coefficient", "logit_weight_decay_coefficient",
                     "minimum_forward_speed"):
            value = checkpoint.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing or invalid AMP {name}")

    def _validate_weights(self, checkpoint):
        for key, module in (("discriminator", self.discriminator), ("norm", self.norm)):
            if not isinstance(checkpoint.get(key), dict):
                raise ValueError(f"Missing AMP {key} checkpoint state")
            saved, current = checkpoint[key], module.state_dict()
            if saved.keys() != current.keys() or any(
                not isinstance(saved[name], torch.Tensor) or saved[name].shape != current[name].shape
                or not saved[name].is_floating_point() for name in current
            ):
                raise ValueError(f"Incompatible AMP {key} state")
            if any(not bool(torch.isfinite(value).all()) for value in saved.values()):
                raise ValueError(f"Nonfinite AMP {key} checkpoint state")
        if bool((checkpoint["norm"]["var"] < 0).any()) or checkpoint["norm"]["count"].item() < 0:
            raise ValueError("Invalid AMP normalization checkpoint")

    @classmethod
    def for_scoring(cls, kin, checkpoint: dict, ground_sampler=None) -> AMPStyle:
        """Build a frozen scorer without source motions, expert cache or Adam.

        Metadata is checked against the supplied robot. Expert identity remains
        the training checkpoint's declaration; inference never reloads experts.
        Device defaults to ``kin.device``, then the checkpoint normalization.
        """
        if not isinstance(checkpoint, dict) or checkpoint.get("version") != cls.CHECKPOINT_VERSION:
            raise ValueError("Unsupported AMP style checkpoint version")
        cls._validate_metadata(checkpoint)
        layout = checkpoint.get("feature_layout")
        if not isinstance(layout, dict):
            raise ValueError("Missing AMP feature layout")
        mode = checkpoint.get("feature_mode", layout.get("feature_mode", "full"))
        if mode not in ("full", "proprioceptive"):
            raise ValueError("Invalid AMP feature mode")
        expert_hash = checkpoint.get("expert_hash")
        if (not isinstance(expert_hash, str) or len(expert_hash) != 64
                or any(character not in "0123456789abcdef" for character in expert_hash)):
            raise ValueError("Invalid AMP expert identity")
        height = layout.get("robot_height")
        if not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
            raise ValueError("Invalid AMP robot height metadata")
        history_steps = checkpoint.get("history_steps", 2)
        if type(history_steps) is not int or history_steps < 2:
            raise ValueError("Invalid AMP history length")
        frame_dim = checkpoint.get("feature_dim")
        if isinstance(frame_dim, bool) or not isinstance(frame_dim, int) or frame_dim <= 0 or frame_dim % history_steps:
            raise ValueError("Invalid AMP feature dimension")
        frame_dim //= history_steps
        joints = getattr(kin, "num_joints", None)
        if isinstance(joints, bool) or not isinstance(joints, int) or joints <= 0:
            raise ValueError("Scoring kinematics must declare num_joints")
        norm = checkpoint.get("norm", {})
        if not isinstance(norm, dict) or not isinstance(norm.get("mean"), torch.Tensor):
            raise ValueError("Missing AMP normalization checkpoint")
        device = torch.device(getattr(kin, "device", norm["mean"].device))
        if ground_sampler is not None and not callable(ground_sampler):
            raise ValueError("ground_sampler must be callable")
        scorer = cls.__new__(cls)
        nn.Module.__init__(scorer)
        scorer._scoring_only, scorer.feature_mode = True, mode
        scorer.minimum_forward_speed = checkpoint["minimum_forward_speed"]
        scorer.robot_height, scorer.num_joints = float(height), joints
        scorer.history_steps = history_steps
        scorer.frame_dim, scorer.feature_dim = frame_dim, history_steps * frame_dim
        object.__setattr__(scorer, "kin", kin)
        object.__setattr__(scorer, "ground_sampler", partial(ground_sampler) if ground_sampler else None)
        scorer._set_default_q(device)
        scorer.num_tracked_bodies = 0
        if mode == "full":
            remaining = frame_dim - (13 + 2 * joints)
            if remaining <= 0 or remaining % 3:
                raise ValueError("Invalid AMP full feature dimension")
            scorer.num_tracked_bodies = remaining // 3
            if (hasattr(kin, "tracked_idx")
                    and len(kin.tracked_idx) != scorer.num_tracked_bodies):
                raise ValueError("Scoring tracked-body layout differs")
        elif frame_dim != 9 + 2 * joints:
            raise ValueError("Invalid AMP proprioceptive feature dimension")
        scorer.feature_layout = scorer._feature_layout()
        if history_steps > 2:
            scorer.feature_layout["history_steps"] = history_steps
            scorer.feature_layout["endpoints"] = "oldest-to-newest full history"
        if scorer.feature_layout != layout:
            raise ValueError("Scoring robot or feature layout differs from the checkpoint")
        weights = checkpoint.get("discriminator")
        if not isinstance(weights, dict) or not weights or len(weights) % 2:
            raise ValueError("Invalid AMP discriminator checkpoint")
        widths, previous = [], scorer.feature_dim
        for index in range(len(weights) // 2):
            weight, bias = weights.get(f"{2 * index}.weight"), weights.get(f"{2 * index}.bias")
            if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
                    or weight.shape[1] != previous or weight.shape[0] <= 0
                    or not isinstance(bias, torch.Tensor) or bias.shape != (weight.shape[0],)):
                raise ValueError("Invalid AMP discriminator layer dimensions")
            widths.append(weight.shape[0])
            previous = weight.shape[0]
        if widths[-1] != 1:
            raise ValueError("AMP discriminator must have one output")
        if tuple(checkpoint["hidden"]) != tuple(widths[:-1]):
            raise ValueError("AMP hidden metadata differs from discriminator weights")
        scorer.norm = RunningNorm(frame_dim, eps=cls.MIN_STD).to(device)
        activation = checkpoint.get("activation", "ELU")
        activation_cls = {"ELU": nn.ELU, "ReLU": nn.ReLU}.get(activation)
        if activation_cls is None:
            raise ValueError("Invalid AMP activation metadata")
        scorer.activation = activation
        scorer.hidden = tuple(widths[:-1])
        scorer.optimizer_name = checkpoint.get("optimizer_name", "Adam")
        scorer.weight_decay_coefficient = checkpoint.get("weight_decay_coefficient", 0.)
        scorer.logit_weight_decay_coefficient = checkpoint.get("logit_weight_decay_coefficient", 0.)
        scorer.gradient_penalty_scope = checkpoint.get("gradient_penalty_scope", "expert")
        scorer.gradient_penalty_weight = checkpoint.get("gradient_penalty_weight", 10.)
        scorer.discriminator = mlp(scorer.feature_dim, widths[:-1], 1, act=activation_cls).to(device)
        scorer._validate_weights(checkpoint)
        scorer.norm.load_state_dict(checkpoint["norm"])
        scorer.discriminator.load_state_dict(checkpoint["discriminator"])
        scorer.expert_hash = expert_hash
        scorer.expert_provenance = copy.deepcopy(checkpoint.get("expert_provenance", {}))
        scorer._replay_size = scorer._replay_cursor = 0
        return scorer.eval().requires_grad_(False)

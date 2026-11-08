"""Opt-in scheduled waist posture, independent of actor waist outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Optional

import torch
from torch import nn


_WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    represented = torch.tensor(result, dtype=torch.float32)
    if not bool(torch.isfinite(represented)) or represented.item() <= 0:
        raise ValueError(f"{name} must be representable as positive float32")
    return result


def _vector(value, length, name):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool or value.is_complex() or value.ndim != 1:
            raise ValueError(f"{name} must be a finite real vector of length {length}")
        values = value.detach().to(device="cpu", dtype=torch.float64)
    else:
        try:
            items = tuple(value)
        except TypeError as error:
            raise ValueError(f"{name} must be a finite real vector of length {length}") from error
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in items):
            raise ValueError(f"{name} must contain real numbers without booleans")
        values = torch.tensor(items, dtype=torch.float64)
    if values.shape != (length,) or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be a finite real vector of length {length}")
    result = values.to(dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be representable as float32")
    return result.clone()


class WaistEntryConstraint(nn.Module):
    """Ramp three waist targets from default to expert, then hold indefinitely.

    Target positions use ``default_q + action * action_scale``. The caller's
    entry clock alone selects the pose, including after key release or command
    changes. Previous actions are accepted for composition, without introducing
    state or restoring raw actor waist outputs after the ramp. Analytic bounds
    concern scheduled targets at a continuously advancing clock, not physical
    joint tracking or arbitrary clock jumps.
    """

    def __init__(
        self,
        joint_names,
        default_q,
        action_scale,
        reference_q,
        ramp_duration_s,
        control_dt=0.02,
        max_joint_rate_rad_s=1.2,
        max_joint_acceleration_rad_s2=3.0,
        action_limit=20.0,
    ):
        super().__init__()
        try:
            names = tuple(joint_names)
        except TypeError as error:
            raise ValueError("joint_names must contain 29 unique names including three waist joints") from error
        if (len(names) != 29 or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names) or any(name not in names for name in _WAIST_NAMES)):
            raise ValueError("joint_names must contain 29 unique names including three waist joints")
        default = _vector(default_q, 29, "default_q")
        scale = _vector(action_scale, 29, "action_scale")
        if bool((scale <= 0).any()):
            raise ValueError("action_scale entries must be positive")
        if not isinstance(reference_q, Mapping) or set(reference_q) != set(_WAIST_NAMES):
            raise ValueError("reference_q must map exactly the three named waist joints")
        reference = _vector([reference_q[name] for name in _WAIST_NAMES], 3, "reference_q")
        self.ramp_duration_s = _positive(ramp_duration_s, "ramp_duration_s")
        self.control_dt = _positive(control_dt, "control_dt")
        self.max_joint_rate_rad_s = _positive(max_joint_rate_rad_s, "max_joint_rate_rad_s")
        self.max_joint_acceleration_rad_s2 = _positive(
            max_joint_acceleration_rad_s2, "max_joint_acceleration_rad_s2")
        self.action_limit = _positive(action_limit, "action_limit")
        indices = torch.tensor([names.index(name) for name in _WAIST_NAMES], dtype=torch.long)
        start = default.index_select(0, indices).double()
        distance = (reference.double() - start).abs()
        rate = (1.875 * distance / self.ramp_duration_s).max().item()
        acceleration = ((10. / math.sqrt(3.)) *
                        (distance / self.ramp_duration_s) / self.ramp_duration_s).max().item()
        if rate > self.max_joint_rate_rad_s * (1. + 1e-6):
            raise ValueError(f"waist ramp exceeds max_joint_rate_rad_s: {rate:.6g}")
        if acceleration > self.max_joint_acceleration_rad_s2 * (1. + 1e-6):
            raise ValueError(f"waist ramp exceeds max_joint_acceleration_rad_s2: {acceleration:.6g}")
        normalized = (reference.double() - start) / scale.index_select(0, indices).double()
        # A monotone blend between zero and the final normalized target stays in range.
        if bool((normalized.abs() > self.action_limit).any()):
            raise ValueError("scheduled normalized waist targets exceed action_limit")
        delta = reference - start.float()
        if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(
                delta / scale.index_select(0, indices)).all()):
            raise ValueError("scheduled waist target arithmetic must remain finite in float32")
        self.joint_names = names
        self.num_joints = len(names)
        self.waist_joint_names = _WAIST_NAMES
        self.register_buffer("waist_indices", indices)
        self.register_buffer("default_q", default)
        self.register_buffer("action_scale", scale)
        self.register_buffer("reference_q", reference)

    @torch.jit.export
    def reference_at(self, time: torch.Tensor) -> torch.Tensor:
        """Named waist-order targets for scalar or batched seconds."""
        if not torch.jit.is_tracing():
            if time.device != self.reference_q.device or time.dtype != self.reference_q.dtype:
                raise ValueError("entry time and constraint buffers must share device and dtype")
            if torch.jit.is_scripting():
                if not bool(torch.isfinite(time).all()):
                    raise ValueError("entry_time must be finite")
            elif not torch.compiler.is_compiling():
                if not bool(torch.isfinite(time).all()):
                    raise ValueError("entry_time must be finite")
        phase = time.clamp(0., self.ramp_duration_s) / self.ramp_duration_s
        blend = (phase.pow(3) * (10. + phase * (-15. + 6. * phase))).clamp(0., 1.)
        default = self.default_q.index_select(0, self.waist_indices)
        target = default + blend.unsqueeze(-1) * (self.reference_q - default)
        target = torch.where((time <= 0.).unsqueeze(-1), default, target)
        return torch.where((time >= self.ramp_duration_s).unsqueeze(-1), self.reference_q, target)

    def forward(
        self,
        raw: torch.Tensor,
        previous: torch.Tensor,
        older: Optional[torch.Tensor] = None,
        entry_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if entry_time is None:
            raise ValueError("entry_time is required for waist posture control")
        if not torch.jit.is_tracing():
            if raw.ndim < 1 or raw.shape[-1] != self.num_joints or previous.shape != raw.shape:
                raise ValueError("actions must have matching shapes ending in num_joints")
            if (raw.device != self.default_q.device or raw.dtype != self.default_q.dtype
                    or previous.device != raw.device or previous.dtype != raw.dtype):
                raise ValueError("actions and constraint buffers must share device and dtype")
            if older is not None:
                if older.shape != raw.shape or older.device != raw.device or older.dtype != raw.dtype:
                    raise ValueError("older must match action shape, device and dtype")
            if entry_time.ndim != 0 and entry_time.shape != raw.shape[:-1]:
                raise ValueError("entry_time must be scalar or match the action batch shape")
        default = self.default_q.index_select(0, self.waist_indices)
        scale = self.action_scale.index_select(0, self.waist_indices)
        target = (self.reference_at(entry_time) - default) / scale
        target = target.expand_as(raw.index_select(-1, self.waist_indices))
        return raw.index_copy(-1, self.waist_indices, target)

    def to_dict(self):
        """JSON-compatible constructor metadata; persistent waist hold is explicit."""
        return {
            "joint_names": list(self.joint_names),
            "default_q": self.default_q.detach().cpu().tolist(),
            "action_scale": self.action_scale.detach().cpu().tolist(),
            "reference_q": dict(zip(self.waist_joint_names, self.reference_q.detach().cpu().tolist())),
            "ramp_duration_s": self.ramp_duration_s,
            "control_dt": self.control_dt,
            "max_joint_rate_rad_s": self.max_joint_rate_rad_s,
            "max_joint_acceleration_rad_s2": self.max_joint_acceleration_rad_s2,
            "action_limit": self.action_limit,
        }

    @classmethod
    def from_dict(cls, config):
        if not isinstance(config, Mapping):
            raise ValueError("waist constraint config must be a mapping")
        return cls(**dict(config))


class CombinedEntryConstraint(nn.Module):
    """Apply the primary constraint followed by persistent waist targets."""

    def __init__(self, primary, waist):
        super().__init__()
        if not isinstance(primary, nn.Module) or not isinstance(waist, WaistEntryConstraint):
            raise ValueError("composition requires a primary module and WaistEntryConstraint")
        self.primary = primary
        self.waist = waist

    def forward(
        self,
        raw: torch.Tensor,
        previous: torch.Tensor,
        older: Optional[torch.Tensor] = None,
        entry_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        constrained = self.primary(raw, previous, older, entry_time=entry_time)
        return self.waist(constrained, previous, older, entry_time=entry_time)


def compose_entry_waist_constraint(primary, waist_config, device=None):
    """Preserve the legacy module exactly unless an explicit config is supplied."""
    if waist_config is None:
        return primary
    if primary is None:
        raise ValueError("waist_config requires an existing primary constraint")
    if not isinstance(primary, nn.Module):
        raise ValueError("primary constraint must be an nn.Module")
    waist = WaistEntryConstraint.from_dict(waist_config)
    inferred_device, inferred_dtype = None, None
    for tensor in (*primary.buffers(), *primary.parameters()):
        if inferred_device is None:
            inferred_device = tensor.device
        if inferred_dtype is None and tensor.is_floating_point():
            inferred_dtype = tensor.dtype
    target_device = device if device is not None else inferred_device
    waist = waist.to(device=target_device, dtype=inferred_dtype)
    return CombinedEntryConstraint(primary, waist)

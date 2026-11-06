"""Upper-body target constraints and optional synchronized stealth entry."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


_UPPER_JOINT = re.compile(r"(?:^|_)(?:shoulder|elbow|wrist)(?:_|$)")


def _finite_vector(value, name):
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional numeric vector") from error
    if tensor.ndim != 1 or tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a finite one-dimensional numeric vector")
    return tensor


def _upper_indices(joint_names):
    names = tuple(joint_names)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))):
        raise ValueError("joint_names must contain unique nonempty names")
    indices = tuple(
        index for index, name in enumerate(names) if _UPPER_JOINT.search(name)
    )
    if not indices:
        raise ValueError("joint_names must include shoulder, elbow or wrist joints")
    return names, indices


def _reference_values(reference_q, names, indices):
    arm_names = tuple(names[index] for index in indices)
    if isinstance(reference_q, Mapping):
        if set(reference_q) != set(arm_names):
            raise ValueError("reference_q must map every selected upper-body joint by name")
        return _finite_vector(
            [reference_q[name] for name in arm_names], "reference_q",
        )
    reference = _finite_vector(reference_q, "reference_q")
    if reference.shape == (len(names),):
        return reference[list(indices)]
    if reference.shape == (len(indices),):
        return reference
    raise ValueError("reference_q must match all joints or selected upper-body joints")


def _entry_schedule(times, poses, names, indices, default, scale, reference,
                    max_rate, max_acceleration, action_limit):
    if (times is None) != (poses is None):
        raise ValueError("entry_times_s and entry_reference_q must be provided together")
    if times is None:
        return torch.empty(0), torch.empty((0, len(indices)))
    times = _finite_vector(times, "entry_times_s")
    if (times.numel() < 2 or times[0].item() != 0.
            or not bool((times[1:] > times[:-1]).all())):
        raise ValueError("entry_times_s must start at zero and strictly increase, with at least two times")
    if not isinstance(poses, (list, tuple)) or len(poses) != times.numel():
        raise ValueError("entry_reference_q must contain one pose per entry time")
    poses = torch.stack([_reference_values(pose, names, indices) for pose in poses])
    default, scale = default[list(indices)], scale[list(indices)]
    if not torch.allclose(poses[0], default, atol=1e-6, rtol=0):
        raise ValueError("entry_reference_q must start at default upper-body joints")
    if not torch.allclose(poses[-1], reference, atol=1e-6, rtol=0):
        raise ValueError("entry_reference_q must end at reference_q")
    poses[0], poses[-1] = default, reference
    if bool((((poses.double() - default.double()) / scale.double()).abs()
             > action_limit + 1e-6).any()):
        raise ValueError("entry_reference_q exceeds action_limit")
    duration = (times[1:].double() - times[:-1].double())[:, None]
    distance = (poses[1:].double() - poses[:-1].double()).abs()
    # For s(u) = 10u^3 - 15u^4 + 6u^5, max s' = 15/8 and max |s''| = 10/sqrt(3).
    peak_rate = (1.875 * distance / duration).max().item()
    peak_acceleration = ((10. / math.sqrt(3.)) * distance / duration.square()).max().item()
    if peak_rate > max_rate * (1. + 1e-6):
        raise ValueError(f"entry schedule exceeds max_joint_rate_rad_s: {peak_rate:.6g}")
    if max_acceleration and peak_acceleration > max_acceleration * (1. + 1e-6):
        raise ValueError(
            f"entry schedule exceeds max_joint_acceleration_rad_s2: {peak_acceleration:.6g}")
    return times, poses


def extract_static_upper_body_pose(
    motion, joint_names, *, frame_start=0, frame_end=None, method="medoid",
):
    """Extract one arm pose from an inclusive contiguous range in one clip."""
    names, indices = _upper_indices(joint_names)
    q = getattr(motion, "q", None)
    clip_id = getattr(motion, "clip_id", None)
    if (not isinstance(q, torch.Tensor) or q.ndim != 2 or q.shape[0] == 0
            or q.shape[1] != len(names) or not q.is_floating_point()):
        raise ValueError("motion.q must be a nonempty floating-point (frames, joints) tensor")
    if (not isinstance(clip_id, torch.Tensor) or clip_id.ndim != 1
            or clip_id.shape[0] != q.shape[0] or clip_id.numel() == 0
            or isinstance(frame_start, bool) or not isinstance(frame_start, int)
            or frame_start < 0 or isinstance(frame_end, bool)
            or (frame_end is not None and not isinstance(frame_end, int))
            or (frame_end is not None and frame_end < frame_start)
            or frame_start >= q.shape[0]):
        raise ValueError("invalid contiguous training clip range")
    end = q.shape[0] - 1 if frame_end is None else frame_end
    if end >= q.shape[0] or torch.unique(clip_id[frame_start:end + 1]).numel() != 1:
        raise ValueError("selected frames must belong to exactly one training clip")
    if method not in ("median", "medoid"):
        raise ValueError("method must be median or medoid")
    selected = q[frame_start:end + 1].index_select(
        -1, torch.tensor(indices, device=q.device),
    )
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("selected motion.q values must be finite")
    median = torch.quantile(selected, .5, dim=0)
    pose = median if method == "median" else selected[(selected - median).abs().sum(-1).argmin()]
    return {names[index]: float(pose[offset].item())
            for offset, index in enumerate(indices)}


def calibrate_upper_body_reference(
    motion, joint_names, reference_q=None, *, frame_start=0, frame_end=None,
    method="median",
):
    """Validate an external static pose or derive one from a single clip."""
    names, indices = _upper_indices(joint_names)
    if reference_q is not None:
        values = _reference_values(reference_q, names, indices)
        return {names[index]: float(values[offset].item())
                for offset, index in enumerate(indices)}
    if motion is None:
        raise ValueError("motion or an external reference_q must be provided")
    return extract_static_upper_body_pose(
        motion, names, frame_start=frame_start, frame_end=frame_end, method=method,
    )


def blend_upper_body_reference(
    joint_names, default_q, aiming_reference_q, aiming_weight=0.5,
):
    """Blend reset and aiming targets into a named static arm pose."""
    names, indices = _upper_indices(joint_names)
    default = _finite_vector(default_q, "default_q")
    if default.shape != (len(names),):
        raise ValueError("default_q must match joint_names")
    if (isinstance(aiming_weight, bool) or not isinstance(aiming_weight, (int, float))
            or not math.isfinite(aiming_weight) or not 0. <= aiming_weight <= 1.):
        raise ValueError("aiming_weight must be finite and between zero and one")
    aiming = _reference_values(aiming_reference_q, names, indices)
    blended = default[list(indices)] + float(aiming_weight) * (
        aiming - default[list(indices)]
    )
    return {names[index]: float(blended[offset].item())
            for offset, index in enumerate(indices)}


class UpperBodyConstraint(nn.Module):
    """Follow a synchronized entry, then bound and rate-limit static arm targets.

    Actions use ``default_q + action * action_scale``. The previous applied
    action is supplied by the caller, so the module retains no rollout state.
    An optional entry schedule replaces arm actions through its final timestamp.
    The caller supplies the time of the action target in seconds, and advances
    beyond the schedule duration to enable static residuals. Scheduled targets
    have zero velocity and acceleration at each waypoint. Their validation
    bounds target motion only; physical tracking and collision margins require
    separate validation.
    """

    def __init__(
        self,
        joint_names,
        default_q,
        action_scale,
        reference_q,
        residual_radius_rad=0.25,
        max_joint_rate_rad_s=1.2,
        control_dt=0.02,
        action_limit=20.0,
        max_joint_acceleration_rad_s2=0.0,
        entry_times_s=None,
        entry_reference_q=None,
        posture_tolerance_rad=None,
    ):
        super().__init__()
        names, indices = _upper_indices(joint_names)
        default = _finite_vector(default_q, "default_q")
        scale = _finite_vector(action_scale, "action_scale")
        if default.shape != (len(names),) or scale.shape != (len(names),):
            raise ValueError("default_q and action_scale must match joint_names")
        if bool((scale <= 0).any()):
            raise ValueError("action_scale entries must be positive")
        if (isinstance(residual_radius_rad, bool) or not isinstance(residual_radius_rad, (int, float))
                or not math.isfinite(residual_radius_rad) or residual_radius_rad < 0):
            raise ValueError("residual_radius_rad must be finite and nonnegative")
        parameters = (
            ("max_joint_rate_rad_s", max_joint_rate_rad_s),
            ("control_dt", control_dt),
            ("action_limit", action_limit),
        )
        for name, value in parameters:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

        reference = _reference_values(reference_q, names, indices)
        if (isinstance(max_joint_acceleration_rad_s2, bool)
                or not isinstance(max_joint_acceleration_rad_s2, (int, float))
                or not math.isfinite(max_joint_acceleration_rad_s2)
                or max_joint_acceleration_rad_s2 < 0):
            raise ValueError("max_joint_acceleration_rad_s2 must be finite and nonnegative")
        if posture_tolerance_rad is not None and (
                isinstance(posture_tolerance_rad, bool)
                or not isinstance(posture_tolerance_rad, (int, float))
                or not math.isfinite(posture_tolerance_rad) or posture_tolerance_rad <= 0):
            raise ValueError("posture_tolerance_rad must be finite and positive")
        if residual_radius_rad == 0. and posture_tolerance_rad is None:
            raise ValueError("Zero residual_radius_rad requires an explicit positive posture_tolerance_rad")
        times, poses = _entry_schedule(
            entry_times_s, entry_reference_q, names, indices, default, scale, reference,
            max_joint_rate_rad_s, max_joint_acceleration_rad_s2, action_limit)

        self.joint_names = names
        self.num_joints = len(names)
        self.upper_joint_names = tuple(names[index] for index in indices)
        self.residual_radius_rad = float(residual_radius_rad)
        self.max_joint_rate_rad_s = float(max_joint_rate_rad_s)
        self.control_dt = float(control_dt)
        self.action_limit = float(action_limit)
        self.max_joint_acceleration_rad_s2 = float(max_joint_acceleration_rad_s2)
        # Retain configured seconds: float32 rounding can spuriously exceed an
        # equal command-entry duration when the trainer compares Python floats.
        self._entry_times_config = (
            torch.as_tensor(entry_times_s, dtype=torch.float64, device="cpu").tolist()
            if times.numel() else [])
        self.entry_duration_s = self._entry_times_config[-1] if times.numel() else 0.
        self._explicit_posture_tolerance = posture_tolerance_rad is not None
        self.posture_tolerance_rad = float(
            residual_radius_rad if posture_tolerance_rad is None else posture_tolerance_rad)
        self.register_buffer("upper_indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("default_q", default)
        self.register_buffer("action_scale", scale)
        self.register_buffer("reference_q", reference)
        self.register_buffer("entry_times_s", times, persistent=bool(times.numel()))
        self.register_buffer("entry_reference_q", poses, persistent=bool(times.numel()))

    @torch.jit.export
    def reference_at(self, time: torch.Tensor) -> torch.Tensor:
        """Arm targets at arbitrary scalar/batched seconds, clamped at endpoints."""
        if not torch.jit.is_tracing():
            if time.device != self.reference_q.device or time.dtype != self.reference_q.dtype:
                raise ValueError("entry time and constraint buffers must share device and dtype")
        if self.entry_duration_s == 0.:
            return self.reference_q + torch.zeros_like(time)[..., None]
        segment = (time[..., None] >= self.entry_times_s[1:]).sum(-1).clamp_max(
            self.entry_times_s.numel() - 2)
        start = self.entry_times_s[segment]
        duration = self.entry_times_s[segment + 1] - start
        phase = ((time - start) / duration).clamp(0, 1)
        blend = (phase.pow(3) * (10. + phase * (-15. + 6. * phase))).clamp(0, 1)
        first, last = self.entry_reference_q[segment], self.entry_reference_q[segment + 1]
        target = first + blend[..., None] * (last - first)
        target = torch.where((time <= 0.)[..., None], self.entry_reference_q[0], target)
        return torch.where((time >= self.entry_duration_s)[..., None], self.reference_q, target)

    def forward(self, raw_action, previous_applied_action,
                previous_previous_action: Optional[torch.Tensor] = None,
                entry_time: Optional[torch.Tensor] = None):
        if self.entry_duration_s > 0. and entry_time is None:
            raise ValueError("entry_time is required for an entry schedule")
        if not torch.jit.is_tracing():
            if (raw_action.ndim < 1 or raw_action.shape[-1] != self.num_joints
                    or previous_applied_action.shape != raw_action.shape):
                raise ValueError("actions must have matching shapes ending in num_joints")
            if (raw_action.device != self.default_q.device
                    or previous_applied_action.device != raw_action.device
                    or raw_action.dtype != self.default_q.dtype
                    or previous_applied_action.dtype != raw_action.dtype):
                raise ValueError("actions and constraint buffers must share device and dtype")
            if entry_time is not None and entry_time.ndim != 0 and (
                    entry_time.shape != raw_action.shape[:-1]):
                raise ValueError("entry_time must be scalar or match the action batch shape")

        scale = self.action_scale.index_select(0, self.upper_indices)
        default = self.default_q.index_select(0, self.upper_indices)
        previous = previous_applied_action.index_select(-1, self.upper_indices)
        raw = raw_action.index_select(-1, self.upper_indices)

        previous_q = default + previous * scale
        desired_q = default + raw * scale
        target_q = self.reference_q
        if self.residual_radius_rad > 0.:
            residual = self.residual_radius_rad * torch.tanh(
                (desired_q - self.reference_q) / self.residual_radius_rad
            )
            target_q = target_q + residual
        max_step = self.max_joint_rate_rad_s * self.control_dt
        step = max_step * torch.tanh((target_q - previous_q) / max_step)
        if self.max_joint_acceleration_rad_s2 > 0.:
            if previous_previous_action is None:
                raise ValueError("Acceleration limiting requires two previous executed actions")
            older = previous_previous_action.index_select(-1, self.upper_indices)
            previous_step = (previous - older) * scale
            max_change = self.max_joint_acceleration_rad_s2 * self.control_dt**2
            step = previous_step + (step - previous_step).clamp(-max_change, max_change)
            step = step.clamp(-max_step, max_step)
        arm_action = ((previous_q + step - default) / scale).clamp(
            -self.action_limit, self.action_limit,
        )
        if self.entry_duration_s > 0. and entry_time is not None:
            scheduled = (self.reference_at(entry_time) - default) / scale
            # Independent joint limiting would change the planned arm path.
            arm_action = torch.where(
                (entry_time <= self.entry_duration_s)[..., None], scheduled, arm_action)
        return raw_action.index_copy(-1, self.upper_indices, arm_action)

    def posture_reward(self, state, blend: Optional[torch.Tensor] = None,
                       entry_time: Optional[torch.Tensor] = None):
        """Reward holding the trained aiming posture without requiring playback."""
        q, qd = state[4], state[5]
        q = q.index_select(-1, self.upper_indices)
        qd = qd.index_select(-1, self.upper_indices)
        target = self.reference_q
        if entry_time is not None:
            if blend is not None:
                raise ValueError("Provide entry_time or blend, not both")
            if entry_time.ndim != 0 and entry_time.shape != q.shape[:-1]:
                raise ValueError("entry_time must be scalar or match the state batch shape")
            target = self.reference_at(entry_time)
        elif self.entry_duration_s > 0.:
            raise ValueError("entry_time is required for an entry schedule")
        elif blend is not None:
            default = self.default_q.index_select(-1, self.upper_indices)
            target = default + blend[..., None] * (target - default)
        pose_loss = F.smooth_l1_loss(
            (q - target) / self.posture_tolerance_rad,
            torch.zeros_like(q), reduction="none",
        ).mean(-1)
        rate_loss = (qd / self.max_joint_rate_rad_s).square().mean(-1)
        return -.004 * pose_loss - .001 * rate_loss

    def to_dict(self):
        """Return a JSON-compatible constructor config with named arm targets."""
        config = {
            "joint_names": list(self.joint_names),
            "default_q": self.default_q.detach().cpu().tolist(),
            "action_scale": self.action_scale.detach().cpu().tolist(),
            "reference_q": {
                name: float(self.reference_q[index].detach().cpu())
                for index, name in enumerate(self.upper_joint_names)
            },
            "residual_radius_rad": self.residual_radius_rad,
            "max_joint_rate_rad_s": self.max_joint_rate_rad_s,
            "control_dt": self.control_dt,
            "action_limit": self.action_limit,
        }
        if self.max_joint_acceleration_rad_s2:
            config["max_joint_acceleration_rad_s2"] = self.max_joint_acceleration_rad_s2
        if self.entry_duration_s:
            config["entry_times_s"] = list(self._entry_times_config)
            config["entry_reference_q"] = [
                dict(zip(self.upper_joint_names, row))
                for row in self.entry_reference_q.detach().cpu().tolist()
            ]
        if self._explicit_posture_tolerance:
            config["posture_tolerance_rad"] = self.posture_tolerance_rad
        return config

    @classmethod
    def from_dict(cls, config):
        """Rebuild a constraint from :meth:`to_dict` output."""
        if not isinstance(config, Mapping):
            raise ValueError("constraint config must be a mapping")
        return cls(**dict(config))

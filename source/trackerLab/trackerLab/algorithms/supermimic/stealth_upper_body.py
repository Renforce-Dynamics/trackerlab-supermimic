"""Static upper-body target constraints for single-clip stealth policies."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

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
    """Bound arm targets around a static pose and rate-limit them in radians.

    Actions use ``default_q + action * action_scale``. The previous applied
    action is supplied by the caller, so the module retains no rollout state.
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
    ):
        super().__init__()
        names, indices = _upper_indices(joint_names)
        default = _finite_vector(default_q, "default_q")
        scale = _finite_vector(action_scale, "action_scale")
        if default.shape != (len(names),) or scale.shape != (len(names),):
            raise ValueError("default_q and action_scale must match joint_names")
        if bool((scale <= 0).any()):
            raise ValueError("action_scale entries must be positive")
        parameters = (
            ("residual_radius_rad", residual_radius_rad),
            ("max_joint_rate_rad_s", max_joint_rate_rad_s),
            ("control_dt", control_dt),
            ("action_limit", action_limit),
        )
        for name, value in parameters:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

        reference = _reference_values(reference_q, names, indices)

        self.joint_names = names
        self.num_joints = len(names)
        self.upper_joint_names = tuple(names[index] for index in indices)
        self.residual_radius_rad = float(residual_radius_rad)
        self.max_joint_rate_rad_s = float(max_joint_rate_rad_s)
        self.control_dt = float(control_dt)
        self.action_limit = float(action_limit)
        self.register_buffer("upper_indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("default_q", default)
        self.register_buffer("action_scale", scale)
        self.register_buffer("reference_q", reference)

    def forward(self, raw_action, previous_applied_action):
        if not torch.jit.is_tracing():
            if (raw_action.ndim < 1 or raw_action.shape[-1] != self.num_joints
                    or previous_applied_action.shape != raw_action.shape):
                raise ValueError("actions must have matching shapes ending in num_joints")
            if (raw_action.device != self.default_q.device
                    or previous_applied_action.device != raw_action.device
                    or raw_action.dtype != self.default_q.dtype
                    or previous_applied_action.dtype != raw_action.dtype):
                raise ValueError("actions and constraint buffers must share device and dtype")

        scale = self.action_scale.index_select(0, self.upper_indices)
        default = self.default_q.index_select(0, self.upper_indices)
        previous = previous_applied_action.index_select(-1, self.upper_indices)
        raw = raw_action.index_select(-1, self.upper_indices)

        previous_q = default + previous * scale
        desired_q = default + raw * scale
        residual = self.residual_radius_rad * torch.tanh(
            (desired_q - self.reference_q) / self.residual_radius_rad
        )
        target_q = self.reference_q + residual
        max_step = self.max_joint_rate_rad_s * self.control_dt
        step = max_step * torch.tanh((target_q - previous_q) / max_step)
        arm_action = ((previous_q + step - default) / scale).clamp(
            -self.action_limit, self.action_limit,
        )
        return raw_action.index_copy(-1, self.upper_indices, arm_action)

    def posture_reward(self, state):
        """Reward holding the trained aiming posture without requiring playback."""
        q, qd = state[4], state[5]
        q = q.index_select(-1, self.upper_indices)
        qd = qd.index_select(-1, self.upper_indices)
        pose_loss = F.smooth_l1_loss(
            (q - self.reference_q) / self.residual_radius_rad,
            torch.zeros_like(q), reduction="none",
        ).mean(-1)
        rate_loss = (qd / self.max_joint_rate_rad_s).square().mean(-1)
        return -.004 * pose_loss - .001 * rate_loss

    def to_dict(self):
        """Return a JSON-compatible constructor config with named arm targets."""
        return {
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

    @classmethod
    def from_dict(cls, config):
        """Rebuild a constraint from :meth:`to_dict` output."""
        if not isinstance(config, Mapping):
            raise ValueError("constraint config must be a mapping")
        return cls(**dict(config))

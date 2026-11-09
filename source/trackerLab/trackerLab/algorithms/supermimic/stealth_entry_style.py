"""Training-only torso entry shaping with source-bound reference metadata."""

from __future__ import annotations

import copy
import json
import math
from numbers import Integral, Real

import torch
from torch import nn
import torch.nn.functional as F

from trackerLab.kinematics.supermimic.rotations import yaw_mat


_SCHEMA = "g1-stealth-entry-style-v1"
_STATE_NAMES = ("root_p", "root_R", "root_v", "root_w", "q", "qd")


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _index(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _hash(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in value)):
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return value


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Entry metadata must be finite JSON values") from error


def _source(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("source must contain prepared expert provenance")
    source = _json_copy(value)
    filename = source.get("forward_csv")
    if (not isinstance(filename, str) or not filename.endswith(".csv")
            or len(filename) <= 4 or "/" in filename or "\\" in filename):
        raise ValueError("source.forward_csv must be a plain CSV filename")
    _hash(source.get("forward_csv_sha256"), "source.forward_csv_sha256")
    _hash(source.get("pool_manifest_sha256"), "source.pool_manifest_sha256")
    _index(source.get("converted_frame"), "source.converted_frame")
    return source


def _torso_layout(kin):
    torso = getattr(getattr(kin, "robot", None), "com_body", None)
    try:
        tracked = [kin.body_names[i] for i in kin.tracked_idx]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("Kinematics must expose named tracked bodies") from error
    if not isinstance(torso, str) or not torso or tracked.count(torso) != 1:
        raise ValueError("Kinematics must track robot.com_body exactly once")
    return torso, tracked.index(torso)


def _torso_up_heading(kin, state, torso_index):
    # Remove yaw only: root pitch/roll must remain in the gravity-relative tilt.
    state = tuple(value.float() if value.dtype in (torch.float16, torch.bfloat16)
                  else value for value in state)
    with torch.autocast(device_type=state[0].device.type, enabled=False):
        rotations = kin.forward(*state)[1]
        torso_up = rotations[..., torso_index, :, 2]
        heading_up = (
            yaw_mat(state[1]).transpose(-1, -2) @ torso_up[..., None]
        ).squeeze(-1)
        return F.normalize(heading_up, dim=-1, eps=1e-8)


def entry_torso_cost(
    torso_up_heading: torch.Tensor,
    target_up: torch.Tensor,
    entry_time: torch.Tensor,
    duration: float = 1.21,
    error_scale: float = .35,
) -> torch.Tensor:
    """Unweighted entry cost in [0, 1] for finite unit heading-frame vectors.

    Inputs end in three forward/left/up coordinates; leading shapes and the
    clock broadcast. Completed clocks give exactly zero cost and gradients.
    This function has no action-control or runtime-reference behavior.
    """
    _positive(duration, "duration")
    _positive(error_scale, "error_scale")
    if (not isinstance(torso_up_heading, torch.Tensor)
            or not torso_up_heading.is_floating_point()
            or torso_up_heading.ndim < 1 or torso_up_heading.shape[-1] != 3):
        raise ValueError("torso_up_heading must be a floating tensor ending in 3")
    dtype = (torch.float32 if torso_up_heading.dtype in (torch.float16, torch.bfloat16)
             else torso_up_heading.dtype)
    up = torso_up_heading.to(dtype=dtype)
    target = torch.as_tensor(target_up, dtype=dtype, device=up.device)
    time = torch.as_tensor(entry_time, dtype=dtype, device=up.device)
    if target.ndim < 1 or target.shape[-1] != 3:
        raise ValueError("target_up must end in 3")
    phase = (time / duration).clamp(0., 1.)
    blend = (phase.pow(3) * (10. + phase * (-15. + 6. * phase))).clamp(0., 1.)
    upright = up.new_tensor((0., 0., 1.))
    desired = F.normalize(
        (1. - blend[..., None]) * upright + blend[..., None] * target,
        dim=-1, eps=1e-8,
    )
    error_squared = (up - desired).square().sum(-1)
    cost = (1. - blend) * error_squared / (error_squared + error_scale**2)
    return torch.where(time >= duration, torch.zeros_like(cost), cost)


class EntryStyleGuidance(nn.Module):
    """Fixed training reward; FK and input-state gradients remain differentiable.

    The caller owns kinematics/device placement and supplies post-step state
    and command clock. Configuration is JSON metadata, independent of any
    runtime action controller. Disabled behavior is owned by the caller.
    """

    def __init__(self, kin, config):
        super().__init__()
        if not isinstance(config, dict):
            raise ValueError("Entry config must be a JSON object")
        config = _json_copy(config)
        if config.get("schema", _SCHEMA) != _SCHEMA:
            raise ValueError("Unsupported entry style schema")
        torso, self.torso_index = _torso_layout(kin)
        if config.get("torso_body") != torso:
            raise ValueError("Entry torso_body must match tracked robot.com_body")
        self.weight = _positive(config.get("weight"), "weight")
        self.duration = _positive(config.get("duration"), "duration")
        self.error_scale = _positive(config.get("error_scale", .35), "error_scale")
        values = config.get("target_up")
        if (not isinstance(values, list) or len(values) != 3
                or any(isinstance(v, bool) or not isinstance(v, Real) for v in values)):
            raise ValueError("target_up must be a finite unit three-vector")
        target = torch.tensor(values, dtype=torch.float64)
        if (not bool(torch.isfinite(target).all())
                or abs(float(torch.linalg.vector_norm(target)) - 1.) > 1e-5
                or target[0] <= 0 or target[2] <= 0):
            raise ValueError("target_up must be unit length with positive forward and up")
        config["source"] = _source(config.get("source"))
        _hash(config.get("expert_hash"), "expert_hash")
        _index(config.get("reference_index"), "reference_index")
        _index(config.get("reference_clip_index"), "reference_clip_index")
        config.update(schema=_SCHEMA, weight=self.weight, duration=self.duration,
                      error_scale=self.error_scale)
        self._config = config
        object.__setattr__(self, "kin", kin)
        self.register_buffer("target_up", target.to(device=getattr(kin, "device", "cpu")))

    def reward(self, state, c):
        """Nonpositive per-step reward; c[..., 6] is elapsed skill-entry time."""
        if c.ndim < 1 or c.shape[-1] != 7:
            raise ValueError("Entry command must end in seven scenario fields")
        up = _torso_up_heading(self.kin, state, self.torso_index)
        return -self.weight * entry_torso_cost(
            up, self.target_up, c[..., 6], self.duration, self.error_scale,
        )

    def to_dict(self):
        """Return independent JSON metadata for checkpoint/resume validation."""
        return copy.deepcopy(self._config)


def calibrate_entry_config(kin, motion, reference_document, weight, duration, expert_hash):
    """Calibrate once from an already verified prepared document and motion.

    The caller invokes prepare_stealth_waist_entry.prepare(pool) outside
    compiled code and verifies that pool supplies this motion. This function
    performs no file imports, hashing, motion loading or state mutation.
    """
    if (not isinstance(reference_document, dict)
            or reference_document.get("schema") != "g1-stealth-waist-entry-v1"):
        raise ValueError("Expected a prepared source-bound waist reference document")
    source = _source(reference_document.get("source"))
    stem = source["forward_csv"][:-4]
    matches = [i for i, name in enumerate(motion.names) if name == stem]
    if len(matches) != 1:
        raise ValueError("Forward CSV must match exactly one motion.names stem")
    clip = matches[0]
    start, end = int(motion.clip_start[clip]), int(motion.clip_end[clip])
    frame = source["converted_frame"]
    index = start + frame
    if start < 0 or end < start or index > end or end >= len(motion.q):
        raise ValueError("Reference frame lies outside its motion clip")
    torso, torso_index = _torso_layout(kin)
    with torch.no_grad():
        state = tuple(getattr(motion, name)[index:index + 1] for name in _STATE_NAMES)
        if not all(bool(torch.isfinite(value).all()) for value in state):
            raise ValueError("Reference motion state must be finite")
        target = _torso_up_heading(kin, state, torso_index)[0].detach().cpu().tolist()
    config = {
        "schema": _SCHEMA, "weight": weight, "duration": duration, "error_scale": .35,
        "target_up": target, "torso_body": torso, "source": source,
        "expert_hash": expert_hash, "reference_index": index, "reference_clip_index": clip,
    }
    return EntryStyleGuidance(kin, config).to_dict()

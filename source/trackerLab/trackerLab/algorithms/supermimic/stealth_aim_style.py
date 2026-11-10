"""Training-only, source-bound guidance toward one command-independent aim pose."""

from __future__ import annotations

import copy
import json
import math
from numbers import Integral, Real

import torch
from torch import nn
import torch.nn.functional as F

from trackerLab.kinematics.supermimic.rotations import yaw_mat
from trackerLab.tracker_env.mdp.supermimic_command import entry_blend
from trackerTask.supermimic.robot import G1_JOINT_NAMES


_SCHEMA = "g1-stealth-aim-style-v1"
_ROBOT_HEIGHT = .76
_TORSO = "torso_link"
_WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_ARMS = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
)
_SCALES = {
    "position_rmse_m": .05, "rotation_chordal": .2, "arm_rmse_rad": .2,
    "torso_up_error": .2, "arm_speed_rms_rad_s": .1,
}
_WEIGHTS = dict(zip(_SCALES, (.35, .2, .2, .2, .05)))


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    represented = torch.tensor(value, dtype=torch.float32)
    if not bool(torch.isfinite(represented)) or represented.item() <= 0:
        raise ValueError(f"{name} must be representable as positive float32")
    return float(value)


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Aim metadata must contain finite JSON values") from error


def _source(value):
    """Keep caller-owned frame identity; require at least one explicit SHA256."""
    if not isinstance(value, dict) or not value:
        raise ValueError("source must contain selected-pose provenance")
    source = _json_copy(value)

    def digest(value, name):
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in value)):
            raise ValueError(f"{name} must be a SHA256 hex digest string")

    def hashes(document, prefix):
        count = 0
        for key, item in document.items():
            name = f"{prefix}.{key}"
            if key == "sha256" or key.endswith("_sha256"):
                digest(item, name)
                count += 1
            elif key == "hashes":
                if not isinstance(item, dict) or not item:
                    raise ValueError(f"{name} must map names to SHA256 digest strings")
                for label, value in item.items():
                    digest(value, f"{name}.{label}")
                    count += 1
            elif isinstance(item, dict):
                count += hashes(item, name)
        return count

    if not hashes(source, "source"):
        raise ValueError("source must contain a sha256, *_sha256 or hashes digest")
    return source


def _array(value, shape, name):
    def numeric(item):
        if isinstance(item, list):
            return all(numeric(child) for child in item)
        return isinstance(item, Real) and not isinstance(item, bool)

    if not numeric(value):
        raise ValueError(f"{name} must contain finite real numbers")
    try:
        result = torch.tensor(value, dtype=torch.float64)
    except (ValueError, TypeError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must have shape {shape}") from error
    if (result.shape != shape or not bool(torch.isfinite(result).all())
            or not bool(torch.isfinite(result.float()).all())):
        raise ValueError(f"{name} must have shape {shape} and finite float32 values")
    return result


def _rotations(value, name):
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    if (not torch.allclose(value.transpose(-1, -2) @ value, identity.expand_as(value),
                           rtol=0., atol=1e-5)
            or not torch.allclose(torch.linalg.det(value),
                                  torch.ones_like(value[..., 0, 0]), rtol=0., atol=1e-5)):
        raise ValueError(f"{name} must contain proper rotation matrices")


def _layout(kin):
    robot = getattr(kin, "robot", None)
    if getattr(robot, "name", None) != "g1" or getattr(robot, "com_body", None) != _TORSO:
        raise ValueError("Aim guidance requires the g1 robot with com_body torso_link")
    names = getattr(kin, "joint_names", ())
    if (not isinstance(names, (tuple, list)) or len(names) != 29
            or any(not isinstance(name, str) for name in names)
            or set(names) != set(G1_JOINT_NAMES)):
        raise ValueError("Kinematics must declare the 29 unique named G1 joints")
    bodies, tracked = getattr(kin, "body_names", ()), getattr(kin, "tracked_idx", ())
    if (not isinstance(bodies, (tuple, list)) or not bodies
            or any(not isinstance(name, str) or not name for name in bodies)
            or len(set(bodies)) != len(bodies)):
        raise ValueError("Kinematics must declare unique body_names")
    if (not isinstance(tracked, (tuple, list)) or not tracked
            or any(isinstance(i, bool) or not isinstance(i, Integral)
                   or not 0 <= i < len(bodies) for i in tracked)
            or len(set(tracked)) != len(tracked)):
        raise ValueError("Kinematics tracked_idx must contain unique valid body indices")
    tracked_names = tuple(bodies[i] for i in tracked)
    if any(name not in tracked_names for name in (_TORSO, *_WRISTS)):
        raise ValueError("Kinematics must track torso_link and both wrist_yaw links")
    return (tuple(names.index(name) for name in _ARMS), tracked_names.index(_TORSO),
            tuple(tracked_names.index(name) for name in _WRISTS), len(tracked_names))


def _state(state):
    if (not isinstance(state, (tuple, list)) or len(state) != 6
            or any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
                   for value in state) or state[4].ndim < 1):
        raise ValueError("state must contain six floating root_p/root_R/root_v/root_w/q/qd tensors")
    q = state[4]
    batch = q.shape[:-1]
    shapes = (batch + (3,), batch + (3, 3), batch + (3,), batch + (3,),
              batch + (29,), batch + (29,))
    if any(value.shape != shape for value, shape in zip(state, shapes)):
        raise ValueError("state shapes must share a batch with 3D root fields and 29 joints")
    if any(value.device != q.device or value.dtype != q.dtype for value in state):
        raise ValueError("state tensors must share device and dtype")
    return tuple(value.float() if value.dtype in (torch.float16, torch.bfloat16)
                 else value for value in state)


def _pose(kin, state, torso_index, wrist_indices, num_tracked):
    with torch.autocast(device_type=state[0].device.type, enabled=False):
        positions, rotations = kin.forward(*state)[:2]
        batch = state[4].shape[:-1]
        if (positions.shape != batch + (num_tracked, 3)
                or rotations.shape != batch + (num_tracked, 3, 3)):
            raise ValueError("FK must match the tracked position and rotation shapes")
        torso_R = rotations[..., torso_index, :, :]
        torso_inverse = torso_R.transpose(-1, -2)
        offsets = positions[..., wrist_indices, :] - positions[..., torso_index, None, :]
        wrist_p = (torso_inverse[..., None, :, :] @ offsets[..., None]).squeeze(-1)
        wrist_R = torso_inverse[..., None, :, :] @ rotations[..., wrist_indices, :, :]
        up = (yaw_mat(state[1]).transpose(-1, -2) @ torso_R[..., :, 2, None]).squeeze(-1)
        return wrist_p, wrist_R, F.normalize(up, dim=-1, eps=1e-8)


def _scaled_norm(value, divisor=1.):
    # Scaling cancels analytically and keeps finite large errors out of squared overflow.
    scale = value.detach().abs().amax(-1, keepdim=True).clamp_min(1.)
    return (torch.linalg.vector_norm(value / scale, dim=-1)
            * (scale.squeeze(-1) / divisor))


class AimStyleGuidance(nn.Module):
    """One frozen aim pose for every command; only c[..., 6] gates activation.

    This module supplies a training reward and diagnostics, with no action or
    reference-playback interface. Runtime states follow the finite FK tensor
    contract. Calibration validates the selected reference outside compiled code.
    """

    def __init__(self, kin, config):
        super().__init__()
        if not isinstance(config, dict):
            raise ValueError("Aim config must be a JSON object")
        config = _json_copy(config)
        if config.get("schema") != _SCHEMA:
            raise ValueError("Unsupported aim style schema")
        arms, self.torso_index, self.wrist_indices, self.num_tracked = _layout(kin)
        if (config.get("robot", "g1") != "g1"
                or config.get("torso_body", _TORSO) != _TORSO
                or config.get("wrist_bodies", list(_WRISTS)) != list(_WRISTS)):
            raise ValueError("Aim robot, torso_body and ordered wrist_bodies must match G1")
        self.weight = _positive(config.get("weight", .01), "weight")
        self.duration = _positive(config.get("duration", .60), "duration")
        self.robot_height = _positive(config.get("robot_height", _ROBOT_HEIGHT), "robot_height")
        if not math.isclose(self.robot_height, _ROBOT_HEIGHT, rel_tol=0., abs_tol=1e-7):
            raise ValueError("Aim robot_height must match the .76 m G1 contract")
        reference = config.get("reference_q")
        if not isinstance(reference, dict) or set(reference) != set(_ARMS):
            raise ValueError("reference_q must map exactly the 14 named G1 arm joints")
        q = _array([reference[name] for name in _ARMS], (14,), "reference_q")
        p = _array(config.get("reference_wrist_positions_torso"), (2, 3),
                   "reference_wrist_positions_torso")
        R = _array(config.get("reference_wrist_rotations_torso"), (2, 3, 3),
                   "reference_wrist_rotations_torso")
        _rotations(R, "reference_wrist_rotations_torso")
        up = _array(config.get("reference_torso_up_heading"), (3,), "reference_torso_up_heading")
        if abs(float(torch.linalg.vector_norm(up)) - 1.) > 1e-5:
            raise ValueError("reference_torso_up_heading must be a unit three-vector")
        scales, weights = config.get("error_scales", _SCALES), config.get("term_weights", _WEIGHTS)
        if not isinstance(scales, dict) or set(scales) != set(_SCALES):
            raise ValueError("error_scales must map exactly the five aim metrics")
        self.error_scales = {name: _positive(scales[name], f"error_scales.{name}") for name in _SCALES}
        if (not isinstance(weights, dict) or set(weights) != set(_WEIGHTS)
                or any(isinstance(w, bool) or not isinstance(w, Real)
                       or not math.isfinite(w) or w < 0 for w in weights.values())
                or not math.isclose(sum(weights.values()), 1., rel_tol=0., abs_tol=1e-8)):
            raise ValueError("term_weights must be five finite nonnegative weights summing to one")
        self.term_weights = {name: float(weights[name]) for name in _WEIGHTS}
        config.update(
            robot="g1", torso_body=_TORSO, wrist_bodies=list(_WRISTS),
            weight=self.weight, duration=self.duration, robot_height=self.robot_height,
            source=_source(config.get("source")), error_scales=self.error_scales.copy(),
            term_weights=self.term_weights.copy(),
        )
        self._config = config
        object.__setattr__(self, "kin", kin)
        device = getattr(kin, "device", "cpu")
        self.register_buffer("arm_indices", torch.tensor(arms, dtype=torch.long, device=device))
        self.register_buffer("reference_q", q.to(device=device))
        self.register_buffer("reference_wrist_positions_torso", p.to(device=device))
        self.register_buffer("reference_wrist_rotations_torso", R.to(device=device))
        self.register_buffer("reference_torso_up_heading", up.to(device=device))

    def metrics(self, state):
        """Per-sample RMS wrist distance, chordal distance, arm pose, tilt and speed.

        Position is RMS Euclidean distance over two wrists (meters); rotation
        is RMS Frobenius distance divided by sqrt(2), approximately radians for
        small rotations. Arm pose/speed are RMS over the 14 named joints. Torso
        error is the Euclidean distance between unit heading-frame up vectors.
        """
        state = _state(state)
        q, qd = state[4:]
        p, R, up = _pose(self.kin, state, self.torso_index, self.wrist_indices, self.num_tracked)
        position = p - self.reference_wrist_positions_torso.to(p)
        rotation = R - self.reference_wrist_rotations_torso.to(R)
        arm = q.index_select(-1, self.arm_indices) - self.reference_q.to(q)
        torso = up - self.reference_torso_up_heading.to(up)
        speed = qd.index_select(-1, self.arm_indices)
        return {
            "position_rmse_m": _scaled_norm(position.flatten(-2), math.sqrt(2.)),
            "rotation_chordal": _scaled_norm(rotation.flatten(-3), 2.),
            "arm_rmse_rad": _scaled_norm(arm, math.sqrt(14.)),
            "torso_up_error": _scaled_norm(torso),
            "arm_speed_rms_rad_s": _scaled_norm(speed, math.sqrt(14.)),
        }

    def reward(self, state, c):
        """Reward in [-weight, 0], quintic activation from zero to duration."""
        if (not isinstance(c, torch.Tensor) or not c.is_floating_point()
                or c.ndim < 1 or c.shape[-1] != 7):
            raise ValueError("Aim command must be a floating tensor ending in seven fields")
        metrics = self.metrics(state)
        sample = metrics["arm_rmse_rad"]
        if c.device != sample.device:
            raise ValueError("Aim command and state must share a device")
        try:
            torch.broadcast_shapes(sample.shape, c.shape[:-1])
        except RuntimeError as error:
            raise ValueError("Aim command batch must broadcast with state batch") from error
        cost = torch.zeros_like(sample)
        for name, value in metrics.items():
            # (value / scale)^2 / (1 + (value / scale)^2), without squaring overflow.
            saturated = (value / torch.hypot(value, value.new_tensor(self.error_scales[name]))).square()
            cost = cost + self.term_weights[name] * saturated
        blend = entry_blend(c.to(sample), self.duration).clamp(0., 1.)
        return -self.weight * blend * cost.clamp(0., 1.)

    def to_dict(self):
        """Return independent JSON metadata for checkpoint and resume identity."""
        return copy.deepcopy(self._config)


@torch.no_grad()
def calibrate_aim_config(kin, stateone, source, weight=.01, duration=.60):
    """Freeze one caller-selected G1 pose; loading and source selection stay outside.

    An unbatched state or any singleton batch is accepted. Pose values are copied
    as JSON numbers. Reference velocity is zero, independent of the source's qd.
    Source must carry an explicit SHA256 digest and may retain asset/frame metadata.
    """
    state = _state(stateone)
    if state[4].numel() != 29:
        raise ValueError("stateone must contain exactly one selected G1 pose")
    if any(not bool(torch.isfinite(value).all()) for value in state):
        raise ValueError("Selected reference state must be finite")
    _rotations(state[1].double(), "Selected root_R")
    arms, torso, wrists, count = _layout(kin)
    p, R, up = _pose(kin, state, torso, wrists, count)
    q = state[4][..., list(arms)].reshape(14).detach().cpu().tolist()
    config = {
        "schema": _SCHEMA, "weight": weight, "duration": duration,
        "robot_height": _ROBOT_HEIGHT,
        "reference_q": dict(zip(_ARMS, q)),
        "reference_wrist_positions_torso": p.reshape(2, 3).detach().cpu().tolist(),
        "reference_wrist_rotations_torso": R.reshape(2, 3, 3).detach().cpu().tolist(),
        "reference_torso_up_heading": up.reshape(3).detach().cpu().tolist(),
        "source": _source(source),
    }
    return AimStyleGuidance(kin, config).to_dict()

"""Shared G1 sagittal reflection for turning-style reference preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_MIRROR = np.diag((1.0, -1.0, 1.0))
# G1 29-DoF rev 1.0 hinge axes: pitch/knee/elbow are Y; roll/yaw are X/Z.
# Keep the complete contract local so reference reflection needs no model runtime.
_G1_JOINT_SIGNS = {
    f"{side}_{part}_joint": sign
    for side in ("left", "right")
    for part, sign in (
        ("hip_pitch", 1.0), ("hip_roll", -1.0), ("hip_yaw", -1.0),
        ("knee", 1.0), ("ankle_pitch", 1.0), ("ankle_roll", -1.0),
        ("shoulder_pitch", 1.0), ("shoulder_roll", -1.0), ("shoulder_yaw", -1.0),
        ("elbow", 1.0), ("wrist_roll", -1.0), ("wrist_pitch", 1.0), ("wrist_yaw", -1.0),
    )
} | {"waist_yaw_joint": -1.0, "waist_roll_joint": -1.0, "waist_pitch_joint": 1.0}


def _mirror_joint_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


@dataclass(frozen=True)
class MirrorReceipt:
    """Source column i writes signs[i] * q[i] to source_to_target[i]."""

    joint_names: tuple[str, ...]
    source_to_target: tuple[int, ...]
    signs: tuple[float, ...]
    reflection_matrix: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict:
        return {
            "joint_names": list(self.joint_names),
            "source_to_target": list(self.source_to_target),
            "signs": list(self.signs),
            "reflection_matrix": [list(row) for row in self.reflection_matrix],
            "convention": "world sagittal reflection y -> -y with named G1 joint symmetry",
        }


def mirror_receipt(joint_names: tuple[str, ...] | list[str]) -> MirrorReceipt:
    """Validate a complete canonical G1 joint layout, in any supplied order."""
    if not isinstance(joint_names, (tuple, list)):
        raise ValueError("G1 joint_names must be a tuple or list of canonical joint names")
    names = tuple(joint_names)
    if any(not isinstance(name, str) for name in names):
        raise ValueError("G1 joint_names must contain strings")
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("G1 joint_names must contain exactly 29 unique canonical joint names")
    if set(names) != _G1_JOINT_SIGNS.keys():
        missing = sorted(_G1_JOINT_SIGNS.keys() - set(names))
        unknown = sorted(set(names) - _G1_JOINT_SIGNS.keys())
        raise ValueError(f"G1 joint_names are incompatible: missing={missing}, unknown={unknown}")
    index = {name: i for i, name in enumerate(names)}
    targets = tuple(index[_mirror_joint_name(name)] for name in names)
    signs = tuple(_G1_JOINT_SIGNS[name] for name in names)
    return MirrorReceipt(
        names, targets, signs, tuple(tuple(float(value) for value in row) for row in _MIRROR)
    )


def mirror_g1_motion(values: np.ndarray, joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
    """Reflect a nonempty (T, 36) G1 XYZ/XYZW/q29 clip across world y=0.

    Joint columns retain the supplied canonical-name order. Quaternion signs
    and frame order are preserved; applying this transform twice recovers the
    input, including its quaternion representatives. Input arrays are untouched.
    This is the rev 1.0 kinematic convention, with no joint-limit clamping.
    """
    receipt = mirror_receipt(joint_names)
    values = np.asarray(values)
    if values.dtype.kind not in "fiu":
        raise ValueError("G1 motion must contain real numeric values")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 36:
        raise ValueError("G1 motion must have shape (T>=1, 36): XYZ/XYZW and 29 named joints")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("G1 motion must be finite")
    quaternions = values[:, 3:7]
    if (np.any(np.abs(quaternions) > 1.0 + 1e-5)
            or not np.allclose(np.linalg.norm(quaternions, axis=1), 1., atol=1e-5, rtol=0)):
        raise ValueError("G1 root quaternions must be unit XYZW")
    output = values.copy()
    output[:, :3] = values[:, :3] @ _MIRROR.T
    # M R M maps XYZW to (-x, y, -z, w) without matrix-to-quaternion sign jumps.
    output[:, [3, 5]] *= -1.0
    output[:, 7 + np.asarray(receipt.source_to_target)] = values[:, 7:] * receipt.signs
    return output


__all__ = ["MirrorReceipt", "mirror_g1_motion", "mirror_receipt"]

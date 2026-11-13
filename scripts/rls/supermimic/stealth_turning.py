"""Shared G1 sagittal reflection for turning-style reference preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


_MIRROR = np.diag((1.0, -1.0, 1.0))
_NEGATIVE_MIRROR_PARTS = frozenset(
    ("hip_roll", "hip_yaw", "ankle_roll", "shoulder_roll",
     "shoulder_yaw", "wrist_roll", "wrist_yaw")
)


def _mirror_joint_name(name: str) -> str:
    if "left_" in name:
        return name.replace("left_", "right_", 1)
    if "right_" in name:
        return name.replace("right_", "left_", 1)
    return name


def _mirror_joint_sign(name: str) -> float:
    base = name.replace("left_", "").replace("right_", "")
    if name.startswith(("waist_yaw_joint", "waist_roll_joint")):
        return -1.0
    return -1.0 if any(part in base for part in _NEGATIVE_MIRROR_PARTS) else 1.0


@dataclass(frozen=True)
class MirrorReceipt:
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
    names = tuple(joint_names)
    index = {name: i for i, name in enumerate(names)}
    targets = tuple(index[_mirror_joint_name(name)] for name in names)
    signs = tuple(_mirror_joint_sign(name) for name in names)
    return MirrorReceipt(
        names, targets, signs, tuple(tuple(float(value) for value in row) for row in _MIRROR)
    )


def mirror_g1_motion(values: np.ndarray, joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
    """Reflect finite G1 XYZ/XYZW/q29 motion across the sagittal plane."""
    values = np.asarray(values, dtype=np.float64)
    names = tuple(joint_names)
    receipt = mirror_receipt(names)
    if values.ndim != 2 or values.shape[1] != 7 + len(names):
        raise ValueError("G1 motion must have XYZ/XYZW followed by the named joint vector")
    if not np.isfinite(values).all():
        raise ValueError("G1 motion must be finite")
    if not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1., atol=1e-5, rtol=0):
        raise ValueError("G1 root quaternions must be unit XYZW")
    output = values.copy()
    output[:, :3] = values[:, :3] @ _MIRROR.T
    rotations = Rotation.from_quat(values[:, 3:7]).as_matrix()
    output[:, 3:7] = Rotation.from_matrix(
        np.einsum("ij,njk,kl->nil", _MIRROR, rotations, _MIRROR)
    ).as_quat()
    output[:, 7:] = values[:, 7:][:, receipt.source_to_target] * np.asarray(
        receipt.signs, dtype=np.float64
    )[None, :]
    return output


__all__ = ["MirrorReceipt", "mirror_g1_motion", "mirror_receipt"]

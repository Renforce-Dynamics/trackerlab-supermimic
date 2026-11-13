"""Turning-reference symmetry and input validation."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from stealth_turning import mirror_g1_motion, mirror_receipt


JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "right_hip_pitch_joint",
    "right_hip_roll_joint", "waist_yaw_joint",
)


def test_receipt_swaps_sides_and_reflects_joint_axes():
    receipt = mirror_receipt(JOINTS)
    assert receipt.source_to_target == (2, 3, 0, 1, 4)
    assert receipt.signs == (1., -1., 1., -1., -1.)


def test_motion_reflection_preserves_root_quaternion_and_named_joint_contract():
    values = np.zeros((2, 12), dtype=np.float64)
    values[:, :3] = [[.2, .3, .7], [.4, -.1, .7]]
    values[:, 3:7] = Rotation.from_euler("xyz", [[.1, .2, .3], [.2, -.1, .4]]).as_quat()
    values[:, 7:] = np.arange(10, dtype=np.float64).reshape(2, 5) / 10.
    mirrored = mirror_g1_motion(values, JOINTS)
    np.testing.assert_allclose(mirrored[:, :3], values[:, :3] * [1., -1., 1.])
    assert np.allclose(np.linalg.norm(mirrored[:, 3:7], axis=1), 1.)
    np.testing.assert_allclose(mirrored[:, 7:], values[:, 7:][:, [2, 3, 0, 1, 4]] * [1., -1., 1., -1., -1.])


@pytest.mark.parametrize("bad", [
    np.zeros((2, 11)),
    np.full((2, 12), np.nan),
])
def test_motion_reflection_rejects_invalid_inputs(bad):
    if bad.shape == (2, 12):
        bad[:, 6] = 1.
    with pytest.raises(ValueError):
        mirror_g1_motion(bad, JOINTS)

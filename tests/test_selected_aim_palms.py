"""Regressions for bilateral pose transfer and visual-palm geometry coverage."""

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("mink", reason="Optional gait-retargeting dependency")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts/rls/supermimic"))
from retarget_gait_references import flat_model, ROLES
from retarget_selected_aim import (
    coupled_arm_targets, palm_clearance_limit, palm_frame, robot_palms, source_palms)


def test_palm_frame_observes_roll_and_rotates_with_anatomy():
    frame = palm_frame((1, 0, 0), (0, 0, 1))
    rotation = Rotation.from_rotvec((.5, .1, -.2)).as_matrix()
    np.testing.assert_allclose(
        palm_frame(rotation @ frame[:, 0], rotation @ frame[:, 1]), rotation @ frame)
    rolled = palm_frame((1, 0, 0), (0, 1, 0))
    np.testing.assert_array_equal(rolled[:, 0], frame[:, 0])
    assert Rotation.from_matrix(rolled @ frame.T).magnitude() > 1.
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.)
    with pytest.raises(ValueError, match="degenerate"):
        palm_frame((1, 0, 0), (2, 0, 0))


def test_shared_wrist_delta_survives_asymmetric_chain_morphology():
    source = np.zeros((14, 3))
    source[[8, 9, 10]] = ((0, .2, .3), (.2, .2, .2), (.3, .05, .4))
    source[[11, 12, 13]] = ((0, -.2, .3), (.2, -.2, .25), (.32, .01, .47))
    rest = source.copy()
    rest[8:11] *= .6
    rest[11:14] *= .8
    target = source.copy()
    target[8], target[11] = (0, .15, .2), (0, -.15, .2)
    changed, metadata = coupled_arm_targets(source, target, rest, np.eye(3))
    scale = metadata["arm_scale"]
    np.testing.assert_allclose(changed[10]-changed[13], (source[10]-source[13])*scale)
    np.testing.assert_array_equal(changed[:8], target[:8])
    np.testing.assert_array_equal(changed[[8, 11]], target[[8, 11]])
    R = Rotation.from_rotvec((.2, -.3, .5)).as_matrix()
    translation = np.array((2., -1., 3.))
    rotated, _ = coupled_arm_targets(source @ R.T+translation,
        target @ R.T+translation, rest @ R.T+translation, R)
    np.testing.assert_allclose(rotated, changed @ R.T+translation, atol=1e-12)


def test_source_palm_calibration_rejects_another_body_before_loading(tmp_path):
    path = tmp_path/"wrong.npz"
    path.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="exact source body"):
        source_palms({"sourceCorrection": {"bodyModelSha256": "0"*64}}, path)


@pytest.fixture(scope="module")
def g1():
    from trackerTask.supermimic.robots import get_robot
    if not Path(get_robot("g1").mjcf_path).is_file():
        pytest.skip("Local G1 mesh assets are required")
    return flat_model("g1")


def test_palm_frames_use_proximal_mesh_and_visual_geoms_remain_unchanged(g1):
    model = g1[0]
    before = model.geom_contype.copy(), model.geom_conaffinity.copy()
    frames, geoms, evidence = robot_palms(model)
    for frame in frames:
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.)
        assert frame[0, 0] > .99 and frame[2, 1] > .99
    assert all(model.geom_contype[g] == model.geom_conaffinity[g] == 0 for g in geoms)
    limit = palm_clearance_limit(model, geoms)
    assert limit.geom_id_pairs == [tuple(sorted(geoms))]
    np.testing.assert_array_equal(model.geom_contype, before[0])
    np.testing.assert_array_equal(model.geom_conaffinity, before[1])
    assert evidence["left"]["core_vertices"] > 1000


def test_published_pose_preserves_full_frames_and_explicit_clearance(g1):
    canonical = REPO.parent/"artifacts/g1-stealth-shared-aim-20261110/selected-library-asset/canonical.json"
    model_path = REPO.parent/"models/smplx/SMPLX_NEUTRAL.npz"
    selected = REPO/"data/style-research-20261110/owner-aim-canonical-v5/selected-pose.csv"
    if not all(path.is_file() for path in (canonical, model_path, selected)):
        pytest.skip("Owner's private source/body and generated pose are required")
    from retarget_selected_aim import body_frame
    document = json.loads(canonical.read_bytes())
    report = json.loads(selected.with_suffix(".json").read_bytes())
    assert report["geometry_acceptance"]["passed"] is True
    assert report["sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    model, _, _, qids, *_ = g1
    data = mujoco.MjData(model)
    pose = np.loadtxt(selected, delimiter=",")
    data.qpos[:3], data.qpos[3:7], data.qpos[qids] = pose[:3], pose[[6, 3, 4, 5]], pose[7:]
    mujoco.mj_forward(model, data)
    source_frames, _ = source_palms(document, model_path)
    robot_frames, geoms, _ = robot_palms(model)
    frame = document["frames"][68]
    p = {name: np.asarray(value["position"]) for name, value in frame.items()}
    root = body_frame(p["LeftUpLeg"]-p["RightUpLeg"], p["Spine2"]-p["Hips"])
    heading = np.arctan2(root[1, 0], root[0, 0])
    H = Rotation.from_euler("Z", heading).as_matrix()
    for i, side in enumerate(("Left", "Right")):
        q = np.asarray(frame[side+"Hand"]["quaternion"])
        wanted = H.T @ Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix() @ source_frames[i]
        actual = data.xmat[model.body(ROLES["g1"][10 if i == 0 else 13]).id].reshape(3, 3) @ robot_frames[i]
        assert Rotation.from_matrix(wanted.T @ actual).magnitude() < .1
    distance = mujoco.mj_geomDistance(model, data, *geoms, 1., np.empty(6))
    assert distance >= .0035
    assert report["bilateral_hands"]["actual_wrist_distance_m"] < .075

"""Reference-frame regression, physical geometry and immutable-input checks."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths

configure_paths()
from full_body_reference import (
    CRITERIA, accept_report, assess_reference, csv_to_qpos, layout, solve_reference, validate_motion,
    assess_staggered_idle, solve_staggered_idle,
)
from prepare_stealth_full_body import local_file, verify_full_body_pool
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import generate, robot_parameters


def good_report():
    return {name: (.003 if name == "minimum_sole_height_m" else 0.) for name in CRITERIA}


@pytest.mark.parametrize("name", CRITERIA)
def test_acceptance_rejects_each_independent_violation(name):
    report = good_report()
    limit = CRITERIA[name]
    report[name] = limit - .001 if name == "minimum_sole_height_m" else limit + .001
    passed, failures = accept_report(report)
    assert not passed and any(name in failure for failure in failures)


@pytest.mark.parametrize("value", [None, True, "0.0", float("nan"), float("inf")])
def test_acceptance_requires_real_finite_measurements(value):
    report = good_report()
    report["max_torso_rotation_error_rad"] = value
    assert not accept_report(report)[0]


def test_contact_aware_acceptance_keeps_strict_foot_error_visible():
    report = good_report()
    report["max_foot_position_error_m"] = .005
    report["max_sole_vertical_error_m"] = .002
    report["max_support_slip_increase_m_s"] = .0012
    report["strict_rigid_foot_1mm_passed"] = False
    snapshot = copy.deepcopy(report)
    assert accept_report(report) == (True, [])
    assert report == snapshot
    report["minimum_sole_height_m"] = -.001
    assert not accept_report(report)[0]


def test_motion_validation_and_joint_order_conversion():
    values = np.zeros((11, 36))
    values[:, 6] = 1
    values[:, 7:] = np.arange(29)
    assert np.array_equal(validate_motion(values), values)
    addresses = np.arange(35, 6, -1)
    poses = csv_to_qpos(values, addresses)
    np.testing.assert_array_equal(poses[:, 3:7], np.tile([1, 0, 0, 0], (11, 1)))
    np.testing.assert_array_equal(poses[:, addresses], values[:, 7:])
    for bad in (values[:10], values[:, :35], values * float("nan")):
        with pytest.raises(ValueError):
            validate_motion(bad)
    values[:, 6] = .5
    with pytest.raises(ValueError):
        validate_motion(values)


@pytest.mark.parametrize("relative", ["../outside.csv", "/outside.csv", "sources/../source.csv",
                                     "sources\\source.csv", "", "./source.csv", "C:/outside.csv"])
def test_source_binding_rejects_escaping_or_ambiguous_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        local_file(tmp_path, relative)


def test_source_binding_resolves_only_existing_pool_files(tmp_path):
    (tmp_path / "sources").mkdir()
    path = tmp_path / "sources/source.csv"
    path.write_text("original\n")
    assert local_file(tmp_path, "sources/source.csv") == path.resolve()
    with pytest.raises(ValueError):
        local_file(tmp_path, "sources/missing.csv")


@pytest.fixture
def physical_g1():
    import mujoco

    spec = get_robot("g1")
    if not Path(spec.mjcf_path).is_file():
        pytest.skip("Local G1 model assets are not installed")
    model, spec, *_ = build_model("g1", generate("stealth", .76, 42), self_collisions=True)
    _, _, motors = robot_parameters("g1")
    _, _, qids, _, feet, _ = layout(model, spec)
    reference = np.r_[0., 0., .76, 0., 0., 0., 1., spec.default_q]
    data = mujoco.MjData(model)
    data.qpos[:] = csv_to_qpos(reference[None], qids)[0]
    mujoco.mj_forward(model, data)
    heights = []
    for body in feet:
        ids = np.flatnonzero((model.geom_bodyid == body)
                            & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        heights.extend((data.geom_xpos[ids, 2] - model.geom_size[ids, 0]).tolist())
    reference[2] += .003 - min(heights)
    return model, spec, motors, np.tile(reference, (12, 1))


def test_same_waist_coordinates_do_not_preserve_selected_world_torso(physical_g1):
    model, spec, motors, source = physical_g1
    waist = spec.joint_names.index("waist_pitch_joint")
    source[:, 7 + waist] = .16
    selected = source[0].copy()
    selected[3:7] = Rotation.from_euler("y", .2).as_quat()
    report = assess_reference(source, source, selected, model, spec, motors)
    assert report["max_arm_coordinate_error_rad"] == 0
    assert report["max_foot_position_error_m"] == 0
    assert report["max_torso_rotation_error_rad"] == pytest.approx(.2)
    assert not report["passed"]
    assert any("torso" in reason for reason in report["failures"])


def test_new_ground_penetration_is_rejected(physical_g1):
    model, spec, motors, source = physical_g1
    candidate = source.copy()
    candidate[:, 2] -= .03
    report = assess_reference(candidate, source, source[0], model, spec, motors)
    assert report["minimum_sole_height_m"] < -.02
    assert not report["passed"]


def test_full_body_solver_preserves_feet_with_selected_torso_and_world_height(physical_g1):
    pytest.importorskip("mink")
    pytest.importorskip("daqp")
    model, spec, motors, source = physical_g1
    selected = source[0].copy()
    selected[2] -= .02
    selected[3:7] = Rotation.from_euler("y", .12).as_quat()
    selected[7 + spec.joint_names.index("waist_pitch_joint")] = .16
    before = source.copy()
    candidate = solve_reference(source, selected, model, spec, motors)
    report = assess_reference(candidate, source, selected, model, spec, motors)
    np.testing.assert_array_equal(source, before)
    assert report["passed"], report["failures"]
    assert report["max_torso_rotation_error_rad"] < .001
    assert report["max_root_height_error_m"] < .001
    assert report["max_foot_position_error_m"] < .001
    assert report["max_arm_coordinate_error_rad"] == 0


@pytest.fixture
def prepared_pool(tmp_path):
    receipt = REPO / "data/style-research-20261112/full-body-staggered-preflight-v2/preparation.json"
    if not receipt.is_file():
        pytest.skip("Private prepared reference fixture is not distributed with code")
    original = Path(json.loads(receipt.read_text(encoding="utf-8"))["pool"])
    return Path(shutil.copytree(original, tmp_path / "pool"))


def test_actual_pool_reverification_is_read_only(prepared_pool):
    before = {p.relative_to(prepared_pool): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in prepared_pool.rglob("*") if p.is_file()}
    result = verify_full_body_pool(prepared_pool)
    assert result["passed"] and not result["visual_policy_acceptance"]
    assert set(result["checks"]) == {"forward", "backward", "idle"}
    after = {p.relative_to(prepared_pool): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in prepared_pool.rglob("*") if p.is_file()}
    assert before == after


def test_recomputed_geometry_rejects_bad_pose_even_after_motion_hash_is_updated(prepared_pool):
    manifest_path = prepared_pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(item for item in manifest["files"] if item.get("clip_role") == "forward")
    path = prepared_pool / row["file"]
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    values[10, 2] -= .03
    np.savetxt(path, values, delimiter=",", fmt="%.17g")
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Full-body geometry failed"):
        verify_full_body_pool(prepared_pool)


def test_bundled_donor_tampering_is_rejected(prepared_pool):
    path = prepared_pool / "sources/backward.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Chronological source bytes"):
        verify_full_body_pool(prepared_pool)


def test_selected_idle_is_staggered_planted_stationary_and_inside_support(physical_g1):
    pytest.importorskip("mink")
    path = REPO / "data/style-research-20261110/owner-aim-canonical-v5/selected-pose.csv"
    if not path.is_file():
        pytest.skip("Owner-selected private pose is not distributed with code")
    model, spec, motors, _ = physical_g1
    selected = np.loadtxt(path, delimiter=",", ndmin=2)[0]
    candidate, donor = solve_staggered_idle(selected, 101, model, spec, motors)
    report = assess_staggered_idle(candidate, donor, selected, model, spec, motors)
    assert report["passed"], report["failures"]
    stance = report["idle_stance"]
    assert stance["pose_change_max"] == 0
    assert stance["front_foot"] == "left"
    assert stance["fore_aft_separation_m"] == pytest.approx(.28, abs=1e-5)
    assert stance["lateral_separation_m"] == pytest.approx(.18, abs=1e-5)
    assert stance["com_support_margin_m"] > .01
    np.testing.assert_allclose(stance["all_sole_poles_height_range_m"], [.003, .003], atol=1e-5)


def test_idle_hold_donor_cannot_silently_change(prepared_pool):
    metadata_path = prepared_pool / "full-body-style.json"
    metadata = json.loads(metadata_path.read_text())
    binding = metadata["sources"]["idle"]
    donor_path = prepared_pool / binding["file"]
    values = np.loadtxt(donor_path, delimiter=",", ndmin=2)
    values[10, 7] += .01
    np.savetxt(donor_path, values, delimiter=",", fmt="%.17g")
    binding["sha256"] = hashlib.sha256(donor_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest_path = prepared_pool / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row = next(item for item in manifest["files"] if item.get("clip_role") == "idle")
    row["parent_csv_sha256"] = binding["sha256"]
    manifest["auxiliary_files"]["full_body_style"]["sha256"] = hashlib.sha256(
        metadata_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact selected-pose hold"):
        verify_full_body_pool(prepared_pool)

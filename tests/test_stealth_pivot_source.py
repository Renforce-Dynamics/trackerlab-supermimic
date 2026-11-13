"""Pinned walk pivots preserve their source timing, geometry and heldout split."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
import prepare_stealth_pivot_source as pivot
import prepare_stealth_turning_pool as turning
from prepare_style_campaign import _json_bytes, digest


@pytest.fixture
def source_path():
    path = REPO / "data/retargeted/lafan1_g1/walk3_subject4.csv"
    if not path.is_file():
        pytest.skip("Private pinned G1 CSV is not distributed with the source")
    assert digest(path) == turning.WALK_PIVOT_SHA256[path.name]
    return path


@pytest.fixture
def prepared(source_path, tmp_path, monkeypatch):
    monkeypatch.setattr(pivot, "_require_ignored_output", lambda path: None)
    result = pivot.prepare(source_path, 5635, 5726, tmp_path / "source")
    return source_path, Path(result["csv"]), Path(result["origin"]), result["record"]


def test_real_crop_retains_clock_and_derives_left_from_recorded_right(prepared):
    original, csv, origin, row = prepared
    assert row["output_frames"] == 151
    assert row["source_frame_range"] == [5635, 5726]
    assert row["reflected_to_left"] is True
    assert row["source_group"] == "walk3" and row["split"] == "train"
    values = np.loadtxt(csv, delimiter=",", ndmin=2)
    expected, _, _, _ = turning.walk_pivot_values(original, 5635, 5726)
    np.testing.assert_array_equal(values, expected)
    assert turning._load_origin(origin, csv, original) == row


def test_modified_crop_is_rejected_even_after_its_hash_is_updated(prepared):
    original, csv, origin, _ = prepared
    values = np.loadtxt(csv, delimiter=",", ndmin=2)
    values[20, 7] += .05
    np.savetxt(csv, values, delimiter=",", fmt="%.17g")
    document = json.loads(origin.read_text())
    document["files"][0]["sha256"] = digest(csv)
    origin.write_bytes(_json_bytes(document))
    with pytest.raises(ValueError, match="does not reproduce"):
        turning._load_origin(origin, csv, original)


@pytest.mark.parametrize("change", [
    ("source_frame_range", [5636, 5727]), ("reflected_to_left", False),
    ("constant_floor_shift_m", 0.5), ("source_total_frames", 10000),
])
def test_false_pivot_derivation_metadata_is_rejected(prepared, change):
    original, csv, origin, _ = prepared
    document = json.loads(origin.read_text())
    document["files"][0][change[0]] = change[1]
    origin.write_bytes(_json_bytes(document))
    with pytest.raises(ValueError, match="does not reproduce"):
        turning._load_origin(origin, csv, original)


def test_pivot_original_must_be_pinned_before_output_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(pivot, "_require_ignored_output", lambda path: None)
    original = tmp_path / "original" / "walk3_subject4.csv"
    original.parent.mkdir()
    original.write_bytes(b"unverified source")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="pinned"):
        pivot.prepare(original, 5635, 5726, output)
    assert not output.exists()


def test_real_corrected_pivot_pool_keeps_the_selected_style(prepared, tmp_path, monkeypatch):
    original, csv, origin, _ = prepared
    base = REPO / "data/style-research-20261112/full-body-staggered-v1/pools/g1/stealth-2976ff08ade9bf213b33"
    if not (base / "manifest.json").is_file():
        pytest.skip("Private selected-pose pool is not distributed")
    monkeypatch.setattr(turning, "_require_ignored_output", lambda path: None)
    output = tmp_path / "pivot-pool"
    result = turning.prepare(base, csv, origin, output, original)
    assert result["verification"]["passed"]
    for role in ("turn_left", "turn_right"):
        check = result["verification"]["checks"][role]
        assert min(check["candidate_lifts"]) >= 1
        assert max(check["candidate_support_slip_m_s"]) < .04
        assert check["max_arm_coordinate_error_rad"] < 1e-8
        assert check["max_torso_rotation_error_rad"] < .01
    assert (output / "aim-style-reference.json").read_bytes() == (base / "aim-style-reference.json").read_bytes()

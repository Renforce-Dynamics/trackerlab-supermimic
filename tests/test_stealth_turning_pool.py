"""Turning pool provenance, generation and independent mirror verification."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
import prepare_stealth_turning_pool as turning
from prepare_stealth_full_body import verify_full_body_pool
from prepare_style_campaign import _identity, _json_bytes, verify_pool
from trackerTask.supermimic.robots import get_robot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    csv = tmp_path / "aiming2_subject2.csv"
    bvh = tmp_path / "aiming2_subject2.bvh"
    origin = tmp_path / "origin.json"
    model = tmp_path / "model.xml"
    bvh.write_bytes(b"synthetic original BVH fixture")
    model.write_bytes(b"synthetic G1 model identity")
    values = np.zeros((101, 36))
    values[:, 2] = .665
    values[:, 3:7] = Rotation.from_euler("z", np.linspace(0, .4, 101)).as_quat()
    np.savetxt(csv, values, delimiter=",", fmt="%.17g")
    names = tuple(get_robot("g1").joint_names)
    spec = SimpleNamespace(joint_names=names, mjcf_path=model)
    monkeypatch.setattr(turning, "get_robot", lambda name: spec)
    monkeypatch.setattr(turning, "ORIGINAL_AIMING_SHA256", {bvh.name: sha(bvh)})
    row = {
        "file": csv.name, "sha256": sha(csv), "source_file": bvh.name,
        "source_sha256": sha(bvh), "source_group": "aiming2", "split": "train",
        "source_fps": 30., "output_fps": 50., "output_frames": 101,
        "source_total_frames": 9172, "source_frame_range": [200, 261],
        "target_model_sha256": sha(model), "joint_names": list(names),
    }
    document = {"schema": "lafan1-g1-origin-v1", "revision": turning.OFFICIAL_REVISION,
                "task_group": "aiming", "files": [row]}
    origin.write_bytes(_json_bytes(document))
    return csv, bvh, origin, document


def test_original_identity_and_sample_clock_are_required(source_fixture):
    csv, bvh, origin, document = source_fixture
    assert turning._load_origin(origin, csv, bvh) == document["files"][0]


@pytest.mark.parametrize("field,value", [
    ("source_group", "aiming1"), ("split", "test"), ("source_fps", 50.),
    ("output_fps", 30.), ("output_frames", 100), ("source_frame_range", [200, 260]),
    ("source_frame_range", [False, 261]), ("source_frame_range", [200, 10000]),
    ("source_file", "aiming1_subject1.bvh"), ("source_sha256", "f" * 64),
    ("target_model_sha256", "f" * 64), ("joint_names", []),
])
def test_origin_rejects_split_time_and_robot_mismatches(source_fixture, field, value):
    csv, bvh, origin, document = source_fixture
    document["files"][0][field] = value
    origin.write_bytes(_json_bytes(document))
    with pytest.raises(ValueError):
        turning._load_origin(origin, csv, bvh)


def test_duplicate_origin_is_rejected(source_fixture):
    csv, bvh, origin, document = source_fixture
    document["files"].append(copy.deepcopy(document["files"][0]))
    origin.write_bytes(_json_bytes(document))
    with pytest.raises(ValueError, match="exactly one"):
        turning._load_origin(origin, csv, bvh)


@pytest.mark.parametrize("changed", ["bvh", "csv", "yaw"])
def test_changed_original_bytes_or_absent_left_turn_are_rejected(source_fixture, changed):
    csv, bvh, origin, document = source_fixture
    if changed == "bvh":
        bvh.write_bytes(b"changed original")
    elif changed == "csv":
        csv.write_bytes(csv.read_bytes() + b"\n")
    else:
        values = np.loadtxt(csv, delimiter=",")
        values[:, 3:7] = (0., 0., 0., 1.)
        np.savetxt(csv, values, delimiter=",", fmt="%.17g")
        document["files"][0]["sha256"] = sha(csv)
        origin.write_bytes(_json_bytes(document))
    with pytest.raises(ValueError):
        turning._load_origin(origin, csv, bvh)


def test_output_must_stay_in_ignored_research_tree(tmp_path):
    output = tmp_path / "publishable-motion"
    with pytest.raises(ValueError, match="ignored"):
        turning.prepare(tmp_path / "base", tmp_path / "clip.csv", tmp_path / "origin.json",
                        output, tmp_path / "clip.bvh")
    assert not output.exists()


@pytest.fixture(scope="module")
def actual_pool(tmp_path_factory):
    base = REPO / "data/style-research-20261112/full-body-staggered-v1/pools/g1/stealth-2976ff08ade9bf213b33"
    donor = REPO / "data/style-research-20261114/turn-audit-v1/aiming2_subject2.csv"
    origin = donor.parent / "origin_manifest.json"
    bvh = REPO.parent / "_pi_plus_extract/GMR/MotionData/lafan1/aiming2_subject2.bvh"
    if any(not path.is_file() for path in (base / "manifest.json", donor, origin, bvh)):
        pytest.skip("Private audited Aiming and full-body fixtures are not distributed")
    output = tmp_path_factory.mktemp("actual-turning") / "pool"
    # The fixture owns this temporary output instead of the production data/ tree.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(turning, "_require_ignored_output", lambda path: None)
        result = turning.prepare(base, donor, origin, output, bvh)
    assert result["manifest_sha256"] == sha(output / "manifest.json")
    return output


def test_real_generation_preserves_base_and_passes_physical_geometry(actual_pool):
    result = verify_full_body_pool(actual_pool)
    assert result["passed"] and not result["visual_policy_acceptance"]
    assert set(result["checks"]) == {"forward", "backward", "idle", "turn_left", "turn_right"}
    for role in ("turn_left", "turn_right"):
        assert min(result["checks"][role]["candidate_lifts"]) >= 1
        assert result["checks"][role]["max_self_penetration_m"] <= .001
    train = json.loads((actual_pool / "train.json").read_text())["files"]
    assert len(train) == 5


@pytest.mark.parametrize("case", ["mirror", "audit", "heldout", "original"])
def test_reverification_detects_false_provenance_even_with_consistent_hashes(actual_pool, tmp_path, case):
    pool = Path(shutil.copytree(actual_pool, tmp_path / "pool"))
    manifest_path, style_path = pool / "manifest.json", pool / "full-body-style.json"
    manifest, style = (json.loads(path.read_text()) for path in (manifest_path, style_path))
    if case == "mirror":
        binding = style["sources"]["turn_right"]
        path = pool / binding["file"]
        values = np.loadtxt(path, delimiter=",")
        values[25, 7] += .1
        np.savetxt(path, values, delimiter=",", fmt="%.17g")
        binding["sha256"] = sha(path)
        row = next(row for row in manifest["files"] if row.get("clip_role") == "turn_right")
        row["parent_csv_sha256"] = sha(path)
        style_path.write_bytes(_json_bytes(style))
        manifest["auxiliary_files"]["full_body_style"]["sha256"] = sha(style_path)
        expected = "sagittal mirror"
    elif case == "audit":
        audit_path = pool / "turning-reference-audit.json"
        audit = json.loads(audit_path.read_text())
        audit["mirror"]["signs"][0] *= -1
        audit_path.write_bytes(_json_bytes(audit))
        manifest["turning_extension"]["audit_sha256"] = sha(audit_path)
        manifest["recipe"]["turning_extension"] = audit
        manifest["recipe_sha256"] = _identity(manifest["recipe"])
        expected = "mirror convention"
    elif case == "heldout":
        row = next(row for row in manifest["files"] if row["split"] == "test")
        path = pool / row["file"]
        path.write_bytes(path.read_bytes() + b"\n")
        row["sha256"] = sha(path)
        style["heldout"][row["file"]] = sha(path)
        style_path.write_bytes(_json_bytes(style))
        manifest["auxiliary_files"]["full_body_style"]["sha256"] = sha(style_path)
        expected = "base motion"
    else:
        audit_path = pool / "turning-reference-audit.json"
        audit = json.loads(audit_path.read_text())
        path = pool / audit["source"]["bvh"]["file"]
        path.write_bytes(path.read_bytes() + b"\n")
        audit["source"]["bvh"]["sha256"] = sha(path)
        audit_path.write_bytes(_json_bytes(audit))
        manifest["turning_extension"]["audit_sha256"] = sha(audit_path)
        manifest["recipe"]["turning_extension"] = audit
        manifest["recipe_sha256"] = _identity(manifest["recipe"])
        expected = "original Aiming2"
    manifest_path.write_bytes(_json_bytes(manifest))
    verify_pool(pool, "g1", "stealth")
    with pytest.raises(ValueError, match=expected):
        verify_full_body_pool(pool)

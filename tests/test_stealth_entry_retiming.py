"""A timing change preserves the actual arm path and embedded runtime controller."""

import copy
import math
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from retime_stealth_entry import retime
from train_terrain import parse_args
from trackerLab.algorithms.supermimic.stealth_upper_body import UpperBodyConstraint
from trackerTask.supermimic.robots import get_robot


def document():
    robot = get_robot("g1")
    initial = {n: q for n, q in zip(robot.joint_names, robot.default_q)
               if any(part in n for part in ("shoulder", "elbow", "wrist"))}
    final = {n: q - .8 for n, q in initial.items()}
    middle = {n: q - .5 for n, q in initial.items()}
    return {"reference_q": final, "entry_times_s": [0., 2., 3.],
            "entry_reference_q": [initial, middle, final],
            "source": {"original": "test-motion"}}


def controller(doc, rate, acceleration):
    robot = get_robot("g1")
    return UpperBodyConstraint(
        robot.joint_names, robot.default_q, robot.action_scale, doc["reference_q"],
        residual_radius_rad=0., posture_tolerance_rad=.25,
        max_joint_rate_rad_s=rate, max_joint_acceleration_rad_s2=acceleration,
        entry_times_s=doc["entry_times_s"], entry_reference_q=doc["entry_reference_q"])


def test_same_geometric_path_scaled_derivatives_and_runtime_roundtrip():
    source = document()
    untouched = copy.deepcopy(source)
    faster = retime(source, .5, "a" * 64)
    original, changed = controller(source, 1.2, 3.), controller(faster, 2.4, 12.)
    times = torch.linspace(0., 4., 301)
    torch.testing.assert_close(original.reference_at(times), changed.reference_at(times * .5))
    scripted = torch.jit.script(UpperBodyConstraint.from_dict(changed.to_dict()))
    torch.testing.assert_close(scripted.reference_at(times), changed.reference_at(times))
    assert faster["entry_reference_q"] == source["entry_reference_q"]
    assert faster["reference_q"] == source["reference_q"]
    assert source == untouched
    assert faster["source"]["parent_reference_sha256"] == "a" * 64
    original_info = retime(source, 1., "a" * 64)["source"]
    for key, multiplier in (("peak_scheduled_rate_rad_s", 2.),
                            ("peak_scheduled_acceleration_rad_s2", 4.)):
        assert math.isclose(faster["source"][key], original_info[key] * multiplier)


@pytest.mark.parametrize("scale", [0., -1., True, float("inf"), float("nan"), 1e-300])
def test_invalid_time_scales_fail(scale):
    with pytest.raises(ValueError):
        retime(document(), scale, "a" * 64)


def test_final_pose_rounding_matches_existing_controller_tolerance():
    source = document()
    source["entry_reference_q"][-1] = source["reference_q"].copy()
    name = next(iter(source["reference_q"]))
    source["entry_reference_q"][-1][name] += 2e-16
    original = copy.deepcopy(source)
    faster = retime(source, .5, "a" * 64)
    assert faster["entry_reference_q"] == original["entry_reference_q"]
    controller(faster, 2.4, 12.)
    source["entry_reference_q"][-1][name] += .01
    with pytest.raises(ValueError, match="Final waypoint"):
        retime(source, .5, "a" * 64)


def test_cli_retains_historical_rate_and_new_preset():
    base = ["--task", "locomotion/g1/stealth", "--run-dir", "unused"]
    assert parse_args(base).upper_body_rate == 1.2
    args = parse_args([*base, "--config", str(
        REPO / "scripts/rls/supermimic/configs/command_stealth_velocity_visual.json")])
    assert args.upper_body_rate == 2.4
    assert args.upper_body_acceleration == 12.
    assert args.entry_duration_s == 2.42
    assert args.style_feature_mode == "postural_end_effectors"

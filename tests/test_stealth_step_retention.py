"""Stepping and speed are independent requirements for a continuation."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"))
from check_stealth_step_retention import compare_steps
from train_terrain import parse_args

BASELINE = [[5, 4], [5, 1], [3, 3], [5, 2], [5, 5], [5, 4]]


def test_same_gait_passes_and_observed_shuffling_regression_fails():
    assert compare_steps(BASELINE, BASELINE)["passed"]
    result = compare_steps(BASELINE, [[0, 0], [0, 0], [1, 1], [1, 0], [1, 0], [2, 0]])
    assert not result["passed"]
    assert len(result["reasons"]) == 2


def test_one_sided_hops_cannot_substitute_for_bilateral_stepping():
    assert not compare_steps(BASELINE, [[0, 20]] * 6)["passed"]
    assert not compare_steps(BASELINE, [[20, 0]] * 6)["passed"]
    assert not compare_steps([[0, 10]] * 6, [[10, 10]] * 6)["passed"]


def test_retention_requires_majority_of_fixed_first_attempts():
    retained = [[3, 2]] * 5 + [[0, 0]]
    assert compare_steps(BASELINE, retained)["passed"]
    assert not compare_steps(BASELINE, [[5, 5]] * 4 + [[0, 0]] * 2)["passed"]


@pytest.mark.parametrize("bad", [[], [[1, 2]] * 5, [[-1, 2]] * 6,
                                  [[1., 2.]] * 6, [[True, False]] * 6,
                                  [[1, 2, 3]] * 6])
def test_invalid_evidence_cannot_pass(bad):
    with pytest.raises(ValueError):
        compare_steps(BASELINE, bad)


def test_recovery_recipe_keeps_original_controller_and_style_contract():
    root = Path(__file__).resolve().parents[1] / "scripts/rls/supermimic/configs"
    base = ["--task", "locomotion/g1/stealth", "--run-dir", "unused"]
    before = parse_args([*base, "--config", str(root / "command_stealth_entry.json")])
    after = parse_args([*base, "--config", str(root / "command_stealth_gait_restore.json")])
    for field in ("entry_duration_s", "waist_pd_scale", "upper_body_residual",
                  "upper_body_rate", "upper_body_acceleration", "self_collisions",
                  "physical_domain_randomization", "foot_slip_mode", "foot_slide_weight",
                  "feet_close_weight", "amp_history", "amp_weight", "style_feature_mode"):
        assert getattr(before, field) == getattr(after, field), field
    assert after.waist_entry_reference is None
    assert after.stealth_upper_body_constraint and not after.upper_body_rate_limit
    assert not after.warm_start_reset_arms and not after.warm_start_reset_world
    assert after.freeze_actor_normalization and after.warm_start_require_control_match
    assert after.stealth_control_mode == "wasd" and after.wasd_reward_version == 2
    assert after.command_training_modes is None

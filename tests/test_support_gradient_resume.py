"""Minimal support-gradient trial preserves every other model objective input."""

from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from train_terrain import parse_args, validate_support_gradient_resume

BASE = ["--task", "locomotion/g1/stealth", "--run-dir", "unused"]
SOURCE = REPO / "scripts/rls/supermimic/configs/command_stealth_free_arms.json"
TARGET = REPO / "scripts/rls/supermimic/configs/command_stealth_support_gradient.json"


def args(config=TARGET, *options):
    return parse_args([*BASE, "--config", str(config), *options])


def checkpoint(**changes):
    return {"task": "locomotion/g1/stealth", "arguments": {
        "foot_slip_mode": "g1_sole", "differentiate_support": False, **changes}}


def test_support_trial_changes_only_optimizer_surrogate_and_budget(tmp_path):
    before, after = args(SOURCE), args(TARGET)
    changes = {name for name in vars(before) if getattr(before, name) != getattr(after, name)}
    assert changes == {"config", "iterations", "differentiate_support",
                       "warm_start_reset_arms", "warm_start_reset_world"}
    assert after.iterations == 25000
    assert after.entry_duration_s == before.entry_duration_s == 1.21
    assert after.style_feature_mode == before.style_feature_mode
    assert not before.differentiate_support and after.differentiate_support
    approved = args(TARGET, "--resume", "source.pt", "--allow-support-gradient-change")
    assert validate_support_gradient_resume(checkpoint(), approved, tmp_path)
    assert not validate_support_gradient_resume(
        {"task": "locomotion/g1/stealth", "arguments": {
            "foot_slip_mode": "g1_sole", "differentiate_support": True}},
        args(TARGET, "--resume", "same.pt"), tmp_path)


def test_existing_run_and_unapproved_changes_are_rejected(tmp_path):
    approved = args(TARGET, "--resume", "source.pt", "--allow-support-gradient-change")
    with pytest.raises(ValueError, match="fresh G1"):
        validate_support_gradient_resume(checkpoint(), args(TARGET, "--resume", "source.pt"), tmp_path)
    (tmp_path / "last.pt").touch()
    with pytest.raises(ValueError, match="fresh G1"):
        validate_support_gradient_resume(checkpoint(), approved, tmp_path)
    (tmp_path / "last.pt").unlink()
    with pytest.raises(ValueError, match="fresh G1"):
        validate_support_gradient_resume(checkpoint(foot_slip_mode="legacy"), approved, tmp_path)
    with pytest.raises(ValueError, match="fresh G1"):
        validate_support_gradient_resume({"task": "locomotion/go2/stealth",
                                         "arguments": checkpoint()["arguments"]}, approved, tmp_path)
    with pytest.raises(ValueError, match="changed support"):
        validate_support_gradient_resume(checkpoint(differentiate_support=True), approved, tmp_path)

"""Command timing is explicitly enabled and separate from synchronized profiling."""

import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"
sys.path.insert(0, str(SCRIPTS))
import train_terrain


BASE = ["--task", "locomotion/g1/stealth", "--run-dir", "unused-profile-options"]


def test_phase_observer_is_disabled_by_default():
    args = train_terrain.parse_args(BASE)
    assert args.phase_profile_updates == 0
    assert not args.profile


@pytest.mark.parametrize("count", [1, 25, 1000])
def test_bounded_optional_phase_window(count):
    args = train_terrain.parse_args([*BASE, "--phase-profile-updates", str(count)])
    assert args.phase_profile_updates == count


@pytest.mark.parametrize("count", [-1, 1001])
def test_invalid_phase_budget_is_rejected_before_runtime(count):
    with pytest.raises(SystemExit):
        train_terrain.parse_args([*BASE, "--phase-profile-updates", str(count)])


def test_synchronized_profiler_cannot_overlap_event_observation():
    with pytest.raises(SystemExit):
        train_terrain.parse_args([*BASE, "--phase-profile-updates", "25", "--profile"])


def test_preset_profile_budget_can_be_explicitly_disabled(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"phase_profile_updates": 25}))
    args = train_terrain.parse_args([*BASE, "--config", str(path)])
    assert args.phase_profile_updates == 25
    args = train_terrain.parse_args([
        *BASE, "--config", str(path), "--phase-profile-updates", "0"])
    assert args.phase_profile_updates == 0


def test_boolean_preset_cannot_become_an_integer_profile_budget(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"phase_profile_updates": True}))
    with pytest.raises(SystemExit):
        train_terrain.parse_args([*BASE, "--config", str(path)])

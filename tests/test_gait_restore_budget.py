"""An early gait regression must not erase an explicitly authorized budget."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"))
from gait_restore_budget import checkpoint_start, plan


def test_resume_1000_adds_5000_actor_updates_without_new_warmup():
    result = plan(1000)
    assert result["target_iteration"] == 6000
    assert result["milestones"] == [3000, 6000]
    assert result["additional_actor_updates"] == 5000
    assert result["remaining_world_only_iterations"] == 0
    assert result["visual_acceptance"] is False


def test_fresh_run_does_not_count_world_only_steps_as_actor_updates():
    result = plan()
    assert result["target_iteration"] == 5500
    assert result["milestones"] == [2500, 5500]
    assert result["remaining_world_only_iterations"] == 500
    assert result["additional_actor_updates"] == 5000


@pytest.mark.parametrize("start", [-1, True, None, 1, 750, 1000., "1000"])
def test_budget_rejects_invalid_or_unsaved_starts(start):
    with pytest.raises(ValueError):
        plan(start)


def checkpoint():
    return {"task": "locomotion/g1/stealth", "training_stage": "command-style",
            "arguments": {"wm_only_iters": 500}, "iteration": 1000,
            **{name: {} for name in ("actor", "world", "critic", "opt_pi", "opt_wm", "opt_v", "amp_style")}}


def test_resume_needs_full_models_and_optimizers():
    assert checkpoint_start(checkpoint()) == 1000
    for name in ("actor", "world", "critic", "opt_pi", "opt_wm", "opt_v", "amp_style"):
        ck = checkpoint()
        del ck[name]
        with pytest.raises(ValueError, match="missing"):
            checkpoint_start(ck)


@pytest.mark.parametrize("change", [
    {"task": "locomotion/g1/velocity"}, {"training_stage": "tracking"},
    {"arguments": {"wm_only_iters": 1500}}, {"iteration": 0}, {"iteration": True},
])
def test_resume_rejects_different_task_or_unfinished_warmup(change):
    with pytest.raises(ValueError):
        checkpoint_start({**checkpoint(), **change})

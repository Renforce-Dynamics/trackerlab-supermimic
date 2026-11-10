"""Cumulative actor accounting survives arbitrary checkpointed handoff iterations."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"))
from combined_training_budget import checkpoint_start, plan


def checkpoint(iteration=1373):
    return {
        "task": "locomotion/g1/stealth", "training_stage": "command-style",
        "iteration": iteration,
        "arguments": {
            "wm_only_iters": 500, "stealth_control_mode": "wasd",
            "style_feature_mode": "postural_end_effectors_vx",
            "entry_duration_s": 1.21, "entry_style_weight": .008,
            "waist_pd_scale": 1., "freeze_actor_normalization": True,
        },
        "stealth_upper_body_constraint": None, "stealth_waist_constraint": None,
        "upper_body_rate_limit": {}, "entry_style_guidance": {},
        **{name: {} for name in ("actor", "world", "critic", "opt_pi", "opt_wm", "opt_v", "amp_style")},
    }


def test_fresh_budget_excludes_world_only_warmup():
    result = plan()
    assert result["target_iteration"] == 50500
    assert result["actor_update_budget"] == result["remaining_actor_updates"] == 50000
    assert result["remaining_world_only_iterations"] == 500
    assert result["milestones"] == [2500, 5500, 10500, 20500, 30500, 40500, 50500]
    assert result["visual_acceptance"] is False


@pytest.mark.parametrize("start", [1, 250, 500, 1373, 2500, 5500, 10500, 50499])
def test_resume_keeps_total_budget_and_omits_finished_milestones(start):
    assert checkpoint_start(checkpoint(start)) == start
    result = plan(start)
    assert result["remaining_actor_updates"] + result["actor_updates_already_completed"] == 50000
    assert (result["remaining_actor_updates"] + result["remaining_world_only_iterations"]
            == 50500 - start)
    assert all(value > start for value in result["milestones"])
    assert result["milestones"][-1] == 50500
    if start >= 500:
        assert result["remaining_world_only_iterations"] == 0


def test_1373_handoff_counts_873_actor_updates_without_resetting_them():
    result = plan(1373)
    assert result["actor_updates_already_completed"] == 873
    assert result["remaining_actor_updates"] == 49127


@pytest.mark.parametrize("bad", [None, True, False, -1, 50500, 60000, 500., "500"])
def test_invalid_start_rejected(bad):
    with pytest.raises(ValueError):
        plan(bad)


def test_only_complete_matching_checkpoint_can_resume():
    for name in ("actor", "world", "critic", "opt_pi", "opt_wm", "opt_v", "amp_style"):
        ck = checkpoint()
        del ck[name]
        with pytest.raises(ValueError, match="missing"):
            checkpoint_start(ck)
    for name, value in (("entry_duration_s", 4.84), ("entry_style_weight", 0.),
                        ("waist_pd_scale", 4.), ("style_feature_mode", "postural_end_effectors"),
                        ("freeze_actor_normalization", False)):
        ck = checkpoint()
        ck["arguments"][name] = value
        with pytest.raises(ValueError, match="matching"):
            checkpoint_start(ck)
    for name in ("stealth_upper_body_constraint", "stealth_waist_constraint"):
        ck = checkpoint()
        ck[name] = {"active": True}
        with pytest.raises(ValueError, match="matching"):
            checkpoint_start(ck)
    with pytest.raises(ValueError, match="saved"):
        checkpoint_start(checkpoint(0))


def test_velocity_led_resume_rejects_original_combined_recipe():
    ck = checkpoint(5500)
    with pytest.raises(ValueError, match="matching"):
        checkpoint_start(ck, "velocity-led")
    ck["arguments"].update(entry_duration_s=.60, wasd_reward_version=3,
                           stealth_gait_weight=.02, amp_weight=.05)
    assert checkpoint_start(ck, "velocity-led") == 5500
    with pytest.raises(ValueError, match="matching"):
        checkpoint_start(ck)

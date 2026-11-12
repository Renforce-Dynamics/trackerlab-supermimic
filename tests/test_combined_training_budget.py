"""Cumulative actor accounting survives arbitrary checkpointed handoff iterations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"))
from combined_training_budget import actor_budget, checkpoint_start, main, plan


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


def test_shared_aim_resume_requires_matching_weight_and_saved_target():
    ck = checkpoint(5500)
    ck["arguments"].update(entry_duration_s=.60, wasd_reward_version=3,
                           stealth_gait_weight=.02, amp_weight=.1)
    with pytest.raises(ValueError, match="guidance"):
        checkpoint_start(ck, "shared-aim")
    ck["aim_style_guidance"] = {"schema": "g1-stealth-aim-style-v1"}
    assert checkpoint_start(ck, "shared-aim") == 5500
    with pytest.raises(ValueError):
        checkpoint_start(ck, "velocity-led")


def test_beyondamp_uses_twenty_thousand_updates_and_versioned_geometry():
    ck = checkpoint(5500)
    ck["arguments"].update(entry_duration_s=.60, wasd_reward_version=3,
                           stealth_gait_weight=.02, amp_weight=.1,
                           stealth_gait_version=2, stealth_swing_height=.04)
    ck["aim_style_guidance"] = {"palm_separation": {}}
    assert checkpoint_start(ck, "beyondamp") == 5500
    budget = plan(5500, actor_updates=20000)
    assert budget["actor_update_budget"] == 20000
    assert budget["remaining_actor_updates"] == 15000
    assert budget["milestones"] == [10500, 20500]
    del ck["aim_style_guidance"]["palm_separation"]
    with pytest.raises(ValueError, match="palm separation"):
        checkpoint_start(ck, "beyondamp")
    with pytest.raises(ValueError):
        plan(20500, actor_updates=20000)


@pytest.mark.parametrize("start", [20500, 21750, 25499])
def test_beyondamp_extension_preserves_completed_actor_updates(start):
    ck = checkpoint(start)
    ck["arguments"].update(entry_duration_s=.6, wasd_reward_version=3,
                           stealth_gait_weight=.02, amp_weight=.1,
                           stealth_gait_version=2, stealth_swing_height=.04)
    ck["aim_style_guidance"] = {"palm_separation": {}}
    with pytest.raises(ValueError):
        checkpoint_start(ck, "beyondamp")
    assert checkpoint_start(ck, "beyondamp", extra_actor_updates=5000) == start
    budget = plan(start, actor_updates=actor_budget("beyondamp", 5000))
    assert budget["target_iteration"] == 25500
    assert budget["remaining_world_only_iterations"] == 0
    assert budget["actor_updates_already_completed"] == start-500
    assert budget["remaining_actor_updates"] == 25500-start
    assert budget["milestones"] == [25500]


def test_extension_requires_resume_and_the_matching_recipe(tmp_path):
    with pytest.raises(SystemExit):
        main(["--recipe", "beyondamp", "--extra-actor-updates", "5000",
              "--output", str(tmp_path/"plan.json")])
    assert not (tmp_path/"plan.json").exists()
    with pytest.raises(ValueError, match="BeyondAMP"):
        actor_budget("shared-aim", 5000)
    for extra in (True, -1, 1000, 10000):
        with pytest.raises(ValueError, match="Additional"):
            actor_budget("beyondamp", extra)


@pytest.fixture
def fullbody_state(tmp_path):
    from bootstrap import configure_paths
    configure_paths()
    from trackerTask.supermimic.robots import get_robot
    from trackerLab.algorithms.supermimic.stealth_action_filter import UpperBodyRateLimiter
    folder = Path(__file__).resolve().parents[1] / "scripts/rls/supermimic/configs"
    preset = folder / "command_stealth_full_body.json"
    args = json.loads(preset.read_text())
    args.update(
        config=str(preset), task="locomotion/g1/stealth", command_training_modes=None,
        command_reward_mode=None, stealth_command=None, upper_body_reference=None,
        waist_entry_reference=None, allow_command_sampling_change=False,
        allow_command_reward_change=False, allow_style_feature_change=False,
        allow_support_gradient_change=False, warm_start_normalizer_prior_count=0.)
    pool = tmp_path / "full body pool"
    pool.mkdir()
    aim = {"source": {"selected_csv_sha256": "a" * 64}, "palm_separation": {}}
    payload = json.dumps(aim).encode()
    (pool / "aim-style-reference.json").write_bytes(payload)
    (pool / "full-body-style.json").write_text("{}\n")
    manifest = {
        "recipe": {"guidance": aim, "source": aim["source"]},
        "auxiliary_files": {"aim_style": {"file": "aim-style-reference.json",
            "sha256": hashlib.sha256(payload).hexdigest()}},
        "files": [{"split": "train", "clip_role": "forward", "file": "forward.csv",
                   "sha256": "b" * 64}],
    }
    payload = json.dumps(manifest).encode()
    (pool / "manifest.json").write_bytes(payload)
    spec = get_robot("g1")
    controller = UpperBodyRateLimiter(
        spec.joint_names, spec.action_scale, max_joint_rate_rad_s=2.4,
        max_joint_acceleration_rad_s2=12., control_dt=.02, action_limit=20.).to_dict()
    ck = checkpoint()
    ck.update(
        robot="g1", arguments=args, upper_body_rate_limit=controller,
        aim_style_guidance=deepcopy(aim),
        entry_style_guidance={
            "weight": .008, "duration": .6, "expert_hash": "c" * 64,
            "source": {"pool_manifest_sha256": hashlib.sha256(payload).hexdigest(),
                       "forward_csv": "forward.csv", "forward_csv_sha256": "b" * 64}},
        amp_style={"expert_hash": "c" * 64, "feature_mode": "postural_end_effectors_vx",
                   "history_steps": 10})
    return ck, pool


def test_fullbody_fresh_budget_reviews_only_after_five_thousand_actor_updates(tmp_path, capsys):
    assert actor_budget("fullbody") == 5000
    result = plan(actor_updates=actor_budget("fullbody"))
    assert result["remaining_world_only_iterations"] == 500
    assert result["actor_update_budget"] == result["remaining_actor_updates"] == 5000
    assert result["actor_updates_already_completed"] == 0
    assert result["target_iteration"] == 5500
    assert result["milestones"] == [5500]
    assert result["visual_acceptance"] is False
    output = tmp_path / "plan.json"
    main(["--recipe", "fullbody", "--output", str(output)])
    assert json.loads(output.read_text()) == result
    assert capsys.readouterr().out == "0 5500 5500\n"
    with pytest.raises(FileExistsError):
        main(["--recipe", "fullbody", "--output", str(output)])


@pytest.mark.parametrize("start", [1, 250, 500, 1373, 2500, 5499])
def test_fullbody_resume_stays_inside_original_budget(fullbody_state, start):
    ck, pool = fullbody_state
    ck["iteration"] = start
    before = deepcopy(ck)
    assert checkpoint_start(ck, "fullbody", pool=pool) == start
    result = plan(start, actor_updates=actor_budget("fullbody"))
    assert result["milestones"] == [5500]
    assert result["remaining_actor_updates"] + result["actor_updates_already_completed"] == 5000
    assert result["remaining_actor_updates"] + result["remaining_world_only_iterations"] == 5500-start
    assert ck == before


@pytest.mark.parametrize("start", [0, 5500, 20500, 25500, True, 1000.])
def test_fullbody_rejects_completed_or_invalid_resume(fullbody_state, start):
    ck, pool = fullbody_state
    ck["iteration"] = start
    with pytest.raises(ValueError):
        checkpoint_start(ck, "fullbody", pool=pool)


@pytest.mark.parametrize("recipe", ["combined", "velocity-led", "shared-aim"])
def test_extra_five_thousand_requires_a_bounded_recipe(recipe):
    with pytest.raises(ValueError, match="BeyondAMP"):
        actor_budget(recipe, 5000)


@pytest.mark.parametrize("name,value", [
    ("target_clearance_m", None), ("target_clearance_m", .7),
    ("config", "command_stealth_beyondamp.json"), ("iterations", 20500),
    ("command_training_modes", ["forward"]), ("command_reward_mode", "legacy"),
    ("upper_body_reference", "fixed-arms.json"), ("waist_entry_reference", "waist.json"),
    ("self_collisions", False), ("physical_domain_randomization", False),
    ("upper_body_rate", 1.2), ("warm_start_reset_arms", True),
    ("warm_start_reset_world", True), ("allow_style_feature_change", True),
    ("allow_command_sampling_change", True), ("warm_start_normalizer_prior_count", 1.),
])
def test_fullbody_resume_rejects_changed_preset_and_control_options(fullbody_state, name, value):
    ck, pool = fullbody_state
    ck["arguments"][name] = value
    with pytest.raises(ValueError, match="preset"):
        checkpoint_start(ck, "fullbody", pool=pool)


@pytest.mark.parametrize("name,value", [
    ("max_joint_rate_rad_s", 1.2), ("response_version", 1),
    ("joint_names", ["unrelated_joint"]), ("action_scale", [.1] * 29),
])
def test_fullbody_resume_checks_actual_saved_controller(fullbody_state, name, value):
    ck, pool = fullbody_state
    ck["upper_body_rate_limit"][name] = value
    with pytest.raises(ValueError, match="29-action controller"):
        checkpoint_start(ck, "fullbody", pool=pool)


@pytest.mark.parametrize("field,name,value", [
    ("entry_style_guidance", "expert_hash", "d" * 64),
    ("entry_style_guidance", "duration", 1.21),
    ("entry_style_guidance", "source", {}),
    ("amp_style", "expert_hash", ""),
    ("amp_style", "feature_mode", "postural"),
    ("amp_style", "history_steps", 2),
    ("aim_style_guidance", "source", {"selected_csv_sha256": "d" * 64}),
])
def test_fullbody_resume_rejects_mismatched_saved_style(fullbody_state, field, name, value):
    ck, pool = fullbody_state
    ck[field][name] = value
    with pytest.raises(ValueError, match="source or saved style"):
        checkpoint_start(ck, "fullbody", pool=pool)


def test_fullbody_resume_requires_current_pool_and_unchanged_sources(fullbody_state):
    ck, pool = fullbody_state
    with pytest.raises(ValueError, match="--pool"):
        checkpoint_start(ck, "fullbody")
    ck["entry_style_guidance"]["source"]["forward_csv_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="forward expert"):
        checkpoint_start(ck, "fullbody", pool=pool)
    ck["entry_style_guidance"]["source"]["forward_csv_sha256"] = "b" * 64
    (pool / "manifest.json").write_bytes((pool / "manifest.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source"):
        checkpoint_start(ck, "fullbody", pool=pool)


def test_fullbody_cli_reads_full_saved_state_on_cpu_and_does_not_reset_budget(fullbody_state, tmp_path, capsys):
    import torch
    ck, pool = fullbody_state
    path = tmp_path / "full.pt"
    output = tmp_path / "resume-plan.json"
    torch.save(ck, path)
    main(["--recipe", "fullbody", "--pool", str(pool), "--resume", str(path),
          "--output", str(output)])
    assert capsys.readouterr().out == "1373 5500 5500\n"
    result = json.loads(output.read_text())
    assert result["remaining_world_only_iterations"] == 0
    assert result["remaining_actor_updates"] == 4127
    assert result["milestones"] == [5500]
    with pytest.raises(ValueError, match="original 5500"):
        main(["--recipe", "fullbody", "--pool", str(pool), "--resume", str(path),
              "--extra-actor-updates", "5000", "--output", str(tmp_path / "extra.json")])
    assert not (tmp_path / "extra.json").exists()


@pytest.mark.parametrize("start", [5500, 6000, 10499])
def test_fullbody_extension_preserves_full_state_and_remaining_budget(fullbody_state, start):
    ck, pool = fullbody_state
    ck["iteration"] = start
    if start > 5500:
        ck["arguments"]["iterations"] = 10500
    before = deepcopy(ck)
    assert checkpoint_start(ck, "fullbody", pool=pool, extra_actor_updates=5000) == start
    result = plan(start, actor_updates=actor_budget("fullbody", 5000))
    assert result["target_iteration"] == 10500
    assert result["remaining_world_only_iterations"] == 0
    assert result["remaining_actor_updates"] == 10500 - start
    assert result["actor_updates_already_completed"] == start - 500
    assert result["milestones"] == [10500]
    assert ck == before


def test_fullbody_extension_cli_requires_completed_source_and_preserves_sources(
        fullbody_state, tmp_path, capsys):
    import torch
    ck, pool = fullbody_state
    ck["iteration"] = 5500
    source = tmp_path / "5500.pt"
    output = tmp_path / "extension.json"
    torch.save(ck, source)
    source_bytes = source.read_bytes()
    main(["--recipe", "fullbody", "--pool", str(pool), "--resume", str(source),
          "--extra-actor-updates", "5000", "--output", str(output)])
    assert capsys.readouterr().out == "5500 10500 10500\n"
    result = json.loads(output.read_text())
    assert result["additional_actor_update_budget"] == result["remaining_actor_updates"] == 5000
    assert result["continuation"].startswith("full checkpoint resume")
    assert source.read_bytes() == source_bytes


def test_fullbody_extension_cannot_expand_a_fresh_run_or_forged_target(fullbody_state, tmp_path):
    with pytest.raises(SystemExit):
        main(["--recipe", "fullbody", "--extra-actor-updates", "5000",
              "--output", str(tmp_path / "unstarted.json")])
    ck, pool = fullbody_state
    ck["iteration"] = 6000
    with pytest.raises(ValueError, match="preset"):
        checkpoint_start(ck, "fullbody", pool=pool, extra_actor_updates=5000)
    ck["arguments"]["iterations"] = 10500
    ck["iteration"] = 10500
    with pytest.raises(ValueError):
        checkpoint_start(ck, "fullbody", pool=pool, extra_actor_updates=5000)

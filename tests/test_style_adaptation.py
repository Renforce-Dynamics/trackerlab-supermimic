"""Explicit style-only continuation keeps the actor and physical recipe intact."""

import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from style_adaptation import bootstrap_adapted_style, restore_resume_style
from train_terrain import parse_args
from test_amp_end_effectors import WristKinematics, motion_fixture
from trackerLab.algorithms.supermimic.amp_style import AMPStyle


@pytest.fixture
def learners():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    kin, motion = WristKinematics(), motion_fixture((12, 13))
    base = AMPStyle(kin, motion, .8, hidden=(16, 8), history_steps=10,
                    feature_mode="postural_end_effectors", replay_capacity=32)
    target = AMPStyle(kin, motion, .8, hidden=(16, 8), history_steps=10,
                      feature_mode="postural_end_effectors_vx", replay_capacity=32)
    yield base, target, motion
    torch.set_num_threads(threads)


def test_explicit_adaptation_verifies_old_expert_prefix_without_restoring_wrong_shape(learners, tmp_path):
    base, target, motion = learners
    checkpoint = {"task": "locomotion/g1/stealth", "amp_style": base.checkpoint()}
    original_weights = {key: value.clone() for key, value in target.state_dict().items()}
    original_metadata = copy.deepcopy(checkpoint["amp_style"]["feature_layout"])
    with pytest.raises(ValueError, match="allow-style"):
        restore_resume_style(target, checkpoint, motion, run_dir=tmp_path)
    receipt = restore_resume_style(target, checkpoint, motion, allow_feature_change=True, run_dir=tmp_path)
    assert receipt["expert_prefix_identical"]
    assert receipt["actor_world_critic_and_optimizers_preserved"]
    assert receipt["target_expert_hash"] != receipt["source_expert_hash"]
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, original_weights[key], rtol=0, atol=0)
    assert checkpoint["amp_style"]["feature_layout"] == original_metadata
    assert target.checkpoint()["updates"] == 0
    (tmp_path / "last.pt").touch()
    with pytest.raises(ValueError, match="fresh run"):
        restore_resume_style(target, checkpoint, motion, allow_feature_change=True, run_dir=tmp_path)


def test_changed_experts_and_unrelated_mode_transition_are_rejected(learners, tmp_path):
    base, target, motion = learners
    checkpoint = {"task": "locomotion/g1/stealth", "amp_style": base.checkpoint()}
    checkpoint["amp_style"]["expert_hash"] = "0" * 64
    with pytest.raises(ValueError, match="identical expert"):
        restore_resume_style(target, checkpoint, motion, allow_feature_change=True, run_dir=tmp_path)
    checkpoint["task"] = "locomotion/a3/stealth"
    with pytest.raises(ValueError, match="G1 stealth"):
        restore_resume_style(target, checkpoint, motion, allow_feature_change=True, run_dir=tmp_path)


def test_same_feature_resume_restores_existing_scorer_without_bootstrap(learners, tmp_path):
    base, target, motion = learners
    checkpoint = {"task": "locomotion/g1/stealth", "amp_style": target.checkpoint()}
    before = copy.deepcopy(target.discriminator.state_dict())
    with torch.no_grad():
        next(target.discriminator.parameters()).fill_(8.)
    assert restore_resume_style(target, checkpoint, motion, run_dir=tmp_path) is None
    for key, value in target.discriminator.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_bootstrap_counts_only_real_amp_optimizer_updates_and_honors_stop(learners):
    _, target, _ = learners
    frames = target._expert_frames[:10][None].expand(4, -1, -1).clone()
    calls = []
    def collect(steps):
        calls.append(steps)
        target.observe_window(frames, torch.ones(4, dtype=torch.bool))
    receipt = bootstrap_adapted_style(target, 3, 4, collect, lambda: False)
    assert calls == [1, 1, 1]
    assert receipt["optimizer_updates"] == 3
    assert receipt["actor_optimizer_updates"] == 0
    with pytest.raises(InterruptedError):
        bootstrap_adapted_style(target, 3, 4, collect, lambda: True)
    assert len(calls) == 3


def test_velocity_preset_preserves_entry_motion_physics_and_optimizer_settings():
    base = ["--task", "locomotion/g1/stealth", "--run-dir", "unused"]
    prior = parse_args([*base, "--config", str(
        REPO / "scripts/rls/supermimic/configs/command_stealth_free_arms.json")])
    target = parse_args([*base, "--config", str(
        REPO / "scripts/rls/supermimic/configs/command_stealth_velocity_style.json")])
    changed = {key for key in vars(prior) if getattr(prior, key) != getattr(target, key)}
    assert changed == {"config", "iterations", "style_feature_mode",
                       "warm_start_reset_arms", "warm_start_reset_world"}
    assert target.entry_duration_s == 1.21
    assert target.freeze_actor_normalization
    assert not target.warm_start_reset_arms


def test_description_succeeds_without_optional_simulator_imports():
    command = [
        sys.executable, "-c",
        "import runpy,sys; sys.modules['mujoco_warp']=None; sys.modules['warp']=None; "
        f"sys.path.insert(0,{str(REPO / 'scripts/rls/supermimic')!r}); "
        "sys.argv=['train_terrain.py','--task','locomotion/g1/stairs',"
        "'--run-dir','unused','--iterations','23','--describe']; "
        f"runpy.run_path({str(REPO / 'scripts/rls/supermimic/train_terrain.py')!r},run_name='__main__')"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '"iterations": 23' in result.stdout

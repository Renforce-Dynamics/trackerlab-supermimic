"""Free-arm training preserves locomotion and learns through its target filter."""

from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from train_terrain import (
    adapt_warm_start_actor, build_upper_body_rate_limiter, configure_cuda_memory, parse_args,
    validate_upper_body_options,
)
from trackerLab.algorithms.supermimic import networks
from trackerLab.algorithms.supermimic.stealth_action_filter import UpperBodyRateLimiter
from trackerLab.algorithms.supermimic.stealth_waist_entry import (
    WaistEntryConstraint, compose_entry_waist_constraint,
)
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import get_task

BASE = ["--task", "locomotion/g1/stealth", "--run-dir", "unused"]
PRESET = REPO / "scripts/rls/supermimic/configs/command_stealth_free_arms.json"


def free_args():
    return parse_args([*BASE, "--config", str(PRESET)])


def test_free_arm_recipe_has_no_pose_reference_and_preserves_reward_coefficients():
    args = free_args()
    prior = parse_args([*BASE, "--config", str(
        REPO / "scripts/rls/supermimic/configs/command_stealth_velocity_visual.json")])
    validate_upper_body_options(args, get_task(args.task))
    assert args.entry_duration_s == prior.entry_duration_s / 2
    assert args.iterations == 20000
    assert args.wm_only_iters == 500
    assert not args.stealth_upper_body_constraint
    assert args.upper_body_reference is None
    assert args.upper_body_rate_limit and args.warm_start_reset_arms
    assert args.warm_start_reset_world and args.freeze_actor_normalization
    assert args.warm_start_normalizer_prior_count == 0
    assert args.command_training_modes is None
    for name in ("wasd_reward_version", "style_feature_mode", "amp_weight",
                 "foot_slide_weight", "feet_close_weight", "clearance_ratio",
                 "actor_lr", "waist_pd_scale", "physical_domain_randomization",
                 "self_collisions", "reference_reset_probability"):
        assert getattr(args, name) == getattr(prior, name)


def test_learned_limits_preserve_legacy_defaults_and_apply_explicit_values():
    spec = get_robot("g1")
    assert build_upper_body_rate_limiter(parse_args(BASE), spec) is None
    legacy = build_upper_body_rate_limiter(parse_args([*BASE, "--upper-body-rate-limit"]), spec)
    assert legacy.max_joint_rate_rad_s == 1.2
    assert legacy.max_joint_acceleration_rad_s2 == 3.
    current = build_upper_body_rate_limiter(free_args(), spec)
    assert current.max_joint_rate_rad_s == 2.4
    assert current.max_joint_acceleration_rad_s2 == 12.
    assert current.response_version == 2
    assert UpperBodyRateLimiter.from_dict(current.to_dict()).to_dict() == current.to_dict()


@pytest.mark.parametrize("options", [
    ["--upper-body-reference", "fixed.json"],
    ["--upper-body-residual", "0"],
    ["--stealth-upper-body-constraint", "--upper-body-rate-limit"],
    ["--upper-body-acceleration", "12"],
    ["--upper-body-rate-limit", "--upper-body-rate", "0"],
    ["--upper-body-rate-limit", "--upper-body-rate", "nan"],
    ["--upper-body-rate-limit", "--upper-body-acceleration", "-1"],
    ["--upper-body-rate-limit", "--upper-body-acceleration", "inf"],
])
def test_invalid_or_fixed_pose_options_are_rejected(options):
    args = parse_args([*BASE, *options])
    with pytest.raises(ValueError):
        validate_upper_body_options(args, get_task(args.task))


def test_released_arms_keep_gradients_with_persistent_waist_and_exported_filter():
    spec = get_robot("g1")
    arms = build_upper_body_rate_limiter(free_args(), spec)
    waist = WaistEntryConstraint(
        spec.joint_names, spec.default_q, spec.action_scale,
        {"waist_yaw_joint": -.03, "waist_roll_joint": -.02, "waist_pitch_joint": .52}, 1.02)
    control = compose_entry_waist_constraint(arms, waist.to_dict())
    portable = torch.jit.script(control)
    raw = torch.full((3, 29), .25, requires_grad=True)
    previous = torch.zeros_like(raw)
    older = previous.clone()
    times = torch.tensor([.1, 1.21, 8.])
    result = control(raw, previous, older, times)
    torch.testing.assert_close(portable(raw, previous, older, times), result)
    gradient = torch.autograd.grad(result.sum(), raw)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient[:, arms.upper_indices] > 0).all()
    torch.testing.assert_close(gradient[:, :12], torch.ones(3, 12))
    torch.testing.assert_close(gradient[:, waist.waist_indices], torch.zeros(3, 3))
    assert not hasattr(arms, "reference_q")
    assert not hasattr(arms, "posture_reward")
    with pytest.raises(ValueError, match="two executed-action"):
        control(raw, previous, entry_time=times)
    with pytest.raises(torch.jit.Error, match="two executed-action"):
        portable(raw, previous, entry_time=times)


def test_optional_cuda_allocator_cap_is_scoped_to_selected_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device: calls.append((fraction, device)))
    configure_cuda_memory(None, "cuda:1")
    assert calls == []
    configure_cuda_memory(.60, "cuda:1")
    assert calls == [(.60, "cuda:1")]
    for value in (0., -1., 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="CUDA memory fraction"):
            configure_cuda_memory(value, "cuda:0")
    assert calls == [(.60, "cuda:1")]


def test_arm_release_preserves_leg_outputs_and_frozen_input_statistics(monkeypatch):
    monkeypatch.setattr(networks, "AMP_DTYPE", None)
    actor = TerrainPolicy(283, 12, hist=10, num_joints=29, action_limit=20.).eval()
    with torch.no_grad():
        actor.obs_norm.update(torch.randn(8, 283))
        actor.goal_norm.update(torch.randn(8, 12))
        actor.dec[-1].bias.fill_(.25)
    state = {name: value.clone() for name, value in actor.state_dict().items()}
    obs, goals = torch.randn(2, 10, 283), torch.randn(2, 12)
    with torch.no_grad():
        before = actor(obs, goals)
    receipt = adapt_warm_start_actor(
        actor, get_robot("g1").joint_names, reset_arms=True, normalizer_prior_count=0.)
    with torch.no_grad():
        after = actor(obs, goals)
    assert len(receipt["arm_output_rows_reset"]) == 14
    assert receipt["normalizer_prior_counts"] == {}
    torch.testing.assert_close(after[:, :15], before[:, :15], rtol=0, atol=0)
    torch.testing.assert_close(after[:, 15:], torch.zeros(2, 14), rtol=0, atol=0)
    for name, value in actor.state_dict().items():
        if name.startswith(("obs_norm.", "goal_norm.")):
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

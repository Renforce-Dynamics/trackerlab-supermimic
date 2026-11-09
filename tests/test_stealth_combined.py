"""Combined stealth recipe: actor authority, preserved source and signed AMP."""

from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from test_amp_end_effectors import WristKinematics, motion_fixture, recorded
from train_terrain import (
    adapt_warm_start_actor, build_upper_body_rate_limiter, can_transfer_world_model,
    parse_args, update_collection_normalizers, validate_upper_body_options,
)
from trackerLab.algorithms.supermimic import networks
from trackerLab.algorithms.supermimic.amp_style import AMPStyle
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import get_task

BASE = ["--task", "locomotion/g1/stealth", "--run-dir", "unused"]
CONFIGS = REPO / "scripts/rls/supermimic/configs"


@pytest.fixture
def combined():
    return parse_args([*BASE, "--config", str(CONFIGS / "command_stealth_combined.json")])


def test_combined_changes_only_amp_and_entry_cost_and_keeps_all_joint_gradients(combined):
    prior = parse_args([*BASE, "--config", str(CONFIGS / "command_stealth_autonomous.json")])
    changed = {name for name in vars(prior)
               if getattr(prior, name) != getattr(combined, name)}
    assert changed == {"config", "style_feature_mode", "entry_style_weight"}
    assert combined.style_feature_mode == "postural_end_effectors_vx"
    assert combined.entry_style_weight == .008
    assert combined.entry_duration_s == 1.21
    assert combined.iterations - combined.wm_only_iters >= 5000
    assert combined.foot_slide_weight == .04 and combined.feet_close_weight == .08
    assert combined.foot_slip_mode == "g1_sole" and not combined.differentiate_support
    assert not combined.stealth_upper_body_constraint
    assert combined.upper_body_reference is None and combined.waist_entry_reference is None
    assert combined.waist_pd_scale == 1. and combined.upper_body_rate_limit
    validate_upper_body_options(combined, get_task(combined.task))

    spec = get_robot("g1")
    control = build_upper_body_rate_limiter(combined, spec)
    raw = torch.linspace(-.3, .3, spec.num_joints).repeat(3, 1).requires_grad_()
    previous = torch.zeros_like(raw)
    times = torch.tensor([0., combined.entry_duration_s, 8.])
    applied = control(raw, previous, previous, entry_time=times)
    gradient, = torch.autograd.grad(applied.sum(), raw)
    assert applied.shape == (3, 29)
    assert torch.isfinite(gradient).all() and (gradient > 0).all()
    torch.testing.assert_close(applied[:, :15], raw[:, :15], rtol=0, atol=0)
    torch.testing.assert_close(gradient[:, :15], torch.ones(3, 15), rtol=0, atol=0)
    torch.testing.assert_close(applied, applied[:1].expand_as(applied), rtol=0, atol=0)
    assert not hasattr(control, "reference_q") and not hasattr(control, "posture_reward")


def test_combined_source_adaptation_preserves_leg_waist_outputs_and_normalization(
        combined, monkeypatch):
    monkeypatch.setattr(networks, "AMP_DTYPE", None)
    spec = get_robot("g1")
    recovery = parse_args([*BASE, "--config", str(CONFIGS / "command_stealth_gait_restore.json")])
    assert not combined.warm_start_reset_world
    assert can_transfer_world_model(vars(recovery), vars(combined))
    assert combined.freeze_actor_normalization and combined.warm_start_reset_arms
    assert combined.warm_start_normalizer_prior_count == 0.

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261110)
        actor = TerrainPolicy(283, 12, hist=combined.hist, num_joints=29,
                              action_limit=combined.action_limit).eval()
        obs, goal = torch.randn(2, combined.hist, 283), torch.randn(2, 12)
        with torch.no_grad():
            actor.obs_norm.update(obs)
            actor.goal_norm.update(goal)
            actor.dec[-1].bias.fill_(.25)
            before = actor(obs, goal)
            saved = {name: value.clone() for name, value in actor.state_dict().items()}
        receipt = adapt_warm_start_actor(
            actor, spec.joint_names, reset_arms=combined.warm_start_reset_arms,
            normalizer_prior_count=combined.warm_start_normalizer_prior_count)
        update_collection_normalizers(
            actor, obs + 10., goal - 10., frozen=combined.freeze_actor_normalization)
        with torch.no_grad():
            after = actor(obs, goal)

    assert receipt["arm_output_rows_reset"] == list(spec.joint_names[15:])
    assert receipt["normalizer_prior_counts"] == {}
    torch.testing.assert_close(after[:, :15], before[:, :15], rtol=0, atol=0)
    torch.testing.assert_close(after[:, 15:], torch.zeros(2, 14), rtol=0, atol=0)
    output_layer = f"dec.{len(actor.dec) - 1}."
    for name, value in actor.state_dict().items():
        if name.startswith(output_layer):
            torch.testing.assert_close(value[:15], saved[name][:15], rtol=0, atol=0)
        else:
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_combined_amp_adds_signed_actual_velocity_and_keeps_reward_gradient(combined):
    kin, motion = WristKinematics(), motion_fixture((12, 13))
    options = dict(history_steps=combined.amp_history, hidden=(), replay_capacity=4)
    old = AMPStyle(kin, motion, .76, feature_mode="postural_end_effectors", **options)
    current = AMPStyle(kin, motion, .76, feature_mode=combined.style_feature_mode, **options)
    assert current.frame_dim == old.frame_dim + 1
    assert current.feature_dim == old.feature_dim + combined.amp_history
    assert current.feature_layout["frame"][-1] == ["root_local_forward_velocity/robot_height", 1]
    torch.testing.assert_close(current._expert_frames[:, :-1], old._expert_frames,
                               rtol=0, atol=0)
    with pytest.raises(ValueError, match="expert identity or feature layout"):
        current.restore(old.checkpoint())

    state = recorded(motion, [0, 0, 0])
    local_velocity = torch.tensor([[.25, 0., 0.], [-.20, 0., 0.], [0., 0., 0.]])
    velocity = (state[1] @ local_velocity[..., None]).squeeze(-1).requires_grad_()
    state = (*state[:2], velocity, *state[3:])
    frames = current.frame_features(state)
    torch.testing.assert_close(frames[:, :-1], old.frame_features(state), rtol=0, atol=0)
    torch.testing.assert_close(frames[:, -1], local_velocity[:, 0] / .76)
    gradient, = torch.autograd.grad(frames[:, -1].sum(), velocity)
    torch.testing.assert_close(gradient, state[1][..., :, 0] / .76)

    # A controlled scorer tests connectivity; it makes no claim about learned gait.
    with torch.no_grad():
        current.norm.mean.zero_()
        current.norm.var.fill_(1.)
        current.discriminator[-1].weight.zero_()
        current.discriminator[-1].weight[0, -1] = 1.
        current.discriminator[-1].bias.zero_()
    with current.freeze():
        frames = current.frame_features(state)
        windows = frames[:, None].expand(-1, combined.amp_history, -1)
        reward = current.reward_from_features(current.window_features(windows))
        gradient, = torch.autograd.grad(reward.sum(), velocity)
    body_gradient = (state[1].transpose(-1, -2) @ gradient[..., None]).squeeze(-1)
    assert torch.isfinite(body_gradient).all() and (body_gradient[:, 0] > 0).all()
    assert current.checkpoint()["updates"] == 0

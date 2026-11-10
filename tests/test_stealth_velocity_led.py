"""Velocity-led rewards must reject the observed stationary command solution."""

import math
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from test_command_locomotion import make_command, make_state
from test_stealth_wasd import wasd_mdp
from train_terrain import parse_args, validate_command_curriculum_options
from trackerLab.tracker_env.mdp.supermimic_command import (
    STEALTH_WASD_MODES, WASD_V3_MAX_STEP_REWARD,
)


def test_all_motion_commands_outweigh_any_static_amp_advantage():
    mdp = wasd_mdp(wasd_reward_version=3)
    c = make_command(mdp, len(STEALTH_WASD_MODES))
    c[:, 1:4] = torch.tensor([row[1:4] for row in STEALTH_WASD_MODES])
    state = list(make_state(mdp, len(c)))
    zero = torch.zeros(len(c), 4)
    static = mdp.reward(state, zero, zero, c)
    moving = [x.clone() for x in state]
    moving[2][:, :2], moving[3][:, 2] = c[:, 1:3], c[:, 3]
    compliant = mdp.reward(moving, zero, zero, c)
    assert ((compliant-static)[:-1] > .05+.02).all()
    torch.testing.assert_close(compliant[-1], static[-1])
    assert (compliant <= WASD_V3_MAX_STEP_REWARD).all()
    mdp.set_style_weight(.05)
    assert math.isclose(mdp.max_step_reward, WASD_V3_MAX_STEP_REWARD+.05)


def test_standing_gradients_follow_forward_reverse_and_turn_requests():
    mdp = wasd_mdp(wasd_reward_version=3)
    c = make_command(mdp, 4)
    c[:, 1:4] = torch.tensor([[.25, 0., 0.], [-.20, 0., 0.],
                             [0., 0., .55], [0., 0., -.55]])
    state = list(make_state(mdp, 4))
    state[2] = torch.zeros(4, 3, requires_grad=True)
    state[3] = torch.zeros(4, 3, requires_grad=True)
    zero = torch.zeros(4, 4)
    v, w = torch.autograd.grad(mdp.reward(state, zero, zero, c).sum(), state[2:4])
    assert v[0, 0] > .1 and v[1, 0] < -.1
    assert w[2, 2] > .05 and w[3, 2] < -.05
    assert torch.isfinite(v).all() and torch.isfinite(w).all()


def test_fast_entry_and_fresh_objective_keep_learned_arms_and_waist():
    args = parse_args([
        "--task", "locomotion/g1/stealth", "--run-dir", "unused",
        "--warm-start", "source.pt", "--config",
        str(REPO / "scripts/rls/supermimic/configs/command_stealth_velocity_led.json"),
    ])
    validate_command_curriculum_options(args)
    assert args.entry_duration_s == .60 and args.amp_weight == .05
    assert args.wasd_reward_version == 3 and args.stealth_gait_weight == .02
    assert args.style_feature_mode == "postural_end_effectors_vx"
    assert not args.warm_start_reset_arms and not args.warm_start_reset_world
    assert args.warm_start_require_control_match and args.freeze_actor_normalization
    assert not args.stealth_upper_body_constraint and args.waist_entry_reference is None
    assert args.iterations - args.wm_only_iters == 50000


def test_reward_change_is_not_a_silent_full_resume():
    args = parse_args([
        "--task", "locomotion/g1/stealth", "--run-dir", "unused",
        "--stealth-control-mode", "wasd", "--wasd-reward-version", "3",
        "--resume", "old.pt", "--allow-command-reward-change",
    ])
    with pytest.raises(ValueError, match="WASD-v2"):
        validate_command_curriculum_options(args)

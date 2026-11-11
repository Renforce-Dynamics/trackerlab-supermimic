"""The clearance adaptation must retain height gradients and legacy behavior."""

from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from train_terrain import parse_args, validate_gait_options, validate_gait_resume
from trackerTask.supermimic.terrain import get_task
from trackerLab.algorithms.supermimic.stealth_gait import command_gait_penalty


def reward(h, v, version=2, command=None, elapsed=3.):
    if command is None:
        command = h.new_tensor((0., .25, 0., 0., .70, 0., elapsed)).expand(*h.shape[:-2], 7)
    return command_gait_penalty(h, v, command, scale=1., weight=.02,
        foot_slide_weight=.04, entry_duration_s=.6, version=version, swing_height=.04)


def test_low_moving_foot_has_height_gradient_even_below_ground():
    for height in (-.04, -.02, 0., .008):
        h = torch.tensor([[[0.]*4, [height]*4]], requires_grad=True)
        v = torch.zeros(1, 2, 4, 3)
        v[:, 1, :, 0] = .4
        old = torch.autograd.grad(reward(h, v, 1).sum(), h, retain_graph=True)[0][:, 1].sum()
        new = torch.autograd.grad(reward(h, v, 2).sum(), h)[0][:, 1].sum()
        assert torch.isfinite(new) and new > old
        assert new > .02


def test_clearance_targets_each_moving_foot_and_respects_stance():
    h = torch.zeros(3, 2, 4)
    h[1, 1] = .04
    h[2, 1] = .08
    v = torch.zeros(3, 2, 4, 3)
    v[:, 1, :, 0] = .5
    r = reward(h, v)
    assert r[1] > r[0] and r[1] > r[2]
    # Stance foot stays grounded without receiving a swing-height cost.
    difference = reward(h[1:2], v[1:2], 2)-reward(h[1:2], v[1:2], 1)
    torch.testing.assert_close(difference, torch.zeros_like(difference))
    torch.testing.assert_close(reward(h.flip(1), v.flip(1)), r)
    torch.testing.assert_close(reward(h, -v), r)


def test_idle_entry_and_flight_guards_remain_active():
    h = torch.zeros(3, 2, 4)
    h[1, 1] = .04
    h[2] = .04
    v = torch.full((3, 2, 4, 3), .25)
    v[..., 2] = 0.
    assert (reward(h, v) <= 0).all()
    assert reward(h, v)[2] < reward(h, v)[0]
    torch.testing.assert_close(reward(h, v, elapsed=0.)[:2], torch.zeros(2))
    # The existing flight/slip guard remains active even at the first entry step.
    assert reward(h, v, elapsed=0.)[2] < 0
    idle = h.new_tensor((0., 0., 0., 0., .7, 0., 3.)).expand(3, 7).clone()
    torch.testing.assert_close(reward(h, v, command=idle), torch.zeros(3))
    idle[:, 3] = .55
    assert (reward(h, v, command=idle) < 0).all()


def test_legacy_checkpoint_cannot_silently_receive_changed_clearance_rewards():
    args = parse_args(["--task", "locomotion/g1/stealth", "--run-dir", "unused",
                      "--config", str(REPO/"scripts/rls/supermimic/configs/command_stealth_shared_aim.json")])
    validate_gait_options(args, get_task(args.task))
    old = {"arguments": {"stealth_gait_weight": .02}}
    validate_gait_resume(old, args)
    args.stealth_gait_version = 2
    with pytest.raises(ValueError, match="stealth_gait_version"):
        validate_gait_resume(old, args)
    args.stealth_gait_weight = 0.
    with pytest.raises(ValueError, match="positive"):
        validate_gait_options(args, get_task(args.task))


def test_beyondamp_preset_changes_only_gait_version_and_trial_budget():
    folder = REPO/"scripts/rls/supermimic/configs"
    common = ["--task", "locomotion/g1/stealth", "--run-dir", "unused",
              "--warm-start", "source.pt", "--aim-style-reference", "reviewed.json"]
    old = parse_args([*common, "--config", str(folder/"command_stealth_shared_aim.json")])
    new = parse_args([*common, "--config", str(folder/"command_stealth_beyondamp.json")])
    changed = {name for name in vars(old) if getattr(old, name) != getattr(new, name)}
    assert changed == {"config", "iterations", "stealth_gait_version"}
    assert new.iterations == 20500 and new.stealth_gait_version == 2
    assert new.stealth_swing_height == .04 and new.amp_weight == .1
    assert not new.warm_start_reset_arms and not new.stealth_upper_body_constraint
    validate_gait_options(new, get_task(new.task))


@pytest.mark.parametrize("value", [True, float("nan"), 0., .2])
def test_invalid_swing_target_is_rejected(value):
    args = parse_args(["--task", "locomotion/g1/stealth", "--run-dir", "unused"])
    args.stealth_swing_height = value
    with pytest.raises(ValueError, match="swing height"):
        validate_gait_options(args, get_task(args.task))

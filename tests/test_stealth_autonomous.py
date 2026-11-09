"""Owner's learned-arm, endpoint-AMP, short-entry, learned-waist combination."""

from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from train_terrain import build_upper_body_rate_limiter, parse_args, validate_upper_body_options
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import get_task

BASE = ["--task", "locomotion/g1/stealth", "--run-dir", "unused"]
PRESET = REPO / "scripts/rls/supermimic/configs/command_stealth_autonomous.json"


def autonomous_args():
    return parse_args([*BASE, "--config", str(PRESET)])


def test_explicit_combination_has_no_arm_or_waist_pose_override():
    args = autonomous_args()
    validate_upper_body_options(args, get_task(args.task))
    assert not args.stealth_upper_body_constraint
    assert args.upper_body_reference is None
    assert args.waist_entry_reference is None
    assert args.waist_pd_scale == 1.
    assert args.style_feature_mode == "postural_end_effectors"
    assert args.entry_duration_s == 1.21
    assert args.upper_body_rate_limit
    assert args.iterations - args.wm_only_iters == 5000
    assert args.freeze_actor_normalization and not args.warm_start_reset_world
    assert args.stealth_control_mode == "wasd" and args.command_training_modes is None


def test_all_waist_outputs_retain_identity_and_arms_retain_learning_gradients():
    spec = get_robot("g1")
    control = build_upper_body_rate_limiter(autonomous_args(), spec)
    raw = torch.full((3, spec.num_joints), .25, requires_grad=True)
    previous, older = torch.zeros_like(raw), torch.zeros_like(raw)
    times = torch.tensor([.1, 1.21, 8.])
    output = control(raw, previous, older, entry_time=times)
    gradient, = torch.autograd.grad(output.sum(), raw)
    waist = [i for i, name in enumerate(spec.joint_names) if name.startswith("waist_")]
    assert len(waist) == 3
    assert torch.equal(output[:, waist], raw[:, waist])
    assert torch.equal(gradient[:, waist], torch.ones_like(gradient[:, waist]))
    assert torch.equal(gradient[:, :12], torch.ones_like(gradient[:, :12]))
    assert torch.isfinite(gradient).all() and (gradient[:, control.upper_indices] > 0).all()
    assert not hasattr(control, "reference_q") and not hasattr(control, "posture_reward")
    torch.testing.assert_close(torch.jit.script(control)(raw, previous, older, times), output)


def test_g1_endpoint_layout_contains_both_feet_and_both_hand_links():
    spec = get_robot("g1")
    assert {"left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link"}.issubset(spec.tracked_bodies)


def test_recipe_preserves_recovery_reward_and_dynamics_settings():
    before = parse_args([*BASE, "--config", str(
        REPO / "scripts/rls/supermimic/configs/command_stealth_gait_restore.json")])
    after = autonomous_args()
    for field in ("actor_lr", "amp_weight", "amp_history", "style_feature_mode",
                  "foot_slide_weight", "feet_close_weight", "foot_slip_mode",
                  "differentiate_support", "waist_pd_scale", "physical_domain_randomization",
                  "self_collisions", "collision_mask_version", "clearance_ratio", "command_resample_steps"):
        assert getattr(before, field) == getattr(after, field), field

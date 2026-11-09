"""Pure cost, differentiable FK guidance and source calibration contracts."""

import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.stealth_entry_style import (
    EntryStyleGuidance, calibrate_entry_config, entry_torso_cost,
)
from trackerLab.kinematics.supermimic.rotations import axis_angle_to_mat


TARGET = torch.tensor(
    [0.4694757618473205, -0.06730751037741539, 0.8803761741918467],
    dtype=torch.float64,
)
UPRIGHT = torch.tensor([0., 0., 1.], dtype=torch.float64)


def pitched(angle):
    return torch.stack((angle.sin(), torch.zeros_like(angle), angle.cos()), -1)


def test_desired_pose_has_zero_cost_at_start_and_midpoint():
    desired = F.normalize(UPRIGHT + TARGET, dim=-1)
    poses = torch.stack((UPRIGHT, desired))
    cost = entry_torso_cost(poses, TARGET, torch.tensor([0., .605], dtype=torch.float64))
    torch.testing.assert_close(cost, torch.zeros_like(cost), rtol=0., atol=1e-30)


def test_backward_lean_costs_more_than_forward_lean_at_midpoint():
    forward = pitched(torch.tensor(math.radians(14.), dtype=torch.float64))
    backward = forward * forward.new_tensor((-1., 1., 1.))
    assert entry_torso_cost(backward, TARGET, .605) > entry_torso_cost(forward, TARGET, .605)


def test_side_tilt_is_penalized():
    desired = F.normalize(UPRIGHT + TARGET, dim=-1)
    side = F.normalize(desired + desired.new_tensor((0., .3, 0.)), dim=-1)
    assert entry_torso_cost(side, TARGET, .605) > entry_torso_cost(desired, TARGET, .605)


def test_gradient_descent_moves_upright_torso_forward():
    angle = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    cost = entry_torso_cost(pitched(angle), TARGET, .605)
    gradient, = torch.autograd.grad(cost, angle)
    updated = angle.detach() - .05 * gradient
    assert torch.isfinite(gradient) and gradient < 0
    assert updated > 0
    assert entry_torso_cost(pitched(updated), TARGET, .605) < cost


def test_completed_and_reference_reset_clocks_have_exact_zero_cost_and_gradients():
    up = (-UPRIGHT).expand(3, 3).clone().requires_grad_()
    target = TARGET.clone().requires_grad_()
    time = torch.tensor([1.21, 1.22, 20.], dtype=torch.float64, requires_grad=True)
    cost = entry_torso_cost(up, target, time)
    gradients = torch.autograd.grad(cost.sum(), (up, target, time))
    assert torch.equal(cost, torch.zeros_like(cost))
    for gradient in gradients:
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_scalar_and_broadcast_batches():
    scalar = entry_torso_cost(UPRIGHT, TARGET, .605)
    assert scalar.shape == ()
    times = torch.tensor([0., .2, .605, 1.21], dtype=torch.float64)
    up = UPRIGHT.expand(2, 4, 3)
    targets = TARGET.expand(2, 1, 3)
    actual = entry_torso_cost(up, targets, times)
    expected = torch.stack([entry_torso_cost(UPRIGHT, TARGET, t) for t in times])
    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected.expand(2, 4))
    torch.testing.assert_close(entry_torso_cost(UPRIGHT, TARGET, times), expected)
    torch.testing.assert_close(entry_torso_cost(up, TARGET, .605), scalar.expand(2, 4))


def test_caller_handles_common_world_yaw_before_cost():
    torso = pitched(torch.tensor(-.2, dtype=torch.float64))
    expected = entry_torso_cost(torso, TARGET, .605)
    for yaw in (-2.4, .7, math.pi):
        c, s = math.cos(yaw), math.sin(yaw)
        heading = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=torch.float64)
        caller_up = heading.T @ (heading @ torso)
        caller_target = heading.T @ (heading @ TARGET)
        torch.testing.assert_close(entry_torso_cost(caller_up, caller_target, .605), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_cost_and_gradients_are_finite_and_bounded(dtype):
    angles = torch.linspace(-math.pi, math.pi, 65, dtype=torch.float32)
    up = pitched(angles).to(dtype).requires_grad_()
    times = torch.linspace(-.1, 1.4, 65, dtype=torch.float32).to(dtype)
    cost = entry_torso_cost(up, TARGET, times)
    gradient, = torch.autograd.grad(cost.sum(), up)
    assert torch.isfinite(cost).all() and torch.isfinite(gradient).all()
    assert ((cost >= 0.) & (cost <= 1.)).all()


def test_custom_duration_and_error_scale():
    desired = F.normalize(UPRIGHT + TARGET, dim=-1)
    e2 = (UPRIGHT - desired).square().sum()
    actual = entry_torso_cost(UPRIGHT, TARGET, .4, duration=.8, error_scale=.5)
    torch.testing.assert_close(actual, .5 * e2 / (e2 + .25))
    assert entry_torso_cost(UPRIGHT, TARGET, .8, duration=.8).item() == 0.


@pytest.mark.parametrize("name", ["duration", "error_scale"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        entry_torso_cost(UPRIGHT, TARGET, .4, **{name: value})


def test_state_gradient_matches_finite_difference():
    up = pitched(torch.tensor(.1, dtype=torch.float64)).requires_grad_()
    time = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda u, t: entry_torso_cost(u, TARGET, t), (up, time))


class TorsoKinematics:
    device = "cpu"
    body_names = ("pelvis", "unused", "torso_link", "wrist")
    tracked_idx = (2, 3, 0)
    robot = SimpleNamespace(root_body="pelvis", com_body="torso_link")

    def forward(self, p, R, v, w, q, qd):
        yaw = axis_angle_to_mat(q.new_tensor((0., 0., 1.)), q[..., 0])
        roll = axis_angle_to_mat(q.new_tensor((1., 0., 0.)), q[..., 1])
        pitch = axis_angle_to_mat(q.new_tensor((0., 1., 0.)), q[..., 2])
        torso = R @ yaw @ roll @ pitch
        rotations = torch.stack((R, R, torso, R), -3)[..., self.tracked_idx, :, :]
        return (p[..., None, :].expand(*p.shape[:-1], 3, 3), rotations,
                v[..., None, :].expand(*v.shape[:-1], 3, 3),
                w[..., None, :].expand(*w.shape[:-1], 3, 3))


def source_document():
    return {
        "schema": "g1-stealth-waist-entry-v1",
        "source": {
            "forward_csv": "forward__aiming2_subject5__f02830-02891.csv",
            "forward_csv_sha256": "a" * 64,
            "pool_manifest_sha256": "b" * 64,
            "converted_frame": 72,
            "source_file": "aiming2_subject5.bvh",
            "source_frame_range": [2830, 2891],
            "selection": "Actual three-waist-joint sample nearest median in L1.",
        },
    }


def guidance_config():
    return {
        "weight": .008, "duration": 1.21, "target_up": TARGET.tolist(),
        "torso_body": "torso_link", "source": source_document()["source"],
        "expert_hash": "c" * 64, "reference_index": 75, "reference_clip_index": 1,
    }


def state_with(q, root_pitch=0., root_yaw=0.):
    pitch = torch.as_tensor(root_pitch, dtype=q.dtype).expand(q.shape[:-1])
    yaw = torch.as_tensor(root_yaw, dtype=q.dtype).expand(q.shape[:-1])
    R = (axis_angle_to_mat(q.new_tensor((0., 0., 1.)), yaw)
         @ axis_angle_to_mat(q.new_tensor((0., 1., 0.)), pitch))
    zeros = torch.zeros_like(q)
    return zeros, R, zeros, zeros, q, zeros


def command(time):
    time = torch.as_tensor(time, dtype=torch.float64)
    return torch.cat((torch.zeros(*time.shape, 6, dtype=time.dtype), time[..., None]), -1)


def test_guidance_uses_tracked_torso_and_penalizes_backward_and_side():
    guidance = EntryStyleGuidance(TorsoKinematics(), guidance_config())
    q = torch.tensor([[0., 0., .24], [0., 0., -.24], [0., -.4, .24]], dtype=torch.float64)
    before = q.clone()
    reward = guidance.reward(state_with(q), command(.605))
    assert reward[0] > reward[1] and reward[0] > reward[2]
    assert ((reward <= 0.) & (reward >= -.008)).all()
    assert torch.equal(q, before)
    assert list(guidance.parameters()) == []


def test_guidance_keeps_root_pitch_and_joint_gradients_and_common_yaw_invariance():
    guidance = EntryStyleGuidance(TorsoKinematics(), guidance_config())
    q = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    pitch = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    reward = guidance.reward(state_with(q, pitch), command(.605))
    joint_grad, root_grad = torch.autograd.grad(reward, (q, pitch))
    assert joint_grad[2] > 0 and root_grad > 0
    assert torch.isfinite(joint_grad).all() and torch.isfinite(root_grad)
    for yaw in (-2.4, .7, math.pi):
        torch.testing.assert_close(
            guidance.reward(state_with(q, .2, yaw), command(.605)),
            -.008 * entry_torso_cost(pitched(q.new_tensor(.2)), TARGET, .605),
        )
    assert (guidance.reward(state_with(q, .2), command(.605))
            > guidance.reward(state_with(q, -.2), command(.605)))


def test_guidance_completed_clock_has_exact_zero_fk_and_clock_gradients():
    guidance = EntryStyleGuidance(TorsoKinematics(), guidance_config())
    q = torch.full((2, 3), -.3, dtype=torch.float64, requires_grad=True)
    time = torch.tensor([1.21, 3.], dtype=torch.float64, requires_grad=True)
    pitch = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    reward = guidance.reward(state_with(q, pitch), command(time))
    assert torch.equal(reward, torch.zeros_like(reward))
    for grad in torch.autograd.grad(reward.sum(), (q, time, pitch)):
        assert torch.equal(grad, torch.zeros_like(grad))


def test_guidance_reward_compiles_with_full_graph():
    guidance = EntryStyleGuidance(TorsoKinematics(), guidance_config())
    q = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    state, c = state_with(q), command(torch.tensor([.4, 1.21]))
    compiled = torch.compile(guidance.reward, backend="eager", fullgraph=True)
    actual = compiled(state, c)
    torch.testing.assert_close(actual, guidance.reward(state, c))
    gradient, = torch.autograd.grad(actual.sum(), q)
    assert gradient[0, 2] > 0
    assert torch.equal(gradient[1], torch.zeros_like(gradient[1]))


@pytest.mark.parametrize("field,value", [
    ("weight", 0.), ("weight", float("inf")), ("duration", -1.),
    ("error_scale", 0.), ("target_up", [0., 0., 2.]),
    ("target_up", [float("nan"), 0., 1.]), ("target_up", [-.6, 0., .8]),
    ("target_up", [.6, 0., -.8]), ("torso_body", "pelvis"),
    ("expert_hash", "invalid"), ("reference_index", -1),
    ("reference_clip_index", True), ("source", {}),
])
def test_guidance_rejects_invalid_config(field, value):
    config = guidance_config()
    config[field] = value
    with pytest.raises(ValueError):
        EntryStyleGuidance(TorsoKinematics(), config)


def motion_fixture():
    q = torch.zeros(107, 3, dtype=torch.float64)
    q[75, 2] = .3
    state = state_with(q, root_pitch=.17, root_yaw=1.3)
    return SimpleNamespace(
        **dict(zip(("root_p", "root_R", "root_v", "root_w", "q", "qd"), state)),
        names=["idle", source_document()["source"]["forward_csv"][:-4], "backward"],
        clip_start=torch.tensor([0, 3, 104]), clip_end=torch.tensor([2, 103, 106]),
    )


def test_calibration_uses_named_clip_offset_actual_root_tilt_and_keeps_provenance():
    kin, motion, document = TorsoKinematics(), motion_fixture(), source_document()
    before = motion.q.clone()
    config = calibrate_entry_config(kin, motion, document, .008, 1.21, "d" * 64)
    assert config["reference_index"] == 75 and config["reference_clip_index"] == 1
    assert config["source"] == document["source"] and config["expert_hash"] == "d" * 64
    torch.testing.assert_close(
        torch.tensor(config["target_up"], dtype=torch.float64),
        pitched(torch.tensor(.47, dtype=torch.float64)),
    )
    assert torch.equal(motion.q, before)
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    guidance = EntryStyleGuidance(kin, config)
    saved = copy.deepcopy(config)
    config["source"]["converted_frame"] = 0
    document["source"]["converted_frame"] = 0
    assert guidance.to_dict() == saved


@pytest.mark.parametrize("case", ["missing_clip", "duplicate_clip", "past_clip", "nonfinite", "hash"])
def test_calibration_rejects_invalid_reference_identity_or_state(case):
    motion, document = motion_fixture(), source_document()
    if case == "missing_clip":
        motion.names[1] = "other"
    elif case == "duplicate_clip":
        motion.names[0] = motion.names[1]
    elif case == "past_clip":
        document["source"]["converted_frame"] = 101
    elif case == "nonfinite":
        motion.q[75, 2] = float("nan")
    elif case == "hash":
        document["source"]["forward_csv_sha256"] = "invalid"
    with pytest.raises(ValueError):
        calibrate_entry_config(TorsoKinematics(), motion, document, .008, 1.21, "d" * 64)


def test_guidance_rejects_untracked_torso_and_nonfinite_metadata():
    kin = TorsoKinematics()
    kin.tracked_idx = (3, 0)
    with pytest.raises(ValueError, match="com_body"):
        EntryStyleGuidance(kin, guidance_config())
    config = guidance_config()
    config["source"]["diagnostic"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        EntryStyleGuidance(TorsoKinematics(), config)

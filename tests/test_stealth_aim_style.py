"""CPU contracts for one source-bound training aim pose across all commands."""

import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.stealth_aim_style import (
    AimStyleGuidance, calibrate_aim_config,
)
from trackerLab.kinematics.supermimic.rotations import axis_angle_to_mat
from trackerLab.tracker_env.mdp.supermimic_command import entry_blend
from trackerTask.supermimic.robot import G1_JOINT_NAMES


class AimKinematics:
    """Differentiable articulated torso/arms with deliberately shuffled tracks."""

    device = "cpu"
    joint_names = tuple(G1_JOINT_NAMES)
    body_names = ("right_wrist_yaw_link", "pelvis", "torso_link", "left_wrist_yaw_link", "unused")
    tracked_idx = (3, 1, 0, 2)
    robot = SimpleNamespace(name="g1", root_body="pelvis", com_body="torso_link")

    def rotate(self, q, name, axis):
        return axis_angle_to_mat(q.new_tensor(axis), q[..., self.joint_names.index(name)])

    def forward(self, p, R, v, w, q, qd):
        torso = (R @ self.rotate(q, "waist_yaw_joint", (0., 0., 1.))
                 @ self.rotate(q, "waist_roll_joint", (1., 0., 0.))
                 @ self.rotate(q, "waist_pitch_joint", (0., 1., 0.)))
        torso_p = p + (R @ q.new_tensor((0., 0., .18))[..., None]).squeeze(-1)
        wrists_p, wrists_R = [], []
        for side, sign in (("left", 1.), ("right", -1.)):
            shoulder = (self.rotate(q, f"{side}_shoulder_pitch_joint", (0., 1., 0.))
                        @ self.rotate(q, f"{side}_shoulder_roll_joint", (1., 0., 0.))
                        @ self.rotate(q, f"{side}_shoulder_yaw_joint", (0., 0., 1.)))
            elbow = shoulder @ self.rotate(q, f"{side}_elbow_joint", (0., 1., 0.))
            wrist = (elbow @ self.rotate(q, f"{side}_wrist_roll_joint", (1., 0., 0.))
                     @ self.rotate(q, f"{side}_wrist_pitch_joint", (0., 1., 0.))
                     @ self.rotate(q, f"{side}_wrist_yaw_joint", (0., 0., 1.)))
            local = (q.new_tensor((0., sign * .17, .15))
                     + (shoulder @ q.new_tensor((0., 0., -.23))[..., None]).squeeze(-1)
                     + (elbow @ q.new_tensor((.08, 0., -.20))[..., None]).squeeze(-1))
            wrists_p.append(torso_p + (torso @ local[..., None]).squeeze(-1))
            wrists_R.append(torso @ wrist)
        positions = torch.stack((wrists_p[1], p, torso_p, wrists_p[0], p), -2)
        rotations = torch.stack((wrists_R[1], R, torso, wrists_R[0], R), -3)
        positions = positions[..., self.tracked_idx, :]
        rotations = rotations[..., self.tracked_idx, :, :]
        return positions, rotations, torch.zeros_like(positions), torch.zeros_like(positions)


def source():
    # Synthetic identity exercises metadata only; the owner supplies the actual asset.
    return {"asset": "selected-aim.bvh", "asset_sha256": "a" * 64,
            "retarget_sha256": "b" * 64, "source_frame": 4564,
            "selection": {"frame_range": [4496, 4676], "frozen": True}}


def reference_q():
    q = torch.zeros(29, dtype=torch.float64)
    values = {"waist_yaw_joint": .12, "waist_roll_joint": -.08, "waist_pitch_joint": .3,
              "left_shoulder_pitch_joint": -.8, "left_shoulder_roll_joint": .2,
              "left_shoulder_yaw_joint": -.15, "left_elbow_joint": 1.1,
              "left_wrist_roll_joint": .1, "left_wrist_pitch_joint": -.2,
              "left_wrist_yaw_joint": .25, "right_shoulder_pitch_joint": -.65,
              "right_shoulder_roll_joint": -.18, "right_shoulder_yaw_joint": .2,
              "right_elbow_joint": .95, "right_wrist_roll_joint": -.1,
              "right_wrist_pitch_joint": .13, "right_wrist_yaw_joint": -.3}
    for name, value in values.items():
        q[G1_JOINT_NAMES.index(name)] = value
    return q


def state_with(q, *, qd=None, root_pitch=0., root_yaw=0., translation=(0., 0., .76)):
    batch = q.shape[:-1]
    zeros = q.new_zeros(batch + (3,))
    pitch = torch.as_tensor(root_pitch, dtype=q.dtype, device=q.device).expand(batch)
    yaw = torch.as_tensor(root_yaw, dtype=q.dtype, device=q.device).expand(batch)
    R = (axis_angle_to_mat(q.new_tensor((0., 0., 1.)), yaw)
         @ axis_angle_to_mat(q.new_tensor((0., 1., 0.)), pitch))
    p = torch.as_tensor(translation, dtype=q.dtype, device=q.device).expand(batch + (3,))
    return p, R, zeros, zeros.clone(), q, torch.zeros_like(q) if qd is None else qd


def command(time, *, dtype=torch.float64):
    time = torch.as_tensor(time, dtype=dtype)
    return torch.cat((time.new_zeros(time.shape + (6,)), time[..., None]), -1)


def guidance_config(kin=None):
    kin = AimKinematics() if kin is None else kin
    return calibrate_aim_config(kin, state_with(reference_q(), root_pitch=.14, root_yaw=.7), source())


def perturbed_q(batch=()):
    q = reference_q().expand(batch + (29,)).clone()
    q[..., G1_JOINT_NAMES.index("left_shoulder_pitch_joint")] += .25
    q[..., G1_JOINT_NAMES.index("right_elbow_joint")] -= .2
    q[..., G1_JOINT_NAMES.index("waist_pitch_joint")] -= .12
    return q


def test_calibration_copies_single_selected_pose_and_roundtrips_json():
    kin, provenance = AimKinematics(), source()
    state = state_with(reference_q().reshape(1, 1, 29), root_pitch=.14, root_yaw=.7)
    before = tuple(value.clone() for value in state)
    config = calibrate_aim_config(kin, state, provenance)
    assert config["schema"] == "g1-stealth-aim-style-v1"
    assert config["weight"] == .01 and config["duration"] == .60 and config["robot_height"] == .76
    assert len(config["reference_q"]) == 14
    for name, value in config["reference_q"].items():
        assert value == reference_q()[G1_JOINT_NAMES.index(name)].item()
    assert config["wrist_bodies"] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert config["source"] == provenance
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    guidance = AimStyleGuidance(kin, json.loads(json.dumps(config)))
    assert guidance.to_dict() == config
    assert list(guidance.parameters()) == []
    assert all(not value.requires_grad for value in guidance.buffers())
    assert guidance.reward(state, command(.6)).abs().max() < 1e-28
    for actual, expected in zip(state, before):
        assert torch.equal(actual, expected)
    saved = copy.deepcopy(config)
    config["reference_q"]["left_elbow_joint"] = 0.
    provenance["selection"]["frame_range"][0] = 0
    exported = guidance.to_dict()
    exported["source"]["source_frame"] = 0
    assert guidance.to_dict() == saved


def test_metrics_have_physical_units_and_correct_wrist_order():
    kin, q = AimKinematics(), reference_q()
    config = calibrate_aim_config(kin, state_with(q), source())
    config["reference_wrist_positions_torso"][0][0] += .06
    config["reference_wrist_positions_torso"][1][1] -= .08
    angle = torch.tensor(.4, dtype=torch.float64)
    turn = axis_angle_to_mat(q.new_tensor((0., 0., 1.)), angle)
    wrist_R = torch.tensor(config["reference_wrist_rotations_torso"], dtype=q.dtype)
    config["reference_wrist_rotations_torso"] = (wrist_R @ turn).tolist()
    config["reference_q"]["left_elbow_joint"] += .2
    config["reference_torso_up_heading"] = [0., 0., 1.]
    qd = torch.zeros_like(q)
    qd[G1_JOINT_NAMES.index("right_wrist_yaw_joint")] = .5
    guidance = AimStyleGuidance(kin, config)
    metrics = guidance.metrics(state_with(q, qd=qd))
    assert set(metrics) == set(config["term_weights"])
    assert metrics["position_rmse_m"].item() == pytest.approx(math.sqrt((.06**2 + .08**2) / 2))
    assert metrics["rotation_chordal"].item() == pytest.approx(2. * math.sin(.2))
    assert metrics["arm_rmse_rad"].item() == pytest.approx(.2 / math.sqrt(14.))
    assert metrics["arm_speed_rms_rad_s"].item() == pytest.approx(.5 / math.sqrt(14.))
    config_up = calibrate_aim_config(kin, state_with(q), source())["reference_torso_up_heading"]
    expected_up = torch.tensor(config_up) - torch.tensor([0., 0., 1.])
    assert metrics["torso_up_error"].item() == pytest.approx(expected_up.norm().item())
    e = {name: (value / config["error_scales"][name]).square() for name, value in metrics.items()}
    expected = -.01 * sum(config["term_weights"][name] * value / (1. + value)
                         for name, value in e.items())
    torch.testing.assert_close(guidance.reward(state_with(q, qd=qd), command(.6)), expected)


@pytest.mark.parametrize("yaw", [-2.4, .7, math.pi])
def test_reward_metrics_and_calibration_are_invariant_to_translation_and_world_yaw(yaw):
    kin, q = AimKinematics(), perturbed_q()
    config = guidance_config(kin)
    guidance = AimStyleGuidance(kin, config)
    state = state_with(q, root_pitch=.2)
    moved = state_with(q, root_pitch=.2, root_yaw=yaw, translation=(12., -8., 4.))
    torch.testing.assert_close(guidance.reward(moved, command(.6)), guidance.reward(state, command(.6)))
    for name, value in guidance.metrics(state).items():
        torch.testing.assert_close(guidance.metrics(moved)[name], value)
    original = calibrate_aim_config(kin, state, source())
    transformed = calibrate_aim_config(kin, moved, source())
    for name in ("reference_wrist_positions_torso", "reference_wrist_rotations_torso",
                 "reference_torso_up_heading"):
        torch.testing.assert_close(torch.tensor(original[name]), torch.tensor(transformed[name]))


def test_all_commands_share_one_pose_and_only_the_entry_clock_affects_reward():
    guidance = AimStyleGuidance(AimKinematics(), guidance_config())
    q = perturbed_q()
    c = command(torch.full((9,), .6))
    c[:, :6] = torch.arange(54, dtype=c.dtype).reshape(9, 6) * .13 - 2.
    original, state = c.clone(), state_with(q)
    reward = guidance.reward(state, c)
    assert torch.equal(reward, reward[0].expand_as(reward))
    assert torch.equal(c, original)
    c.requires_grad_()
    grad, = torch.autograd.grad(guidance.reward(state, c).sum(), c)
    assert torch.equal(grad[:, :6], torch.zeros_like(grad[:, :6]))
    times = torch.tensor([-.1, 0., .15, .3, .45, .6, .9, 2., 20.], dtype=c.dtype)
    actual = guidance.reward(state, command(times))
    torch.testing.assert_close(actual, reward * entry_blend(command(times), .6))
    assert actual[0] == actual[1] == 0.
    assert actual[2] > actual[3] > actual[4] > actual[5]
    assert torch.equal(actual[5:], actual[5].expand_as(actual[5:]))


def test_zero_and_completed_clock_have_zero_clock_gradients_and_persistent_pose_cost():
    guidance = AimStyleGuidance(AimKinematics(), guidance_config())
    q = perturbed_q((4,)).requires_grad_()
    time = torch.tensor([0., .3, .6, 8.], dtype=q.dtype, requires_grad=True)
    reward = guidance.reward(state_with(q), command(time))
    joints, clock = torch.autograd.grad(reward.sum(), (q, time))
    assert torch.equal(joints[0], torch.zeros_like(joints[0]))
    assert reward[0] == 0. and reward[2] == reward[3] < 0.
    assert clock[0] == clock[2] == clock[3] == 0. and clock[1] < 0.
    assert joints[2].abs().sum() > 0. and joints[3].abs().sum() > 0.


def test_reward_preserves_arm_waist_root_and_velocity_gradients_without_state_mutation():
    guidance = AimStyleGuidance(AimKinematics(), guidance_config())
    q = perturbed_q().requires_grad_()
    qd = torch.full_like(q, .08, requires_grad=True)
    pitch = torch.tensor(.02, dtype=q.dtype, requires_grad=True)
    state = state_with(q, qd=qd, root_pitch=pitch)
    before = tuple(value.detach().clone() for value in state)
    reward = guidance.reward(state, command(.6))
    joints, velocity, root = torch.autograd.grad(reward, (q, qd, pitch))
    assert torch.isfinite(joints).all() and torch.isfinite(velocity).all() and torch.isfinite(root)
    assert joints[guidance.arm_indices].abs().sum() > 0.
    assert joints[G1_JOINT_NAMES.index("waist_pitch_joint")].abs() > 0.
    assert joints[G1_JOINT_NAMES.index("waist_roll_joint")].abs() > 0.
    assert root.abs() > 0.
    assert (velocity[guidance.arm_indices] < 0.).all()
    assert torch.equal(velocity[:15], torch.zeros_like(velocity[:15]))
    assert torch.equal(joints[:12], torch.zeros_like(joints[:12]))
    assert guidance.reward(state_with(q.detach() + .01 * joints, root_pitch=.02), command(.6)) > reward
    for actual, expected in zip(state, before):
        assert torch.equal(actual.detach(), expected)


def test_handwaving_penalty_uses_only_arm_speed_with_a_small_bounded_contribution():
    kin, q = AimKinematics(), reference_q()
    guidance = AimStyleGuidance(kin, calibrate_aim_config(kin, state_with(q), source()))
    qd = torch.zeros_like(q)
    qd[:15] = 15.
    idle = guidance.reward(state_with(q, qd=qd), command(.6))
    assert idle.abs() < 1e-28
    qd[guidance.arm_indices] = .1
    torch.testing.assert_close(guidance.reward(state_with(q, qd=qd), command(.6)), q.new_tensor(-.00025))
    qd[guidance.arm_indices] = 1e6
    moving = guidance.reward(state_with(q, qd=qd), command(.6))
    assert -.0005 <= moving < -.00049


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_batched_reward_and_gradients_are_finite_and_bounded(dtype):
    guidance = AimStyleGuidance(AimKinematics(), guidance_config())
    generator = torch.Generator().manual_seed(18)
    q = (torch.randn(3, 5, 29, generator=generator) * 4.).to(dtype).requires_grad_()
    qd = (torch.randn(3, 5, 29, generator=generator) * 8.).to(dtype).requires_grad_()
    times = torch.tensor([0., .15, .3, .6, 12.], dtype=dtype)
    reward = guidance.reward(state_with(q, qd=qd), command(times, dtype=dtype))
    assert reward.shape == (3, 5)
    assert torch.isfinite(reward).all() and ((reward >= -.01) & (reward <= 0.)).all()
    for gradient in torch.autograd.grad(reward.sum(), (q, qd)):
        assert torch.isfinite(gradient).all()
    assert all(value.shape == (3, 5) for value in guidance.metrics(state_with(q)).values())


@pytest.mark.parametrize("dtype,magnitude", [(torch.float32, 1e25), (torch.float64, 1e200)])
def test_large_finite_arm_errors_have_finite_metrics_bounded_reward_and_gradients(dtype, magnitude):
    guidance = AimStyleGuidance(AimKinematics(), guidance_config())
    q = reference_q().to(dtype)
    q[guidance.arm_indices] = magnitude
    q.requires_grad_()
    qd = torch.full_like(q, magnitude, requires_grad=True)
    state = state_with(q, qd=qd)
    metrics = guidance.metrics(state)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert metrics["arm_rmse_rad"].item() == pytest.approx(magnitude)
    assert metrics["arm_speed_rms_rad_s"].item() == pytest.approx(magnitude)
    reward = guidance.reward(state, command(.6))
    assert torch.isfinite(reward) and -.01 <= reward < 0.
    for gradient in torch.autograd.grad(reward, (q, qd)):
        assert torch.isfinite(gradient).all()


def test_exact_reference_has_finite_zero_reward_and_metric_gradients():
    kin = AimKinematics()
    config = calibrate_aim_config(kin, state_with(reference_q()), source())
    guidance = AimStyleGuidance(kin, config)
    q, qd = reference_q().requires_grad_(), torch.zeros(29, dtype=torch.float64, requires_grad=True)
    state = state_with(q, qd=qd)
    reward = guidance.reward(state, command(.6))
    for grad in torch.autograd.grad(reward, (q, qd), retain_graph=True):
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0., atol=1e-14)
    total = sum(guidance.metrics(state).values())
    for grad in torch.autograd.grad(total, (q, qd)):
        assert torch.isfinite(grad).all()


def test_reward_matches_finite_differences_and_compiles_as_a_full_graph():
    guidance = AimStyleGuidance(AimKinematics(), guidance_config())
    q = perturbed_q().requires_grad_()
    time = torch.tensor(.23, dtype=q.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda q, t: guidance.reward(state_with(q), command(t)), (q, time))
    compiled = torch.compile(guidance.reward, backend="eager", fullgraph=True)
    actual = compiled(state_with(q), command(time))
    torch.testing.assert_close(actual, guidance.reward(state_with(q), command(time)))
    grad, = torch.autograd.grad(actual, q)
    assert torch.isfinite(grad).all() and grad[guidance.arm_indices].abs().sum() > 0.


def test_joint_order_is_resolved_by_names_and_tracks_can_be_minimal():
    kin = AimKinematics()
    config = guidance_config(kin)
    q = perturbed_q()
    order = list(reversed(range(29)))
    kin.joint_names = tuple(G1_JOINT_NAMES[i] for i in order)
    kin.tracked_idx = (2, 0, 3)
    shuffled = AimStyleGuidance(kin, config)
    original = AimStyleGuidance(AimKinematics(), config)
    torch.testing.assert_close(shuffled.reward(state_with(q[order]), command(.6)),
                               original.reward(state_with(q), command(.6)))


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("weight", 0.), ("weight", -1.), ("weight", True),
    ("weight", float("inf")), ("weight", 1e-60), ("weight", 1e50),
    ("duration", 0.), ("duration", float("nan")), ("robot_height", .8),
    ("robot", "go2"), ("torso_body", "pelvis"),
    ("wrist_bodies", ["right_wrist_yaw_link", "left_wrist_yaw_link"]),
    ("reference_q", {}), ("reference_wrist_positions_torso", [[0., 0., 0.]]),
    ("reference_wrist_rotations_torso", [[[1., 0., 0.]]]),
    ("reference_torso_up_heading", [0., 0., 2.]),
    ("reference_torso_up_heading", [True, 0., 0.]),
    ("reference_torso_up_heading", [float("nan"), 0., 1.]),
    ("source", {}), ("source", {"asset": "unbound.bvh"}),
    ("source", {"asset_sha256": "invalid"}), ("source", {"asset_sha256": 17}),
    ("source", {"hashes": {"asset": None}}), ("source", {"hashes": []}),
    ("error_scales", {}), ("term_weights", {}),
])
def test_invalid_config_is_rejected(field, value):
    config = guidance_config()
    config[field] = value
    with pytest.raises(ValueError):
        AimStyleGuidance(AimKinematics(), config)


@pytest.mark.parametrize("case", ["joint_bool", "joint_extra", "position_nonfinite", "reflection",
                                  "nonorthogonal", "scale_zero", "scale_bool", "weight_negative",
                                  "weight_sum", "weight_bool", "source_nonfinite", "source_tensor"])
def test_invalid_nested_values_are_rejected(case):
    config = guidance_config()
    if case == "joint_bool":
        config["reference_q"]["left_elbow_joint"] = True
    elif case == "joint_extra":
        config["reference_q"]["waist_pitch_joint"] = .2
    elif case == "position_nonfinite":
        config["reference_wrist_positions_torso"][0][0] = float("inf")
    elif case in ("reflection", "nonorthogonal"):
        config["reference_wrist_rotations_torso"][0] = [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        if case == "nonorthogonal":
            config["reference_wrist_rotations_torso"][0][0][0] = 2.
    elif case in ("scale_zero", "scale_bool"):
        config["error_scales"]["position_rmse_m"] = 0. if case == "scale_zero" else True
    elif case in ("weight_negative", "weight_sum", "weight_bool"):
        config["term_weights"]["position_rmse_m"] = {"weight_negative": -.35,
                                                       "weight_sum": .9, "weight_bool": True}[case]
    elif case == "source_nonfinite":
        config["source"]["extra"] = float("nan")
    elif case == "source_tensor":
        config["source"]["extra"] = torch.ones(1)
    with pytest.raises(ValueError):
        AimStyleGuidance(AimKinematics(), config)


@pytest.mark.parametrize("case", ["robot", "com_body", "joints", "duplicate_joint", "missing_wrist",
                                  "missing_torso", "duplicate_track", "negative_track", "bool_track",
                                  "duplicate_body"])
def test_invalid_kinematic_identity_is_rejected(case):
    config, kin = guidance_config(), AimKinematics()
    if case in ("robot", "com_body"):
        kin.robot = SimpleNamespace(name="go2" if case == "robot" else "g1", com_body="pelvis")
    elif case in ("joints", "duplicate_joint"):
        kin.joint_names = list(kin.joint_names)
        kin.joint_names[-1] = "wrong_joint" if case == "joints" else kin.joint_names[0]
    elif case == "missing_wrist":
        kin.tracked_idx = (1, 2, 3)
    elif case == "missing_torso":
        kin.tracked_idx = (0, 1, 3)
    elif case == "duplicate_track":
        kin.tracked_idx = (0, 1, 2, 3, 3)
    elif case == "negative_track":
        kin.tracked_idx = (-1, 0, 2, 3)
    elif case == "bool_track":
        kin.tracked_idx = (False, 1, 2, 3)
    elif case == "duplicate_body":
        kin.body_names = (*kin.body_names[:-1], kin.body_names[0])
    with pytest.raises(ValueError):
        AimStyleGuidance(kin, config)


@pytest.mark.parametrize("case", ["batch", "q_shape", "root_shape", "dtype", "nonfinite", "root_rotation"])
def test_calibration_rejects_invalid_selected_state(case):
    kin = AimKinematics()
    state = list(state_with(reference_q().expand(2, 29) if case == "batch" else reference_q()))
    if case == "q_shape":
        state[4] = state[4][:-1]
    elif case == "root_shape":
        state[0] = torch.zeros(4, dtype=state[4].dtype)
    elif case == "dtype":
        state[5] = state[5].long()
    elif case == "nonfinite":
        state[5][0] = float("nan")
    elif case == "root_rotation":
        state[1] = torch.zeros_like(state[1])
    with pytest.raises(ValueError):
        calibrate_aim_config(kin, state, source())


@pytest.mark.parametrize("bad", [torch.zeros(6), torch.zeros(2, 6), torch.zeros(7, dtype=torch.int64),
                                 torch.zeros(3, 7), [0.] * 7])
def test_invalid_command_shapes_and_types_are_rejected(bad):
    guidance = AimStyleGuidance(AimKinematics(), guidance_config())
    with pytest.raises(ValueError):
        guidance.reward(state_with(perturbed_q((2,))), bad)


def test_source_digest_container_and_custom_reward_settings_are_portable():
    kin = AimKinematics()
    metadata = {"hashes": {"asset": "c" * 64, "pose": "D" * 64}, "frame": 27}
    config = calibrate_aim_config(kin, state_with(reference_q()), metadata, weight=.02, duration=1.2)
    config["error_scales"]["arm_rmse_rad"] = .4
    config["term_weights"] = {name: float(name == "arm_rmse_rad") for name in config["term_weights"]}
    guidance = AimStyleGuidance(kin, config)
    q = reference_q()
    q[guidance.arm_indices] += .4
    torch.testing.assert_close(guidance.reward(state_with(q), command(.6)), q.new_tensor(-.005))
    torch.testing.assert_close(guidance.reward(state_with(q), command(1.2)), q.new_tensor(-.01))
    assert guidance.to_dict() == json.loads(json.dumps(config))

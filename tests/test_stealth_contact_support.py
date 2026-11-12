"""Grounded support must not encourage sinking, while landing gradients remain."""

from pathlib import Path
import sys

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"
sys.path.insert(0, str(SCRIPTS))
from bootstrap import configure_paths
configure_paths()
from train_terrain import parse_args, validate_gait_options, validate_gait_resume
from trackerTask.supermimic.terrain import get_task
from trackerLab.algorithms.supermimic.stealth_gait import command_gait_penalty


def inputs(heights, speeds, *, dtype=torch.float64):
    h = torch.tensor(heights, dtype=dtype).reshape(1, 2, 1).expand(1, 2, 4).clone()
    v = torch.zeros(1, 2, 4, 3, dtype=dtype)
    v[..., 0] = torch.tensor(speeds, dtype=dtype)[None, :, None]
    c = torch.tensor([[0., 0., 0., .55, .665, 0., .8]], dtype=dtype)
    return h, v, c


def reward(h, v, c, *, version=3, scale=1.):
    return command_gait_penalty(
        h, v, c, scale=scale, weight=.02, foot_slide_weight=.04,
        entry_duration_s=.6, version=version, swing_height=.04)


# Frozen outputs and uniform-foot-height derivatives from the adb8f82 implementation.
LEGACY = {
    1: [
        (-.019629796628215578, [-.10685027562784401, -.10685027562784398]),
        (-.01301793243849418, [-2.5913296599051456, .0005853593169191855]),
        (-.10993377802559542, [-.008273419443658288, -.008273419443658288]),
    ],
    2: [
        (-.027524809437211743, [-.008162615515392005, -.008162615515391977]),
        (-.01301793243849418, [-2.5913296599051456, .0005853593169191855]),
        (-.10993377802559542, [-.008273419443658288, -.008273419443658288]),
    ],
}


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("index,heights,speeds", [
    (0, [0., 0.], [.1, .1]), (1, [0., .04], [0., .25]), (2, [.04, .04], [.25, .25]),
])
def test_saved_legacy_versions_keep_values_and_gradients(version, index, heights, speeds):
    h, v, c = inputs(heights, speeds)
    h.requires_grad_()
    r = reward(h, v, c, version=version)
    gradient, = torch.autograd.grad(r.sum(), h)
    value, expected = LEGACY[version][index]
    torch.testing.assert_close(r, r.new_tensor([value]), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradient.sum(-1), r.new_tensor([expected]),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("height", [-.02, -.003, 0., .003, .004])
def test_stationary_planted_foot_has_no_sinking_gradient(height):
    h, v, c = inputs([height, .04], [0., .25])
    h.requires_grad_()
    r = reward(h, v, c)
    gradient, = torch.autograd.grad(r.sum(), h)
    assert torch.isfinite(gradient).all()
    assert gradient[:, 0].sum().item() == 0.
    assert gradient[:, 1].sum() > 0
    reference, reference_v, reference_c = inputs([0., .04], [0., .25])
    torch.testing.assert_close(r, reward(reference, reference_v, reference_c), rtol=0, atol=0)


def test_low_sliding_feet_get_ascent_gradients_instead_of_downward_pressure():
    h, v, c = inputs([0., 0.], [.1, .1])
    h.requires_grad_()
    previous = torch.autograd.grad(reward(h, v, c, version=2).sum(), h, retain_graph=True)[0]
    current = torch.autograd.grad(reward(h, v, c).sum(), h)[0]
    assert (previous.sum(-1) < 0).all()
    assert (current.sum(-1) > .09).all()


@pytest.mark.parametrize("height", [.02, .03, .04, .08])
def test_two_airborne_feet_keep_finite_downward_landing_gradients(height):
    h, v, c = inputs([height, height], [.25, .25])
    h.requires_grad_()
    gradient, = torch.autograd.grad(reward(h, v, c).sum(), h)
    assert torch.isfinite(gradient).all() and (gradient.sum(-1) < 0).all()
    one, one_v, one_c = inputs([0., .04], [0., .25])
    assert reward(h, v, c) < reward(one, one_v, one_c)


def test_idle_entry_and_airborne_slip_compensation_are_preserved():
    h, v, c = inputs([.04, .04], [.25, .25])
    c[:, 3] = 0.
    h.requires_grad_()
    r = reward(h, v, c)
    gradient, = torch.autograd.grad(r.sum(), h)
    assert r.item() == 0. and (gradient == 0).all()
    c[:, 3], c[:, 6] = .55, 0.
    assert reward(h, v, c) < -.009
    ground, _, _ = inputs([0., 0.], [.25, .25])
    assert reward(ground, v, c).item() == 0.
    c[:, 6] = .8
    below, _, _ = inputs([.019999, .019999], [.25, .25])
    above, _, _ = inputs([.020001, .020001], [.25, .25])
    assert reward(above, v, c) <= reward(below, v, c) - .04*.25 + 1e-7


@pytest.mark.parametrize("scale", [.5, 1., 1.5])
def test_morphology_scaling_and_left_right_symmetry(scale):
    h, v, c = inputs([.003, .012], [.02, .4])
    baseline = reward(h, v, c)
    scaled_c = c.clone()
    scaled_c[:, 1:3] *= scale
    scaled_c[:, 4] *= scale
    torch.testing.assert_close(reward(h*scale, v*scale, scaled_c, scale=scale), baseline)
    torch.testing.assert_close(reward(h.flip(1), v.flip(1), c), baseline)
    c[:, 3] *= -1
    torch.testing.assert_close(reward(h, v, c), baseline)


@pytest.mark.parametrize("height", [-.003, .004, .006, .012, .03, .04])
def test_contact_derivatives_match_finite_differences(height):
    h, v, c = inputs([height, .04], [.1, .25])
    delta = torch.zeros(1, 2, dtype=h.dtype, requires_grad=True)
    derivative, = torch.autograd.grad(reward(h+delta[..., None], v, c).sum(), delta)
    epsilon = 1e-8
    for foot in range(2):
        plus, minus = h.clone(), h.clone()
        plus[:, foot] += epsilon
        minus[:, foot] -= epsilon
        numerical = (reward(plus, v, c)-reward(minus, v, c))/(2*epsilon)
        torch.testing.assert_close(numerical, derivative[:, foot], rtol=1e-5, atol=3e-5)


def test_large_height_coordinates_remain_finite():
    for height in (-1e6, 1e6):
        h, v, c = inputs([height, .04], [.1, .25], dtype=torch.float32)
        h.requires_grad_()
        r = reward(h, v, c)
        gradient, = torch.autograd.grad(r.sum(), h)
        assert torch.isfinite(r).all() and (r <= 0).all()
        assert torch.isfinite(gradient).all()


def test_contact_version_is_scoped_and_cannot_silently_resume_version_two():
    args = parse_args([
        "--task", "locomotion/g1/stealth", "--run-dir", "unused-contact",
        "--stealth-gait-version", "3", "--stealth-gait-weight", ".02",
        "--foot-slip-mode", "g1_sole", "--wasd-reward-version", "3"])
    validate_gait_options(args, get_task(args.task))
    saved = {"arguments": {"stealth_gait_weight": .02, "stealth_gait_version": 2,
                           "stealth_swing_height": .04}}
    with pytest.raises(ValueError, match="stealth_gait_version"):
        validate_gait_resume(saved, args)
    saved["arguments"]["stealth_gait_version"] = 3
    validate_gait_resume(saved, args)
    with pytest.raises(ValueError, match="G1 stealth"):
        validate_gait_options(args, get_task("locomotion/g1/velocity"))
    args.stealth_gait_weight = 0.
    with pytest.raises(ValueError, match="positive"):
        validate_gait_options(args, get_task(args.task))


def test_evaluator_and_metadata_use_the_saved_contact_version(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from evaluate_terrain import attach_locomotion_quality
    from export_command_policy import command_training_metadata
    from test_locomotion_sole import Plane, RigidFeetKinematics, sole_xml

    monkeypatch.chdir(tmp_path)
    Path("sole.xml").write_text(sole_xml(), encoding="ascii")
    kin, field = RigidFeetKinematics("sole.xml"), Plane()
    checkpoint = {
        "arguments": {"foot_slip_mode": "g1_sole", "stealth_gait_weight": .02,
                      "stealth_gait_version": 3, "stealth_swing_height": .04,
                      "entry_duration_s": .6, "stealth_control_mode": "wasd",
                      "wasd_reward_version": 3},
        "quality_sole_height": [0., 0.],
    }
    env = SimpleNamespace()
    quality = attach_locomotion_quality(env, kin, field, [(20., 12., 0., 0.)]*6,
                                        checkpoint)
    assert quality.stealth_gait_version == 3
    assert command_training_metadata(checkpoint)["stealth_gait_version"] == 3
    q = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
    qd = torch.tensor([[.1, 0., 0., .1, 0., 0.]], dtype=torch.float64)
    state = (torch.tensor([[0., 0., .035]], dtype=torch.float64),
             torch.eye(3, dtype=torch.float64)[None], torch.zeros(1, 3, dtype=torch.float64),
             torch.zeros(1, 3, dtype=torch.float64), q, qd)
    command = torch.tensor([[0., 0., 0., .55, .665, 0., .8]], dtype=torch.float64)
    action, context = torch.zeros_like(q), torch.zeros(1, 10, dtype=torch.float64)
    value = env.command_locomotion_reward_fn(state, state, action, context, command)
    gradient, = torch.autograd.grad(value.sum(), q)
    assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
    assert (gradient[:, [2, 5]] > .09).all()

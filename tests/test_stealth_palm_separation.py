"""Focused tests for the optional differentiable G1 palm separator."""

import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rls" / "supermimic"))
from bootstrap import configure_paths

configure_paths()

from trackerLab.algorithms.supermimic.stealth_palm_separation import (
    PalmSeparationGuidance,
    calibrate_palm_separation_config,
)


def _box(extent):
    x, y, z = extent
    return [
        [-x, -y, -z], [-x, -y, z], [-x, y, -z], [-x, y, z],
        [x, -y, -z], [x, -y, z], [x, y, -z], [x, y, z],
    ]


def _source():
    return {
        "selected_pose_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "provenance": {"canonical_sha256": "c" * 64},
    }


def _config():
    hulls = [_box((.02, .03, .04)), _box((.02, .03, .04))]
    config = {
        "schema": "g1-stealth-palm-separation-v1",
        "robot": "g1",
        "torso_body": "torso_link",
        "wrist_bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        "hand_meshes": ["left_rubber_hand", "right_rubber_hand"],
        "weight": .006,
        "margin_m": .010,
        "scale_m": .010,
        "separator_normal_torso": [1., 0., 0.],
        "hull_vertices_wrist": hulls,
        "reference_wrist_positions_torso": [[0., 0., 0.], [.06, 0., 0.]],
        "reference_wrist_rotations_torso": [
            np.eye(3).tolist(), np.eye(3).tolist(),
        ],
        "reference_gap_m": .020,
        "reference_geom_distance_m": .020,
        "mesh_provenance": {
            side: {
                "mesh_name": mesh,
                "mesh_vertex_count": 8,
                "hull_vertex_count": 8,
                "mesh_sha256": digest,
                "hull_sha256": hashlib.sha256(
                    np.asarray(hulls[index], dtype="<f8", order="C").tobytes()
                ).hexdigest(),
            }
            for index, (side, mesh, digest) in enumerate((
                ("left", "left_rubber_hand", "d" * 64),
                ("right", "right_rubber_hand", "e" * 64),
            ))
        },
        "calibration": {
            "hull_method": "scipy.spatial.ConvexHull/all-vertices",
            "separator_method": "mujoco.mj_geomDistance/left-to-right",
            "array_hash_encoding": "little-endian float64 C order",
            "model_mjb_sha256": "f" * 64,
            "reference_qpos_sha256": "1" * 64,
            "reference_mocap_sha256": "2" * 64,
            "mujoco_version": "synthetic",
            "scipy_version": "synthetic",
        },
        "source": _source(),
    }
    return config


def _runtime(right_x, *, left_x=0., dtype=torch.float64):
    p = torch.tensor(
        [[left_x, 0., 0.], [right_x, 0., 0.]],
        dtype=dtype,
    )
    R = torch.eye(3, dtype=dtype).expand(2, 3, 3).clone()
    return p, R


def _dense_gap(guidance, wrist_p, wrist_R):
    normal = guidance.separator_normal_torso.to(wrist_p)
    left_vertices = guidance.left_hull_vertices_wrist.to(wrist_p)
    right_vertices = guidance.right_hull_vertices_wrist.to(wrist_p)
    local_normal = wrist_R.transpose(-1, -2) @ normal
    left = local_normal[..., 0, :] @ left_vertices.T
    right = local_normal[..., 1, :] @ right_vertices.T
    translation = ((wrist_p[..., 1, :] - wrist_p[..., 0, :]) * normal).sum(-1)
    return translation + right.amin(-1) - left.amax(-1)


def _generic_rotation(dtype=torch.float64):
    vector = torch.tensor((.37, -.81, .29), dtype=dtype)
    angle = torch.tensor(.47, dtype=dtype)
    axis = vector / vector.norm()
    x, y, z = axis
    c, s = angle.cos(), angle.sin()
    cross = torch.tensor(((0., -z, y), (z, 0., -x), (-y, x, 0.)), dtype=dtype)
    return c * torch.eye(3, dtype=dtype) + (1. - c) * torch.outer(axis, axis) + s * cross


def test_interpenetration_has_separating_position_gradients_and_bounded_reward():
    guidance = PalmSeparationGuidance(_config())
    wrist_p, wrist_R = _runtime(.015)
    before = wrist_p.clone(), wrist_R.clone()
    wrist_p.requires_grad_()
    reward = guidance.reward(wrist_p, wrist_R)
    (gradient,) = torch.autograd.grad(reward, wrist_p)
    assert reward.item() < 0.
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0] < 0.
    assert gradient[1, 0] > 0.
    assert reward.item() >= -.006
    assert guidance(wrist_p.detach(), wrist_R).item() == pytest.approx(reward.item())
    updated = wrist_p.detach() + .01 * gradient
    assert guidance.gap(updated, wrist_R) > guidance.gap(wrist_p, wrist_R)
    assert guidance.reward(updated, wrist_R) > reward
    torch.testing.assert_close(wrist_p.detach(), before[0], rtol=0, atol=0)
    torch.testing.assert_close(wrist_R, before[1], rtol=0, atol=0)


def test_reward_and_gradients_are_zero_at_and_beyond_margin():
    config = _config()
    # Binary fractions make equality at the threshold exact.
    config["margin_m"] = .0625
    guidance = PalmSeparationGuidance(config)
    p, R = _runtime(.04 + .0625)
    actual_margin = guidance.gap(p, R).item()
    config["margin_m"] = actual_margin
    guidance = PalmSeparationGuidance(config)
    p = torch.stack((p, _runtime(.25)[0])).requires_grad_()
    R = R.expand(2, 2, 3, 3).clone().requires_grad_()
    reward = guidance.reward(p, R)
    gradients = torch.autograd.grad(reward.sum(), (p, R))
    assert torch.equal(reward, torch.zeros_like(reward))
    assert all(torch.equal(grad, torch.zeros_like(grad)) for grad in gradients)


@pytest.mark.parametrize("gap", [-.05, -.01, 0., .004, .009, .02])
def test_reward_matches_requested_formula(gap):
    guidance = PalmSeparationGuidance(_config())
    p, R = _runtime(gap + .04)
    violation = max((.010 - gap) / .010, 0.)
    expected = -.006 * violation**2 / (1. + violation**2)
    assert guidance.gap(p, R).item() == pytest.approx(gap, abs=1e-14)
    assert guidance.reward(p, R).item() == pytest.approx(expected, abs=1e-14)


def _z_rotation(angle):
    c, s, z = angle.cos(), angle.sin(), torch.zeros_like(angle)
    return torch.stack((
        torch.stack((c, -s, z)), torch.stack((s, c, z)),
        torch.stack((z, z, torch.ones_like(z))),
    ))


def test_rotating_an_overlapping_hull_has_a_finite_useful_gradient():
    guidance = PalmSeparationGuidance(_config())
    p, R = _runtime(.035)
    angle = torch.tensor(.2, dtype=p.dtype, requires_grad=True)
    R = torch.stack((_z_rotation(angle), R[1]))
    reward = guidance.reward(p, R)
    (gradient,) = torch.autograd.grad(reward, angle)
    assert torch.isfinite(gradient) and gradient < 0.
    updated_R = torch.stack((_z_rotation(angle.detach() + gradient * .1), R[1].detach()))
    assert guidance.reward(p, updated_R) > reward
    assert torch.autograd.gradcheck(
        lambda a: guidance.reward(p, torch.stack((_z_rotation(a), R[1].detach()))),
        (angle,), atol=1e-7,
    )


def test_projection_overlap_can_penalize_hulls_separated_along_another_plane():
    guidance = PalmSeparationGuidance(_config())
    p, R = _runtime(.015)
    p[1, 1] = .2
    assert p[1, 1] - .03 > p[0, 1] + .03
    assert guidance.gap(p, R) < 0.
    assert guidance.reward(p, R) < 0.


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_batches_low_precision_and_large_violations_have_finite_gradients(dtype):
    guidance = PalmSeparationGuidance(_config())
    p, R = _runtime(.015, dtype=dtype)
    p = p.expand(2, 3, 2, 3).clone()
    p[0, 0, 1, 0] = -1e4 if dtype == torch.float16 else -1e30
    p[0, 1, 1, 0] = 1.
    p.requires_grad_()
    R = R.expand(2, 3, 2, 3, 3).clone().requires_grad_()
    reward = guidance.reward(p, R)
    assert reward.shape == (2, 3)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert reward.dtype == expected_dtype and guidance.gap(p, R).dtype == expected_dtype
    assert torch.isfinite(reward).all()
    assert ((reward >= reward.new_tensor(-.006)) & (reward <= 0.)).all()
    assert reward[0, 1] == 0.
    gradients = torch.autograd.grad(reward.sum(), (p, R))
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert gradients[0][1, 2, 0, 0] < 0. < gradients[0][1, 2, 1, 0]


def test_extreme_finite_config_avoids_division_square_and_subtraction_overflow():
    config = _config()
    config.update(margin_m=torch.finfo(torch.float32).max, scale_m=torch.finfo(torch.float32).tiny)
    guidance = PalmSeparationGuidance(config)
    p, R = _runtime(-torch.finfo(torch.float32).max / 2., dtype=torch.float32)
    p.requires_grad_()
    reward = guidance.reward(p, R)
    (gradient,) = torch.autograd.grad(reward, p)
    assert reward == reward.new_tensor(-.006)
    assert torch.isfinite(gradient).all()
    config["margin_m"] = .01
    guidance = PalmSeparationGuidance(config)
    p, R = _runtime(.2, dtype=torch.float32)
    p.requires_grad_()
    reward = guidance.reward(p, R)
    (gradient,) = torch.autograd.grad(reward, p)
    assert reward == 0.
    assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_cpu_autocast_and_fullgraph_compile_preserve_the_reward(backend):
    guidance = PalmSeparationGuidance(_config())
    p, R = _runtime(.035, dtype=torch.float32)
    p = p.expand(4097, 2, 3).clone().requires_grad_()
    R = torch.stack((_generic_rotation(p.dtype), _generic_rotation(p.dtype).T))
    R = R.expand(4097, 2, 3, 3).clone().requires_grad_()
    expected = guidance.reward(p, R)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = guidance.reward(p, R)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    compiled = torch.compile(guidance, backend=backend, fullgraph=True)
    result = compiled(p, R)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected.sum(), (p, R), retain_graph=True)
    actual_grad = torch.autograd.grad(result.sum(), (p, R))
    for actual, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda config: config.update(weight=float("nan")),
        lambda config: config.update(margin_m=float("inf")),
        lambda config: config["hull_vertices_wrist"][0].__setitem__(0, [0., 0., float("nan")]),
        lambda config: config["mesh_provenance"]["left"].update(hull_sha256="0" * 64),
        lambda config: config.update(source={"source_frame": 68}),
        lambda config: config.update(source={"frames": [{"sha256": "invalid"}]}),
        lambda config: config.update(source={"hashes": []}),
        lambda config: config.update(source={"sha256": "invalid"}),
        lambda config: config.update(wrist_bodies=None),
        lambda config: config.update(wrist_bodies=config["wrist_bodies"][::-1]),
        lambda config: config.update(hand_meshes=config["hand_meshes"][::-1]),
        lambda config: config.update(robot="go2"),
        lambda config: config.update(schema="unknown"),
        lambda config: config.update(unused=True),
        lambda config: config.pop("reference_gap_m"),
        lambda config: config.update(separator_normal_torso=[0., 0., 0.]),
        lambda config: config.update(separator_normal_torso=[True, 0., 0.]),
        lambda config: config.update(separator_normal_torso=[1e39, 0., 0.]),
        lambda config: config.update(hull_vertices_wrist=config["hull_vertices_wrist"][:1]),
        lambda config: config.update(hull_vertices_wrist=[[], []]),
        lambda config: config["hull_vertices_wrist"][0].__setitem__(0, [0., 0.]),
        lambda config: config["hull_vertices_wrist"][0].__setitem__(0, [True, 0., 0.]),
        lambda config: config["hull_vertices_wrist"][0].__setitem__(0, config["hull_vertices_wrist"][0][1]),
        lambda config: config["mesh_provenance"]["right"].update(hull_vertex_count=7),
        lambda config: config["mesh_provenance"]["right"].update(mesh_vertex_count=4),
        lambda config: config["mesh_provenance"]["right"].update(mesh_vertex_count=True),
        lambda config: config["mesh_provenance"]["left"].update(mesh_sha256="z" * 64),
        lambda config: config["calibration"].update(model_mjb_sha256=""),
        lambda config: config["calibration"].update(scipy_version=None),
        lambda config: config["calibration"].update(hull_method="sampled"),
        lambda config: config.update(reference_wrist_rotations_torso=[np.zeros((3, 3)).tolist()] * 2),
        lambda config: config.update(reference_wrist_rotations_torso=[np.diag([1., -1., 1.]).tolist()] * 2),
        lambda config: config.update(reference_gap_m=.01),
        lambda config: config["reference_wrist_positions_torso"][1].__setitem__(0, .07),
    ],
)
def test_config_rejects_nonfinite_or_invalid_provenance(mutator):
    config = _config()
    mutator(config)
    with pytest.raises(ValueError):
        PalmSeparationGuidance(config)


@pytest.mark.parametrize("name,value", itertools.product(
    ("weight", "margin_m", "scale_m"),
    (True, False, 0., -.01, float("nan"), float("inf"), -float("inf"),
     "0.01", None, 1e-50, 1e39, 10**400),
))
def test_config_rejects_invalid_scalar_parameters(name, value):
    config = _config()
    config[name] = value
    with pytest.raises(ValueError):
        PalmSeparationGuidance(config)


def test_runtime_rejects_wrong_shapes_and_types():
    guidance = PalmSeparationGuidance(_config())
    p, R = _runtime(.06)
    with pytest.raises(ValueError, match=r"\(\.\.\., 2, 3\)"):
        guidance.gap(torch.zeros(3, 3), R)
    with pytest.raises(ValueError, match=r"\(\.\.\., 2, 3, 3\)"):
        guidance.gap(p, torch.zeros(2, 3, 2))
    with pytest.raises(ValueError, match="batch shapes"):
        guidance.gap(p.unsqueeze(0), R)
    for bad_p, bad_R in ((p.tolist(), R), (p.long(), R), (p, R.tolist()), (p, R.long())):
        with pytest.raises(ValueError, match="floating tensor"):
            guidance.gap(bad_p, bad_R)
    with pytest.raises(ValueError, match="dtype"):
        guidance.gap(p.float(), R)
    with pytest.raises(ValueError, match="device"):
        guidance.gap(p.to("meta"), R)


def test_gap_is_covariant_under_a_rigid_rotation_of_the_torso_frame():
    base = PalmSeparationGuidance(_config())
    angle = torch.tensor(.63, dtype=torch.float64)
    c, s = torch.cos(angle), torch.sin(angle)
    Q = torch.stack((
        torch.stack((c, -s, torch.zeros_like(c))),
        torch.stack((s, c, torch.zeros_like(c))),
        torch.tensor((0., 0., 1.), dtype=torch.float64),
    ))
    rotated_config = _config()
    rotated_config["separator_normal_torso"] = (
        Q @ torch.tensor([1., 0., 0.], dtype=torch.float64)
    ).tolist()
    rotated_config["reference_wrist_positions_torso"] = (
        Q @ torch.tensor(
            rotated_config["reference_wrist_positions_torso"],
            dtype=torch.float64,
        ).T
    ).T.tolist()
    rotated_config["reference_wrist_rotations_torso"] = (
        Q @ torch.tensor(
            rotated_config["reference_wrist_rotations_torso"],
            dtype=torch.float64,
        )
    ).tolist()
    rotated = PalmSeparationGuidance(rotated_config)

    p, R = _runtime(.015)
    rotated_p = (Q @ p.T).T + torch.tensor([.2, -.3, .1], dtype=p.dtype)
    rotated_R = Q @ R
    torch.testing.assert_close(base.gap(p, R), rotated.gap(rotated_p, rotated_R))
    torch.testing.assert_close(base.reward(p, R), rotated.reward(rotated_p, rotated_R))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (2, 5), (4097,), (0, 2)])
def test_chunked_exact_support_matches_dense_oracle_and_generic_gradients(dtype, batch):
    guidance = PalmSeparationGuidance(_config())
    p, _ = _runtime(.015, dtype=dtype)
    R = torch.stack((
        _generic_rotation(dtype),
        _generic_rotation(dtype).transpose(0, 1),
    ))
    local_n = R.transpose(-1, -2) @ guidance.separator_normal_torso.to(p)
    for side, hull in enumerate((guidance.left_hull_vertices_wrist, guidance.right_hull_vertices_wrist)):
        values = (local_n[side] @ hull.to(p).T).sort().values
        assert values[1] - values[0] > 1e-4
        assert values[-1] - values[-2] > 1e-4
    p = p.expand(batch + (2, 3)).clone().requires_grad_()
    R = R.expand(batch + (2, 3, 3)).clone().requires_grad_()
    actual_gap = guidance.gap(p, R)
    dense_gap = _dense_gap(guidance, p, R)
    atol = 1e-8 if dtype == torch.float32 else 1e-14
    torch.testing.assert_close(actual_gap, dense_gap, rtol=0., atol=atol)
    actual_grad = torch.autograd.grad(actual_gap.sum(), (p, R), retain_graph=True)
    dense_grad = torch.autograd.grad(dense_gap.sum(), (p, R), retain_graph=True)
    for actual, wanted in zip(actual_grad, dense_grad):
        torch.testing.assert_close(actual, wanted, rtol=0., atol=atol)
    violation = torch.relu((guidance.margin_m - dense_gap) / guidance.scale_m)
    dense_reward = -guidance.weight * violation.square() / (1. + violation.square())
    actual_reward = guidance.reward(p, R)
    torch.testing.assert_close(actual_reward, dense_reward, rtol=1e-6, atol=atol)
    actual_grad = torch.autograd.grad(actual_reward.sum(), (p, R))
    dense_grad = torch.autograd.grad(dense_reward.sum(), (p, R))
    for actual, wanted in zip(actual_grad, dense_grad):
        torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=atol)


def test_tied_support_chooses_one_stored_vertex_with_a_valid_subgradient():
    guidance = PalmSeparationGuidance(_config())
    p, R = _runtime(.015)
    p.requires_grad_()
    R.requires_grad_()
    gap = guidance.gap(p, R)
    dense = _dense_gap(guidance, p, R)
    torch.testing.assert_close(gap, dense, rtol=0., atol=0.)
    gradient_p, gradient_R = torch.autograd.grad(gap, (p, R))
    assert torch.equal(gradient_p[0], torch.tensor((-1., 0., 0.), dtype=p.dtype))
    assert torch.equal(gradient_p[1], torch.tensor((1., 0., 0.), dtype=p.dtype))
    # For the x-directed plane, four vertices tie on each box. Vertex 4 is the
    # first left maximum, and vertex 0 is the first right minimum.
    expected_R = torch.zeros_like(R)
    expected_R[0, 0] = -guidance.left_hull_vertices_wrist[4]
    expected_R[1, 0] = guidance.right_hull_vertices_wrist[0]
    torch.testing.assert_close(gradient_R, expected_R, rtol=0, atol=0)
    dense_R, = torch.autograd.grad(dense, R)
    assert not torch.equal(gradient_R, dense_R)


def test_to_dict_is_json_safe_and_independent():
    guidance = PalmSeparationGuidance(_config())
    exported = guidance.to_dict()
    assert json.loads(json.dumps(exported, allow_nan=False)) == exported
    exported["source"]["model_sha256"] = "0" * 64
    exported["hull_vertices_wrist"][0][0][0] = 999.
    assert guidance.to_dict()["source"]["model_sha256"] == "b" * 64
    assert guidance.to_dict()["hull_vertices_wrist"][0][0][0] == -.02
    assert copy.deepcopy(guidance.to_dict()) == guidance.to_dict()
    assert list(guidance.parameters()) == []
    assert all(not value.requires_grad for value in guidance.buffers())
    recreated = PalmSeparationGuidance(json.loads(json.dumps(guidance.to_dict())))
    p, R = _runtime(.015)
    torch.testing.assert_close(recreated(p, R), guidance(p, R), rtol=0, atol=0)


@pytest.fixture
def calibration_model():
    """Small in-memory named meshes exercise calibration without private assets."""
    mujoco = pytest.importorskip("mujoco")
    vertices = " ".join(str(v) for point in _box((.02, .03, .04)) for v in point)
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <compiler angle="radian"/>
      <asset>
        <mesh name="left_rubber_hand" vertex="{vertices}" scale="1.2 .8 1.1"/>
        <mesh name="right_rubber_hand" vertex="{vertices}" scale=".9 1.1 .8"/>
      </asset>
      <worldbody>
        <body name="torso_link" pos=".1 -.2 .3" euler=".2 .1 -.3">
          <freejoint/>
          <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
          <body name="left_wrist_yaw_link" pos="-.1 0 0" euler=".1 .2 .3">
            <joint axis="0 0 1"/>
            <geom name="left_palm" type="mesh" mesh="left_rubber_hand"
                  pos=".002 .004 -.003" euler=".3 -.2 .1" contype="0" conaffinity="0"/>
          </body>
          <body name="right_wrist_yaw_link" pos=".1 .01 .02" euler="-.1 -.2 -.3">
            <joint axis="0 0 1"/>
            <geom name="right_palm" type="mesh" mesh="right_rubber_hand"
                  pos="-.003 -.002 .004" euler="-.2 .1 -.3" contype="0" conaffinity="0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """)
    return model, mujoco.MjData(model)


def test_calibration_reads_configured_mesh_transforms_and_preserves_unforwarded_data(calibration_model):
    import mujoco
    from scipy.spatial import ConvexHull

    model, data = calibration_model
    data.qpos[-2:] = (.18, -.13)
    original_qpos, original_xpos = data.qpos.copy(), data.xpos.copy()
    original_mesh = model.mesh_vert.copy()
    original_masks = model.geom_contype.copy(), model.geom_conaffinity.copy()
    config = calibrate_palm_separation_config(model, data, _source())
    assert config == calibrate_palm_separation_config(model, data, _source())
    mujoco.mj_kinematics(model, data)
    expected = calibrate_palm_separation_config(model, data, _source())
    assert expected == config
    for index, name in enumerate(("left_palm", "right_palm")):
        geom = model.geom(name).id
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        local = vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom]
        np.testing.assert_array_equal(
            config["hull_vertices_wrist"][index], local[ConvexHull(local).vertices])
        assert len(config["hull_vertices_wrist"][index]) == 8
    # Restore the deliberately stale test data and verify that calibration itself
    # leaves every observed input array intact.
    data.xpos[:] = original_xpos
    config = calibrate_palm_separation_config(model, data, _source())
    np.testing.assert_array_equal(data.qpos, original_qpos)
    np.testing.assert_array_equal(data.xpos, original_xpos)
    np.testing.assert_array_equal(model.mesh_vert, original_mesh)
    np.testing.assert_array_equal(model.geom_contype, original_masks[0])
    np.testing.assert_array_equal(model.geom_conaffinity, original_masks[1])


@pytest.mark.parametrize("failure", ["overlap", "collision_mask", "nonfinite_qpos", "degenerate_mesh"])
def test_calibration_failures_preserve_the_original_masks_and_state(calibration_model, failure):
    model, data = calibration_model
    if failure == "overlap":
        model.body_pos[model.body("right_wrist_yaw_link").id] = model.body_pos[
            model.body("left_wrist_yaw_link").id]
    elif failure == "collision_mask":
        model.geom_contype[model.geom("left_palm").id] = 1
    elif failure == "nonfinite_qpos":
        data.qpos[0] = float("nan")
    else:
        mesh = model.mesh("left_rubber_hand").id
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        model.mesh_vert[start:start + count] = 0.
    before = tuple(array.copy() for array in (
        model.geom_contype, model.geom_conaffinity, model.mesh_vert, data.qpos, data.xpos))
    with pytest.raises(ValueError):
        calibrate_palm_separation_config(model, data, _source())
    for actual, expected in zip((
        model.geom_contype, model.geom_conaffinity, model.mesh_vert, data.qpos, data.xpos,
    ), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.fixture(scope="module")
def real_g1_reference():
    selected = REPO / "data" / "style-research-20261110" / "owner-aim-canonical-v5" / "selected-pose.csv"
    report_path = selected.with_suffix(".json")
    if not selected.is_file() or not report_path.is_file():
        pytest.skip("Owner selected G1 reference is unavailable")
    try:
        import mujoco
        from dataclasses import replace
        from trackerTask.supermimic.mujoco_terrain import build_model
        from trackerTask.supermimic.robots import get_robot
        from trackerTask.supermimic.terrain import ROBOT_HEIGHTS, generate
    except ImportError as error:
        pytest.skip(f"MuJoCo G1 calibration dependencies are unavailable: {error}")
    if not Path(get_robot("g1").mjcf_path).is_file():
        pytest.skip("Local G1 mesh assets are required")
    terrain = generate("stairs", ROBOT_HEIGHTS["g1"], 0)
    model, spec, _, qids, *_rest = build_model(
        "g1",
        replace(terrain, heights=np.zeros_like(terrain.heights)),
        ground_plane=True,
    )
    data = mujoco.MjData(model)
    pose = np.loadtxt(selected, delimiter=",")
    if pose.shape != (36,):
        raise AssertionError(f"selected-pose.csv has unexpected shape {pose.shape}")
    data.qpos[:3] = pose[:3]
    data.qpos[3:7] = pose[[6, 3, 4, 5]]
    data.qpos[qids] = pose[7:]
    mujoco.mj_forward(model, data)
    report = json.loads(report_path.read_text())
    source = {
        "selected_pose_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "model_sha256": report["model_sha256"],
        "canonical_sha256": report["source"]["canonical_sha256"],
        "source_frame": report["source"]["source_frame"],
    }
    assert source["selected_pose_sha256"] == report["sha256"]
    assert source["model_sha256"] == hashlib.sha256(Path(spec.mjcf_path).read_bytes()).hexdigest()
    return model, data, spec, source, report


@pytest.mark.parametrize("compiled", [False, True])
def test_full_hull_backward_storage_scales_with_samples_not_vertex_count(real_g1_reference, compiled):
    model, data, _, source, _ = real_g1_reference
    config = calibrate_palm_separation_config(model, data, source)
    guidance = PalmSeparationGuidance(config)
    batch = 4097
    p = guidance.reference_wrist_positions_torso.float().expand(batch, 2, 3).clone().requires_grad_()
    R = guidance.reference_wrist_rotations_torso.float().expand(batch, 2, 3, 3).clone().requires_grad_()
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.numel() * tensor.element_size()))
        return tensor

    reward_fn = torch.compile(guidance, backend="aot_eager", fullgraph=True) if compiled else guidance
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        reward = reward_fn(p, R)
    assert saved
    assert sum(size for _, size in saved) < batch * 128
    # The torso-to-wrist matmul can save normals expanded over both wrists.
    assert max(size for _, size in saved) <= batch * 2 * 3 * p.element_size()
    assert all(3191 not in shape and 2966 not in shape for shape, _ in saved)
    gradients = torch.autograd.grad(reward.sum(), (p, R))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_real_reference_hulls_gap_and_visual_masks_are_unchanged(real_g1_reference):
    from scipy.spatial import ConvexHull
    import mujoco

    model, data, spec, source, report = real_g1_reference
    before_contype = model.geom_contype.copy()
    before_conaffinity = model.geom_conaffinity.copy()
    before_mesh = model.mesh_vert.copy()
    before_qpos = data.qpos.copy()
    before_xpos = data.xpos.copy()
    config = calibrate_palm_separation_config(model, data, source)
    assert [item["hull_vertex_count"] for item in config["mesh_provenance"].values()] == [
        3191, 2966,
    ]
    assert [item["mesh_vertex_count"] for item in config["mesh_provenance"].values()] == [
        22876, 21928,
    ]
    assert config["reference_gap_m"] == pytest.approx(
        report["bilateral_hands"]["visual_palm_convex_clearance_m"],
        abs=2e-7,
    )
    assert config["reference_gap_m"] == pytest.approx(.004, abs=2e-7)
    assert config["reference_geom_distance_m"] == pytest.approx(config["reference_gap_m"])
    assert np.array_equal(model.geom_contype, before_contype)
    assert np.array_equal(model.geom_conaffinity, before_conaffinity)
    assert np.array_equal(model.mesh_vert, before_mesh)
    assert np.array_equal(data.qpos, before_qpos)
    assert np.array_equal(data.xpos, before_xpos)
    assert config["source"] == source
    assert config["calibration"]["reference_qpos_sha256"] == hashlib.sha256(
        np.asarray(data.qpos, dtype="<f8").tobytes()).hexdigest()

    guidance = PalmSeparationGuidance(config)
    torso_id = model.body("torso_link").id
    torso_R = data.xmat[torso_id].reshape(3, 3)
    torso_p = data.xpos[torso_id]
    wrist_p, wrist_R = [], []
    for i, name in enumerate(("left_wrist_yaw_link", "right_wrist_yaw_link")):
        body_id = model.body(name).id
        wrist_p.append(torso_R.T @ (data.xpos[body_id] - torso_p))
        wrist_R.append(torso_R.T @ data.xmat[body_id].reshape(3, 3))
        mesh = model.mesh(config["hand_meshes"][i]).id
        geom, = np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_dataid == mesh)
                               & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))
        mesh_p = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        mesh_R = np.empty(9)
        mujoco.mju_quat2Mat(mesh_R, model.geom_quat[geom])
        local = mesh_p @ mesh_R.reshape(3, 3).T + model.geom_pos[geom]
        expected_hull = local[ConvexHull(local).vertices]
        np.testing.assert_array_equal(config["hull_vertices_wrist"][i], expected_hull)
    wrist_p = torch.tensor(np.asarray(wrist_p), dtype=torch.float64)
    wrist_R = torch.tensor(np.asarray(wrist_R), dtype=torch.float64)
    assert guidance.gap(wrist_p, wrist_R).item() == pytest.approx(
        config["reference_gap_m"],
        abs=2e-7,
    )
    assert guidance.reward(wrist_p, wrist_R).item() == pytest.approx(-.006 * .36 / 1.36, abs=1e-10)
    batched_p = wrist_p.float().expand(2, 4, 2, 3).clone().requires_grad_()
    batched_R = wrist_R.float().expand(2, 4, 2, 3, 3)
    reward = guidance.reward(batched_p, batched_R)
    assert reward.shape == (2, 4)
    torch.testing.assert_close(
        guidance.gap(batched_p, batched_R), torch.full((2, 4), .004), rtol=0, atol=1e-7)
    (gradient,) = torch.autograd.grad(reward.sum(), batched_p)
    normal = guidance.separator_normal_torso.float()
    assert ((gradient[..., 1, :] - gradient[..., 0, :]) @ normal > 0).all()
    print(f"\nG1 hulls: left=3191 right=2966; reference gap={config['reference_gap_m']:.16g} m")

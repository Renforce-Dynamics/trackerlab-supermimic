"""Turning-reference symmetry, layout validation, and optional G1 FK checks."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from stealth_turning import mirror_g1_motion, mirror_receipt


JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)  # fmt: skip

MIRROR = np.diag((1.0, -1.0, 1.0))
AXIAL_MIRROR = -MIRROR


def random_motion(frames=7, seed=19):
    rng = np.random.default_rng(seed)
    values = np.empty((frames, 36), dtype=np.float64)
    values[:, :3] = rng.normal(size=(frames, 3))
    values[:, 3:7] = Rotation.random(frames, random_state=rng).as_quat()
    values[:, 7:] = rng.normal(size=(frames, 29))
    return values


def test_receipt_covers_the_complete_g1_layout_and_reflects_each_axis():
    receipt = mirror_receipt(JOINTS)
    assert receipt.source_to_target == (
        6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14,
        22, 23, 24, 25, 26, 27, 28, 15, 16, 17, 18, 19, 20, 21,
    )
    assert receipt.signs == (
        1., -1., -1., 1., 1., -1., 1., -1., -1., 1., 1., -1.,
        -1., -1., 1., 1., -1., -1., 1., -1., 1., -1.,
        1., -1., -1., 1., -1., 1., -1.,
    )
    assert receipt.reflection_matrix == ((1., 0., 0.), (0., -1., 0.), (0., 0., 1.))
    permutation = np.asarray(receipt.source_to_target)
    np.testing.assert_array_equal(permutation[permutation], np.arange(29))
    np.testing.assert_array_equal(np.asarray(receipt.signs)[permutation] * receipt.signs, 1.)
    document = json.loads(json.dumps(receipt.to_dict(), allow_nan=False))
    assert document["joint_names"] == list(JOINTS)
    assert document["source_to_target"] == list(permutation)
    assert document["signs"] == list(receipt.signs)
    assert document["reflection_matrix"] == MIRROR.tolist()


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("seed", [23, 47, 83])
def test_receipt_accepts_a_canonical_name_reorder(container, seed):
    order = np.random.default_rng(seed).permutation(29)
    names = container(JOINTS[index] for index in order)
    before_names = names.copy() if isinstance(names, list) else names
    receipt = mirror_receipt(names)
    assert len(receipt.source_to_target) == len(names)
    assert sorted(receipt.source_to_target) == list(range(len(names)))
    assert receipt.joint_names == tuple(names)

    canonical = random_motion(seed=seed)
    reordered = canonical.copy()
    for column, name in enumerate(names):
        reordered[:, 7 + column] = canonical[:, 7 + JOINTS.index(name)]
    mirrored = mirror_g1_motion(reordered, names)
    expected = mirror_g1_motion(canonical, JOINTS)
    for column, name in enumerate(names):
        expected_column = JOINTS.index(name)
        np.testing.assert_array_equal(mirrored[:, 7 + column], expected[:, 7 + expected_column])
    np.testing.assert_array_equal(mirrored[:, :7], expected[:, :7])
    np.testing.assert_array_equal(mirror_g1_motion(mirrored, names), reordered)
    for source, target in enumerate(receipt.source_to_target):
        np.testing.assert_array_equal(
            mirrored[:, 7 + target], reordered[:, 7 + source] * receipt.signs[source],
        )
    assert names == before_names


def test_motion_reflection_preserves_root_quaternion_representatives_and_is_involutive():
    values = random_motion(frames=128, seed=29)
    values[1::2, 3:7] *= -1.0
    before = values.copy()

    mirrored = mirror_g1_motion(values, JOINTS)
    np.testing.assert_array_equal(values, before)
    np.testing.assert_allclose(mirrored[:, :3], values[:, :3] @ MIRROR.T)
    np.testing.assert_array_equal(mirrored[:, 3:7], values[:, 3:7] * [-1., 1., -1., 1.])
    np.testing.assert_array_equal(mirror_g1_motion(mirrored, JOINTS), values)
    source_rotation = Rotation.from_quat(values[:, 3:7]).as_matrix()
    target_rotation = Rotation.from_quat(mirrored[:, 3:7]).as_matrix()
    np.testing.assert_allclose(
        target_rotation, MIRROR @ source_rotation @ MIRROR, atol=1e-14, rtol=0,
    )
    np.testing.assert_allclose(np.linalg.det(target_rotation), 1., atol=1e-14, rtol=0)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_root_quaternion_continuity_across_half_and_full_turns(axis):
    values = random_motion(frames=513)
    values[:, 3:7] = Rotation.from_euler(axis, np.linspace(-2 * np.pi, 2 * np.pi, 513)).as_quat()
    mirrored = mirror_g1_motion(values, JOINTS)
    source_dot = np.sum(values[1:, 3:7] * values[:-1, 3:7], axis=1)
    target_dot = np.sum(mirrored[1:, 3:7] * mirrored[:-1, 3:7], axis=1)
    np.testing.assert_array_equal(source_dot, target_dot)
    assert target_dot.min() > .999
    np.testing.assert_allclose(
        Rotation.from_quat(mirrored[:, 3:7]).as_matrix(),
        MIRROR @ Rotation.from_quat(values[:, 3:7]).as_matrix() @ MIRROR,
        atol=1e-14, rtol=0,
    )
    np.testing.assert_array_equal(mirror_g1_motion(mirrored, JOINTS), values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_read_only_strided_single_frame_input_returns_independent_float64(dtype):
    storage = random_motion(frames=6)
    if dtype == np.int64:
        storage[:, 3:7] = [0., 0., 0., 1.]
    storage = storage.astype(dtype)
    values = storage[::2]
    snapshot = storage.copy()
    values.setflags(write=False)
    output = mirror_g1_motion(values, list(JOINTS))
    assert output.dtype == np.float64
    assert not np.shares_memory(output, storage)
    np.testing.assert_array_equal(storage, snapshot)
    np.testing.assert_array_equal(mirror_g1_motion(output, JOINTS), values)
    np.testing.assert_array_equal(mirror_g1_motion(values[:1], JOINTS), output[:1])


def test_temporal_root_and_joint_velocities_transform_linearly():
    values = random_motion(frames=19, seed=31)
    dt = .02
    mirrored = mirror_g1_motion(values, JOINTS)
    source_velocity = np.gradient(values, dt, axis=0, edge_order=2)
    mirrored_velocity = np.gradient(mirrored, dt, axis=0, edge_order=2)

    np.testing.assert_allclose(
        mirrored_velocity[:, :3], source_velocity[:, :3] @ MIRROR.T, atol=1e-12, rtol=0,
    )
    np.testing.assert_allclose(
        mirrored_velocity[:, 3:7],
        source_velocity[:, 3:7] * [-1., 1., -1., 1.],
        atol=1e-12,
        rtol=0,
    )
    expected_joint_velocity = np.empty_like(source_velocity[:, 7:])
    receipt = mirror_receipt(JOINTS)
    for source, target in enumerate(receipt.source_to_target):
        expected_joint_velocity[:, target] = source_velocity[:, 7 + source] * receipt.signs[source]
    np.testing.assert_allclose(
        mirrored_velocity[:, 7:], expected_joint_velocity, atol=1e-12, rtol=0,
    )
    source_rotation = Rotation.from_quat(values[:, 3:7])
    target_rotation = Rotation.from_quat(mirrored[:, 3:7])
    source_omega = (source_rotation[1:] * source_rotation[:-1].inv()).as_rotvec() / dt
    target_omega = (target_rotation[1:] * target_rotation[:-1].inv()).as_rotvec() / dt
    np.testing.assert_allclose(target_omega, source_omega @ AXIAL_MIRROR.T, atol=1e-12, rtol=0)


@pytest.mark.parametrize("bad_names", [
    (),
    JOINTS[:5],
    JOINTS[:-1],
    JOINTS + ("neck_pitch_joint",),
    JOINTS[:-1] + ("left_hip_pitch_joint",),
    ("unknown_joint",) + JOINTS[1:],
    JOINTS[:12] + ("waist_yaw_joint_extra",) + JOINTS[13:],
    ("left_hip_pitch_joint_extra",) + JOINTS[1:],
    tuple(name.replace("left_", "middle_", 1) if name.startswith("left_") else name
          for name in JOINTS),
    (None,) + JOINTS[1:],
    (["left_hip_pitch_joint"],) + JOINTS[1:],
    (b"left_hip_pitch_joint",) + JOINTS[1:],
    None,
    " ".join(JOINTS),
    set(JOINTS),
    dict.fromkeys(JOINTS),
    np.asarray(JOINTS),
])
def test_receipt_rejects_incompatible_or_malformed_layouts(bad_names):
    with pytest.raises(ValueError, match="G1 joint_names"):
        mirror_receipt(bad_names)
    with pytest.raises(ValueError, match="G1 joint_names"):
        mirror_g1_motion(random_motion(), bad_names)


@pytest.mark.parametrize("shape", [(), (36,), (0, 36), (2, 35), (2, 37), (1, 2, 36)])
def test_motion_reflection_rejects_malformed_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        mirror_g1_motion(np.zeros(shape), JOINTS)


@pytest.mark.parametrize("dtype", [str, complex, object, bool])
def test_motion_reflection_rejects_nonreal_or_coerced_dtypes(dtype):
    with pytest.raises(ValueError, match="real numeric"):
        mirror_g1_motion(random_motion().astype(dtype), JOINTS)


@pytest.mark.parametrize("column", [0, 4, 35])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_motion_reflection_rejects_nonfinite_values(column, bad_value):
    values = random_motion()
    values[1, column] = bad_value
    with pytest.raises(ValueError, match="finite"):
        mirror_g1_motion(values, JOINTS)


@pytest.mark.parametrize("scale", [0., .5, 2., 1e300, 1. + 2e-5])
def test_motion_reflection_rejects_nonunit_root_quaternions(scale):
    values = random_motion(frames=2, seed=37)
    values[0, 3:7] *= scale
    with pytest.raises(ValueError, match="unit XYZW"):
        mirror_g1_motion(values, JOINTS)


def test_accepted_quaternion_roundoff_is_preserved_without_normalization():
    values = random_motion()
    values[:, 3:7] *= 1. + 5e-6
    mirrored = mirror_g1_motion(values, JOINTS)
    np.testing.assert_array_equal(mirror_g1_motion(mirrored, JOINTS), values)
    np.testing.assert_array_equal(
        np.linalg.norm(mirrored[:, 3:7], axis=1), np.linalg.norm(values[:, 3:7], axis=1),
    )


@pytest.fixture
def installed_g1(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from bootstrap import configure_paths

    configure_paths()
    from trackerTask.supermimic.robots import get_robot

    monkeypatch.chdir(REPO)
    spec = get_robot("g1")
    xml = Path(spec.mjcf_path).resolve()
    if not xml.is_file():
        pytest.skip("Optional configured G1 assets absent")
    # Use only the filename in MuJoCo's Windows fopen path, even on another drive.
    monkeypatch.chdir(xml.parent)
    model = mujoco.MjModel.from_xml_path(xml.name)
    spec = replace(spec, mjcf_path=xml.name)
    assert set(spec.joint_names) == set(JOINTS)
    assert model.nq == 36 and model.nv == 35 and model.njnt == 30
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.jnt_bodyid[0] == model.body(spec.root_body).id
    assert model.jnt_qposadr[0] == 0
    assert sorted(int(model.joint(name).qposadr[0]) for name in JOINTS) == list(range(7, 36))
    assert np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE)
    return mujoco, spec, model


def _random_model_motion(model, spec, frames=128, seed=41):
    rng = np.random.default_rng(seed)
    values = np.empty((frames, 36), dtype=np.float64)
    values[:, :3] = rng.normal(size=(frames, 3)) * [.35, .35, .2] + [0., 0., .8]
    values[:, 3:7] = Rotation.random(frames, random_state=rng).as_quat()
    for column, name in enumerate(spec.joint_names):
        lo, hi = model.joint(name).range
        values[:, 7 + column] = rng.uniform(lo, hi, frames)
        values[:2, 7 + column] = [lo, hi]
    values[:7, 3:7] = Rotation.from_rotvec(np.vstack((
        np.zeros((1, 3)), np.pi * np.eye(3), -np.pi * np.eye(3),
    ))).as_quat()
    return values


def _smooth_model_motion(model, spec, frames=101):
    time = np.arange(frames, dtype=np.float64) * .01
    values = np.zeros((frames, 36), dtype=np.float64)
    values[:, :3] = np.column_stack((
        .15 * np.sin(.8 * time),
        .11 * np.cos(.6 * time),
        .8 + .04 * np.sin(.5 * time),
    ))
    values[:, 3:7] = Rotation.from_euler(
        "xyz",
        np.column_stack((
            .18 * np.sin(.7 * time),
            .14 * np.cos(.55 * time),
            np.pi - .15 + .7 * time,
        )),
    ).as_quat()
    for column, name in enumerate(spec.joint_names):
        lo, hi = model.joint(name).range
        center = .5 * (lo + hi)
        amplitude = .12 * (hi - lo)
        values[:, 7 + column] = center + amplitude * np.sin(.35 * time + .17 * column)
    return values


def _qpos_from_motion(model, spec, value):
    qpos = model.qpos0.copy()
    qpos[:3] = value[:3]
    qpos[3:7] = value[3:7][[3, 0, 1, 2]]
    for column, name in enumerate(spec.joint_names):
        qpos[model.joint(name).qposadr[0]] = value[7 + column]
    return qpos


def _fk_sequence(mujoco, model, spec, values):
    data = mujoco.MjData(model)
    body_positions = np.empty((len(values), model.nbody, 3))
    body_rotations = np.empty((len(values), model.nbody, 3, 3))
    geom_positions = np.empty((len(values), model.ngeom, 3))
    geom_rotations = np.empty((len(values), model.ngeom, 3, 3))
    for frame, value in enumerate(values):
        data.qpos[:] = _qpos_from_motion(model, spec, value)
        mujoco.mj_kinematics(model, data)
        body_positions[frame] = data.xpos
        body_rotations[frame] = data.xmat.reshape(model.nbody, 3, 3)
        geom_positions[frame] = data.geom_xpos
        geom_rotations[frame] = data.geom_xmat.reshape(model.ngeom, 3, 3)
    return body_positions, body_rotations, geom_positions, geom_rotations


def _body_pairs(model):
    ids = {model.body(index).name: index for index in range(model.nbody)}
    assert len(ids) == model.nbody and all(ids)
    pairs = []
    for name, source in ids.items():
        if name.startswith("left_"):
            target_name = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            target_name = "left_" + name[len("right_"):]
        else:
            target_name = name
        assert target_name in ids
        pairs.append((source, ids[target_name]))
    return pairs


def _shoulder_offset(model, body, torso_rotation):
    # rev 1.0 shoulder origins use y=+0.10022/-0.10021 m. Their descendants
    # share this 10 um translation in the torso frame; rotations remain symmetric.
    shoulders = ("left_shoulder_pitch_link", "right_shoulder_pitch_link")
    while body:
        name = model.body(body).name
        if name in shoulders:
            other = model.body(shoulders[1 - shoulders.index(name)]).id
            delta = model.body_pos[body] - MIRROR @ model.body_pos[other]
            np.testing.assert_allclose(delta[[0, 2]], 0., atol=1e-14, rtol=0)
            assert abs(delta[1]) <= 1.01e-5
            return (torso_rotation @ delta[..., None])[..., 0]
        body = int(model.body_parentid[body])
    return np.zeros(torso_rotation.shape[:-2] + (3,))


def _sole_pairs(mujoco, model):
    left_body = model.body("left_ankle_roll_link").id
    right_body = model.body("right_ankle_roll_link").id
    left = np.flatnonzero(
        (model.geom_bodyid == left_body)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    right = np.flatnonzero(
        (model.geom_bodyid == right_body)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    assert len(left) == len(right) == 4
    assert np.all(model.geom_type[np.r_[left, right]] == mujoco.mjtGeom.mjGEOM_SPHERE)

    remaining = set(int(index) for index in right)
    pairs = []
    for source in left:
        expected_local = MIRROR @ model.geom_pos[source]
        target = min(remaining, key=lambda index: np.linalg.norm(
            model.geom_pos[index] - expected_local
        ))
        remaining.remove(target)
        np.testing.assert_allclose(model.geom_pos[target], expected_local, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            model.geom_size[target], model.geom_size[source], atol=1e-12, rtol=0,
        )
        pairs.append((int(source), target))
    assert not remaining
    return pairs + [(target, source) for source, target in pairs]


def _angular_velocity(rotation, dt):
    derivative = np.gradient(rotation, dt, axis=0, edge_order=2)
    skew = derivative @ np.swapaxes(rotation, -1, -2)
    return .5 * np.stack((
        skew[..., 2, 1] - skew[..., 1, 2],
        skew[..., 0, 2] - skew[..., 2, 0],
        skew[..., 1, 0] - skew[..., 0, 1],
    ), axis=-1)


def test_installed_g1_joint_axes_limits_and_bind_frames_match_the_contract(installed_g1):
    mujoco, spec, model = installed_g1
    receipt = mirror_receipt(spec.joint_names)
    for source, target in enumerate(receipt.source_to_target):
        left = model.joint(spec.joint_names[source])
        right = model.joint(spec.joint_names[target])
        sign = receipt.signs[source]
        np.testing.assert_allclose(sign * right.axis, AXIAL_MIRROR @ left.axis, atol=1e-14, rtol=0)
        np.testing.assert_allclose(right.pos, MIRROR @ left.pos, atol=1e-14, rtol=0)
        np.testing.assert_allclose(right.range, np.sort(sign * left.range), atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            model.qpos0[right.qposadr], sign * model.qpos0[left.qposadr], atol=1e-14, rtol=0,
        )
    partners = dict(_body_pairs(model))
    for source, target in partners.items():
        assert partners[int(model.body_parentid[source])] == model.body_parentid[target]
        tolerance = 1.01e-5 if model.body(source).name.endswith("shoulder_pitch_link") else 1e-14
        np.testing.assert_allclose(
            model.body_pos[target], MIRROR @ model.body_pos[source], atol=tolerance, rtol=0,
        )
        rotations = Rotation.from_quat(model.body_quat[[source, target]][:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(rotations[1], MIRROR @ rotations[0] @ MIRROR, atol=1e-14, rtol=0)
    assert len(_sole_pairs(mujoco, model)) == 8


@pytest.mark.parametrize("reordered", [False, True])
def test_installed_g1_fk_matches_reflection_for_all_bodies_and_sole_geometry(installed_g1, reordered):
    mujoco, spec, model = installed_g1
    values = _random_model_motion(model, spec)
    if reordered:
        order = np.random.default_rng(97).permutation(29)
        values[:, 7:] = values[:, 7:][:, order]
        spec = replace(spec, joint_names=tuple(spec.joint_names[index] for index in order))
    mirrored = mirror_g1_motion(values, spec.joint_names)
    inverse = mirror_g1_motion(mirrored, spec.joint_names)
    np.testing.assert_array_equal(inverse, values)
    for column, name in enumerate(spec.joint_names):
        lo, hi = model.joint(name).range
        assert np.all((mirrored[:, 7 + column] >= lo) & (mirrored[:, 7 + column] <= hi))
    source = _fk_sequence(mujoco, model, spec, values)
    target = _fk_sequence(mujoco, model, spec, mirrored)
    recovered = _fk_sequence(mujoco, model, spec, inverse)
    for before, after in zip(source, recovered):
        np.testing.assert_array_equal(before, after)

    source_positions, source_rotations, source_geom_positions, source_geom_rotations = source
    target_positions, target_rotations, target_geom_positions, target_geom_rotations = target
    torso_rotation = target_rotations[:, model.body(spec.com_body).id]
    for source_body, target_body in _body_pairs(model):
        offset = _shoulder_offset(model, target_body, torso_rotation)
        np.testing.assert_allclose(
            target_positions[:, target_body],
            source_positions[:, source_body] @ MIRROR.T + offset,
            atol=2e-12,
            rtol=0,
            err_msg=model.body(target_body).name,
        )
        expected_rotation = np.einsum(
            "ij,tjk,kl->til", MIRROR, source_rotations[:, source_body], MIRROR,
        )
        np.testing.assert_allclose(
            target_rotations[:, target_body], expected_rotation, atol=2e-12, rtol=0,
            err_msg=model.body(target_body).name,
        )

    for source_geom, target_geom in _sole_pairs(mujoco, model):
        np.testing.assert_allclose(
            target_geom_positions[:, target_geom],
            source_geom_positions[:, source_geom] @ MIRROR.T,
            atol=2e-12,
            rtol=0,
        )
        expected_rotation = np.einsum(
            "ij,tjk,kl->til", MIRROR, source_geom_rotations[:, source_geom], MIRROR,
        )
        np.testing.assert_allclose(
            target_geom_rotations[:, target_geom], expected_rotation, atol=2e-12, rtol=0,
        )
        pole = np.array([0., 0., -model.geom_size[source_geom, 0]])
        np.testing.assert_allclose(
            target_geom_positions[:, target_geom] + pole,
            (source_geom_positions[:, source_geom] + pole) @ MIRROR.T,
            atol=2e-12, rtol=0,
        )


def test_installed_g1_fk_temporal_velocities_reflect_for_bodies_and_sole_geometry(installed_g1):
    mujoco, spec, model = installed_g1
    dt = .01
    values = _smooth_model_motion(model, spec)
    mirrored = mirror_g1_motion(values, spec.joint_names)
    source = _fk_sequence(mujoco, model, spec, values)
    target = _fk_sequence(mujoco, model, spec, mirrored)

    source_positions, source_rotations, source_geom_positions, source_geom_rotations = source
    target_positions, target_rotations, target_geom_positions, target_geom_rotations = target
    source_body_linear = np.gradient(source_positions, dt, axis=0, edge_order=2)
    target_body_linear = np.gradient(target_positions, dt, axis=0, edge_order=2)
    source_body_angular = _angular_velocity(source_rotations, dt)
    target_body_angular = _angular_velocity(target_rotations, dt)
    torso_rotation = target_rotations[:, model.body(spec.com_body).id]
    for source_body, target_body in _body_pairs(model):
        offset = _shoulder_offset(model, target_body, torso_rotation)
        np.testing.assert_allclose(
            target_body_linear[:, target_body],
            source_body_linear[:, source_body] @ MIRROR.T
            + np.gradient(offset, dt, axis=0, edge_order=2),
            atol=2e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            target_body_angular[:, target_body],
            source_body_angular[:, source_body] @ AXIAL_MIRROR.T,
            atol=2e-10,
            rtol=0,
        )

    source_geom_linear = np.gradient(source_geom_positions, dt, axis=0, edge_order=2)
    target_geom_linear = np.gradient(target_geom_positions, dt, axis=0, edge_order=2)
    source_geom_angular = _angular_velocity(source_geom_rotations, dt)
    target_geom_angular = _angular_velocity(target_geom_rotations, dt)
    for source_geom, target_geom in _sole_pairs(mujoco, model):
        np.testing.assert_allclose(
            target_geom_linear[:, target_geom],
            source_geom_linear[:, source_geom] @ MIRROR.T,
            atol=2e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            target_geom_angular[:, target_geom],
            source_geom_angular[:, source_geom] @ AXIAL_MIRROR.T,
            atol=2e-10,
            rtol=0,
        )
        pole = np.array([0., 0., -model.geom_size[source_geom, 0]])
        source_material_velocity = source_geom_linear[:, source_geom] + np.cross(
            source_geom_angular[:, source_geom], pole,
        )
        target_material_velocity = target_geom_linear[:, target_geom] + np.cross(
            target_geom_angular[:, target_geom], pole,
        )
        np.testing.assert_allclose(
            target_material_velocity, source_material_velocity @ MIRROR.T,
            atol=2e-10, rtol=0,
        )

    source_joint_velocity = np.gradient(values[:, 7:], dt, axis=0, edge_order=2)
    target_joint_velocity = np.gradient(mirrored[:, 7:], dt, axis=0, edge_order=2)
    expected_joint_velocity = np.empty_like(source_joint_velocity)
    receipt = mirror_receipt(spec.joint_names)
    for source_joint, target_joint in enumerate(receipt.source_to_target):
        expected_joint_velocity[:, target_joint] = (
            source_joint_velocity[:, source_joint] * receipt.signs[source_joint]
        )
    np.testing.assert_allclose(target_joint_velocity, expected_joint_velocity, atol=1e-12, rtol=0)

"""Optional differentiable separation guidance for the configured G1 palms.

Calibration is intentionally separate from the runtime module. It reads the
already-configured MuJoCo model/data, freezes a torso-frame separator plane,
and stores all vertices of each rubber-hand convex hull. Runtime only evaluates
two support functions, so it does not perform pairwise vertex distances or
change actions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn


_SCHEMA = "g1-stealth-palm-separation-v1"
_TORSO_BODY = "torso_link"
_WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_HAND_MESHES = ("left_rubber_hand", "right_rubber_hand")
_DEFAULT_WEIGHT = 0.006
_DEFAULT_MARGIN = 0.010
_DEFAULT_SCALE = 0.010
_FLOAT32_MAX = float(torch.finfo(torch.float32).max)
_FLOAT32_TINY = float(torch.finfo(torch.float32).tiny)
# Batch rows per temporary projection; all hull vertices participate in each row.
_SUPPORT_SEARCH_CHUNK = 4096


def _json_copy(value, name="config"):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)):
        raise ValueError(f"{name} must be a SHA256 hex digest string")


def _source(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("source must contain palm calibration provenance")
    source = _json_copy(value, "source")

    def visit(document, prefix):
        if isinstance(document, list):
            return sum(visit(item, f"{prefix}[{i}]") for i, item in enumerate(document))
        if not isinstance(document, dict):
            return 0
        count = 0
        for key, item in document.items():
            name = f"{prefix}.{key}"
            if key == "sha256" or key.endswith("_sha256"):
                _digest(item, name)
                count += 1
            elif key == "hashes":
                if not isinstance(item, dict) or not item:
                    raise ValueError(f"{name} must map names to SHA256 digest strings")
                for label, digest_value in item.items():
                    _digest(digest_value, f"{name}.{label}")
                    count += 1
            elif isinstance(item, (dict, list)):
                count += visit(item, name)
        return count

    if not visit(source, "source"):
        raise ValueError("source must contain a sha256, *_sha256 or hashes digest")
    return source


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    try:
        represented = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(represented) or not _FLOAT32_TINY <= represented <= _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and representable as positive normal float32")
    return represented


def _real_nested(value):
    if isinstance(value, (list, tuple)):
        return all(_real_nested(item) for item in value)
    return isinstance(value, Real) and not isinstance(value, bool)


def _array(value, shape, name):
    if not _real_nested(value):
        raise ValueError(f"{name} must contain finite real numbers")
    try:
        result = torch.tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must have shape {shape}") from error
    if (tuple(result.shape) != tuple(shape)
            or not bool(torch.isfinite(result).all())
            or not bool(torch.isfinite(result.float()).all())):
        raise ValueError(f"{name} must have shape {shape} and finite float32 values")
    return result


def _proper_rotations(value, name):
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    if (not bool(torch.allclose(
            value.transpose(-1, -2) @ value,
            identity.expand_as(value),
            rtol=0.,
            atol=1e-5,
    )) or not bool(torch.allclose(
        torch.linalg.det(value),
        torch.ones_like(value[..., 0, 0]),
        rtol=0.,
        atol=1e-5,
    ))):
        raise ValueError(f"{name} must contain proper rotation matrices")


def _unit_vector(value, name):
    length = torch.linalg.vector_norm(value)
    if (not bool(torch.isfinite(length)) or abs(float(length) - 1.) > 1e-5):
        raise ValueError(f"{name} must be a finite unit three-vector")


def _hash_array(value):
    array = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_count(value, name, minimum=1):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_runtime_inputs(wrist_p, wrist_R):
    if not isinstance(wrist_p, torch.Tensor) or not wrist_p.is_floating_point():
        raise ValueError("wrist_p must be a floating tensor")
    if not isinstance(wrist_R, torch.Tensor) or not wrist_R.is_floating_point():
        raise ValueError("wrist_R must be a floating tensor")
    if wrist_p.ndim < 2 or wrist_p.shape[-2:] != (2, 3):
        raise ValueError("wrist_p must have shape (..., 2, 3)")
    if wrist_R.ndim < 3 or wrist_R.shape[-3:] != (2, 3, 3):
        raise ValueError("wrist_R must have shape (..., 2, 3, 3)")
    if wrist_p.shape[:-2] != wrist_R.shape[:-3]:
        raise ValueError("wrist_p and wrist_R batch shapes must match")
    if wrist_p.device != wrist_R.device or wrist_p.dtype != wrist_R.dtype:
        raise ValueError("wrist_p and wrist_R must share device and dtype")


def _support_from_witness(local_normal, vertices, *, maximize):
    """Select an exact support vertex without retaining projection matrices.

    Chunk the batch, scanning every stored hull vertex for every sample.
    argmax/argmin choose the first stored vertex at a tie; its gradient is
    one valid support subgradient, rather than the average over tied vertices.
    Only the selected witnesses enter the differentiable dot products.
    """
    flat_normal = local_normal.reshape(-1, 3)
    vertices = vertices.to(device=flat_normal.device, dtype=flat_normal.dtype)
    with torch.no_grad():
        search_normal = flat_normal.detach()
        vertex_transpose = vertices.detach().transpose(0, 1)
        indices = []
        for start in range(0, search_normal.shape[0], _SUPPORT_SEARCH_CHUNK):
            projections = search_normal[start:start + _SUPPORT_SEARCH_CHUNK] @ vertex_transpose
            indices.append(projections.argmax(-1) if maximize else projections.argmin(-1))
            del projections
        best_index = (torch.cat(indices) if indices else
                      torch.empty(0, dtype=torch.long, device=search_normal.device))
        witness = vertices.index_select(0, best_index)
    return (flat_normal * witness).sum(dim=-1).reshape(local_normal.shape[:-1])


def _model_ids(model):
    try:
        torso_id = int(model.body(_TORSO_BODY).id)
        wrist_ids = tuple(int(model.body(name).id) for name in _WRIST_BODIES)
        mesh_ids = tuple(int(model.mesh(name).id) for name in _HAND_MESHES)
    except (AttributeError, KeyError, RuntimeError, ValueError) as error:
        raise ValueError(
            "The MuJoCo model must provide the configured G1 torso, wrists and rubber hands"
        ) from error
    return torso_id, wrist_ids, mesh_ids


def _hand_geom(model, wrist_id, mesh_id, side):
    import mujoco

    matches = np.flatnonzero(
        (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
        & (model.geom_dataid == mesh_id)
        & (model.geom_bodyid == wrist_id)
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one {side} rubber-hand visual mesh on its wrist body")
    geom_id = int(matches[0])
    if (int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0):
        raise ValueError(f"{side} rubber-hand mesh must remain visual-only")
    return geom_id


def _body_local_mesh(model, mesh_id, geom_id):
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
    rotation = np.empty(9, dtype=np.float64)
    import mujoco

    mujoco.mju_quat2Mat(rotation, np.asarray(model.geom_quat[geom_id], dtype=float))
    return vertices @ rotation.reshape(3, 3).T + np.asarray(model.geom_pos[geom_id])


def _body_pose(data, body_id):
    position = np.asarray(data.xpos[body_id], dtype=np.float64)
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    if (position.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(position).all() or not np.isfinite(rotation).all()):
        raise ValueError("MuJoCo data must contain finite forwarded body poses")
    return position, rotation


def _reference_support_gap(hulls, wrist_positions, wrist_rotations, normal):
    projections = []
    for vertices, position, rotation in zip(hulls, wrist_positions, wrist_rotations):
        world_vertices = position + vertices @ rotation.T
        projections.append(world_vertices @ normal)
    return float(np.min(projections[1]) - np.max(projections[0]))


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Palm separation config must be a JSON object")
    config = _json_copy(config)
    expected = {
        "schema", "robot", "torso_body", "wrist_bodies", "hand_meshes",
        "weight", "margin_m", "scale_m", "separator_normal_torso",
        "hull_vertices_wrist", "reference_wrist_positions_torso",
        "reference_wrist_rotations_torso", "reference_gap_m",
        "reference_geom_distance_m", "mesh_provenance", "calibration", "source",
    }
    unknown = set(config) - expected
    missing = expected - set(config)
    if missing:
        raise ValueError(f"Palm separation config is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Palm separation config has unsupported fields: {sorted(unknown)}")
    if config["schema"] != _SCHEMA:
        raise ValueError("Unsupported palm separation schema")
    if config["robot"] != "g1" or config["torso_body"] != _TORSO_BODY:
        raise ValueError("Palm separation requires the configured g1 torso_link")
    if config["wrist_bodies"] != list(_WRIST_BODIES):
        raise ValueError("wrist_bodies must be the ordered G1 wrist_yaw links")
    if config["hand_meshes"] != list(_HAND_MESHES):
        raise ValueError("hand_meshes must be the ordered G1 rubber-hand meshes")

    weight = _positive(config["weight"], "weight")
    margin = _positive(config["margin_m"], "margin_m")
    scale = _positive(config["scale_m"], "scale_m")
    normal = _array(config["separator_normal_torso"], (3,), "separator_normal_torso")
    _unit_vector(normal, "separator_normal_torso")
    reference_p = _array(
        config["reference_wrist_positions_torso"],
        (2, 3),
        "reference_wrist_positions_torso",
    )
    reference_R = _array(
        config["reference_wrist_rotations_torso"],
        (2, 3, 3),
        "reference_wrist_rotations_torso",
    )
    _proper_rotations(reference_R, "reference_wrist_rotations_torso")
    reference_gap = _positive(config["reference_gap_m"], "reference_gap_m")
    reference_distance = _positive(
        config["reference_geom_distance_m"],
        "reference_geom_distance_m",
    )
    if not math.isclose(reference_gap, reference_distance, rel_tol=1e-5, abs_tol=2e-6):
        raise ValueError("reference gap and MuJoCo closest-point distance must agree")

    if (not isinstance(config["hull_vertices_wrist"], list)
            or len(config["hull_vertices_wrist"]) != 2):
        raise ValueError("hull_vertices_wrist must contain exactly two ordered hulls")
    hulls = []
    for side, value in zip(("left", "right"), config["hull_vertices_wrist"]):
        if not isinstance(value, list) or len(value) < 4:
            raise ValueError(f"{side} hull must contain at least four vertices")
        hull = _array(value, (len(value), 3), f"hull_vertices_wrist.{side}")
        if (len(torch.unique(hull, dim=0)) != len(hull)
                or torch.linalg.matrix_rank(hull - hull[0]).item() != 3):
            raise ValueError(f"{side} hull must contain distinct vertices spanning 3D")
        hulls.append(hull)

    provenance = config["mesh_provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"left", "right"}:
        raise ValueError("mesh_provenance must contain left and right records")
    for side, hull in zip(("left", "right"), hulls):
        record = provenance[side]
        required = {
            "mesh_name", "mesh_vertex_count", "hull_vertex_count",
            "mesh_sha256", "hull_sha256",
        }
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"mesh_provenance.{side} has an invalid field set")
        if record["mesh_name"] != _HAND_MESHES[0 if side == "left" else 1]:
            raise ValueError(f"mesh_provenance.{side}.mesh_name is not a G1 rubber hand")
        mesh_count = _validate_count(
            record["mesh_vertex_count"], f"mesh_provenance.{side}.mesh_vertex_count", minimum=4)
        hull_count = _validate_count(
            record["hull_vertex_count"],
            f"mesh_provenance.{side}.hull_vertex_count",
            minimum=4,
        )
        if hull_count != hull.shape[0] or hull_count > mesh_count:
            raise ValueError(f"mesh_provenance.{side}.hull_vertex_count disagrees with hull")
        _digest(record["mesh_sha256"], f"mesh_provenance.{side}.mesh_sha256")
        _digest(record["hull_sha256"], f"mesh_provenance.{side}.hull_sha256")
        if record["hull_sha256"].lower() != _hash_array(hull.numpy()):
            raise ValueError(f"mesh_provenance.{side}.hull_sha256 disagrees with hull vertices")
    source = _source(config["source"])
    calibration = config["calibration"]
    if (not isinstance(calibration, dict)
            or calibration.get("hull_method") != "scipy.spatial.ConvexHull/all-vertices"
            or calibration.get("separator_method") != "mujoco.mj_geomDistance/left-to-right"
            or calibration.get("array_hash_encoding") != "little-endian float64 C order"):
        raise ValueError("calibration must identify hull, separator and hash conventions")
    for name in ("model_mjb_sha256", "reference_qpos_sha256", "reference_mocap_sha256"):
        _digest(calibration.get(name), f"calibration.{name}")
    for name in ("mujoco_version", "scipy_version"):
        if not isinstance(calibration.get(name), str) or not calibration[name]:
            raise ValueError(f"calibration.{name} must identify the calibration runtime")
    actual_gap = _reference_support_gap(
        [hull.numpy() for hull in hulls], reference_p.numpy(),
        reference_R.numpy(), normal.numpy(),
    )
    if not math.isclose(actual_gap, reference_gap, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("reference_gap_m disagrees with the stored hulls and reference poses")

    config.update(
        schema=_SCHEMA,
        robot="g1",
        torso_body=_TORSO_BODY,
        wrist_bodies=list(_WRIST_BODIES),
        hand_meshes=list(_HAND_MESHES),
        weight=weight,
        margin_m=margin,
        scale_m=scale,
        separator_normal_torso=normal.tolist(),
        reference_wrist_positions_torso=reference_p.tolist(),
        reference_wrist_rotations_torso=reference_R.tolist(),
        reference_gap_m=reference_gap,
        reference_geom_distance_m=reference_distance,
        source=source,
    )
    return config, normal, tuple(hulls), reference_p, reference_R


class PalmSeparationGuidance(nn.Module):
    """Optional bounded negative palm-separation reward.

    Positions have shape (..., 2, 3), rotations (..., 2, 3, 3), both in
    torso_link coordinates and ordered left/right wrist_yaw. Rotations map
    wrist-local vectors into torso coordinates. Output shape is (...,).
    Runtime poses must be finite proper rigid transforms, as for the caller's
    FK contract; value checks stay outside the compiled training path.

    Support searches are detached and batch-chunked, with the two hulls
    processed sequentially. Backward retains one witness vertex per wrist
    and sample, never the full projection matrices. Ties use the first stored
    extreme vertex, one valid subgradient rather than an average over ties.

    A positive gap certifies convex separation along the frozen source plane.
    A negative gap is only projection overlap: hulls can be separated along
    another plane. This conservative reward has no action or command interface;
    callers apply entry_blend themselves. Omitting it leaves existing rewards
    unchanged.
    """

    def __init__(self, config):
        super().__init__()
        (
            config,
            normal,
            hulls,
            reference_p,
            reference_R,
        ) = _validate_config(config)
        self.weight = float(config["weight"])
        self.margin_m = float(config["margin_m"])
        self.scale_m = float(config["scale_m"])
        self.register_buffer("separator_normal_torso", normal)
        self.register_buffer("left_hull_vertices_wrist", hulls[0])
        self.register_buffer("right_hull_vertices_wrist", hulls[1])
        self.register_buffer("reference_wrist_positions_torso", reference_p)
        self.register_buffer("reference_wrist_rotations_torso", reference_R)
        self._config = config

    def gap(self, wrist_p, wrist_R):
        """Signed right-minimum minus left-maximum support gap, in meters.

        Support indices are selected from detached full-hull projections in
        bounded sequential chunks. The differentiable path gathers one exact
        witness vertex per sample and evaluates its dot product.
        """
        _validate_runtime_inputs(wrist_p, wrist_R)
        with torch.autocast(device_type=wrist_p.device.type, enabled=False):
            if wrist_p.dtype in (torch.float16, torch.bfloat16):
                wrist_p, wrist_R = wrist_p.float(), wrist_R.float()
            normal = self.separator_normal_torso.to(wrist_p)
            local_normals = wrist_R.transpose(-1, -2) @ normal
            # Each side is searched and released before the other side starts.
            left = _support_from_witness(
                local_normals[..., 0, :],
                self.left_hull_vertices_wrist,
                maximize=True,
            )
            right = _support_from_witness(
                local_normals[..., 1, :],
                self.right_hull_vertices_wrist,
                maximize=False,
            )
            translation = ((wrist_p[..., 1, :] - wrist_p[..., 0, :]) * normal).sum(-1)
            return translation + right - left

    def reward(self, wrist_p, wrist_R):
        """Return a bounded nonpositive penalty; callers apply entry blending."""
        gap = self.gap(wrist_p, wrist_R)
        # Halve before subtraction, and use hypot before division: even a large
        # finite margin/error or tiny scale cannot overflow an intermediate square.
        violation = torch.relu(gap.new_tensor(self.margin_m * .5) - gap * .5)
        bounded_square = (
            violation / torch.hypot(violation, gap.new_tensor(self.scale_m * .5))
        ).square().clamp(0., 1.)
        return -self.weight * bounded_square

    def forward(self, wrist_p, wrist_R):
        return self.reward(wrist_p, wrist_R)

    def to_dict(self):
        """Return independent JSON metadata suitable for a checkpoint."""
        return copy.deepcopy(self._config)


def calibrate_palm_separation_config(
    model,
    data,
    source,
    *,
    weight=_DEFAULT_WEIGHT,
    margin=_DEFAULT_MARGIN,
    scale=_DEFAULT_SCALE,
):
    """Calibrate a frozen G1 separator from the current MuJoCo reference pose.

    The model must expose the configured visual ``left_rubber_hand`` and
    ``right_rubber_hand`` meshes on the two G1 wrist bodies. ``data`` must
    contain the selected reference qpos. Kinematics runs on a private MjData
    copy; input arrays, visual collision masks and source files are untouched.
    Mesh hashes cover all compiled vertices transformed into wrist coordinates;
    hull hashes cover every ConvexHull vertex in its returned order.

    ``source`` is caller-owned JSON provenance with at least one explicit
    SHA256 (e.g. selected_pose_sha256 and model_sha256). Additional calibration
    hashes identify the actual compiled model, reference qpos and mocap state.
    The returned dictionary can be stored under AimStyleGuidance's optional
    ``palm_separation`` key. All three reward parameters are in reward/meters
    units as appropriate, with defaults .006/.010/.010.
    """
    import mujoco
    import scipy
    from scipy.spatial import ConvexHull, QhullError

    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise ValueError("model and data must be MuJoCo MjModel and MjData objects")
    source = _source(source)
    weight = _positive(weight, "weight")
    margin = _positive(margin, "margin")
    scale = _positive(scale, "scale")
    torso_id, wrist_ids, mesh_ids = _model_ids(model)
    if (data.qpos.shape != (model.nq,) or data.xpos.shape != (model.nbody, 3)
            or data.mocap_pos.shape != (model.nmocap, 3)
            or data.mocap_quat.shape != (model.nmocap, 4)
            or any(not np.isfinite(value).all()
                   for value in (data.qpos, data.mocap_pos, data.mocap_quat))):
        raise ValueError("data must contain a finite reference state for the configured model")
    reference = mujoco.MjData(model)
    reference.qpos[:] = data.qpos
    reference.mocap_pos[:] = data.mocap_pos
    reference.mocap_quat[:] = data.mocap_quat
    mujoco.mj_kinematics(model, reference)

    torso_position, torso_rotation = _body_pose(reference, torso_id)
    wrist_world_positions, wrist_world_rotations = [], []
    hulls, geoms, records = [], [], {}
    for side, wrist_id, mesh_id, mesh_name in zip(
        ("left", "right"),
        wrist_ids,
        mesh_ids,
        _HAND_MESHES,
    ):
        geom_id = _hand_geom(model, wrist_id, mesh_id, side)
        geoms.append(geom_id)
        mesh_vertices = _body_local_mesh(model, mesh_id, geom_id)
        try:
            hull = ConvexHull(mesh_vertices)
        except (QhullError, ValueError) as error:
            raise ValueError(f"{side} rubber-hand mesh has no 3D convex hull") from error
        hull_vertices = mesh_vertices[np.asarray(hull.vertices, dtype=np.int64)]
        hulls.append(hull_vertices)
        wrist_position, wrist_rotation = _body_pose(reference, wrist_id)
        wrist_world_positions.append(wrist_position)
        wrist_world_rotations.append(wrist_rotation)
        records[side] = {
            "mesh_name": mesh_name,
            "mesh_vertex_count": int(mesh_vertices.shape[0]),
            "hull_vertex_count": int(hull_vertices.shape[0]),
            "mesh_sha256": _hash_array(mesh_vertices),
            "hull_sha256": _hash_array(hull_vertices),
        }

    segment = np.zeros(6, dtype=np.float64)
    search_distance = float(
        np.linalg.norm(reference.geom_xpos[geoms[1]] - reference.geom_xpos[geoms[0]])
        + sum(model.geom_rbound[geoms]) + 1.
    )
    distance = float(mujoco.mj_geomDistance(
        model, reference, *geoms, search_distance, segment,
    ))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("Reference rubber-hand meshes must have a positive closest-point gap")
    closest_points = segment.reshape(2, 3)
    normal_world = closest_points[1] - closest_points[0]
    normal_length = float(np.linalg.norm(normal_world))
    if (not math.isfinite(normal_length) or normal_length <= 1e-12
            or not math.isclose(normal_length, distance, rel_tol=1e-5, abs_tol=2e-6)):
        raise ValueError("Reference closest points do not define a separator normal")
    normal_world /= normal_length
    normal_torso = torso_rotation.T @ normal_world
    normal_torso /= np.linalg.norm(normal_torso)

    wrist_positions_torso = np.stack([
        torso_rotation.T @ (position - torso_position)
        for position in wrist_world_positions
    ])
    wrist_rotations_torso = np.stack([
        torso_rotation.T @ rotation
        for rotation in wrist_world_rotations
    ])
    reference_gap = _reference_support_gap(
        hulls,
        wrist_positions_torso,
        wrist_rotations_torso,
        normal_torso,
    )
    if not math.isfinite(reference_gap) or reference_gap <= 0:
        raise ValueError("Reference support gap must be finite and positive")
    if not math.isclose(reference_gap, distance, rel_tol=1e-5, abs_tol=2e-6):
        raise ValueError(
            "Reference closest-point distance and convex support gap disagree"
        )

    model_bytes = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, model_bytes)
    config = {
        "schema": _SCHEMA,
        "robot": "g1",
        "torso_body": _TORSO_BODY,
        "wrist_bodies": list(_WRIST_BODIES),
        "hand_meshes": list(_HAND_MESHES),
        "weight": weight,
        "margin_m": margin,
        "scale_m": scale,
        "separator_normal_torso": normal_torso.tolist(),
        "hull_vertices_wrist": [hull.tolist() for hull in hulls],
        "reference_wrist_positions_torso": wrist_positions_torso.tolist(),
        "reference_wrist_rotations_torso": wrist_rotations_torso.tolist(),
        "reference_gap_m": reference_gap,
        "reference_geom_distance_m": distance,
        "mesh_provenance": records,
        "calibration": {
            "hull_method": "scipy.spatial.ConvexHull/all-vertices",
            "separator_method": "mujoco.mj_geomDistance/left-to-right",
            "array_hash_encoding": "little-endian float64 C order",
            "model_mjb_sha256": hashlib.sha256(model_bytes.tobytes()).hexdigest(),
            "reference_qpos_sha256": _hash_array(reference.qpos),
            "reference_mocap_sha256": _hash_array(np.concatenate((
                reference.mocap_pos.ravel(), reference.mocap_quat.ravel()))),
            "mujoco_version": mujoco.__version__,
            "scipy_version": scipy.__version__,
        },
        "source": source,
    }
    return PalmSeparationGuidance(config).to_dict()


__all__ = [
    "PalmSeparationGuidance",
    "calibrate_palm_separation_config",
]

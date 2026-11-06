"""Privileged physical randomization derived from existing command context."""

import torch


DOMAIN_SCALE_RANGES = {
    "body_mass": (0.90, 1.10),
    "actuator_kp": (0.85, 1.15),
    "actuator_kd": (0.85, 1.15),
    "actuator_effort": (0.90, 1.10),
}


def command_domain_scales(xi, num_joints):
    """Map four fixed joint-offset context channels to physical scale factors."""
    if (not isinstance(xi, torch.Tensor) or xi.ndim != 2
            or xi.shape[1] != num_joints + 4 or xi.shape[0] < 1
            or not xi.is_floating_point() or not bool(torch.isfinite(xi).all())):
        raise ValueError("xi must be finite with shape (batch, num_joints + 4)")
    latent = (xi[:, 1:5] / .01).clamp(-1, 1)
    return torch.stack((
        1. + .10 * latent[:, 0],
        1. + .15 * latent[:, 1],
        1. + .15 * latent[:, 2],
        1. + .10 * latent[:, 3],
    ), dim=-1)


def apply_command_domain_scales(
    ids,
    actuator_ids,
    scales,
    *,
    body_mass,
    body_inertia,
    actuator_gainprm,
    actuator_biasprm,
    actuator_forcerange,
    base_body_mass,
    base_body_inertia,
    base_actuator_gainprm,
    base_actuator_biasprm,
    base_actuator_forcerange,
):
    """Restore selected worlds from immutable baselines and apply their scales."""
    if ids.ndim != 1 or scales.shape != (len(ids), 4):
        raise ValueError("ids and scales must have shapes (batch,) and (batch, 4)")
    if len(ids) == 0 or not bool(torch.isfinite(scales).all()):
        raise ValueError("at least one finite domain sample is required")
    mass, kp, kd, effort = scales.unbind(-1)
    body_mass[ids] = base_body_mass[ids] * mass[:, None]
    body_inertia[ids] = base_body_inertia[ids] * mass[:, None, None]
    actuator_gainprm[ids] = base_actuator_gainprm[ids]
    actuator_biasprm[ids] = base_actuator_biasprm[ids]
    actuator_forcerange[ids] = base_actuator_forcerange[ids]
    gain_index = actuator_ids[None, :]
    gain = actuator_gainprm[ids[:, None], gain_index, 0]
    actuator_gainprm[ids[:, None], gain_index, 0] = gain * kp[:, None]
    damping = actuator_biasprm[ids[:, None], gain_index, 2]
    actuator_biasprm[ids[:, None], gain_index, 1] *= kp[:, None]
    actuator_biasprm[ids[:, None], gain_index, 2] = damping * kd[:, None]
    actuator_forcerange[ids[:, None], gain_index, :] *= effort[:, None, None]


def command_domain_randomization_contract(enabled):
    return {
        "enabled": bool(enabled),
        "sampling": "once per simulator world; retained across episode resets",
        "source": (
            "four fixed joint-offset channels in privileged context xi"
            if enabled else None
        ),
        "actor_observes_physical_scales_directly": False,
        "scale_ranges": {
            name: list(bounds) for name, bounds in DOMAIN_SCALE_RANGES.items()
        } if enabled else {},
        "mass_and_inertia": (
            "scaled together per world" if enabled else "nominal"
        ),
        "motor_parameters": (
            "kp, kd and symmetric effort limits scaled per world"
            if enabled else "nominal"
        ),
    }

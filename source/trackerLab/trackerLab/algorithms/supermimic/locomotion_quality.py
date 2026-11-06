"""Reference-free foot and actuator regularization for command locomotion."""

import math

import torch

from trackerTask.supermimic.gait import FOOT_BODIES, FOOT_OFFSETS
from trackerLab.tracker_env.mdp.supermimic_command import CONTROL_DT


def _g1_sole_geometry(mjcf_path, device):
    """Read the configured G1's spherical foot colliders, including MJCF defaults."""
    import mujoco

    if not mjcf_path:
        raise ValueError("g1_sole requires robot.mjcf_path")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    centers, radii = [], []
    for name in FOOT_BODIES["g1"]:
        body = model.body(name).id
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
        if not geoms or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE
                            for g in geoms):
            raise ValueError(f"g1_sole requires spherical collision geoms on {name}")
        centers.append(model.geom_pos[geoms].tolist())
        radii.append(model.geom_size[geoms, 0].tolist())
    if len(centers[0]) != len(centers[1]):
        raise ValueError("g1_sole requires the same number of collision spheres on both feet")
    return (torch.tensor(centers, device=device),
            torch.tensor(radii, device=device))


class LocomotionQuality:
    """Locomotion regularization, with an opt-in G1 collision-sole model.

    ``foot_slip_mode="legacy"`` preserves ankle metrics, calibration and rewards.
    ``"g1_sole"`` uses the configured MJCF's foot spheres, without motion-derived
    height calibration. It measures horizontal material velocity at each sphere's
    world-down pole, including angular motion, and uses signed left/right spacing
    for the feet-close reward. This approximates support on a static heightfield;
    it is not a force/contact sensor or a terrain-normal friction measurement.
    """

    def __init__(self, kin, field, motors, *, foot_slide_weight=.008,
                 feet_close_weight=.008, foot_slip_mode="legacy"):
        for name, value in (("foot_slide_weight", foot_slide_weight),
                            ("feet_close_weight", feet_close_weight)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if foot_slip_mode not in ("legacy", "g1_sole"):
            raise ValueError("foot_slip_mode must be 'legacy' or 'g1_sole'")
        if foot_slip_mode == "g1_sole" and kin.robot.name != "g1":
            raise ValueError("g1_sole is supported only for G1")
        self.kin, self.field = kin, field
        self.foot_slip_mode = foot_slip_mode
        self.foot_slide_weight = float(foot_slide_weight)
        self.feet_close_weight = float(feet_close_weight)
        spec = kin.robot
        tracked = [kin.body_names[index] for index in kin.tracked_idx]
        self.feet = [tracked.index(name) for name in FOOT_BODIES[spec.name]]
        device = kin.device
        self.offset = torch.tensor(FOOT_OFFSETS.get(spec.name, (0., 0., 0.)), device=device)
        self.default_q = torch.tensor(spec.default_q, device=device)
        self.action_scale = torch.tensor(spec.action_scale, device=device)
        motor = torch.tensor(motors, device=device)
        self.effort, self.velocity_limit, self.kp, self.kd = motor[:, :4].unbind(1)
        self.qlo, self.qhi = kin.q_lo, kin.q_hi
        self.biped = len(self.feet) == 2
        self.sole_height = torch.zeros(len(self.feet), device=device)
        if foot_slip_mode == "g1_sole":
            self.sole_centers, self.sole_radii = _g1_sole_geometry(
                getattr(spec, "mjcf_path", None), device)

    @torch.no_grad()
    def calibrate(self, motion):
        if self.foot_slip_mode == "g1_sole":
            self.sole_height.zero_()
            return
        positions, *_ = self.foot_state(
            tuple(getattr(motion, name) for name in
                  ("root_p", "root_R", "root_v", "root_w", "q", "qd")))
        self.sole_height = positions[..., 2].quantile(.1, dim=0).clamp(
            -.03 * self.field.scale, .08 * self.field.scale)

    def foot_state(self, state):
        """Legacy representative foot pose and velocity (also in sole mode)."""
        return self._foot_kinematics(state)[:3]

    def _foot_kinematics(self, state):
        p, R, v, w = self.kin.forward(*state)
        arm = (R[..., self.feet, :, :] @ self.offset.to(R)[..., None]).squeeze(-1)
        return (p[..., self.feet, :] + arm, R[..., self.feet, :, :],
                v[..., self.feet, :] + torch.cross(w[..., self.feet, :], arm, dim=-1),
                w[..., self.feet, :])

    def sole_state(self, state):
        """World sphere-pole positions and material velocities, (..., 2, N, 3).

        The lowest spatial pole changes material identity as a sphere rotates.
        Its material velocity is v + omega x r, not the time derivative of the
        moving sampling location. On flat ground this is the friction velocity.
        """
        if self.foot_slip_mode != "g1_sole":
            raise ValueError("sole_state requires foot_slip_mode='g1_sole'")
        return self._sole_state(*self._foot_kinematics(state))

    def _sole_state(self, position, rotation, velocity, angular):
        arm = (rotation[..., None, :, :] @
               self.sole_centers.to(rotation)[..., None]).squeeze(-1)
        arm = arm - self.sole_radii.to(rotation)[..., None] * rotation.new_tensor((0., 0., 1.))
        return (position[..., None, :] + arm,
                velocity[..., None, :] + torch.cross(
                    angular[..., None, :].expand_as(arm), arm, dim=-1))

    def measurements(self, state, previous=None):
        position, rotation, velocity, angular = self._foot_kinematics(state)
        scale = self.field.scale
        extra = {}
        if self.foot_slip_mode == "g1_sole":
            points, point_velocity = self._sole_state(position, rotation, velocity, angular)
            height = points[..., 2] - self.field.sample(points[..., :2])
            # Full support at/below terrain, exactly zero at 2 scale-cm clearance.
            # Detach both numerator and denominator gates to remove the actor's
            # direct reward gradient for lifting/sloping a foot out of support.
            fraction = (1 - height / (.02 * scale)).clamp(0, 1)
            point_support = (fraction.square() * (3 - 2 * fraction)).detach()
            mass = point_support.sum((-1, -2))
            speed = torch.linalg.vector_norm(point_velocity[..., :2], dim=-1) / scale
            # A conditional mean has no discount for a single weakly supported
            # point. Fully airborne samples contribute exactly zero.
            weights = point_support / torch.where(
                mass > 0, mass, torch.ones_like(mass))[..., None, None]
            slip = (speed * weights).sum((-1, -2))
            support = point_support.amax(-1)
            extra["sole_support_heuristic"] = point_support
        else:
            height = position[..., 2] - self.field.sample(position[..., :2]) - self.sole_height
            support = torch.sigmoid((.02 * scale - height) / (.01 * scale))
            speed = torch.sqrt(velocity[..., :2].square().sum(-1) + 1e-8) / scale
            slip = (speed * support).sum(-1) / support.sum(-1).clamp_min(1)
        flat = (rotation[..., 2, :2].square().sum(-1) * support).mean(-1)
        separation = torch.zeros_like(slip)
        if self.biped:
            delta = position[..., 0, :2] - position[..., 1, :2]
            heading = state[1][..., :2, 0]
            if self.foot_slip_mode == "g1_sole":
                heading = heading / torch.linalg.vector_norm(
                    heading, dim=-1, keepdim=True).clamp_min(1e-6)
            lateral = torch.stack((-heading[..., 1], heading[..., 0]), -1)
            signed_separation = (delta * lateral).sum(-1) / scale
            separation = signed_separation.abs()
            if self.foot_slip_mode == "g1_sole":
                extra["foot_lateral_separation_scale_m"] = signed_separation
        acceleration = torch.zeros_like(slip)
        if previous is not None:
            acceleration = ((state[5] - previous[5]) / CONTROL_DT).square().mean(-1)
        return {"support_foot_slip_scale_m_s": slip,
                "support_foot_tilt_squared": flat,
                "foot_separation_scale_m": separation,
                "joint_acceleration_squared": acceleration,
                "support_heuristic": support, **extra}

    def reward(self, previous_state, state, action, xi):
        metrics = self.measurements(state, previous_state)
        target = (self.default_q + xi[..., 1:-3] + action * self.action_scale).clamp(
            self.qlo, self.qhi)
        torque = (self.kp * (target - state[4]) - self.kd * state[5]).clamp(
            -self.effort, self.effort)
        separation = metrics.get("foot_lateral_separation_scale_m",
                                 metrics["foot_separation_scale_m"])
        terms = {
            "foot_slide": -self.foot_slide_weight * metrics["support_foot_slip_scale_m_s"],
            "foot_tilt": -.008 * metrics["support_foot_tilt_squared"] if self.biped else
                         torch.zeros_like(metrics["support_foot_tilt_squared"]),
            "feet_close": -self.feet_close_weight * (
                torch.relu(.1 - separation) / .1
            ).square() if self.biped else torch.zeros_like(metrics["foot_separation_scale_m"]),
            "joint_acceleration": -2.5e-9 * metrics["joint_acceleration_squared"],
            "motor_torque": -3e-9 * torque.square().sum(-1),
            "joint_speed_limit": -.02 * torch.relu(
                state[5].abs() - .9 * self.velocity_limit).square().mean(-1),
            "body_roll_pitch_rate": -.001 * (
                state[1].transpose(-1, -2) @ state[3][..., None]
            ).squeeze(-1)[..., :2].square().sum(-1),
        }
        return sum(terms.values())

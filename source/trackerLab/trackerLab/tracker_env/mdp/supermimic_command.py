"""Reference-free locomotion heads shared by real and imagined transitions.

Scenario ``c`` contains seven physical quantities: lane center world Y,
body vx/vy/wz, target root clearance, jump height, and jump elapsed seconds.
The jump pulse is a clearance target; only the physics/world model changes S.
"""

import math

import torch

from trackerLab.kinematics.supermimic.rotations import yaw_mat
from trackerLab.tracker_env.mdp.supermimic_terrain import (
    FALL_PENALTY, MAX_STEP_REWARD, SUCCESS_BONUS, TerrainMDP,
)
from trackerTask.supermimic.terrain import COMMAND_PROFILES, PROFILES


COMMAND_VERSION = 4
SCENARIO_DIM = 7
GOAL_DIM = 12
CONTROL_DT = .02
JUMP_DURATION = .8
ACTION_LIMIT = 4.
COMMAND_MAX_STEP_REWARD = .152
COMMAND_FIELDS = (
    "lane_center_world_y", "body_vx", "body_vy", "body_wz",
    "target_clearance_m", "jump_height_m", "jump_elapsed_seconds",
)
STEALTH_KEYBOARD_MODES = (
    ("forward", .25, 0., 0., .40),
    ("turn_left", 0., 0., .55, .15),
    ("turn_right", 0., 0., -.55, .15),
    ("forward_left", .25, 0., .45, .15),
    ("forward_right", .25, 0., -.45, .15),
)
GOAL_FIELDS = (
    "body_vx", "body_vy", "body_wz", "target_clearance", "jump_height",
    "jump_pulse", "jump_clock", "lane_error", "remaining_world_x",
    "forward_in_heading_x", "forward_in_heading_y", "terrain_navigation",
)
TERMS = (
    "body_velocity", "yaw_rate", "alive", "upright", "height", "posture",
    "action_rate", "joint_velocity", "joint_limit", "progress", "lane", "heading",
    "dont_wait",
)


def advance_command(c, dt=CONTROL_DT):
    """Pure, differentiable command-clock update for one control action.

    Translational/yaw commands remain fixed in imagination. Real collection
    resamples them independently after 100--250 actions.
    """
    if c.shape[-1] != SCENARIO_DIM:
        raise ValueError(f"Command scenario must have {SCENARIO_DIM} columns")
    if not 0 <= dt < math.inf:
        raise ValueError("Command timestep must be finite and nonnegative")
    elapsed = (c[..., 6:7] + dt).clamp(0, JUMP_DURATION)
    return torch.cat((c[..., :6], elapsed), -1)


def jump_pulse(c):
    """Smooth jump clearance in meters, with finite 0.8-second support."""
    elapsed = c[..., 6]
    phase = elapsed.clamp(0, JUMP_DURATION) / JUMP_DURATION
    pulse = c[..., 5] * torch.sin(math.pi * phase).square()
    return torch.where((elapsed > 0) & (elapsed < JUMP_DURATION), pulse,
                       torch.zeros_like(pulse))


def target_clearance(c):
    return c[..., 4] + jump_pulse(c)


def body_velocities(S):
    """Convert the state's world velocities to the pelvis/body frame."""
    Rt = S[1].transpose(-1, -2)
    return tuple((Rt @ value.unsqueeze(-1)).squeeze(-1) for value in S[2:4])


class CommandMDP(TerrainMDP):
    scenario_dim = SCENARIO_DIM
    goal_dim = GOAL_DIM

    def __init__(self, field, default_q, qlo, qhi, gamma=.99, profile=None,
                 clearance_ratio=1.0, *, target_clearance_m=None, jump_height=0.,
                 action_limit=ACTION_LIMIT):
        profile = field.terrain.profile if profile is None else profile
        if profile not in (*PROFILES, *COMMAND_PROFILES) or profile != field.terrain.profile:
            raise ValueError("Command profile must match the heightfield profile")
        if not math.isfinite(clearance_ratio) or clearance_ratio <= 0:
            raise ValueError("Clearance ratio must be finite and positive")
        clearance = field.height * clearance_ratio if target_clearance_m is None else target_clearance_m
        if not math.isfinite(clearance) or clearance <= 0:
            raise ValueError("Physical target clearance must be finite and positive")
        if not math.isfinite(jump_height) or jump_height < 0:
            raise ValueError("Jump height must be finite and nonnegative")
        if not math.isfinite(action_limit) or action_limit <= 0:
            raise ValueError("Action limit must be finite and positive")
        super().__init__(field, default_q, qlo, qhi, gamma)
        self.profile = profile
        self.terrain_navigation = profile in PROFILES
        self.gamma = float(gamma)
        self.target_clearance_m = float(clearance)
        self.clearance_ratio = self.target_clearance_m / field.height
        self.jump_height, self.action_limit = float(jump_height), float(action_limit)
        self.goal_dim = GOAL_DIM
        self.obs_dim = 3 * self.num_joints + 9 + len(field.offsets)
        self.critic_dim = 13 + 2 * self.num_joints + self.num_joints + 4 + GOAL_DIM + len(field.model_offsets)
        self.set_style_weight(0.)
        n, dev = self.num_joints, default_q.device
        self.obs_noise = torch.cat((
            torch.full((n,), .01), torch.full((n,), .5), torch.full((3,), .05),
            torch.full((3,), .05), torch.zeros(n), torch.full((3,), .05),
            torch.full((len(field.offsets),), .01 / field.scale),
        )).to(dev)

    def set_style_weight(self, weight):
        """Replace the lost task and bounded AMP continuation at terrain goals."""
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Style weight must be finite and nonnegative")
        maximum = COMMAND_MAX_STEP_REWARD + weight
        success = SUCCESS_BONUS + self.gamma * maximum / (1 - self.gamma)
        if not math.isfinite(success):
            raise ValueError("Style weight produces a nonfinite terminal payout")
        self.style_weight, self.max_step_reward = float(weight), maximum
        self.success_reward = success if self.terrain_navigation else 0.

    def advance_command(self, c, dt=CONTROL_DT):
        return advance_command(c, dt)

    def obs(self, S, a_prev, xi, c):
        _, R, _, _, q, qd = S
        v_body, w_body = body_velocities(S)
        proprio = torch.cat((
            q - self.default_q - xi[..., 1:1+self.num_joints], qd * .1,
            w_body * .25, -R[..., 2, :], a_prev, v_body / self.field.scale,
        ), -1)
        return torch.cat((proprio, self.field.features(S)), -1)

    def goal(self, S, c):
        scale = self.field.scale
        commands = torch.cat((
            c[..., 1:3] / scale, c[..., 3:4], c[..., 4:6] / scale,
            jump_pulse(c).unsqueeze(-1) / scale, c[..., 6:7] / JUMP_DURATION,
        ), -1)
        if self.terrain_navigation:
            H = yaw_mat(S[1])
            navigation = torch.stack((
                (c[..., 0] - S[0][..., 1]) / scale,
                ((self.field.terrain.goal_x - S[0][..., 0]) / scale).clamp(-1, 3),
                H[..., 0, 0], -H[..., 1, 0], torch.ones_like(c[..., 0]),
            ), -1)
        else:
            navigation = torch.zeros_like(c[..., :5])
        return torch.cat((commands, navigation), -1)

    def reward(self, S, a, a_prev, c, terms=False):
        p, R, _, _, q, qd = S
        scale = self.field.scale
        v_body, w_body = body_velocities(S)
        clearance = self._clearance(S)
        ev = ((v_body[..., :2] - c[..., 1:3]) / scale).square().sum(-1)
        ew = (w_body[..., 2] - c[..., 3]).square()
        zero = torch.zeros_like(ev)
        if self.terrain_navigation:
            progress = .008 * (S[2][..., 0] / scale).clamp(-1, 1.5)
            lane = -.001 * ((p[..., 1] - c[..., 0]) / scale).square()
            heading = -.003 * (1 - R[..., 0, 0])
        else:
            progress, lane, heading = zero, zero, zero
        values = (
            .04 * torch.exp(-ev / .25), .04 * torch.exp(-ew / .25),
            torch.full_like(ev, .06), -.06 * (1 - R[..., 2, 2]),
            -.004 * ((clearance - target_clearance(c)) / (.2 * self.field.height)).square(),
            -.0006 * (q - self.default_q).square().mean(-1),
            -.0008 * (a - a_prev).square().mean(-1), -.00002 * qd.square().mean(-1),
            -.04 * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1),
            progress, lane, heading,
            -.01 * (c[..., 1] / scale > .15).to(ev.dtype)
            * torch.sigmoid((.1 - v_body[..., 0] / scale) / .03),
        )
        reward = sum(values)
        return (reward, dict(zip(TERMS, values))) if terms else reward

    def _finite(self, S, c):
        finite = torch.isfinite(S[1]).all((-1, -2)) & torch.isfinite(c).all(-1)
        for value in (S[0], *S[2:]):
            finite = finite & torch.isfinite(value).all(-1)
        return finite & (c[..., 4] > 0) & (c[..., 5] >= 0) & (c[..., 6] >= 0)

    def _minimum_clearance(self, c):
        return torch.minimum(torch.full_like(c[..., 4], .42 * self.field.height),
                             .6 * c[..., 4])

    def _clearance(self, S):
        # Invalid physics states must reach the failure mask without unsafe indexing.
        p = S[0]
        return p[..., 2] - self.field.sample(torch.nan_to_num(p[..., :2]))

    def termination(self, S, c):
        p, R = S[:2]
        clearance = self._clearance(S)
        failure = ((clearance < self._minimum_clearance(c)) | (R[..., 2, 2] < .35)
                   | ~self.field.inside(p[..., :2]) | ~self._finite(S, c))
        if self.terrain_navigation:
            failure = failure | (((p[..., 1] - c[..., 0]) / self.field.scale).abs() > 1.75)
        return failure

    def soft_outcome_probabilities(self, S, c):
        p, R = S[:2]
        scale, terrain = self.field.scale, self.field.terrain
        clearance = self._clearance(S)
        fall = torch.sigmoid((self._minimum_clearance(c) - clearance) / (.02 * scale))
        tilt = torch.sigmoid((.35 - R[..., 2, 2]) / .04)
        inside = (
            torch.sigmoid((p[..., 0] - terrain.x_min) / (.1 * scale))
            * torch.sigmoid((terrain.x_max - p[..., 0]) / (.1 * scale))
            * torch.sigmoid((p[..., 1] - terrain.y_min) / (.1 * scale))
            * torch.sigmoid((terrain.y_max - p[..., 1]) / (.1 * scale))
        )
        alive = (1 - fall) * (1 - tilt) * inside
        if self.terrain_navigation:
            lane = ((p[..., 1] - c[..., 0]) / scale).abs()
            alive = alive * (1 - torch.sigmoid((lane - 1.75) / .05))
        fall = torch.where(self._finite(S, c), 1 - alive, torch.ones_like(alive))
        if self.terrain_navigation:
            goal = torch.sigmoid((p[..., 0] - terrain.goal_x) / (.1 * scale))
            success = torch.where(self._finite(S, c), goal * (1 - fall), torch.zeros_like(fall))
        else:
            success = torch.zeros_like(fall)
        return success, fall

    def terminal_reward(self, success, fall):
        return self.success_reward * success - FALL_PENALTY * fall

    def command_metrics(self, S, c):
        v_body, w_body = body_velocities(S)
        velocity_error = torch.linalg.vector_norm(v_body[..., :2] - c[..., 1:3], dim=-1)
        command_speed = torch.linalg.vector_norm(c[..., 1:3], dim=-1)
        commanded_progress = (v_body[..., :2] * c[..., 1:3]).sum(-1) / command_speed.clamp_min(
            .01 * self.field.scale)
        moving_command = command_speed > .1 * self.field.scale
        yaw_error = (w_body[..., 2] - c[..., 3]).abs()
        clearance = self._clearance(S)
        clearance_error = (clearance - target_clearance(c)).abs()
        tracking = ((velocity_error <= .25 * self.field.scale) & (yaw_error <= .25)
                    & (clearance_error <= .12 * self.field.height)
                    & (S[1][..., 2, 2] >= .7) & ~self.termination(S, c)
                    & (~moving_command | (commanded_progress >= .5 * command_speed)))
        return {
            "velocity_error_mps": velocity_error, "yaw_rate_error_rps": yaw_error,
            "clearance_error_m": clearance_error, "command_tracking": tracking,
            "commanded_progress_mps": commanded_progress, "command_speed_mps": command_speed,
            "physical_clearance_m": clearance,
        }

    def transition_outputs(self, S, action, previous_action, command):
        reward, terms = self.reward(S, action, previous_action, command, terms=True)
        return reward, terms, self.command_metrics(S, command), self.termination(S, command)

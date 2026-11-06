"""Derive a training command envelope from expert velocity coverage."""

import math

import torch

from trackerLab.tracker_env.mdp.supermimic_command import stealth_keyboard_modes, wasd_keyboard_metadata


def fixed_command_envelope(command):
    if (not isinstance(command, (tuple, list)) or len(command) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in command)):
        raise ValueError("A fixed command needs three finite SI values: vx, vy, yaw rate")
    vx, vy, yaw_rate = (float(value) for value in command)
    return validate_envelope({
        "version": 1,
        "ranges": [[value, value] for value in (vx, vy, yaw_rate)],
        "source": "fixed body-frame command supplied for this training run",
        "fixed_command": {
            "body_vx_m_s": vx,
            "body_vy_m_s": vy,
            "body_yaw_rate_rad_s": yaw_rate,
        },
        "history_steps": 0,
        "minimum_forward_speed_m_s": 0.,
        "minimum_moving_frames_per_window": 0,
        "forward_only": vx > 0 and vy == 0,
        "vertical_speed_frame": "body",
        "moving_frames": 0,
        "terrain_navigation": False,
        "standing_commands": vx == 0 and vy == 0 and yaw_rate == 0,
    })


def keyboard_stealth_envelope(control_mode="keyboard"):
    modes = [
        {"name": name, "command": [vx, vy, yaw_rate], "probability": probability}
        for name, vx, vy, yaw_rate, probability in stealth_keyboard_modes(control_mode)
    ]
    values = [mode["command"] for mode in modes]
    ranges = [[min(row[axis] for row in values), max(row[axis] for row in values)]
              for axis in range(3)]
    envelope = {
        "version": 1,
        "ranges": ranges,
        "source": "discrete keyboard command modes for low-profile locomotion",
        "command_modes": modes,
        "history_steps": 0,
        "minimum_forward_speed_m_s": 0.,
        "minimum_moving_frames_per_window": 0,
        "forward_only": False,
        "vertical_speed_frame": "body",
        "moving_frames": 0,
        "terrain_navigation": False,
        "standing_commands": False,
    }
    if control_mode == "wasd":
        envelope["keyboard"] = wasd_keyboard_metadata()
        envelope["stationary_commands"] = True
        envelope["stationary_posture"] = "stealth"
    return validate_envelope(envelope)


def reference_envelope(motion, *, terrain_navigation=False, minimum_forward_speed=.15,
                       history_steps=10, forward_only=True):
    if type(history_steps) is not int or history_steps < 2:
        raise ValueError("history_steps must be at least two")
    if not math.isfinite(minimum_forward_speed) or minimum_forward_speed < 0:
        raise ValueError("minimum_forward_speed must be finite and nonnegative")
    rotation = motion.root_R.transpose(-1, -2)
    velocity = (rotation @ motion.root_v[..., None]).squeeze(-1)
    angular = (rotation @ motion.root_w[..., None]).squeeze(-1)
    values = torch.cat((velocity[:, :2], angular[:, 2:3]), -1)
    clip_id = motion.clip_id
    if clip_id.shape != (len(values),):
        raise ValueError("Motion clip identities must match the reference frames")
    if len(values) < history_steps:
        raise ValueError("Style pool has no complete reference history window")
    speed = velocity[:, 0] if forward_only else velocity[:, :2].norm(dim=-1)
    moving = ((speed >= minimum_forward_speed)
              & (motion.root_v[:, 2].abs() <= .3)
              & (angular.norm(dim=-1) <= 1.))
    span = torch.nn.functional.conv1d(
        moving.float()[None, None], torch.ones(1, 1, history_steps, device=values.device))
    continuous = torch.zeros_like(moving)
    if span.numel():
        good = span[0, 0] >= history_steps - 2
        starts = good.nonzero().flatten()
        starts = starts[clip_id[starts] == clip_id[starts + history_steps - 1]]
        for offset in range(history_steps):
            continuous[starts + offset] = True
    continuous &= moving
    values = values[continuous]
    if not len(values):
        raise ValueError("Style pool has no sustained moving reference windows")
    if terrain_navigation:
        values[:, 1:] = 0.
    lower, upper = values.quantile(.1, dim=0), values.quantile(.9, dim=0)
    ranges = torch.stack((lower, upper), -1).cpu().tolist()
    return validate_envelope({
        "version": 1, "ranges": ranges,
        "source": "10th-to-90th-percentile sustained moving windows within each clip",
        "history_steps": history_steps,
        "minimum_forward_speed_m_s": minimum_forward_speed,
        "minimum_moving_frames_per_window": history_steps - 2,
        "forward_only": forward_only,
        "vertical_speed_frame": "world",
        "moving_frames": int(continuous.sum().item()),
        "terrain_navigation": terrain_navigation,
        "standing_commands": False,
    })


def validate_envelope(envelope):
    ranges = envelope.get("ranges")
    if (envelope.get("version") != 1 or not isinstance(ranges, list) or len(ranges) != 3
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in ranges)
            or any(type(value) not in (float, int) or not math.isfinite(value)
                   for pair in ranges for value in pair)
            or any(pair[0] > pair[1] for pair in ranges)):
        raise ValueError("Command envelope requires three finite ordered velocity ranges")
    return envelope

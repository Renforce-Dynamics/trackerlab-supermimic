"""Guard active-mode falls in a 90-world, 1500-step WASD cycle report.

All nine modes must carry valid evidence. Direction, tracking and slip remain
diagnostics; this guard cannot establish command capability or visual quality.
Only the Python standard library is required.
"""

import argparse
import json
import math
from pathlib import Path
import sys


# Mirror STEALTH_WASD_MODES_V1 without importing the Torch runtime.
COMMAND_MODES = (
    ("forward", (.25, 0., 0.)),
    ("backward", (-.20, 0., 0.)),
    ("turn_left", (0., 0., .55)),
    ("turn_right", (0., 0., -.55)),
    ("forward_left", (.25, 0., .45)),
    ("forward_right", (.25, 0., -.45)),
    ("backward_left", (-.20, 0., .45)),
    ("backward_right", (-.20, 0., -.45)),
    ("idle", (0., 0., 0.)),
)
MODE_NAMES = tuple(name for name, _ in COMMAND_MODES)
MAX_FALL_FRACTION = .10
COUNTS = ("episodes", "falls", "successes", "timeouts")
SIGNED_METRICS = (
    "body_vx_m_s", "body_vy_m_s", "body_yaw_rate_rad_s",
    "mean_projected_velocity_m_s", "mean_clearance_m",
)
NONNEGATIVE_METRICS = (
    "mean_planar_speed_m_s", "mean_planar_drift_m_s", "desired_planar_speed_m_s",
    "mean_support_sole_slip_m_s", "velocity_rmse_m_s", "yaw_rate_rmse_rad_s",
)
FRACTION_METRICS = ("tracking_fraction", "supported_fraction")


def _required(row, field, path):
    if not isinstance(row, dict) or field not in row:
        raise ValueError(f"{path}.{field}: missing required field")
    return row[field]


def _number(value, path, *, count=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{path}: expected a finite number")
    if count and (value < 0 or int(value) != value):
        raise ValueError(f"{path}: expected a nonnegative integer count")
    return int(value) if count else value


def _finite_numbers(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_numbers(child, f"{path}[{index}]")
    elif type(value) in (int, float):
        _number(value, path)


def _training_modes(value, path):
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(name, str) or name not in MODE_NAMES for name in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{path}: expected unique canonical training mode names")
    return [name for name in MODE_NAMES if name in value]


def _mode_rows(value, path):
    if not isinstance(value, list) or len(value) != len(MODE_NAMES):
        raise ValueError(f"{path}: expected all nine canonical mode rows")
    rows = {}
    for row in value:
        name = _required(row, "name", path)
        if not isinstance(name, str) or name not in MODE_NAMES or name in rows:
            raise ValueError(f"{path}: expected each canonical mode exactly once")
        rows[name] = row
    return rows


def _counts(row, path):
    return {key: _number(_required(row, key, path), f"{path}.{key}", count=True)
            for key in COUNTS}


def _observe(report, training_modes):
    if not isinstance(report, dict):
        raise ValueError("cycle: expected a JSON object")
    _finite_numbers(report, "cycle")
    for field, expected in (
            ("task", "locomotion/g1/stealth"), ("stealth_control_mode", "wasd"),
            ("command_reward_mode", "legacy"), ("command_mode_layout", "cycle"),
            ("fixed_command_mode", None), ("command_sequence", None),
            ("num_envs", 90), ("physics_steps", 1500),
            ("policy_exploration_noise", False), ("observation_noise", True)):
        value = _required(report, field, "cycle")
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"cycle.{field}: expected {expected!r}")
    seed = _required(report, "evaluation_seed", "cycle")
    if type(seed) is not int:
        raise ValueError("cycle.evaluation_seed: expected an integer")
    duration = _number(_required(report, "entry_duration_s", "cycle"), "cycle.entry_duration_s")
    if duration < 0:
        raise ValueError("cycle.entry_duration_s: expected a nonnegative duration")
    randomization = _required(report, "physical_domain_randomization", "cycle")
    if not isinstance(randomization, dict) or randomization.get("enabled") is not True:
        raise ValueError("cycle.physical_domain_randomization: expected enabled physical randomization")
    recorded = _required(report, "training_command_modes", "cycle")
    recorded = list(MODE_NAMES) if recorded is None else _training_modes(
        recorded, "cycle.training_command_modes")
    if recorded != training_modes:
        raise ValueError("cycle.training_command_modes: differs from requested training modes")
    rows = _mode_rows(_required(report, "keyboard_mode_metrics", "cycle"),
                      "cycle.keyboard_mode_metrics")
    episodes = _required(report, "keyboard_episode_metrics", "cycle")
    attribution = _required(episodes, "attribution", "cycle.keyboard_episode_metrics")
    if attribution != "environment_id_modulo_cycle_mode; stable across resets":
        raise ValueError("cycle.keyboard_episode_metrics.attribution: expected stable cycle attribution")
    episode_rows = _mode_rows(
        _required(episodes, "per_keyboard_mode", "cycle.keyboard_episode_metrics"),
        "cycle.keyboard_episode_metrics.per_keyboard_mode")
    observed, total, samples = [], dict.fromkeys(COUNTS, 0), 0
    for name, command in COMMAND_MODES:
        row, path = rows[name], f"cycle.keyboard_mode_metrics.{name}"
        twist = _required(row, "command", path)
        if not isinstance(twist, list) or len(twist) != 3:
            raise ValueError(f"{path}.command: expected canonical twist {list(command)}")
        for index, component in enumerate(twist):
            _number(component, f"{path}.command[{index}]")
        if twist != list(command):
            raise ValueError(f"{path}.command: differs from canonical twist")
        count = _number(_required(row, "samples", path), f"{path}.samples", count=True)
        if not 0 < count <= 10 * report["physics_steps"]:
            raise ValueError(f"{path}.samples: expected positive samples within the mode's step budget")
        metrics = {}
        for field in (*SIGNED_METRICS, *NONNEGATIVE_METRICS, *FRACTION_METRICS):
            value = _number(_required(row, field, path), f"{path}.{field}")
            if field in NONNEGATIVE_METRICS and value < 0:
                raise ValueError(f"{path}.{field}: expected a nonnegative metric")
            if field in FRACTION_METRICS and not 0 <= value <= 1:
                raise ValueError(f"{path}.{field}: expected a fraction in [0, 1]")
            metrics[field] = value
        counts = _counts(episode_rows[name], f"cycle.keyboard_episode_metrics.{name}")
        if not 0 < counts["episodes"] <= 10 * report["physics_steps"]:
            raise ValueError(f"{path}.episodes: expected positive completed episodes within the step budget")
        if counts["episodes"] != counts["falls"] + counts["successes"] + counts["timeouts"]:
            raise ValueError(f"{path}.episodes: differs from falls + successes + timeouts")
        for field in ("episodes", "falls", "successes"):
            value = _number(_required(row, field, path), f"{path}.{field}", count=True)
            if value != counts[field]:
                raise ValueError(f"{path}.{field}: differs from per-mode episode counts")
        active = name in training_modes
        fraction = counts["falls"] / counts["episodes"]
        observed.append({
            "name": name, "command": list(command), "samples": count, **metrics, **counts,
            "active_training_mode": active, "diagnostics_only": not active,
            "fall_fraction": fraction,
            "fall_guard_passed": fraction <= MAX_FALL_FRACTION if active else None,
            "directional_capability": {
                "scope": "signs_of_mean_motion_only", "readiness_assessed": False,
                "translation_sign_matches": (metrics["body_vx_m_s"] * command[0] > 0
                                             if command[0] else None),
                "yaw_sign_matches": (metrics["body_yaw_rate_rad_s"] * command[2] > 0
                                     if command[2] else None),
                "zero_command_note": "Zero axes require drift/settling review, not a sign test",
            },
        })
        for field in COUNTS:
            total[field] += counts[field]
        samples += count
    declared = _counts(_required(episodes, "total", "cycle.keyboard_episode_metrics"),
                       "cycle.keyboard_episode_metrics.total")
    if total != declared:
        raise ValueError("cycle.keyboard_episode_metrics.total: differs from per-mode sums")
    if _number(_required(report, "command_metric_samples", "cycle"),
               "cycle.command_metric_samples", count=True) != samples:
        raise ValueError("cycle.command_metric_samples: differs from per-mode sample sum")
    lanes = _required(report, "lanes", "cycle")
    if not isinstance(lanes, list) or len(lanes) != 6:
        raise ValueError("cycle.lanes: expected six terrain-level counters")
    lane_total, levels = dict.fromkeys(COUNTS, 0), set()
    for index, lane in enumerate(lanes):
        path = f"cycle.lanes[{index}]"
        level = _required(lane, "level", path)
        if type(level) is not int or level not in range(6) or level in levels:
            raise ValueError(f"{path}.level: expected each level 0..5 exactly once")
        levels.add(level)
        counts = _counts(lane, path)
        # Flat-task lane timeouts include successes; keyboard timeouts exclude them.
        if (counts["episodes"] != counts["falls"] + counts["timeouts"]
                or counts["successes"] > counts["timeouts"]):
            raise ValueError(f"{path}: inconsistent flat-task episode counts")
        for field, outcome in (("fall_fraction", "falls"), ("success_fraction", "successes")):
            if field in lane:
                value = _number(lane[field], f"{path}.{field}")
                if not math.isclose(value, counts[outcome] / max(counts["episodes"], 1),
                                    rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"{path}.{field}: differs from episode counts")
        for field in COUNTS:
            lane_total[field] += counts[field]
    expected_lanes = {**total, "timeouts": total["timeouts"] + total["successes"]}
    if lane_total != expected_lanes:
        raise ValueError("cycle.lanes: totals differ from per-mode episode sums")
    return {
        "protocol": {key: report[key] for key in (
            "evaluation_seed", "num_envs", "physics_steps", "entry_duration_s",
            "command_mode_layout", "observation_noise", "policy_exploration_noise",
            "physical_domain_randomization")},
        "total": total, "samples": samples, "per_mode": observed,
    }


def evaluate_curriculum(cycle, training_modes):
    """Return a JSON-safe continuation receipt; malformed evidence rejects."""
    reasons, observed, active = [], None, None
    try:
        active = _training_modes(training_modes, "training_modes")
        observed = _observe(cycle, active)
    except (ValueError, RecursionError) as error:
        reasons.append(str(error))
    if observed is not None:
        for row in observed["per_mode"]:
            if row["fall_guard_passed"] is False:
                reasons.append(f"{row['name']}.fall_fraction: {row['fall_fraction']} > {MAX_FALL_FRACTION}")
    return {
        "passed": not reasons, "reasons": reasons, "observed": observed,
        "training_command_modes": active,
        "thresholds": {"max_active_mode_fall_fraction": MAX_FALL_FRACTION},
        "scope": "active_mode_fall_guard_only", "all_commands_accepted": False,
        "visual_review_required": True,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--training-modes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cycle, errors = None, []
    try:
        cycle = json.loads(args.cycle.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    except (OSError, ValueError, RecursionError) as error:
        errors.append(f"cycle: could not read evaluation report: {error}")
    receipt = evaluate_curriculum(cycle, args.training_modes)
    receipt["reasons"][:0] = errors
    receipt["passed"] = not receipt["reasons"]
    try:
        if (args.output.resolve() == args.cycle.resolve()
                or (args.output.exists() and args.cycle.exists() and args.output.samefile(args.cycle))):
            raise OSError("output must differ from the cycle report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except OSError as error:
        receipt["passed"] = False
        receipt["reasons"].append(f"output: could not write receipt: {error}")
        print(receipt["reasons"][-1], file=sys.stderr)
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

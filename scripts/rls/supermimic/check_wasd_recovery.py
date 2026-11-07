"""Gate preservation of forward gait using two evaluate_terrain.py reports.

Other WASD modes remain unevaluated. Velocity and samples come from the
forward keyboard row; falls are pooled over completed episodes in all lanes.
This checker uses only the Python standard library.
"""

import argparse
import json
import math
from pathlib import Path
import sys


PROTOCOL_FIELDS = (
    "evaluation_seed", "num_envs", "physics_steps", "entry_duration_s",
    "observation_noise", "physical_domain_randomization",
)
MIN_VX = .15
RETAINED_VX = .75
MAX_VX = .40
MAX_FALL_FRACTION = .10


def _required(row, field, path):
    if field not in row:
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
    return value


def _finite_numbers(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_numbers(child, f"{path}[{index}]")
    elif type(value) in (int, float):
        _number(value, path)


def _same_value(left, right):
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right))
    return left == right


def _observe(report, label):
    if not isinstance(report, dict):
        raise ValueError(f"{label}: expected a JSON object")
    _finite_numbers(report, label)
    for field, expected in (("task", "locomotion/g1/stealth"),
                            ("fixed_command_mode", "forward")):
        if _required(report, field, label) != expected:
            raise ValueError(f"{label}.{field}: expected {expected!r}")
    protocol = {field: _required(report, field, label) for field in PROTOCOL_FIELDS}
    for field in ("evaluation_seed", "num_envs", "physics_steps"):
        value = protocol[field]
        if type(value) is not int or (field != "evaluation_seed" and value <= 0):
            raise ValueError(f"{label}.{field}: expected "
                             + ("an integer" if field == "evaluation_seed" else "a positive integer"))
    if _number(protocol["entry_duration_s"], f"{label}.entry_duration_s") < 0:
        raise ValueError(f"{label}.entry_duration_s: expected a nonnegative duration")
    if type(protocol["observation_noise"]) is not bool:
        raise ValueError(f"{label}.observation_noise: expected a boolean")
    randomization = protocol["physical_domain_randomization"]
    if (not isinstance(randomization, dict)
            or type(randomization.get("enabled")) is not bool):
        raise ValueError(f"{label}.physical_domain_randomization: "
                         "expected a contract object with boolean enabled")

    rows = _required(report, "keyboard_mode_metrics", label)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label}.keyboard_mode_metrics: expected a list of objects")
    forward = [row for row in rows if row.get("name") == "forward"]
    if len(forward) != 1:
        raise ValueError(f"{label}.keyboard_mode_metrics: expected exactly one forward row")
    forward = forward[0]
    path = f"{label}.keyboard_mode_metrics.forward"
    command = _required(forward, "command", path)
    if not isinstance(command, list) or len(command) != 3:
        raise ValueError(f"{path}.command: expected [0.25, 0, 0]")
    for index, value in enumerate(command):
        _number(value, f"{path}.command[{index}]")
    if command != [.25, 0., 0.]:
        raise ValueError(f"{path}.command: expected [0.25, 0, 0]")
    samples = _number(_required(forward, "samples", path), f"{path}.samples", count=True)
    if samples <= 0:
        raise ValueError(f"{path}.samples: expected positive actual samples")
    vx = _number(_required(forward, "body_vx_m_s", path), f"{path}.body_vx_m_s")

    lanes = _required(report, "lanes", label)
    if not isinstance(lanes, list) or not lanes:
        raise ValueError(f"{label}.lanes: expected a nonempty list")
    episodes, falls = 0, 0
    for index, lane in enumerate(lanes):
        lane_path = f"{label}.lanes[{index}]"
        if not isinstance(lane, dict):
            raise ValueError(f"{lane_path}: expected an object")
        count = _number(_required(lane, "episodes", lane_path),
                        f"{lane_path}.episodes", count=True)
        fallen = _number(_required(lane, "falls", lane_path),
                         f"{lane_path}.falls", count=True)
        if fallen > count:
            raise ValueError(f"{lane_path}.falls: exceeds completed episodes")
        if "fall_fraction" in lane:
            fraction = _number(lane["fall_fraction"], f"{lane_path}.fall_fraction")
            if not 0 <= fraction <= 1 or not math.isclose(
                    fraction, fallen / max(count, 1), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{lane_path}.fall_fraction: differs from episode counts")
        episodes += int(count)
        falls += int(fallen)
    if episodes <= 0:
        raise ValueError(f"{label}.lanes.episodes: expected positive completed episodes")

    # These redundant counters are emitted by current evaluator versions.
    for field, expected in (("episodes", episodes), ("falls", falls)):
        if field in forward:
            value = _number(forward[field], f"{path}.{field}", count=True)
            if value != expected:
                raise ValueError(f"{path}.{field}: differs from held-forward lane totals")
    if "command_metric_samples" in report:
        value = _number(report["command_metric_samples"],
                        f"{label}.command_metric_samples", count=True)
        if value != samples:
            raise ValueError(f"{label}.command_metric_samples: differs from forward samples")
    return {
        "protocol": protocol, "command": command, "forward_samples": int(samples),
        "body_vx_m_s": vx, "episodes": episodes, "falls": falls,
        "fall_fraction": falls / episodes,
    }


def evaluate_recovery(baseline, candidate):
    """Return a JSON-safe receipt; malformed reports always reject."""
    reasons, observed = [], {}
    thresholds = {
        "baseline_min_body_vx_m_s": MIN_VX,
        "retained_baseline_velocity_fraction": RETAINED_VX,
        "candidate_min_body_vx_m_s": None,
        "candidate_max_body_vx_m_s": MAX_VX,
        "max_fall_fraction": MAX_FALL_FRACTION,
        "command": [.25, 0., 0.],
    }
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        observed[label] = None
        try:
            observed[label] = _observe(report, label)
        except (ValueError, RecursionError) as error:
            reasons.append(str(error))
    before, after = observed["baseline"], observed["candidate"]
    if before is not None:
        thresholds["candidate_min_body_vx_m_s"] = max(MIN_VX, RETAINED_VX * before["body_vx_m_s"])
        if before["body_vx_m_s"] < MIN_VX:
            reasons.append(f"baseline.body_vx_m_s: {before['body_vx_m_s']} < {MIN_VX}")
        if before["fall_fraction"] > MAX_FALL_FRACTION:
            reasons.append(f"baseline.fall_fraction: {before['fall_fraction']} > {MAX_FALL_FRACTION}")
    if before is not None and after is not None:
        for field in PROTOCOL_FIELDS:
            if not _same_value(before["protocol"][field], after["protocol"][field]):
                reasons.append(f"protocol.{field}: baseline and candidate differ")
    if after is not None:
        minimum = thresholds["candidate_min_body_vx_m_s"]
        minimum = MIN_VX if minimum is None else minimum
        if after["body_vx_m_s"] < minimum:
            reasons.append(f"candidate.body_vx_m_s: {after['body_vx_m_s']} < {minimum}")
        if after["body_vx_m_s"] > MAX_VX:
            reasons.append(f"candidate.body_vx_m_s: {after['body_vx_m_s']} > {MAX_VX}")
        if after["fall_fraction"] > MAX_FALL_FRACTION:
            reasons.append(f"candidate.fall_fraction: {after['fall_fraction']} > {MAX_FALL_FRACTION}")
    return {
        "passed": not reasons, "reasons": reasons, "observed": observed,
        "thresholds": thresholds, "scope": "forward_gait_preservation_only",
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
    for name in ("baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    reports, errors = {}, []
    for label in ("baseline", "candidate"):
        try:
            reports[label] = json.loads(
                getattr(args, label).read_text(encoding="utf-8-sig"),
                object_pairs_hook=_unique_object)
        except (OSError, ValueError, RecursionError) as error:
            reports[label] = None
            errors.append(f"{label}: could not read evaluation report: {error}")
    receipt = evaluate_recovery(reports["baseline"], reports["candidate"])
    receipt["reasons"][:0] = errors
    receipt["passed"] = not receipt["reasons"]
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except OSError as error:
        receipt["passed"] = False
        receipt["reasons"].append(f"output: could not write receipt: {error}")
        print(receipt["reasons"][-1], file=sys.stderr)
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

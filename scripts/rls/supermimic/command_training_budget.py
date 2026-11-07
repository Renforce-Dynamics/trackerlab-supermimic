"""Decide continuation from gate evidence and an owner-selected review floor.

Numerical gate failures remain recorded during the deferred review window.
Malformed evidence always blocks continuation. Only the standard library is
required; this decision does not change either source gate receipt.
"""

import argparse
import json
import math
from pathlib import Path
import sys


def _finite_json(value, path):
    if type(value) in (int, float):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{path}: expected a finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: expected string object keys")
            _finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json(child, f"{path}[{index}]")
    elif value is not None and type(value) not in (bool, str):
        raise ValueError(f"{path}: expected JSON-compatible evidence")


def _iteration(value, path, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path}: expected a finite integer >= {minimum}")
    _finite_json(value, path)
    return value


def _same_value(left, right):
    # Protocol booleans must retain their type; numeric JSON encodings may vary.
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


def _validate_gate(gate, scope, path):
    if not isinstance(gate, dict):
        raise ValueError(f"{path}: expected a gate receipt object")
    _finite_json(gate, path)
    if gate.get("scope") != scope:
        raise ValueError(f"{path}.scope: expected {scope!r}")
    if type(gate.get("passed")) is not bool:
        raise ValueError(f"{path}.passed: expected a boolean")
    reasons = gate.get("reasons")
    if (not isinstance(reasons, list)
            or any(not isinstance(reason, str) or not reason.strip() for reason in reasons)):
        raise ValueError(f"{path}.reasons: expected a list of nonempty reason strings")
    if gate["passed"] != (len(reasons) == 0):
        raise ValueError(f"{path}: passed and reasons disagree")
    observed = gate.get("observed")
    if not isinstance(observed, dict) or not observed:
        raise ValueError(f"{path}.observed: expected nonempty observed evidence")
    if scope == "forward_gait_preservation_only":
        for label in ("baseline", "candidate"):
            if not isinstance(observed.get(label), dict) or not observed[label]:
                raise ValueError(f"{path}.observed.{label}: expected nonempty observed evidence")
        before, after = observed["baseline"], observed["candidate"]
        if "protocol" in before or "protocol" in after:
            if (not isinstance(before.get("protocol"), dict) or not before["protocol"]
                    or not isinstance(after.get("protocol"), dict) or not after["protocol"]
                    or not _same_value(before["protocol"], after["protocol"])):
                raise ValueError(f"{path}.observed: baseline and candidate protocols must match")


def decide(forward_gate, active_gate, iteration, minimum_review_iteration=0):
    """Allow deferred numerical failures only with valid gate evidence."""
    errors, gate_failures = [], []
    current, floor = None, None
    for value, name, minimum in (
            (iteration, "iteration", 1),
            (minimum_review_iteration, "minimum_review_iteration", 0)):
        try:
            validated = _iteration(value, name, minimum)
            if name == "iteration":
                current = validated
            else:
                floor = validated
        except ValueError as error:
            errors.append(str(error))
    for gate, scope, path in (
            (forward_gate, "forward_gait_preservation_only", "forward_gate"),
            (active_gate, "active_mode_fall_guard_only", "active_gate")):
        if isinstance(gate, dict) and gate.get("passed") is False:
            reasons = gate.get("reasons")
            if isinstance(reasons, list):
                gate_failures.extend(reason for reason in reasons if isinstance(reason, str))
        try:
            _validate_gate(gate, scope, path)
        except (ValueError, RecursionError) as error:
            errors.append(f"{path}: {error}" if isinstance(error, RecursionError) else str(error))
    valid = not errors
    deferred = valid and current < floor
    passed = valid and (deferred or (forward_gate["passed"] and active_gate["passed"]))
    return {
        "passed": passed, "conclusion_deferred": deferred,
        "gate_failures": gate_failures,
        "minimum_review_iteration": floor, "iteration": current,
        "valid_evidence": valid,
        "reasons": errors if errors else ([] if passed else list(gate_failures)),
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
    parser.add_argument("--forward-gate", type=Path, required=True)
    parser.add_argument("--active-mode-gate", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--minimum-review-iteration", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gates, errors = [], []
    for path, label in ((args.forward_gate, "forward_gate"),
                        (args.active_mode_gate, "active_gate")):
        try:
            gate = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
        except (OSError, ValueError, RecursionError) as error:
            gate = None
            errors.append(f"{label}: could not read gate receipt: {error}")
        gates.append(gate)
    decision = decide(*gates, args.iteration, args.minimum_review_iteration)
    if errors:
        decision["reasons"][:0] = errors
        decision.update(passed=False, conclusion_deferred=False, valid_evidence=False)
    try:
        for path in (args.forward_gate, args.active_mode_gate):
            if (args.output.resolve() == path.resolve()
                    or (args.output.exists() and path.exists() and args.output.samefile(path))):
                raise OSError("output must differ from both input gate receipts")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except OSError as error:
        decision["passed"] = False
        decision["reasons"].append(f"output: could not write decision receipt: {error}")
        print(decision["reasons"][-1], file=sys.stderr)
    print(json.dumps(decision, allow_nan=False), flush=True)
    return 0 if decision["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Reject lost foot lifting even when a physical forward-speed gate passes."""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_stealth_trajectory import g1_geometry, load_record, measure_poses, sha256
from check_wasd_recovery import PROTOCOL_FIELDS, _same_value, evaluate_recovery


def valid_speed_evidence(speed):
    """A valid weak result can be observed through an authorized training budget."""
    before = speed["observed"]["baseline"]
    after = speed["observed"]["candidate"]
    return bool(before is not None and after is not None and all(
        _same_value(before["protocol"][field], after["protocol"][field])
        for field in PROTOCOL_FIELDS))


def review_exit_code(result, diagnostic_only=False):
    if type(result.get("passed")) is not bool or type(result.get("valid_evidence")) is not bool:
        return 2
    return 0 if result["valid_evidence"] and (result["passed"] or diagnostic_only) else 2


def compare_steps(baseline, candidate):
    """A regression barrier, not certification of natural gait."""
    before, after = np.asarray(baseline), np.asarray(candidate)
    for values in (before, after):
        if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 6
                or values.dtype.kind not in "iu" or np.any(values < 0)):
            raise ValueError("Require at least six first-attempt nonnegative left/right lift counts")
    if before.shape != after.shape:
        raise ValueError("Baseline and candidate recorded worlds differ")
    bmed, amed = np.median(before, axis=0), np.median(after, axis=0)
    bilateral_before = float((before > 0).all(1).mean())
    bilateral_after = float((after > 0).all(1).mean())
    reasons = []
    if np.any(bmed < 1) or bilateral_before < .75:
        reasons.append("Baseline lacks bilateral stepping in the recorded first attempts")
    if np.any(amed < np.maximum(1., .5 * bmed)):
        reasons.append("Candidate lost at least half of a foot's median sustained lift count")
    if bilateral_after < .75 * bilateral_before:
        reasons.append("Candidate lost bilateral stepping in too many recorded first attempts")
    return {"passed": not reasons, "reasons": reasons,
            "baseline_counts": before.tolist(), "candidate_counts": after.tolist(),
            "baseline_median": bmed.tolist(), "candidate_median": amed.tolist(),
            "bilateral_fractions": [bilateral_before, bilateral_after],
            "minimum_retained_count_fraction": .5, "minimum_retained_bilateral_fraction": .75}


def measured_steps(path, evaluation, geometry):
    model, bodies, centers, radii, _ = geometry
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["recorded_env_ids"]
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 6:
            raise ValueError("Record at least six fixed first attempts")
        if int(archive["seed"]) != evaluation["evaluation_seed"]:
            raise ValueError("Trajectory seed differs from evaluation")
        if int(archive["iteration"]) != evaluation["trained_iterations"]:
            raise ValueError("Trajectory checkpoint iteration differs from evaluation")
        dt = float(archive["control_dt"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Trajectory control timestep must be finite and positive")
        commands = archive["command_sequence"]
        if commands.shape != (*archive["qpos"].shape[:2], 7):
            raise ValueError("Command and pose sequence shapes differ")
        first, last = round(5 / dt), round(18 / dt)
        if last >= len(commands):
            raise ValueError("Need complete 5..18 second recording")
        terminals = archive["terminal_steps"]
        if terminals.shape != (len(ids),):
            raise ValueError("Terminal steps must match recorded worlds")
        complete = (terminals < 0) | (terminals >= last)
        if not np.allclose(commands[first:last+1][:, complete, 1:4],
                           [.25, 0, 0], atol=1e-6, rtol=0):
            raise ValueError("Lift comparison requires held forward commands")
    counts, ended = [], []
    for lane in range(len(ids)):
        poses, dt, source = load_record(path, lane, 0., 18.)
        if source["observed_interval_s"][1] < 18. - 1e-8:
            counts.append([0, 0])
            ended.append(int(ids[lane]))
            continue
        metrics = measure_poses(poses[first:], dt, first, model, bodies, centers, radii)["metrics"]
        counts.append([metrics["feet"][side]["sustained_lift_count"] for side in ("left", "right")])
    return {"counts": counts, "recorded_env_ids": ids.tolist(), "early_terminal_ids": ended,
            "trajectory_sha256": sha256(path)}


def check(baseline_evaluation, candidate_evaluation, baseline_trajectory, candidate_trajectory):
    speed = evaluate_recovery(baseline_evaluation, candidate_evaluation)
    geometry = g1_geometry()
    before = measured_steps(baseline_trajectory, baseline_evaluation, geometry)
    after = measured_steps(candidate_trajectory, candidate_evaluation, geometry)
    if before["recorded_env_ids"] != after["recorded_env_ids"]:
        raise ValueError("Baseline and candidate must use the same recorded environment IDs")
    lifts = compare_steps(before["counts"], after["counts"])
    return {"passed": speed["passed"] and lifts["passed"],
            "valid_evidence": valid_speed_evidence(speed), "speed": speed, "lifts": lifts,
            "baseline": before, "candidate": after, "model": geometry[-1],
            "scope": "forward regression barrier only; visual entry/style and all WASD modes remain unqualified",
            "window_s": [5, 18], "lift_definition": "whole sole above 2cm for at least 80ms"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Preserve failed quality results without shortening an authorized budget")
    args = parser.parse_args()
    try:
        result = check(
            json.loads((args.baseline / "evaluation.json").read_text(encoding="utf-8")),
            json.loads((args.candidate / "evaluation.json").read_text(encoding="utf-8")),
            args.baseline / "trajectories.npz", args.candidate / "trajectories.npz")
    except (ValueError, KeyError, OSError) as error:
        result = {"passed": False, "valid_evidence": False,
                  "error": str(error), "scope": "invalid regression evidence"}
    result["diagnostic_only"] = args.diagnostic_only
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}))
    raise SystemExit(review_exit_code(result, args.diagnostic_only))


if __name__ == "__main__":
    main()

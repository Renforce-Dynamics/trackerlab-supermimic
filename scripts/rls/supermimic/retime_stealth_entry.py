"""Uniformly retime an existing arm path while retaining its poses and provenance.

The resulting document changes target timing only. Collision-free target paths
do not establish collision-free physics or natural motion.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def retime(document, scale, source_sha256):
    if type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if not isinstance(document, dict):
        raise ValueError("Arm reference must be an object")
    times, poses = document.get("entry_times_s"), document.get("entry_reference_q")
    if (not isinstance(times, list) or len(times) < 2 or times[0] != 0
            or any(type(t) not in (float, int) or not math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))
            or not isinstance(poses, list) or len(poses) != len(times)):
        raise ValueError("Expected a complete increasing arm waypoint schedule")
    reference = document.get("reference_q")
    if (not isinstance(reference, dict) or not reference
            or any(not isinstance(p, dict) or set(p) != set(reference) for p in poses)):
        raise ValueError("Every waypoint must use the reference joint names")
    if any(type(q) not in (int, float) or not math.isfinite(q)
           for pose in [reference, *poses] for q in pose.values()):
        raise ValueError("Joint values must be finite")
    if any(abs(poses[-1][name] - reference[name]) > 1e-6 for name in reference):
        raise ValueError("Final waypoint must equal the held arm pose")
    result = copy.deepcopy(document)
    result["entry_times_s"] = [round(float(t) * scale, 10) for t in times]
    peak_rate = peak_acceleration = 0.
    for start, end, a, b in zip(result["entry_times_s"], result["entry_times_s"][1:],
                               poses, poses[1:]):
        dt = end - start
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Retimed duration must remain finite and positive")
        distance = max(abs(b[name] - a[name]) for name in reference)
        peak_rate = max(peak_rate, 1.875 * distance / dt)
        peak_acceleration = max(peak_acceleration, (10 / math.sqrt(3)) * distance / dt**2)
    if not all(math.isfinite(value) for value in (peak_rate, peak_acceleration)):
        raise ValueError("Retimed derivatives must remain finite")
    result["source"] = {
        "kind": "uniformly-retimed-existing-arm-path",
        "parent_reference_sha256": source_sha256,
        "parent_source": document.get("source"),
        "time_scale": scale,
        "original_entry_times_s": times,
        "peak_scheduled_rate_rad_s": peak_rate,
        "peak_scheduled_acceleration_rad_s2": peak_acceleration,
        "scope": "Identical joint-space path and final pose; timing changed. "
                 "Actual physical tracking and visual quality require a new rollout.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--time-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.reference.read_bytes()
    result = retime(json.loads(payload), args.time_scale, hashlib.sha256(payload).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "entry_times_s": result["entry_times_s"],
                      "source": result["source"]}, indent=2))


if __name__ == "__main__":
    main()

"""Audit unchanged controller/normalizers/expert and exact optimizer continuity."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def optimizer_continuity(before, after, increment, name):
    old, new = before["state"], after["state"]
    if (not old or old.keys() != new.keys()
            or any("step" not in state for state in (*old.values(), *new.values()))
            or [group["params"] for group in before["param_groups"]]
            != [group["params"] for group in after["param_groups"]]):
        raise ValueError(f"Optimizer entries differ: {name}")
    if any(float(new[key]["step"]) - float(old[key]["step"]) != increment for key in old):
        raise ValueError(f"Optimizer counter continuity differs: {name}")
    left, right = ([float(state["step"]) for state in values.values()] for values in (old, new))
    return {"source_min": min(left), "source_max": max(left),
            "current_min": min(right), "current_max": max(right),
            "expected_increment": increment}


def verify(before, after, source_iteration):
    if before["iteration"] != source_iteration:
        raise ValueError("Unexpected source iteration")
    updates = after["iteration"] - before["iteration"]
    if updates <= 0:
        raise ValueError("Expected completed continuation updates")
    keys = [key for key in before["actor"] if key.startswith(("obs_norm.", "goal_norm."))]
    if not keys or any(not torch.equal(before["actor"][key], after["actor"][key]) for key in keys):
        raise ValueError("Actor normalization changed")
    if (before["stealth_upper_body_constraint"] != after["stealth_upper_body_constraint"]
            or before.get("upper_body_rate_limit") != after.get("upper_body_rate_limit")):
        raise ValueError("Arm controller changed")
    if before["amp_style"]["expert_hash"] != after["amp_style"]["expert_hash"]:
        raise ValueError("AMP expert identity changed")
    steps = {}
    for name, multiplier in (
        ("opt_pi", 1), ("opt_wm", after["arguments"]["wm_batches"]),
        ("opt_v", after["arguments"]["critic_epochs"] * after["arguments"]["critic_minibatches"]),
    ):
        steps[name] = optimizer_continuity(before[name], after[name], updates * multiplier, name)
    amp_increment = after["amp_style"]["updates"] - before["amp_style"]["updates"]
    if amp_increment != updates * after["arguments"]["amp_updates"]:
        raise ValueError("AMP update continuity differs")
    steps["amp_optimizer"] = optimizer_continuity(
        before["amp_style"]["optimizer"], after["amp_style"]["optimizer"],
        amp_increment, "amp_optimizer")
    return {
        "source_iteration": before["iteration"], "iteration": after["iteration"],
        "additional_updates": updates, "actor_normalization_preserved": True,
        "arm_controller_preserved": True, "expert_identity_preserved": True,
        "optimizer_steps": steps, "amp_update_increment": amp_increment,
        "resume_receipt": after.get("resume_receipt"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-iteration", type=int, required=True)
    args = parser.parse_args(argv)
    before, after = (torch.load(path, map_location="cpu", weights_only=False)
                     for path in (args.source, args.checkpoint))
    report = verify(before, after, args.source_iteration)
    report["hashes"] = {}
    for name in ("source", "checkpoint"):
        with getattr(args, name).open("rb") as stream:
            report["hashes"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

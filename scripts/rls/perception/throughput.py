"""Summarize completed PPO updates without equating them to SuperMimic updates."""

import argparse
import csv
import statistics
from pathlib import Path

from common import load, save


def summarize(run, warmup=100):
    run = Path(run)
    config = load(run / "run_config.json")
    with (run / "train_log.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row["completed_updates"]) > warmup]
    if not rows:
        raise ValueError("No measured updates after warmup")
    durations = [float(row["iteration_seconds"]) for row in rows]
    elapsed = sum(durations)
    transitions = config["arguments"]["num_envs"] * config["ppo"]["num_steps_per_env"]
    return {
        "task": config["arguments"]["task"], "num_envs": config["arguments"]["num_envs"],
        "measured_updates": len(rows), "warmup_updates_excluded": warmup,
        "first_completed_update": int(rows[0]["completed_updates"]),
        "last_completed_update": int(rows[-1]["completed_updates"]),
        "elapsed_seconds": elapsed, "ppo_updates_per_second": len(rows) / elapsed,
        "transitions_per_second": len(rows) * transitions / elapsed,
        "median_update_seconds": statistics.median(durations),
        "peak_allocated_gib": max(float(row["peak_allocated_gib"]) for row in rows),
        "tele_reference_outer_updates_per_second": 2.47115,
        "tele_reference_seconds": 404.67, "tele_reference_updates": 1000,
        "algorithm_iteration_parity": False,
        "protocol": "Wall intervals between logged updates; previous logging and checkpoint IO included. Initialization and selected warmup excluded.",
        "comparison_limit": "PPO uses 24 simulator control steps and 20 minibatch optimizer steps per update. Tele uses a world model and imagined policy updates; iteration rates measure different workloads.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    result = summarize(args.run_dir, args.warmup)
    save(args.run_dir / "throughput.json", result)
    print(__import__("json").dumps(result, indent=2))


if __name__ == "__main__":
    main()

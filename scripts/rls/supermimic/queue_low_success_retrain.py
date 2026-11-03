"""Queue 10,000-iteration follow-ups for low-success terrain jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.terrain import TASKS
from supervise_terrain import save


def now():
    return datetime.now(timezone.utc).isoformat()


def build_followup_plan(campaign, iterations=10000, threshold=.9):
    """Select completed jobs below the validation threshold for fresh training."""
    jobs = []
    initial_checkpoints = {}
    for order, job in enumerate(campaign.get("jobs", [])):
        task = job["task"]
        if task not in TASKS or job.get("phase") != "completed":
            continue
        metrics = job.get("validation_metrics", {})
        score = metrics.get("min_seed_macro_success")
        checkpoint = job.get("best_checkpoint")
        if score is None or score >= threshold or not checkpoint:
            continue
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            continue
        jobs.append({
            "task": task,
            "iterations": iterations,
            "phase": "pending",
            "reason": "low_validation_success",
            "source_order": order,
            "source_checkpoint": str(checkpoint_path),
            "source_min_seed_macro_success": score,
        })
        initial_checkpoints[task] = str(checkpoint_path)
    jobs.sort(key=lambda row: (row["source_min_seed_macro_success"], row["source_order"]))
    for job in jobs:
        job.pop("source_order")
    return {
        "source_campaign": campaign.get("campaign_path"),
        "source_campaign_phase": campaign.get("phase"),
        "threshold_min_seed_macro_success": threshold,
        "iterations": iterations,
        "jobs": jobs,
        "initial_checkpoints": initial_checkpoints,
    }


def read_campaign(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    value["campaign_path"] = str(Path(path).resolve())
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=8096)
    parser.add_argument("--threshold", type=float, default=.9)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--warm-start-min-success", type=float, default=.5)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.iterations, args.num_envs, args.poll_seconds) < 1:
        raise SystemExit("Iterations, environment count and poll interval must be positive")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("Low-success threshold must be between zero and one")
    if not 0 <= args.warm_start_min_success <= 1:
        raise SystemExit("Warm-start threshold must be between zero and one")
    if len(args.validation_seeds) < 2 or len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise SystemExit("Validation seeds must be unique and contain at least two values")
    source = args.source_campaign.resolve()
    root = args.run_dir.resolve()
    if source == root or source in root.parents:
        raise SystemExit("Follow-up output must be separate from the source campaign")
    if not source.is_file():
        raise SystemExit(f"Source campaign does not exist: {source}")

    while True:
        campaign = read_campaign(source)
        phase = campaign.get("phase")
        if phase != "running" or args.plan:
            break
        print(f"{now()} waiting for source campaign: {phase}", flush=True)
        time.sleep(args.poll_seconds)

    plan = build_followup_plan(campaign, args.iterations, args.threshold)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "low_success_retrain_plan.json"
    save(plan_path, plan)
    if not plan["jobs"]:
        save(root / "campaign.json", {
            "phase": "no_low_success_jobs",
            "created_at": now(),
            "source_campaign": str(source),
            "threshold_min_seed_macro_success": args.threshold,
        })
        print("No completed terrain job is below the low-success threshold.", flush=True)
        return 0

    initial_path = root / "initial_checkpoints.json"
    save(initial_path, plan["initial_checkpoints"])
    command = [
        sys.executable, "-u",
        str(Path(__file__).with_name("supervise_terrain.py")),
        "--run-dir", str(root),
        "--num-envs", str(args.num_envs),
        "--iterations", str(args.iterations),
        "--initial-checkpoints", str(initial_path),
        "--warm-start-min-success", str(args.warm_start_min_success),
        "--validation-seeds", *[str(seed) for seed in args.validation_seeds],
        "--tasks", *[job["task"] for job in plan["jobs"]],
    ]
    save(root / "launch_receipt.json", {
        "created_at": now(),
        "source_campaign": str(source),
        "command": command,
    })
    return subprocess.call(command, cwd=REPO)


if __name__ == "__main__":
    raise SystemExit(main())

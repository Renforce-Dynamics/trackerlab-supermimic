"""Queue 10,000-iteration follow-ups for low-success terrain jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
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
    if iterations < 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid follow-up iteration count or success threshold")
    jobs = []
    initial_checkpoints = {}
    for order, job in enumerate(campaign.get("jobs", [])):
        task = job["task"]
        if job.get("phase") != "completed":
            continue
        if task not in TASKS:
            raise ValueError(f"Unknown completed terrain task: {task}")
        metrics = job.get("validation_metrics", {})
        score = metrics.get("min_seed_macro_success")
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Missing or invalid validation success for {task}")
        if score >= threshold:
            continue
        checkpoint = job.get("best_checkpoint")
        if not checkpoint:
            raise ValueError(f"Missing best checkpoint for low-success task {task}")
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if task in initial_checkpoints:
            raise ValueError(f"Duplicate terrain task: {task}")
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
    if 42 in args.validation_seeds:
        raise SystemExit("Validation seeds must differ from training seed 42")
    source = args.source_campaign.resolve()
    root = args.run_dir.resolve()
    baseline = source.parent
    if root == baseline or baseline in root.parents or root in baseline.parents:
        raise SystemExit("Follow-up output must be separate from the source campaign")
    if not source.is_file():
        raise SystemExit(f"Source campaign does not exist: {source}")
    if args.plan:
        campaign = read_campaign(source)
        plan = build_followup_plan(campaign, args.iterations, args.threshold)
        print(json.dumps(plan, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".low-success-followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {
            "source_campaign": str(source), "iterations": args.iterations,
            "num_envs": args.num_envs, "threshold": args.threshold,
            "warm_start_min_success": args.warm_start_min_success,
            "validation_seeds": args.validation_seeds,
        }
        config_path = root / "followup_config.json"
        if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise SystemExit("Follow-up settings differ from the saved queue")
        save(config_path, config)
        stop = {"requested": False}
        signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
        signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
        status = {"pid": os.getpid(), "started_at": now(), **config}

        def record(phase, **extra):
            status.update(phase=phase, updated_at=now(), **extra)
            save(root / "followup_status.json", status)

        try:
            plan_path = root / "low_success_retrain_plan.json"
            frozen = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
            while True:
                if stop["requested"] or (root / "STOP").exists():
                    record("stopped")
                    return 0
                campaign = read_campaign(source)
                plan = (frozen if frozen and frozen.get("source_campaign_phase") == "completed"
                        else build_followup_plan(campaign, args.iterations, args.threshold))
                save(plan_path, plan)
                if (campaign.get("phase") == "completed"
                        and all(job.get("phase") == "completed" for job in campaign.get("jobs", []))):
                    break
                phase = ("waiting_for_source" if campaign.get("phase") == "running"
                         else "waiting_for_source_recovery")
                record(phase, source_phase=campaign.get("phase"),
                       source_updated_at=campaign.get("updated_at"), queued_jobs=plan["jobs"])
                print(f"{now()} {phase}; {len(plan['jobs'])} low-success jobs queued", flush=True)
                time.sleep(args.poll_seconds)
            if not plan["jobs"]:
                record("no_low_success_jobs", queued_jobs=[])
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
                "--prefer-initial-checkpoints", "--wait-for-lock",
                "--warm-start-min-success", str(args.warm_start_min_success),
                "--validation-seeds", *[str(seed) for seed in args.validation_seeds],
                "--tasks", *[job["task"] for job in plan["jobs"]],
            ]
            save(root / "launch_receipt.json", {
                "created_at": now(), "source_campaign": str(source), "command": command,
            })
            with subprocess.Popen(command, cwd=REPO) as child:
                record("running", child_pid=child.pid, queued_jobs=plan["jobs"])
                while child.poll() is None:
                    if stop["requested"]:
                        (root / "STOP").touch()
                    time.sleep(1)
                result = root / "campaign.json"
                phase = json.loads(result.read_text(encoding="utf-8")).get("phase") if result.is_file() else "failed"
                record(phase if child.returncode == 0 else "failed", exit_code=child.returncode)
                return child.returncode
        except BaseException as error:
            record("failed", error=repr(error))
            raise


if __name__ == "__main__":
    raise SystemExit(main())

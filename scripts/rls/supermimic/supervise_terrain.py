"""Run the 18 SuperMimic terrain jobs sequentially with resumable receipts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.terrain import TASKS


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-steps", type=int, default=1500)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    plan = [{"task": name, "iterations": args.iterations, "phase": "pending"} for name in args.tasks]
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if min(args.num_envs, args.iterations) < 1:
        parser.error("Environment count and iteration target must be positive")
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "campaign.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        for job in plan:
            for old in previous.get("jobs", []):
                if (old["task"] == job["task"] and old.get("phase") == "completed"
                        and old.get("completed_iterations", 0) >= args.iterations):
                    job.update(old)
        (root / "STOP").unlink(missing_ok=True)
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    import fcntl
    (REPO / "logs").mkdir(exist_ok=True)
    lock = (REPO / "logs" / ".supermimic-terrain-gpu0.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"phase": "running", "pid": os.getpid(), "started_at": now(),
              "num_envs": args.num_envs, "jobs": plan}

    def record():
        status["updated_at"] = now()
        save(receipt_path, status)

    def evaluate(job, checkpoint, run):
        job["phase"] = "evaluating"
        status["active_job"] = job
        record()
        evaluation = run / "evaluation.json"
        with (run / "evaluation_stdout.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).with_name("evaluate_terrain.py")),
                 "--checkpoint", str(checkpoint), "--output", str(evaluation),
                 "--steps", str(args.evaluation_steps), "--seed", str(args.seed + 1)],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            )
            status["child_pid"] = child.pid
            while child.poll() is None:
                record()
                time.sleep(5)
        job.update(evaluation_exit_code=child.returncode, evaluation=str(evaluation),
                   phase="completed" if child.returncode == 0 and evaluation.exists() else "failed",
                   finished_at=now())
        record()
        return job["phase"] == "completed"

    try:
        record()
        previous_by_robot = {}
        for job in plan:
            if stop["requested"] or (root / "STOP").exists():
                break
            if job.get("phase") == "completed":
                previous_by_robot[TASKS[job["task"]].robot] = Path(job["checkpoint"])
                continue
            task = TASKS[job["task"]]
            run = root / task.robot / task.profile
            run.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(Path(__file__).with_name("train_terrain.py")),
                       "--task", task.id, "--run-dir", str(run),
                       "--config", str(REPO / f"scripts/rls/supermimic/configs/terrain_{task.profile}.json"),
                       "--num-envs", str(args.num_envs), "--iterations", str(args.iterations),
                       "--seed", str(args.seed)]
            checkpoint = run / "last.pt"
            summary_path = run / "run_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("phase") == "completed" and summary.get("completed_iterations") >= args.iterations:
                    job.update(phase="completed", checkpoint=str(checkpoint),
                               completed_iterations=summary["completed_iterations"])
                    previous_by_robot[task.robot] = checkpoint
                    if not evaluate(job, checkpoint, run):
                        break
                    continue
            (run / "STOP").unlink(missing_ok=True)
            if checkpoint.exists():
                command.extend(("--resume", str(checkpoint)))
            elif task.robot in previous_by_robot:
                command.extend(("--warm-start", str(previous_by_robot[task.robot])))
            job.update(phase="training", run_dir=str(run), started_at=now(), command=command)
            status["active_job"] = job
            record()
            with (run / "train_stdout.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                status["child_pid"] = child.pid
                record()
                while child.poll() is None:
                    if stop["requested"] or (root / "STOP").exists():
                        stop["requested"] = True
                        (run / "STOP").touch()
                    log_path = run / "train_log.csv"
                    if log_path.exists():
                        import csv
                        with log_path.open(newline="", encoding="utf-8") as stream:
                            last = next(reversed(list(csv.DictReader(stream))), None)
                        if last:
                            job.update(completed_iterations=int(last["iteration"]),
                                       iterations_per_second=float(last["iterations_per_second"]),
                                       terrain_level_mean=float(last["terrain_level_mean"]),
                                       success_fraction=float(last["success_fraction"]),
                                       fall_fraction=float(last["fall_fraction"]))
                    record()
                    time.sleep(5)
            job["train_exit_code"] = child.returncode
            if child.returncode != 0 or not summary_path.exists():
                job.update(phase="failed", finished_at=now())
                record()
                break
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            job.update(phase=summary["phase"], checkpoint=summary["checkpoint"],
                       completed_iterations=summary["completed_iterations"],
                       finished_at=now())
            if summary["phase"] == "completed":
                previous_by_robot[task.robot] = checkpoint
                if not evaluate(job, checkpoint, run):
                    break
            record()
            if summary["phase"] != "completed":
                break
        failures = sum(job.get("phase") == "failed" for job in plan)
        unfinished = any(job.get("phase") != "completed" for job in plan)
        phase = ("completed_with_failures" if failures else
                 "stopped" if stop["requested"] or (root / "STOP").exists() or unfinished else "completed")
        status.update(phase=phase,
                      failed_jobs=failures, finished_at=now())
        record()
    except BaseException as error:
        status.update(phase="failed", error=repr(error), finished_at=now())
        record()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()

"""Run the 18 SuperMimic terrain jobs sequentially with resumable receipts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
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


def evaluation_metrics(result):
    lanes = result["lanes"]
    if len(lanes) != 6 or any(row["episodes"] <= 0 for row in lanes):
        raise ValueError("Evaluation must contain completed episodes at all six difficulty levels")
    episodes = sum(row["episodes"] for row in lanes)
    return {
        "episodes": episodes,
        "success_fraction": sum(row["successes"] for row in lanes) / episodes,
        "fall_fraction": sum(row["falls"] for row in lanes) / episodes,
        "macro_success": sum(row["successes"] / row["episodes"] for row in lanes) / len(lanes),
        "macro_fall": sum(row["falls"] / row["episodes"] for row in lanes) / len(lanes),
    }


def choose_checkpoint(candidates):
    return max(candidates, key=lambda row: (row["metrics"]["macro_success"],
                                            -row["metrics"]["macro_fall"],
                                            row["iteration"]))


def trusted_checkpoint(job, threshold):
    metrics = job.get("validation_metrics", {})
    if metrics.get("min_seed_macro_success", -1) >= threshold:
        return Path(job["best_checkpoint"])
    return None


def warm_start_source(task, initial_checkpoints, previous_by_robot, prefer_initial=False):
    if prefer_initial and task.id in initial_checkpoints:
        return initial_checkpoints[task.id]
    if task.robot in previous_by_robot:
        return previous_by_robot[task.robot][1]
    return initial_checkpoints.get(task.id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--num-envs", type=int, default=8096)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-steps", type=int, default=1500)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--warm-start-min-success", type=float, default=.5)
    parser.add_argument("--initial-checkpoints", type=Path)
    parser.add_argument("--prefer-initial-checkpoints", action="store_true")
    parser.add_argument("--wait-for-lock", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    plan = [{"task": name, "iterations": args.iterations, "phase": "pending"} for name in args.tasks]
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if min(args.num_envs, args.iterations) < 1:
        parser.error("Environment count and iteration target must be positive")
    if (len(args.validation_seeds) < 2 or len(set(args.validation_seeds)) != len(args.validation_seeds)
            or args.seed in args.validation_seeds):
        parser.error("Use at least two unique validation seeds distinct from training")
    if not 0 <= args.warm_start_min_success <= 1:
        parser.error("Warm-start threshold must be between zero and one")
    initial_checkpoints = {}
    if args.initial_checkpoints:
        initial_checkpoints = json.loads(args.initial_checkpoints.read_text(encoding="utf-8"))
        if any(name not in TASKS for name in initial_checkpoints):
            parser.error("Initial checkpoint map contains an unknown task")
        initial_checkpoints = {name: Path(path).resolve()
                               for name, path in initial_checkpoints.items()}
        if any(not path.is_file() for path in initial_checkpoints.values()):
            parser.error("An initial checkpoint is missing")
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
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not args.wait_for_lock:
                raise
            if stop["requested"] or (root / "STOP").exists():
                lock.close()
                return
            time.sleep(1)
    status = {"phase": "running", "pid": os.getpid(), "started_at": now(),
              "num_envs": args.num_envs, "jobs": plan,
              "validation_seeds": args.validation_seeds,
              "warm_start_min_success": args.warm_start_min_success,
              "prefer_initial_checkpoints": args.prefer_initial_checkpoints}

    def record():
        status["updated_at"] = now()
        save(receipt_path, status)

    def evaluate_one(job, checkpoint, output, seed, run):
        with (run / "evaluation_stdout.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).with_name("evaluate_terrain.py")),
                 "--checkpoint", str(checkpoint), "--output", str(output),
                 "--steps", str(args.evaluation_steps), "--seed", str(seed)],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            )
            status["child_pid"] = child.pid
            while child.poll() is None:
                record()
                time.sleep(5)
        job["evaluation_exit_code"] = child.returncode
        if child.returncode != 0 or not output.exists():
            job.update(phase="failed", failed_evaluation=str(output), finished_at=now())
            record()
            return None
        result = json.loads(output.read_text(encoding="utf-8"))
        return {"checkpoint": str(checkpoint), "iteration": result["trained_iterations"],
                "evaluation": str(output), "seed": seed, "metrics": evaluation_metrics(result)}

    def evaluate(job, checkpoint, run):
        job["phase"] = "evaluating"
        status["active_job"] = job
        record()
        selection_seed, *test_seeds = args.validation_seeds
        milestones = sorted(run.glob("iteration_*.pt"))
        latest_milestone = run / f"iteration_{job['completed_iterations']:06d}.pt"
        if latest_milestone not in milestones:
            milestones.append(checkpoint)
        candidates = []
        for candidate in milestones:
            if stop["requested"] or (root / "STOP").exists():
                job["phase"] = "stopped"
                record()
                return False
            row = evaluate_one(job, candidate, run / f"selection_{candidate.stem}.json",
                               selection_seed, run)
            if row is None:
                return False
            candidates.append(row)
            save(run / "checkpoint_selection.json", {"seed": selection_seed, "candidates": candidates})
        best = choose_checkpoint(candidates)
        best_path = run / "best.pt"
        shutil.copy2(best["checkpoint"], run / "best.pt.tmp")
        (run / "best.pt.tmp").replace(best_path)
        shutil.copy2(best["evaluation"], run / "evaluation.json")
        validation = []
        for seed in test_seeds:
            if stop["requested"] or (root / "STOP").exists():
                job["phase"] = "stopped"
                record()
                return False
            row = evaluate_one(job, best_path, run / f"evaluation_seed_{seed}.json", seed, run)
            if row is None:
                return False
            validation.append(row)
        all_rows = [best, *validation]
        metrics = {
            "min_seed_macro_success": min(row["metrics"]["macro_success"] for row in all_rows),
            "test_macro_success": sum(row["metrics"]["macro_success"] for row in validation) / len(validation),
            "test_macro_fall": sum(row["metrics"]["macro_fall"] for row in validation) / len(validation),
        }
        save(run / "validation_summary.json",
             {"selected": best, "test_seeds": validation, "metrics": metrics})
        job.update(evaluation=str(run / "evaluation.json"), best_checkpoint=str(best_path),
                   best_iteration=best["iteration"], validation_metrics=metrics,
                   phase="completed", finished_at=now())
        record()
        return True

    def remember(job, previous_by_robot):
        checkpoint = trusted_checkpoint(job, args.warm_start_min_success)
        if checkpoint is not None:
            robot = TASKS[job["task"]].robot
            score = job["validation_metrics"]["min_seed_macro_success"]
            if robot not in previous_by_robot or score > previous_by_robot[robot][0]:
                previous_by_robot[robot] = (score, checkpoint)

    try:
        record()
        previous_by_robot = {}
        for job in plan:
            if stop["requested"] or (root / "STOP").exists():
                break
            if job.get("phase") == "completed":
                remember(job, previous_by_robot)
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
                    if not evaluate(job, checkpoint, run):
                        break
                    remember(job, previous_by_robot)
                    continue
            (run / "STOP").unlink(missing_ok=True)
            if checkpoint.exists():
                command.extend(("--resume", str(checkpoint)))
            else:
                source = warm_start_source(task, initial_checkpoints, previous_by_robot,
                                           args.prefer_initial_checkpoints)
                if source is not None:
                    command.extend(("--warm-start", str(source)))
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
                if not evaluate(job, checkpoint, run):
                    break
                remember(job, previous_by_robot)
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

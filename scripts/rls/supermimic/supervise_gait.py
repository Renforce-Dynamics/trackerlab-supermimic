"""Train and qualify six SuperMimic gait foundations before advanced tasks."""

import argparse
import csv
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
from trackerTask.supermimic.gait import QUALITY_LIMITS, ROBOTS
from supervise_terrain import save


def now():
    return datetime.now(timezone.utc).isoformat()


def candidate_score(evaluation):
    metrics = evaluation["metrics"]
    return (evaluation["quality_passed"], metrics["full_clip_survival"],
            -metrics["foot_position_rmse_scale_m"], -metrics["joint_rmse_rad"])


def find_training_attempt(parent, target):
    """Return a completed attempt, an unfinished attempt to resume, or a fresh slot."""
    attempts = sorted(parent.glob("attempt_*"), reverse=True)
    for run in attempts:
        summary_path = run / "run_summary.json"
        checkpoint = run / "last.pt"
        if not checkpoint.is_file():
            continue
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        completed = int(summary.get("completed_iterations", 0))
        if summary.get("stop_reason") == "iterations" and completed >= target:
            return run, checkpoint, "completed"
        if completed < target:
            return run, checkpoint, "resume"
    attempt = len(list(parent.glob("attempt_*"))) + 1
    run = parent / f"attempt_{attempt:02d}"
    run.mkdir(parents=True, exist_ok=True)
    return run, None, "fresh"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=8096)
    parser.add_argument("--robots", nargs="+", choices=ROBOTS, default=list(ROBOTS))
    args = parser.parse_args()
    if min(args.iterations, args.max_rounds, args.num_envs) < 1:
        parser.error("Queue dimensions must be positive")
    reference_path = args.references.resolve()
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    references = {robot: (reference_path.parent / path).resolve()
                  for robot, path in references.items()}
    if any(robot not in references for robot in args.robots):
        parser.error("Every requested robot needs a reference directory")
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    (REPO / "logs").mkdir(exist_ok=True)
    gpu_lock = (REPO / "logs/.supermimic-terrain-gpu0.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    receipt = root / "campaign.json"
    status = (json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists()
              else {"stage": "gait-foundation", "jobs": [
                  {"robot": robot, "round": 1, "phase": "pending", "new_iterations": args.iterations}
                  for robot in args.robots], "quality_limits": QUALITY_LIMITS,
                    "advanced_tasks": {"phase": "held_until_gait_qualification",
                                       "count": 18, "old_terrain_objective": "retired"}})
    status.update(phase="running", pid=os.getpid(), started_at=now(), num_envs=args.num_envs)

    def record():
        status["updated_at"] = now()
        save(receipt, status)

    def stopping():
        return stop["requested"] or (root / "STOP").exists()

    def invoke(command, run, kind, job):
        with (run / (kind + "_stdout.log")).open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            status.update(active_job=job, child_pid=child.pid)
            record()
            while child.poll() is None:
                if stopping():
                    (run / "STOP").touch()
                if kind == "train" and (run / "train_log.csv").exists():
                    with (run / "train_log.csv").open(newline="") as log:
                        rows = list(csv.DictReader(log))
                    if rows:
                        row = rows[-1]
                        job.update(completed_iterations=int(row["iteration"]),
                                   iterations_per_second=float(row["it_per_s"]),
                                   normalized_return=float(row["ep_ret_norm"]))
                record()
                time.sleep(5)
        job[kind + "_exit_code"] = child.returncode
        record()
        return child.returncode == 0

    try:
        record()
        index = 0
        while index < len(status["jobs"]):
            if stopping():
                break
            job = status["jobs"][index]
            index += 1
            if job.get("phase") == "completed":
                continue
            robot = job["robot"]
            ref = references[robot]
            job["reference_directory"] = str(ref)
            while not (ref / "gait_audit.json").exists():
                job["phase"] = "waiting_references"
                record()
                if stopping():
                    break
                time.sleep(10)
            if stopping():
                break
            audit = json.loads((ref / "gait_audit.json").read_text(encoding="utf-8"))
            if audit["status"] != "passed":
                job.update(phase="reference_review_required", finished_at=now())
                record()
                continue
            parent = root / robot / f"round_{job['round']:02d}"
            parent.mkdir(parents=True, exist_ok=True)
            previous_source = Path(job["source_checkpoint"]) if job.get("source_checkpoint") else None
            initial_iteration = job.get("source_iteration", 0)
            target = job.setdefault("target_iteration", initial_iteration + args.iterations)
            run, resume_checkpoint, attempt_state = find_training_attempt(parent, target)
            if attempt_state == "resume":
                source = resume_checkpoint
            elif attempt_state == "fresh":
                source = previous_source
            else:
                source = None
            if attempt_state != "completed":
                job.update(phase="training", run_dir=str(run), started_at=now())
                command = [
                    sys.executable, "-u", str(Path(__file__).with_name("train_gait.py")),
                    "--task", robot, "--motion_config", str(ref / "train.json"),
                    "--log_dir", str(run), "--num_envs", str(args.num_envs),
                    "--iterations", str(target), "--lr_decay_iters", str(args.max_rounds * args.iterations),
                ]
                if source:
                    command.extend(("--resume", str(source)))
                job["command"] = command
                if not invoke(command, run, "train", job):
                    job.update(phase="failed", finished_at=now())
                    break
            summary = json.loads((run / "run_summary.json").read_text())
            if summary["stop_reason"] != "iterations":
                job.update(phase="stopped", completed_iterations=summary["completed_iterations"])
                break
            job["phase"] = "evaluating"
            candidates = []
            checkpoints = ([previous_source] if previous_source is not None else [])
            checkpoints.extend(sorted(parent.glob("attempt_*/ckpt_*.pt")))
            unique, seen = [], set()
            for checkpoint in checkpoints:
                if checkpoint is None or not checkpoint.is_file() or checkpoint.resolve() in seen:
                    continue
                unique.append(checkpoint)
                seen.add(checkpoint.resolve())
            checkpoints = unique
            if not checkpoints:
                raise RuntimeError(f"No selectable gait checkpoints were saved for {robot}")
            for checkpoint in checkpoints:
                output = run / f"selection_{checkpoint.stem}.json"
                if stopping():
                    break
                command = [sys.executable, "-u", str(Path(__file__).with_name("evaluate_gait.py")),
                           "--checkpoint", str(checkpoint), "--motion-config", str(ref / "train.json"),
                           "--output", str(output), "--seed", "43"]
                if not invoke(command, run, "evaluation", job):
                    raise RuntimeError(f"Selection evaluation failed for {robot}")
                candidates.append((checkpoint, json.loads(output.read_text())))
            if stopping():
                break
            checkpoint, selection = max(candidates, key=lambda row: candidate_score(row[1]))
            best = run / "best.pt"
            if checkpoint.resolve() != best.resolve():
                shutil.copy2(checkpoint, best)
            save(run / "checkpoint_selection.json", {
                "selection_split": "training recordings", "selected": str(checkpoint),
                "candidates": [value for _, value in candidates],
            })
            tests = []
            for seed in (44, 45):
                output = run / f"evaluation_seed_{seed}.json"
                command = [sys.executable, "-u", str(Path(__file__).with_name("evaluate_gait.py")),
                           "--checkpoint", str(best), "--motion-config", str(ref / "test.json"),
                           "--output", str(output), "--seed", str(seed)]
                if seed == 44:
                    command.extend(("--trajectory-output", str(run / "trajectories.npz")))
                if not invoke(command, run, "evaluation", job):
                    raise RuntimeError(f"Held-out evaluation failed for {robot}")
                tests.append(json.loads(output.read_text()))
                if stopping():
                    break
            if stopping():
                break
            demo_command = [
                sys.executable, "-u", str(Path(__file__).with_name("render_demo.py")),
                "--trajectory", str(run / "trajectories.npz"),
                "--evaluation", str(run / "evaluation_seed_44.json"),
                "--output", str(run / "demo.mp4"), "--seconds", "30",
            ]
            if not invoke(demo_command, run, "demo", job):
                raise RuntimeError(f"Gait demo rendering failed for {robot}")
            passed = selection["quality_passed"] and all(value["quality_passed"] for value in tests)
            validation = {"quality_passed": passed, "selection": selection,
                          "held_out_tests": tests, "limits": QUALITY_LIMITS}
            save(run / "gait_quality.json", validation)
            job.update(phase="completed", quality_passed=passed, best_checkpoint=str(best),
                       best_iteration=selection["iteration"], validation=str(run / "gait_quality.json"),
                       demo=str(run / "demo.mp4"), finished_at=now())
            if not passed and job["round"] < args.max_rounds:
                retry_id = (robot, job["round"] + 1)
                if not any((j["robot"], j["round"]) == retry_id for j in status["jobs"]):
                    status["jobs"].append({
                        "robot": robot, "round": retry_id[1], "phase": "pending",
                        "new_iterations": args.iterations, "source_checkpoint": str(best),
                        "source_iteration": selection["iteration"],
                    })
            record()
        all_qualified = all(any(
            j["robot"] == robot and j.get("quality_passed") for j in status["jobs"])
            for robot in args.robots)
        if all_qualified:
            foundations = {}
            for robot in args.robots:
                selected = max(
                    (job for job in status["jobs"] if job["robot"] == robot and job.get("quality_passed")),
                    key=lambda job: (json.loads(Path(job["validation"]).read_text())["selection"]["iteration"],
                                     job["round"]),
                )
                foundations[robot] = {
                    "checkpoint": selected["best_checkpoint"],
                    "quality": selected["validation"],
                    "iteration": selected["best_iteration"],
                }
            save(root / "gait_foundations.json", foundations)
        status.update(phase="stopped" if stopping() else
                      "failed" if any(j["phase"] == "failed" for j in status["jobs"]) else
                      "gait_qualified" if all_qualified else "gait_review_required",
                      finished_at=now())
        record()
    except BaseException as error:
        status.update(phase="failed", error=repr(error), finished_at=now())
        record()
        raise
    finally:
        gpu_lock.close()


if __name__ == "__main__":
    main()

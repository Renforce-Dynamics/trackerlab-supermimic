"""Schedule command/style foundations, stealth, and perception terrain training."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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
from trackerTask.supermimic.terrain import ALL_TASKS, ROBOTS, TASKS
from supervise_terrain import save


def now():
    return datetime.now(timezone.utc).isoformat()


def build_plan(iterations=10000):
    foundations = [
        {"task": f"locomotion/{robot}/velocity", "robot": robot, "profile": "velocity",
         "style": "walk", "depends_on": None}
        for robot in ROBOTS
    ]
    stealth = [
        {"task": f"locomotion/{robot}/stealth", "robot": robot, "profile": "stealth",
         "style": "stealth", "depends_on": f"locomotion/{robot}/velocity"}
        for robot in ROBOTS if robot != "go2"
    ]
    terrain = [
        {"task": task.id, "robot": task.robot, "profile": task.profile,
         "style": "parkour", "depends_on": f"locomotion/{task.robot}/velocity"}
        for task in TASKS.values()
    ]
    return [dict(job, phase="pending", iterations=iterations, round=1)
            for job in foundations + stealth + terrain]


def command_quality(reports, profile):
    """Physical tracking and survival gate; discriminator scores alone never qualify."""
    if len(reports) < 2:
        return {"passed": False, "reason": "two held-out physical evaluations required"}
    rows, style_scores, gait_rows = [], [], []
    for report in reports:
        metrics = report.get("command_metrics", {})
        lanes = report.get("lanes", [])
        if len(lanes) != 6 or any(
            any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key])
                for key in ("episodes", "success_fraction", "fall_fraction"))
            or row["episodes"] < 1 or not 0 <= row["success_fraction"] <= 1
            or not 0 <= row["fall_fraction"] <= 1 for row in lanes
        ):
            return {"passed": False, "reason": "completed episodes missing"}
        rows.append({
            "macro_success": sum(row["success_fraction"] for row in lanes) / 6,
            "macro_fall": sum(row["fall_fraction"] for row in lanes) / 6,
            "velocity_rmse_scale_m_s": metrics.get("velocity_rmse_scale_m_s", float("inf")),
            "yaw_rate_rmse_rad_s": metrics.get("yaw_rate_rmse_rad_s", float("inf")),
        })
        score = report.get("style_metrics", {}).get("mean_reward_rate", -1.)
        if (not isinstance(score, (int, float)) or not math.isfinite(score)
                or not 0 <= score <= 1
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in rows[-1].values())):
            return {"passed": False, "reason": "nonfinite or invalid evaluation metric"}
        style_scores.append(score)
        gait = report.get("gait_metrics", {})
        gait_limits = {"support_foot_slip_scale_m_s": .12,
                       "close_feet_fraction": .1,
                       "joint_acceleration_squared": 2.5e5}
        if any(type(gait.get(name)) not in (int, float) or not math.isfinite(gait[name])
               or gait[name] < 0 for name in gait_limits):
            return {"passed": False, "reason": "independent foot and acceleration measurements missing"}
        gait_rows.append({name: gait[name] for name in gait_limits})
    limits = {"success": .9 if profile in ("velocity", "stealth") else .8,
              "fall": .1 if profile in ("velocity", "stealth") else .2,
              "velocity_rmse_scale_m_s": .35, "yaw_rate_rmse_rad_s": .5}
    passed = min(style_scores) >= .55 and all(
        row["macro_success"] >= limits["success"]
        and row["macro_fall"] <= limits["fall"]
        and row["velocity_rmse_scale_m_s"] <= limits["velocity_rmse_scale_m_s"]
        and row["yaw_rate_rmse_rad_s"] <= limits["yaw_rate_rmse_rad_s"]
        for row in rows
    ) and all(row[name] <= limit for row in gait_rows for name, limit in gait_limits.items())
    return {"passed": passed, "seeds": rows, "limits": limits,
            "style_reward_rate_by_seed": style_scores, "minimum_style_reward_rate": .55,
            "gait_measurements": gait_rows, "gait_limits": gait_limits,
            "visual_style_review": "required; AMP score is not a visual-quality certificate"}


def resolve_styles(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != 1 or not isinstance(value.get("robots"), dict):
        raise ValueError("Style map must contain version=1 and robots")
    result = {}
    for robot, styles in value["robots"].items():
        result[robot] = {}
        for style, name in styles.items():
            if isinstance(name, str):
                result[robot][style] = (path.parent / name).resolve()
    return result


def recover_jobs(status):
    """A restarted owner retries failed work and completes interrupted exports."""
    revalidate_qualifications(status)
    for job in status["jobs"]:
        if job["phase"] == "failed":
            job["phase"] = "pending"
            job["restarted_after_failure"] = True
        elif (job["phase"] == "completed" and job.get("quality_passed")
              and job["task"] not in status["qualified"]):
            job["phase"] = "awaiting_visual_review"


def visual_review_binding(directory, checkpoint):
    """Return the current approval's artifact identities, or fail closed."""
    directory, checkpoint = Path(directory).resolve(), Path(checkpoint).resolve()
    receipt = directory / "visual_review.json"
    trajectory = directory / "trajectories.npz"
    try:
        raw_review = receipt.read_bytes()
        review = json.loads(raw_review.decode("utf-8"))
        if (not isinstance(review, dict) or review.get("passed") is not True
                or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
                or not isinstance(review.get("notes"), str) or not review["notes"].strip()):
            return None
        binding = {
            "checkpoint": str(checkpoint), "trajectory": str(trajectory),
            "visual_review": str(receipt),
            "visual_review_sha256": hashlib.sha256(raw_review).hexdigest(),
        }
        for path, key in ((checkpoint, "checkpoint_sha256"), (trajectory, "trajectory_sha256")):
            with path.open("rb") as stream:
                binding[key] = hashlib.file_digest(stream, "sha256").hexdigest()
            if review.get(key) != binding[key]:
                return None
        return binding
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def visual_review_passed(directory, checkpoint):
    return visual_review_binding(directory, checkpoint) is not None


def revalidate_qualifications(status):
    """Revoke stale approvals and hold dependent jobs without discarding outputs."""
    qualified, removed = status["qualified"], {}
    for task, entry in list(qualified.items()):
        checkpoint = entry.get("checkpoint") if isinstance(entry, dict) else None
        binding = (visual_review_binding(Path(checkpoint).parent, checkpoint)
                   if isinstance(checkpoint, str) and checkpoint else None)
        if binding is None or any(entry.get(key) != value for key, value in binding.items()):
            removed[task] = qualified.pop(task)
    dependencies = {job["task"]: job.get("depends_on") for job in status["jobs"]}
    while True:
        stale = [task for task in qualified
                 if dependencies.get(task) and dependencies[task] not in qualified]
        if not stale:
            break
        for task in stale:
            removed[task] = qualified.pop(task)
    active_phases = {
        "pending", "training", "evaluating", "waiting_style_data",
        "held_for_velocity_quality", "stopped", "exporting", "awaiting_visual_review",
    }
    for job in status["jobs"]:
        if job["task"] in removed and job.get("quality_passed"):
            entry = removed[job["task"]]
            previous = entry.get("checkpoint") if isinstance(entry, dict) else None
            current = job.get("checkpoint")
            if current is None and job.get("run_dir"):
                current = str(Path(job["run_dir"]) / "last.pt")
            if not previous or not current or Path(current).resolve() == Path(previous).resolve():
                job["phase"] = "awaiting_visual_review"
                job.pop("visual_review_binding", None)
        dependency = job.get("depends_on")
        if dependency and dependency not in qualified and job["phase"] in active_phases:
            job["phase"] = "held_for_velocity_quality"
    return set(removed)


def dispatch_ready(status, job, directory, kind):
    """Recheck approvals immediately before training/export and after export."""
    revalidate_qualifications(status)
    dependency = job.get("depends_on")
    if dependency and dependency not in status["qualified"]:
        job["phase"] = "held_for_velocity_quality"
        return False
    if kind == "export":
        binding = visual_review_binding(directory, Path(directory) / "last.pt")
        if (not job.get("quality_passed") or binding is None
                or binding != job.get("visual_review_binding")):
            job["phase"] = "awaiting_visual_review"
            job.pop("visual_review_binding", None)
            return False
    return True


def checkpoint_iteration(path, task):
    """Recover progress if a process died between atomic checkpoint and summary."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from trackerLab.tracker_env.mdp.supermimic_command import COMMAND_VERSION
    if (checkpoint.get("training_stage") != "command-style"
            or checkpoint.get("objective_version") != COMMAND_VERSION
            or checkpoint.get("task") != task):
        raise ValueError("Campaign checkpoint task/objective differs")
    value = checkpoint.get("iteration")
    if type(value) is not int or value < 1:
        raise ValueError("Checkpoint iteration is invalid")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--style-map", type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=8096)
    parser.add_argument("--evaluation-steps", type=int, default=1500)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[44, 45])
    parser.add_argument("--robots", nargs="+", choices=ROBOTS, default=list(ROBOTS))
    parser.add_argument("--tasks", nargs="+", choices=tuple(ALL_TASKS))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    jobs = [job for job in build_plan(args.iterations) if job["robot"] in args.robots]
    if args.tasks:
        selected = set(args.tasks)
        selected.update(f"locomotion/{ALL_TASKS[name].robot}/velocity" for name in args.tasks)
        jobs = [job for job in jobs if job["task"] in selected]
    if not jobs:
        parser.error("Robot and task filters select no jobs")
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return
    if min(args.iterations, args.max_rounds, args.num_envs, args.evaluation_steps) < 1:
        parser.error("Queue sizes must be positive")
    if (len(set(args.validation_seeds)) < 2
            or len(set(args.validation_seeds)) != len(args.validation_seeds)
            or 42 in args.validation_seeds):
        parser.error("Use two distinct held-out seeds excluding the training seed 42")
    if args.style_map is None:
        parser.error("--style-map is required for training")
    styles = resolve_styles(args.style_map.resolve())
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "campaign.json"
    status = (json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else
              {"stage": "supermimic-command-amp-v4", "jobs": jobs, "qualified": {}})
    if status["stage"] != "supermimic-command-amp-v4":
        raise ValueError("Use a fresh campaign for the new command/style objective")
    recover_jobs(status)
    if (root / "STOP").exists():
        parser.error("Campaign STOP marker exists; remove it explicitly before resuming")
    import fcntl
    (REPO / "logs").mkdir(exist_ok=True)
    lock = (REPO / "logs/.supermimic-terrain-gpu0.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stopping.update(requested=True))
    status.update(phase="running", pid=os.getpid(), started_at=now(),
                  style_map=str(args.style_map.resolve()), num_envs=args.num_envs)

    def record():
        status["updated_at"] = now()
        save(receipt, status)

    def stop_requested():
        return stopping["requested"] or (root / "STOP").exists()

    def invoke(command, directory, kind, job):
        if kind in ("train", "export") and not dispatch_ready(status, job, directory, kind):
            record()
            return None
        with (directory / f"{kind}_stdout.log").open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=REPO, stdout=stream,
                                     stderr=subprocess.STDOUT)
            status.update(child_pid=child.pid, active_job=job)
            job["command"] = command
            record()
            while child.poll() is None:
                if stop_requested():
                    (directory / "STOP").touch()
                log = directory / "train_log.csv"
                if kind == "train" and log.exists():
                    with log.open(newline="") as source:
                        rows = list(csv.DictReader(source))
                    if rows:
                        last = rows[-1]
                        job.update(completed_iterations=int(last["iteration"]),
                                   iterations_per_second=float(last["iterations_per_second"]),
                                   amp_reward=float(last["amp_reward"]))
                record()
                time.sleep(5)
        job[kind + "_exit_code"] = child.returncode
        record()
        return child.returncode == 0

    try:
        record()
        while not stop_requested():
            revalidate_qualifications(status)
            ready = None
            for job in status["jobs"]:
                if job["phase"] not in ("pending", "training", "evaluating",
                                         "waiting_style_data", "held_for_velocity_quality",
                                         "stopped", "exporting", "awaiting_visual_review"):
                    continue
                if job["phase"] == "awaiting_visual_review":
                    directory = Path(job["run_dir"])
                    if not visual_review_passed(directory, directory / "last.pt"):
                        continue
                dependency = job["depends_on"]
                if dependency and dependency not in status["qualified"]:
                    job["phase"] = "held_for_velocity_quality"
                    continue
                config = styles.get(job["robot"], {}).get(job["style"])
                if config is None or not config.is_file():
                    job["phase"] = "waiting_style_data"
                    continue
                ready = job
                break
            record()
            if ready is None:
                pending_data = any(job["phase"] == "waiting_style_data" for job in status["jobs"])
                if pending_data:
                    styles = resolve_styles(args.style_map.resolve())
                    time.sleep(15)
                    continue
                break
            job = ready
            run = root / job["robot"] / job["profile"] / f"round_{job['round']:02d}"
            run.mkdir(parents=True, exist_ok=True)
            if (run / "STOP").exists():
                (run / "STOP").unlink()
            job.update(run_dir=str(run), started_at=job.get("started_at", now()))
            source = Path(job["source_checkpoint"]) if job.get("source_checkpoint") else None
            checkpoint = run / "last.pt"
            summary_path = run / "run_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            finished = summary.get("phase") == "completed" and summary.get("completed_iterations", 0) >= job["iterations"]
            if not finished and checkpoint.is_file():
                completed = checkpoint_iteration(checkpoint, job["task"])
                if completed >= job["iterations"]:
                    summary = {"phase": "completed", "completed_iterations": completed,
                               "recovered_from_atomic_checkpoint": True,
                               "checkpoint": str(checkpoint)}
                    save(summary_path, summary)
                    finished = True
            if not finished:
                command = [
                    sys.executable, "-u", str(Path(__file__).with_name("train_terrain.py")),
                    "--task", job["task"], "--run-dir", str(run),
                    "--amp-motion-config", str(styles[job["robot"]][job["style"]]),
                    "--num-envs", str(args.num_envs), "--iterations", str(job["iterations"]),
                    "--amp-weight", ".25",
                ]
                if job["profile"] == "stealth":
                    command.extend(("--clearance-ratio", ".8"))
                if checkpoint.is_file():
                    command.extend(("--resume", str(checkpoint)))
                elif source:
                    command.extend(("--resume", str(source)))
                elif job["depends_on"]:
                    command.extend(("--warm-start", status["qualified"][job["depends_on"]]["checkpoint"]))
                job["phase"] = "training"
                training_result = invoke(command, run, "train", job)
                if training_result is None:
                    continue
                if not training_result:
                    job.update(phase="failed", finished_at=now())
                    status["phase"] = "failed"
                    record()
                    return
                summary = json.loads(summary_path.read_text())
            job["completed_iterations"] = summary["completed_iterations"]
            if summary["phase"] != "completed" or stop_requested():
                job["phase"] = "stopped"
                break
            reviewed = job["phase"] == "awaiting_visual_review"
            job["phase"] = "evaluating"
            reports = []
            for seed in args.validation_seeds:
                output = run / f"evaluation_seed_{seed}.json"
                if reviewed and output.is_file():
                    reports.append(json.loads(output.read_text()))
                    continue
                command = [
                    sys.executable, "-u", str(Path(__file__).with_name("evaluate_terrain.py")),
                    "--checkpoint", str(checkpoint), "--output", str(output),
                    "--steps", str(args.evaluation_steps), "--seed", str(seed),
                ]
                if seed == args.validation_seeds[0]:
                    command.extend(("--trajectory-output", str(run / "trajectories.npz")))
                if not invoke(command, run, "evaluation", job):
                    raise RuntimeError(f"Physical evaluation failed for {job['task']}")
                reports.append(json.loads(output.read_text()))
                if stop_requested():
                    break
            if stop_requested():
                break
            quality = command_quality(reports, job["profile"])
            if not reviewed and (run / "trajectories.npz").exists():
                import shutil
                shutil.copy2(run / f"evaluation_seed_{args.validation_seeds[0]}.json",
                             run / "evaluation.json")
                command = [
                    sys.executable, "-u", str(Path(__file__).with_name("render_style.py")),
                    "--run-dir", str(run),
                ]
                if not invoke(command, run, "demo", job):
                    raise RuntimeError(f"Demo rendering failed for {job['task']}")
            save(run / "quality.json", quality)
            review_binding = visual_review_binding(run, checkpoint) if quality["passed"] else None
            job.update(phase=("exporting" if review_binding else
                              "awaiting_visual_review" if quality["passed"] else "completed"),
                       quality_passed=quality["passed"],
                       visual_review_binding=review_binding,
                       checkpoint=str(checkpoint), quality=str(run / "quality.json"),
                       finished_at=now())
            if review_binding:
                export_command = [
                    sys.executable, "-u", str(Path(__file__).with_name("export_command_policy.py")),
                    "--checkpoint", str(checkpoint), "--output", str(run / "export"),
                ]
                export_result = invoke(export_command, run, "export", job)
                if export_result is None:
                    continue
                if not export_result:
                    raise RuntimeError(f"Actor export failed for {job['task']}")
                if not dispatch_ready(status, job, run, "export"):
                    record()
                    continue
                status["qualified"][job["task"]] = {
                    **review_binding, "quality": str(run / "quality.json"),
                }
                job["phase"] = "completed"
            elif not quality["passed"] and job["round"] < args.max_rounds:
                next_round = job["round"] + 1
                if not any(other["task"] == job["task"] and other["round"] == next_round
                           for other in status["jobs"]):
                    status["jobs"].append({
                        "task": job["task"], "robot": job["robot"], "profile": job["profile"],
                        "style": job["style"], "depends_on": job["depends_on"],
                        "phase": "pending", "round": next_round,
                        "iterations": job["iterations"] + args.iterations,
                        "source_checkpoint": str(checkpoint),
                    })
            record()
        revalidate_qualifications(status)
        requested_tasks = {job["task"] for job in jobs}
        status["phase"] = ("stopped" if stop_requested() else
                           "completed" if requested_tasks.issubset(status["qualified"]) else
                           "quality_review_required")
        status["finished_at"] = now()
        record()
    except BaseException as error:
        status.update(phase="failed", error=repr(error), finished_at=now())
        record()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()

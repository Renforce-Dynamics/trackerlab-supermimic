"""Sequential multi-robot training with exclusive GPU ownership and durable receipts."""

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time
import os

from common import REPO, load, now, save
from trackerTask.native.locomotion.perception.registry import TASKS


def jobs(tasks, stages):
    return [(task, updates) for updates in stages for task in tasks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--stages", type=int, nargs="+", default=(1000, 10000))
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.stages != sorted(set(args.stages)) or min(args.stages) < 1:
        parser.error("Stages must be positive and strictly increasing")
    if min(args.num_envs, args.eval_steps) < 1:
        parser.error("World count and evaluation horizon must be positive")
    plan = [{"task": t, "updates": n} for t, n in jobs(args.tasks, args.stages)]
    if args.plan:
        print(__import__("json").dumps(plan, indent=2))
        return
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign.json").exists():
        parser.error("Campaign already exists; previous outputs are preserved")
    stopped = {"value": False}
    signal.signal(signal.SIGTERM, lambda *unused: stopped.update(value=True))
    signal.signal(signal.SIGINT, lambda *unused: stopped.update(value=True))
    import fcntl
    gpu_lock = (REPO / "logs/.perception-gpu-0.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"phase": "running", "pid": os.getpid(), "started_at": now(),
              "num_envs": args.num_envs, "jobs": plan}

    def record():
        status["updated_at"] = now()
        save(root / "campaign.json", status)

    def run(command, directory, logname, may_stop=False):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / logname).open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            status.update(child_pid=child.pid, command=command, child_log=str(directory / logname))
            record()
            sent = False
            while child.poll() is None:
                if stopped["value"] or (root / "STOP").exists():
                    stopped["value"] = True
                    if may_stop:
                        (directory / "STOP").touch()
                    elif not sent:
                        child.terminate()
                        sent = True
                record()
                time.sleep(5)
            return child.returncode

    previous, robot_seed = {}, {}
    try:
        record()
        for job in plan:
            if stopped["value"] or (root / "STOP").exists():
                break
            task, updates = job["task"], job["updates"]
            spec = TASKS[task]
            directory = root / f"{spec.robot}-{spec.profile}-{updates:06d}"
            job.update(phase="training", run_dir=str(directory), started_at=now())
            status["active_job"] = job
            python = sys.executable
            command = [python, "-u", str(Path(__file__).with_name("train.py")), "--task", task]
            command += ["--run-dir", str(directory), "--num-envs", str(args.num_envs),
                        "--iterations", str(updates)]
            if task in previous:
                command += ["--resume", str(previous[task])]
            elif spec.robot in robot_seed:
                command += ["--warm-start", str(robot_seed[spec.robot])]
            code = run(command, directory, "train_stdout.log", may_stop=True)
            job["train_exit_code"] = code
            summary_path = directory / "run_summary.json"
            if code or not summary_path.exists():
                job.update(phase="failed", finished_at=now())
                record()
                continue
            summary = load(summary_path)
            if summary["phase"] != "completed":
                job.update(phase="stopped", finished_at=now())
                stopped["value"] = True
                break
            checkpoint = Path(summary["checkpoint"])
            previous[task] = checkpoint
            robot_seed.setdefault(spec.robot, checkpoint)
            job.update(phase="evaluating", checkpoint=str(checkpoint))
            command = [python, "-u", str(Path(__file__).with_name("evaluate.py"))]
            command += ["--checkpoint", str(checkpoint), "--output", str(directory / "evaluation"),
                        "--steps", str(args.eval_steps)]
            job["eval_exit_code"] = run(command, directory, "evaluation_stdout.log")
            if stopped["value"] or (root / "STOP").exists():
                job.update(phase="stopped", finished_at=now())
                record()
                break
            if job["eval_exit_code"]:
                job.update(phase="artifact_failed", finished_at=now())
                record()
                continue
            export = [python, str(Path(__file__).with_name("export.py")),
                      "--checkpoint", str(checkpoint), "--output", str(directory / "export")]
            job["export_exit_code"] = run(export, directory, "export_stdout.log")
            if stopped["value"] or (root / "STOP").exists():
                job.update(phase="stopped", finished_at=now())
                record()
                break
            if not job["export_exit_code"]:
                try:
                    from throughput import summarize
                    save(directory / "throughput.json", summarize(directory, min(100, updates // 2)))
                except (ValueError, OSError) as error:
                    job["throughput_error"] = str(error)
            good = not job["eval_exit_code"] and job["export_exit_code"] == 0
            job.update(phase="completed" if good else "artifact_failed", finished_at=now())
            record()
        failures = sum(j.get("phase") in ("failed", "artifact_failed") for j in plan)
        status.update(phase="stopped" if stopped["value"] else
                      ("completed_with_failures" if failures else "completed"),
                      failed_jobs=failures, finished_at=now())
        record()
    except BaseException as error:
        status.update(phase="failed", error=repr(error), finished_at=now())
        record()
        raise
    finally:
        gpu_lock.close()


if __name__ == "__main__":
    main()

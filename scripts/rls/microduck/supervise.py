"""Own one native MicroDuck training process and publish durable status."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def parse_progress(text):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    match = re.search(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", clean)
    if match:
        return {"reported_iteration_index": int(match[1]),
                "completed_updates": int(match[1]) + 1,
                "target_updates": int(match[2])}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_config.json").exists():
        parser.error("This run already exists; use a new directory")
    lock = output / ".supervisor.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    status = {
        "phase": "starting", "supervisor_pid": os.getpid(),
        "target_updates": args.iterations, "num_envs": args.num_envs,
        "completed_updates": 0, "algorithm": "PPO", "backend": "mjlab",
        "run_dir": str(output), "fresh_initialization": True,
    }

    def save():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = output / "job_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        os.replace(temporary, output / "job_status.json")

    command = [sys.executable, "-u", str(Path(__file__).with_name("train.py")),
               "--upstream", str(args.upstream.resolve()), "--run-dir", str(output),
               "--num-envs", str(args.num_envs), "--iterations", str(args.iterations)]
    status["command"] = command
    save()
    child = None
    try:
        with (output / "train_stdout.log").open("w", buffering=1) as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
            status.update(phase="training", trainer_pid=child.pid)
            save()
            for line in child.stdout:
                log.write(line)
                progress = parse_progress(line)
                if progress:
                    status.update(progress)
                    save()
            code = child.wait()
        status["exit_code"] = code
        if code == 0 and (output / "training_finished.json").exists():
            status.update(phase="evaluating", completed_updates=args.iterations)
            checkpoint = output / f"model_{args.iterations - 1}.pt"
            evaluation = output / "milestones" / f"{args.iterations:06d}"
            eval_command = [
                sys.executable, "-u", str(Path(__file__).with_name("evaluate.py")),
                "--checkpoint", str(checkpoint), "--output", str(evaluation),
                "--expected-updates", str(args.iterations),
            ]
            status["evaluation_command"] = eval_command
            save()
            with (output / "evaluation_stdout.log").open("w") as log:
                child = subprocess.Popen(eval_command, stdout=log, stderr=subprocess.STDOUT)
                status["evaluator_pid"] = child.pid
                save()
                status["evaluation_exit_code"] = child.wait()
            if status["evaluation_exit_code"] == 0 and (evaluation / "ready.json").exists():
                status.update(phase="completed", milestone=str(evaluation),
                              visual_review="pending")
            else:
                status["phase"] = "evaluation_failed"
        else:
            status["phase"] = "failed"
        save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        status.update(phase="failed", error=repr(error))
        save()
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

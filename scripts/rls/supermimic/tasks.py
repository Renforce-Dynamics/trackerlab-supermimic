"""Discover and run a registered task in standalone or installed TrackerLab."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.registry import TASK_REGISTRY, get_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "show", "prepare", "smoke", "train", "supervise",
                                            "gait-supervise"))
    parser.add_argument("--task")
    args, extra = parser.parse_known_args()
    if args.command == "list":
        for entry in TASK_REGISTRY.values():
            print(f"{entry.id:28s} robot={entry.robot:3s} runtime={entry.runtime_task}")
        return
    if not args.task:
        parser.error("--task is required")
    if args.command == "gait-supervise":
        command = [
            sys.executable, str(Path(__file__).with_name("supervise_gait.py")),
            "--references", str(REPO / "logs/gait-foundation-v1/references.json"),
            "--run-dir", str(REPO / "logs/gait-foundation-v1/runs"),
            "--robots", args.task, *extra,
        ]
        subprocess.run(command, check=True, cwd=REPO)
        return
    entry = get_task(args.task)
    if args.command == "show":
        print(json.dumps(entry.describe(), indent=2))
        return
    if entry.runtime_task == "terrain":
        if args.command == "prepare":
            print("Terrain tasks use the pinned robot assets; motion clips are not required.")
            return
        folder = Path(__file__).resolve().parent
        if args.command == "supervise":
            run_dir = REPO / "logs" / "terrain-campaign" / entry.robot / entry.profile
            command = [sys.executable, str(folder / "supervise_terrain.py"),
                       "--tasks", entry.id, "--run-dir", str(run_dir), *extra]
        else:
            command = [sys.executable, str(folder / "train_terrain.py"),
                       "--task", entry.id, "--config", str(REPO / entry.preset), *extra]
        if args.command == "smoke":
            command += ["--num-envs", "16", "--iterations", "4", "--wm-only-iters", "1",
                        "--wm-batches", "1", "--wm-batch", "8", "--pi-batch", "8",
                        "--horizon", "4", "--hist", "2", "--wm-window", "2",
                        "--buffer-steps", "16", "--critic-epochs", "1",
                        "--critic-minibatches", "1", "--no-compile",
                        "--run-dir", str(REPO / "logs" / "terrain-smoke" / entry.robot / entry.profile)]
        elif args.command == "train" and "--run-dir" not in extra:
            command += ["--run-dir", str(REPO / "logs" / "terrain" / entry.robot / entry.profile)]
        subprocess.run(command, check=True, cwd=REPO)
        return
    if entry.runtime_task == "gait-foundation":
        if args.command not in ("train", "smoke"):
            parser.error("Use train or smoke for a gait foundation")
        references = REPO / "logs/gait-foundation-v1/references" / entry.robot
        command = [
            sys.executable, str(Path(__file__).with_name("train_gait.py")),
            "--task", entry.robot, "--motion_config", str(references / "train.json"),
            "--log_dir", str(REPO / "logs/gait-foundation-v1/runs" / entry.robot / "single-run"),
            *extra,
        ]
        subprocess.run(command, check=True, cwd=REPO)
        return
    script = {"prepare": "prepare_resources", "smoke": "smoke_test"}.get(args.command, args.command)
    command = [sys.executable, str(Path(__file__).with_name(script + ".py"))]
    if args.command == "prepare":
        command += ["--task", entry.robot]
    else:
        command += ["--motion_config", str(REPO / entry.motion_config)]
        if args.command == "smoke":
            command += ["--task", entry.robot]
        else:
            command += ["--config", str(REPO / entry.preset)]
    subprocess.run([*command, *extra], check=True, cwd=REPO)


if __name__ == "__main__":
    main()

"""Discover and run a registered task in standalone or installed TrackerLab."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.registry import TASK_REGISTRY, get_task
from trackerTask.native.locomotion.perception.registry import TASKS as PERCEPTION_TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "show", "prepare", "smoke", "train", "supervise"))
    parser.add_argument("--task")
    args, extra = parser.parse_known_args()
    if args.command == "list":
        for entry in TASK_REGISTRY.values():
            print(f"{entry.id:28s} robot={entry.robot:3s} runtime={entry.runtime_task}")
        for entry in PERCEPTION_TASKS.values():
            print(f"{entry.id:28s} robot={entry.robot:3s} runtime={entry.backend}/{entry.algorithm}")
        return
    if not args.task:
        parser.error("--task is required")
    if args.task in PERCEPTION_TASKS:
        entry = PERCEPTION_TASKS[args.task]
        if args.command == "show":
            print(json.dumps(entry.describe(), indent=2))
            return
        folder = Path(__file__).resolve().parent.parent / "perception"
        if args.command == "prepare":
            command = [sys.executable, str(folder / "prepare.py"), "--robot", entry.robot]
        elif args.command == "supervise":
            command = [sys.executable, str(folder / "campaign.py"), "--tasks", entry.id]
        else:
            command = [sys.executable, str(folder / "train.py"), "--task", entry.id]
            if args.command == "smoke":
                command.append("--smoke")
        subprocess.run([*command, *extra], check=True, cwd=REPO)
        return
    entry = get_task(args.task)
    if args.command == "show":
        print(json.dumps(entry.describe(), indent=2))
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

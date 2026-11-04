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
                                            "gait-supervise", "supervise-style"))
    parser.add_argument("--task")
    args, extra = parser.parse_known_args()
    if args.command == "list":
        for entry in TASK_REGISTRY.values():
            print(f"{entry.id:28s} robot={entry.robot:3s} runtime={entry.runtime_task}")
        return
    if not args.task and args.command != "supervise-style":
        parser.error("--task is required")
    if args.command == "gait-supervise":
        from trackerTask.supermimic.gait import ROBOTS
        robot = args.task.removeprefix("gait/")
        selected = list(ROBOTS) if robot == "all" else [robot]
        if any(name not in ROBOTS for name in selected):
            parser.error("--task must be all or gait/<robot>")
        command = [
            sys.executable, str(Path(__file__).with_name("supervise_gait.py")),
            "--references", str(REPO / "data/gait-foundation-v5/references.json"),
            "--run-dir", str(REPO / "logs/gait-foundation-v2/runs"),
            "--robots", *selected, *extra,
        ]
        subprocess.run(command, check=True, cwd=REPO)
        return
    if args.command == "supervise-style":
        command = [
            sys.executable, str(Path(__file__).with_name("supervise_style.py")),
            "--run-dir", str(REPO / "logs/supermimic-command-style-v4-20261105"),
            "--style-map", str(REPO / "data/style-references-v9-stealth/style-map.json"), *extra,
        ]
        subprocess.run(command, check=True, cwd=REPO)
        return
    entry = get_task(args.task)
    if args.command == "show":
        print(json.dumps(entry.describe(), indent=2))
        return
    if entry.runtime_task in ("terrain", "command"):
        if args.command == "prepare":
            print(f"{entry.profile} task uses robot models and audited motion-style references.")
            return
        folder = Path(__file__).resolve().parent
        if args.command == "supervise":
            command = [sys.executable, str(folder / "supervise_style.py"),
                       "--run-dir", str(REPO / "logs/supermimic-command-style-v4-20261105"),
                       "--style-map", str(REPO / "data/style-references-v9-stealth/style-map.json"),
                       "--tasks", entry.id, *extra]
        else:
            command = [sys.executable, str(folder / "train_terrain.py"),
                       "--task", entry.id, *extra]
            if "--amp-motion-config" not in extra:
                from supervise_style import resolve_styles
                style = ("stealth" if entry.profile == "stealth" else
                         "parkour" if entry.profile in ("stairs", "mountain", "parkour") else "walk")
                mapping = REPO / "data/style-references-v9-stealth/style-map.json"
                if not mapping.is_file():
                    parser.error("Prepare a style-map or supply --amp-motion-config")
                config = resolve_styles(mapping).get(entry.robot, {}).get(style)
                if config is None:
                    parser.error(f"No prepared {style} pool for {entry.robot}")
                command.extend(("--amp-motion-config", str(config)))
            if entry.profile == "stealth" and "--clearance-ratio" not in extra:
                command.extend(("--clearance-ratio", ".8"))
            if args.command == "train" and "--run-dir" not in extra:
                command.extend(("--run-dir", str(REPO / "logs/command-style" /
                                                  entry.robot / entry.profile)))
        if args.command == "smoke":
            command += ["--num-envs", "16", "--iterations", "4", "--wm-only-iters", "1",
                        "--wm-batches", "1", "--wm-batch", "8", "--pi-batch", "8",
                        "--horizon", "4", "--hist", "10", "--wm-window", "2",
                        "--buffer-steps", "32", "--critic-epochs", "1",
                        "--critic-minibatches", "1", "--no-compile",
                        "--run-dir", str(REPO / "logs" / "terrain-smoke" / entry.robot / entry.profile)]
        elif args.command == "train" and "--run-dir" not in extra:
            command += ["--run-dir", str(REPO / "logs" / "terrain" / entry.robot / entry.profile)]
        subprocess.run(command, check=True, cwd=REPO)
        return
    if entry.runtime_task == "gait-foundation":
        if args.command not in ("train", "smoke"):
            parser.error("Use train or smoke for a gait foundation")
        references = REPO / "data/gait-foundation-v5" / entry.robot
        command = [
            sys.executable, str(Path(__file__).with_name("train_gait.py")),
            "--task", entry.robot, "--motion_config", str(references / "train.json"),
            "--log_dir", str(REPO / "logs/gait-foundation-v2/runs" / entry.robot / "single-run"),
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

"""Exercise every command/style task through real physics and learner updates."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from bootstrap import configure_paths
REPO = configure_paths()
from supervise_style import build_plan, resolve_styles
from supervise_terrain import save


def smoke_command(job, config, run):
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("train_terrain.py")),
        "--task", job["task"], "--run-dir", str(run),
        "--amp-motion-config", str(config), "--num-envs", "16",
        "--iterations", "4", "--wm-only-iters", "1", "--wm-batches", "1",
        "--wm-batch", "8", "--pi-batch", "8", "--horizon", "4", "--hist", "10",
        "--wm-window", "2", "--buffer-steps", "32", "--critic-epochs", "1",
        "--critic-minibatches", "1", "--warmup-steps", "16", "--log-every", "1",
        "--amp-batch", "16", "--no-compile",
    ]
    if job["profile"] == "stealth":
        command.extend(("--clearance-ratio", ".8"))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style-map", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Use a new validation directory")
    root.mkdir(parents=True, exist_ok=True)
    styles = resolve_styles(args.style_map.resolve())
    report = {"scope": "16-world physical/optimizer smoke, not learned-policy quality",
              "passed": False, "tasks": []}
    for job in build_plan():
        config = styles.get(job["robot"], {}).get(job["style"])
        run = root / job["robot"] / job["profile"]
        run.mkdir(parents=True, exist_ok=True)
        if config is None or not config.is_file():
            report["tasks"].append({"task": job["task"], "passed": False,
                                    "error": "missing style references"})
            continue
        command = smoke_command(job, config, run)
        started = time.monotonic()
        with (run / "stdout.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        summary_path = run / "run_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        passed = result.returncode == 0 and summary.get("completed_iterations") == 4
        row = {"task": job["task"], "passed": passed, "exit_code": result.returncode,
               "seconds": time.monotonic() - started, "summary": summary, "command": command}
        report["tasks"].append(row)
        save(root / "validation.json", report)
        print(json.dumps({"task": job["task"], "passed": passed}), flush=True)
    report["passed"] = len(report["tasks"]) == 29 and all(row["passed"] for row in report["tasks"])
    save(root / "validation.json", report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

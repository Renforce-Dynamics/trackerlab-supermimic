"""Build an auditable repair campaign without changing the baseline artifacts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.terrain import TASKS
from supervise_terrain import evaluation_metrics, save


def build_retrain_plan(baseline, iterations=10000, threshold=.5):
    baseline = Path(baseline).resolve()
    completed, trusted = {}, {}
    for task in TASKS.values():
        run = baseline / task.robot / task.profile
        evaluation, checkpoint = run / "evaluation.json", run / "last.pt"
        if not evaluation.is_file() or not checkpoint.is_file():
            continue
        result = json.loads(evaluation.read_text(encoding="utf-8"))
        if result["task"] != task.id:
            raise ValueError(f"Evaluation task mismatch in {evaluation}")
        metrics = evaluation_metrics(result)
        if result["trained_iterations"] >= iterations and metrics["macro_success"] >= threshold:
            completed[task.id] = {"checkpoint": str(checkpoint), "metrics": metrics}
            previous = trusted.get(task.robot)
            if previous is None or metrics["macro_success"] > previous["metrics"]["macro_success"]:
                trusted[task.robot] = completed[task.id]

    jobs, initial = [], {}
    for task in TASKS.values():
        if task.id in completed:
            continue
        run = baseline / task.robot / task.profile
        source = run / "last.pt"
        job = {"task": task.id, "iterations": iterations,
               "reason": "retrain_existing" if source.is_file() else "previously_untrained"}
        if task.robot in trusted:
            initial[task.id] = trusted[task.robot]["checkpoint"]
            job["initialization"] = "validated_baseline_same_robot"
        elif source.is_file():
            initial[task.id] = str(source)
            job["initialization"] = "same_task_weights_new_objective"
        else:
            job["initialization"] = "validated_predecessor_or_scratch"
        jobs.append(job)
    return {"baseline": str(baseline), "selection_threshold_macro_success": threshold,
            "retained": completed, "jobs": jobs, "initial_checkpoints": initial}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=8096)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or min(args.iterations, args.num_envs) < 1:
        parser.error("Invalid threshold or training size")
    baseline, root = args.baseline.resolve(), args.run_dir.resolve()
    if not baseline.is_dir() or baseline == root or baseline in root.parents:
        parser.error("Use an existing baseline and a separate output directory")
    plan_path = root / "retrain_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["baseline"] != str(baseline):
            parser.error("This repair directory belongs to a different baseline")
        if any(job["iterations"] != args.iterations for job in plan["jobs"]):
            parser.error("Iteration target differs from the saved repair plan")
    else:
        plan = build_retrain_plan(baseline, args.iterations, args.threshold)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if not plan["jobs"]:
        print("All baseline tasks already satisfy the selection threshold.")
        return
    root.mkdir(parents=True, exist_ok=True)
    save(plan_path, plan)
    initial_path = root / "initial_checkpoints.json"
    save(initial_path, plan["initial_checkpoints"])
    command = [sys.executable, "-u", str(Path(__file__).with_name("supervise_terrain.py")),
               "--run-dir", str(root), "--num-envs", str(args.num_envs),
               "--iterations", str(args.iterations), "--initial-checkpoints", str(initial_path),
               "--warm-start-min-success", str(args.threshold),
               "--tasks", *[job["task"] for job in plan["jobs"]]]
    raise SystemExit(subprocess.call(command, cwd=REPO))


if __name__ == "__main__":
    main()

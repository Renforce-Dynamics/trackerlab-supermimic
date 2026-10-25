"""Run the pinned native MicroDuck task in its own Python 3.12 environment."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
TASK_DIR = REPO / "source/trackerTask/trackerTask/native/locomotion/robots/microduck"


def load_settings(smoke=False):
    task = json.loads((TASK_DIR / "task.json").read_text())
    preset = json.loads((TASK_DIR / "agents/ppo.json").read_text())
    if smoke:
        preset.update(num_envs=64, iterations=5, save_interval=1)
    return task, preset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    task, preset = load_settings(args.smoke)
    for argument, key in ((args.num_envs, "num_envs"), (args.iterations, "iterations")):
        if argument is not None:
            if argument <= 0:
                parser.error(f"{key} must be positive")
            preset[key] = argument
    if args.describe:
        print(json.dumps({"task": task, "preset": preset}, indent=2))
        return
    if args.upstream is None or args.run_dir is None:
        parser.error("--upstream and --run-dir are required")
    if sys.version_info[:2] != (3, 12):
        parser.error("Use the isolated MicroDuck Python 3.12 environment")
    upstream, output = args.upstream.resolve(), args.run_dir.resolve()
    if (output / "run_config.json").exists() or list(output.glob("model_*.pt")):
        parser.error("Use a new run directory; existing runs are preserved")
    for name, expected in task["versions"].items():
        actual = importlib.metadata.version(name).split("+")[0]
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {actual}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["WANDB_MODE"] = "disabled"
    import mjlab_microduck
    if not Path(mjlab_microduck.__file__).resolve().is_relative_to(upstream / "src"):
        raise RuntimeError("Installed MicroDuck package differs from --upstream")
    from mjlab.scripts.train import TrainConfig, run_train
    import mjlab.tasks  # Load the official plugin registry.

    cfg = TrainConfig.from_task(task["runtime_task"])
    cfg.env.scene.num_envs = preset["num_envs"]
    cfg.agent.max_iterations = preset["iterations"]
    cfg.agent.seed = preset["seed"]
    cfg.agent.save_interval = preset["save_interval"]
    cfg.agent.logger = preset["logger"]
    cfg.agent.upload_model = False
    cfg.agent.resume = False
    cfg.agent.experiment_name = "microduck_flat"
    cfg.agent.run_name = output.name
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for path in sorted(upstream.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".xml", ".toml", ".lock", ".json"):
            if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(upstream).parts):
                continue
            source_hashes[str(path.relative_to(upstream))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "source_sha256.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    (output / "run_config.json").write_text(json.dumps({
        "task": task, "preset": preset, "upstream": str(upstream),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "algorithm_changes": [], "smoke_only": args.smoke,
        "iteration_convention": "RSL-RL checkpoint indices are zero-based",
    }, indent=2) + "\n")
    run_train(task["runtime_task"], cfg, output)
    (output / "training_finished.json").write_text(json.dumps({
        "requested_updates": preset["iterations"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_pending": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

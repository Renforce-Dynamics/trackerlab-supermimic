"""Train a terrain-perception task using upstream RSL-RL PPO and Warp physics."""

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import math
import os
from pathlib import Path
import signal
import sys
import time

from common import REPO, digest, load, now, save
from trackerTask.native.locomotion.perception.registry import TASKS, get_task, ppo_config


class StopTraining(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--max-minutes", type=float, default=0)
    parser.add_argument("--solver-iterations", type=int, default=50)
    parser.add_argument("--level", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.solver_iterations) < 1 or args.max_minutes < 0:
        parser.error("World count, iterations and solver iterations must be positive")
    if args.smoke:
        args.num_envs, args.iterations = 16, 4
    if args.resume and args.warm_start:
        parser.error("--resume and --warm-start are mutually exclusive")
    task = get_task(args.task)
    config = ppo_config()
    if args.smoke:
        config["save_interval"] = 1
    if args.describe:
        print(__import__("json").dumps({"task": asdict(task), "ppo": config}, indent=2))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required")
    run = args.run_dir.resolve()
    if (run / "run_config.json").exists() or list(run.glob("*.pt")):
        parser.error("Use a fresh run directory; previous outputs are preserved")
    if args.resume:
        previous = load(args.resume.resolve().parent / "run_config.json")
        for name in ("task", "num_envs", "solver_iterations", "seed", "level"):
            if previous["arguments"][name] != getattr(args, name):
                parser.error(f"Resume configuration differs: {name}")
    if args.warm_start:
        previous = load(args.warm_start.resolve().parent / "run_config.json")
        if get_task(previous["arguments"]["task"]).robot != task.robot:
            parser.error("Warm starts require the same robot")
    run.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WARP_CACHE_PATH", str(REPO / "data/cache/perception-warp"))
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from trackerLab.tracker_env.perception.env import PerceptionEnv

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    versions = {n: importlib.metadata.version(n)
                for n in ("torch", "numpy", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")}
    expected = {"mujoco": "3.16.0", "mujoco-warp": "3.16.0",
                "warp-lang": "1.18.0", "rsl-rl-lib": "2.3.3"}
    for package, version in expected.items():
        if versions[package] != version:
            raise RuntimeError(f"{package}: expected {version}, found {versions[package]}")
    arguments = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save(run / "run_config.json", {
        "arguments": arguments, "task": task.describe(), "ppo": config,
        "versions": versions, "started_at": now(), "pid": os.getpid(),
        "iteration_convention": "completed_updates = RSL-RL zero-based checkpoint iter + 1",
    })
    source = {}
    for folder in ("scripts/rls/perception", "source/trackerLab/trackerLab/tracker_env/perception",
                   "source/trackerTask/trackerTask/native/locomotion/perception"):
        source.update({p.relative_to(REPO).as_posix(): digest(p)
                       for p in (REPO / folder).rglob("*.py")})
    save(run / "source_sha256.json", source)
    start = time.monotonic()
    stopped = {"requested": False}

    def request_stop(signum, frame):
        stopped["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    save(run / "status.json", {"phase": "initializing", "pid": os.getpid(), "updated_at": now()})
    env = PerceptionEnv(args.task, args.num_envs, args.device, args.seed,
                        level=args.level, solver_iterations=args.solver_iterations)
    save(run / "simulator_contract.json", env.contract)
    np = __import__("numpy")
    np.save(run / "terrain.npy", env.terrain.heights)

    class Runner(OnPolicyRunner):
        def log(self, locs, width=80, pad=35):
            torch.cuda.synchronize()
            completed = locs["it"] + 1
            losses = {k: float(v) for k, v in locs["loss_dict"].items()}
            if not all(math.isfinite(v) for v in losses.values()):
                raise FloatingPointError(f"Nonfinite PPO losses: {losses}")
            measured = time.monotonic()
            duration = measured - self.last_update
            self.last_update = measured
            metrics = {
                "completed_updates": completed, "iteration_seconds": duration,
                "collection_seconds": locs["collection_time"], "learning_seconds": locs["learn_time"],
                "transitions_per_second": args.num_envs * config["num_steps_per_env"] / duration,
                "terrain_level_mean": float(env.levels.float().mean()),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                **{f"loss_{k}": v for k, v in losses.items()},
            }
            with (run / "train_log.csv").open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(metrics))
                if stream.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)
            self.current_learning_iteration = locs["it"]
            if completed % 25 == 0 or completed == 1:
                save(run / "status.json", {"phase": "training", "task": args.task,
                     "pid": os.getpid(), "updated_at": now(), **metrics})
                print(f"updates={completed} seconds={duration:.4f} "
                      f"transitions/s={metrics['transitions_per_second']:.0f} "
                      f"level={metrics['terrain_level_mean']:.3f}", flush=True)
            if completed % 50 == 0 or args.smoke:
                super().log(locs, width, pad)
            else:
                self.tot_time += locs["collection_time"] + locs["learn_time"]
                self.tot_timesteps += args.num_envs * self.num_steps_per_env
                for k, v in losses.items():
                    self.writer.add_scalar("Loss/" + k, v, completed)
            if stopped["requested"] or (run / "STOP").exists():
                raise StopTraining("stop_requested")
            if args.max_minutes and measured - start >= args.max_minutes * 60:
                raise StopTraining("time_limit")

        def save(self, path, infos=None):
            if not all(bool(torch.isfinite(p).all()) for p in self.alg.policy.parameters()):
                raise FloatingPointError("Nonfinite policy parameters")
            super().save(path, {
                "task": args.task, "completed_updates": self.current_learning_iteration + 1,
                "simulator_contract_sha256": digest(run / "simulator_contract.json"),
            })

    runner = Runner(env, deepcopy(config), str(run), device=args.device)
    runner.git_status_repos = []
    if args.resume or args.warm_start:
        checkpoint = args.resume or args.warm_start
        runner.load(str(checkpoint.resolve()), load_optimizer=bool(args.resume))
    if args.resume:
        runner.current_learning_iteration += 1
    elif args.warm_start:
        runner.current_learning_iteration = 0
    initial = runner.current_learning_iteration
    if initial >= args.iterations:
        raise ValueError("Target updates must exceed the resumed checkpoint")
    runner.last_update = time.monotonic()
    reason = "iterations"
    try:
        runner.learn(args.iterations - initial, init_at_random_ep_len=False)
    except StopTraining as error:
        reason = str(error)
    except BaseException as error:
        save(run / "status.json", {"phase": "failed", "updated_at": now(),
                                  "error": repr(error), "pid": os.getpid()})
        raise
    runner.save(str(run / "last.pt"))
    completed = runner.current_learning_iteration + 1
    summary = {
        "phase": "completed" if reason == "iterations" else "stopped",
        "task": args.task, "completed_updates": completed, "start_updates": initial,
        "requested_updates": args.iterations, "stop_reason": reason,
        "elapsed_seconds_including_initialization": time.monotonic() - start,
        "checkpoint": str(run / "last.pt"), "checkpoint_sha256": digest(run / "last.pt"),
        "finished_at": now(), "evaluation_pending": True,
        "ppo_iteration_is_comparable_to_supermimic_outer_iteration": False,
    }
    save(run / "run_summary.json", summary)
    save(run / "status.json", summary)
    if runner.writer:
        runner.writer.close()
    print(summary, flush=True)


if __name__ == "__main__":
    main()

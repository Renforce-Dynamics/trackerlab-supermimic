"""Finish an existing G1 run, then train and review robust aiming locomotion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]


def source_checkpoint(run, expected_iterations):
    """Admit only a naturally completed run and its immutable final checkpoint."""
    run = Path(run)
    summary = run / "run_summary.json"
    if not summary.exists():
        return None
    receipt = json.loads(summary.read_text(encoding="utf-8"))
    if receipt.get("task") != "locomotion/g1/stealth":
        raise ValueError("Source run is not G1 stealth")
    if receipt.get("phase") != "completed":
        raise ValueError("Source training has an unfinished terminal receipt")
    if receipt.get("completed_iterations") != expected_iterations:
        raise ValueError("Source completed iterations differ from the scheduled handoff")
    path = run / f"iteration_{expected_iterations:06d}.pt"
    if not path.is_file():
        return None
    return path


def process_matches(pid, start_ticks):
    if pid is None:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] != "Z" and (start_ticks is None or int(fields[19]) == start_ticks)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-iterations", type=int, default=20000)
    parser.add_argument("--source-pid", type=int)
    parser.add_argument("--source-start-ticks", type=int)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--probe-iterations", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=1024)
    args = parser.parse_args()
    if not 0 < args.probe_iterations < args.iterations:
        parser.error("Probe iterations must be positive and less than total iterations")
    if args.probe_iterations % 2000 or args.iterations % 2000:
        parser.error("Probe and total iterations must be multiples of the saved 2000-iteration milestones")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    gpu_path = args.gpu_lock.resolve()
    gpu_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    controller_lock = (root / "controller.lock").open("a")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = gpu_path.open("a")
    state = {
        "task": "locomotion/g1/stealth",
        "source_run": str(args.source_run.resolve()),
        "source_iterations": args.source_iterations,
        "source_pid": args.source_pid,
        "source_start_ticks": args.source_start_ticks,
        "controller_pid": os.getpid(),
        "code_commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False,
        ).stdout.strip() or os.environ.get("SUPERMIMIC_CODE_REVISION", "uncommitted"),
        "training_iterations": args.iterations,
        "phase": "waiting_source",
    }

    def update(phase, **values):
        state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        temporary = root / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root / "status.json")
        print(json.dumps(state), flush=True)

    def execute(phase, command, log):
        update(phase, command=command, log=str(log))
        with Path(log).open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                pass_fds=(controller_lock.fileno(), gpu_lock.fileno()),
            )
            update(phase, child_pid=process.pid)
            result = process.wait()
        if result:
            raise RuntimeError(f"{phase} exited with code {result}; see {log}")
        update(phase, child_pid=None, last_returncode=0)

    def completed_or_saved_training(run, target_iterations):
        run = Path(run)
        summary_path = run / "run_summary.json"
        receipt = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        if (receipt.get("phase") == "completed"
                and receipt.get("completed_iterations") == target_iterations):
            milestone = run / f"iteration_{target_iterations:06d}.pt"
            return milestone if milestone.is_file() else run / "last.pt"
        checkpoint = run / "last.pt"
        if checkpoint.is_file():
            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (saved.get("task") == "locomotion/g1/stealth"
                    and saved.get("iteration", 0) >= target_iterations):
                return checkpoint
        return None

    def review(checkpoint, directory, seed):
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "evaluation.json").is_file():
            execute("evaluate", [
                sys.executable, str(REPO / "scripts/rls/supermimic/evaluate_terrain.py"),
                "--checkpoint", str(checkpoint),
                "--output", str(directory / "evaluation.json"),
                "--trajectory-output", str(directory / "trajectories.npz"),
                "--num-envs", "96", "--steps", "1500", "--seed", str(seed),
            ], directory / "evaluate.stdout.log")
        for lane, name in ((0, "forward"), (5, "turning")):
            output = directory / name
            output.mkdir(parents=True, exist_ok=True)
            for filename in ("evaluation.json", "trajectories.npz"):
                shutil.copy2(directory / filename, output / filename)
            if not (output / "demo.json").is_file():
                execute("render", [
                    sys.executable, str(REPO / "scripts/rls/supermimic/render_terrain.py"),
                    "--directory", str(output), "--output", str(output), "--lane", str(lane),
                ], output / "render.stdout.log")

    try:
        update("waiting_source")
        deadline = time.monotonic() + 12 * 3600
        missing_process_count = 0
        while True:
            checkpoint = source_checkpoint(args.source_run, args.source_iterations)
            alive = process_matches(args.source_pid, args.source_start_ticks)
            if checkpoint is not None and not alive:
                break
            missing_process_count = 0 if alive else missing_process_count + 1
            if args.source_pid is not None and checkpoint is None and missing_process_count >= 3:
                raise RuntimeError("Source process exited without the expected completion receipt")
            if time.monotonic() > deadline:
                raise TimeoutError("Source training handoff exceeded twelve hours")
            update("waiting_source", source_process_alive=alive)
            time.sleep(30)

        while True:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                update("waiting_gpu_queue_lock", gpu_lock=str(gpu_path))
                time.sleep(15)

        import torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved["iteration"] != args.source_iterations
                or saved["task"] != "locomotion/g1/stealth"):
            raise ValueError("Source checkpoint metadata differs from its completion receipt")
        motion = saved["arguments"]["amp_motion_config"]
        del saved
        update("source_complete", source_checkpoint=str(checkpoint),
               source_checkpoint_sha256=digest(checkpoint))
        review(checkpoint, root / "source-review" / "seed61", 61)
        run = root / "training"
        common = [
            sys.executable, "-u", str(REPO / "scripts/rls/supermimic/train_terrain.py"),
            "--config", str(REPO / "scripts/rls/supermimic/configs/command_stealth_robust.json"),
            "--task", "locomotion/g1/stealth", "--run-dir", str(run),
            "--amp-motion-config", motion, "--num-envs", str(args.num_envs),
            "--no-compile", "--no-compile-critic", "--fused-optimizers",
            "--ckpt-every", "500", "--keep-every", "2000", "--log-every", "25",
        ]
        probe = completed_or_saved_training(run, args.probe_iterations)
        if probe is None:
            current = run / "last.pt"
            if current.is_file():
                command = [*common, "--resume", str(current),
                           "--iterations", str(args.probe_iterations)]
            else:
                command = [*common, "--warm-start", str(checkpoint),
                           "--iterations", str(args.probe_iterations)]
            execute("train_probe", command, root / "train-probe.stdout.log")
            probe = completed_or_saved_training(run, args.probe_iterations)
            if probe is None:
                raise RuntimeError("Probe completed without its expected checkpoint")
        review(probe, root / "probe-review" / "seed67", 67)
        final = completed_or_saved_training(run, args.iterations)
        if final is None:
            current = run / "last.pt"
            if not current.is_file():
                raise RuntimeError("Probe checkpoint vanished before remainder training")
            execute("train_remainder", [
                *common, "--resume", str(current), "--iterations", str(args.iterations),
            ], root / "train-remainder.stdout.log")
            final = completed_or_saved_training(run, args.iterations)
            if final is None:
                raise RuntimeError("Remainder training completed without its final checkpoint")
        for seed in (71, 73):
            review(final, root / "final-review" / f"seed{seed}", seed)
        if not (root / "export" / "metadata.json").is_file():
            execute("export", [
                sys.executable, str(REPO / "scripts/rls/supermimic/export_command_policy.py"),
                "--checkpoint", str(final), "--output", str(root / "export"),
            ], root / "export.stdout.log")
        update("trained_and_evaluated", checkpoint=str(final),
               checkpoint_sha256=digest(final), quality_acceptance="requires owner video review")
    except Exception as error:
        update("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()

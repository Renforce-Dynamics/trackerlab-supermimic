"""Supervise a fresh long run and package evaluated checkpoints without restarting training."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from configuration import load_config

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
TERMS = (
    "body_position", "body_orientation", "body_linear_velocity", "body_angular_velocity",
    "root_position", "root_orientation", "action_smoothness", "joint_limit_penalty",
)
FATAL = ("PhysX error", "CUDA error", "CUDA out of memory", "Traceback (most recent call last)")


def now():
    return datetime.now().astimezone().isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append({key: float(value) for key, value in row.items()})
            except (ValueError, TypeError):
                continue
    return rows


def runtime_environment():
    env = os.environ.copy()
    for key, value in (("OMP_NUM_THREADS", "8"), ("TORCHINDUCTOR_COMPILE_THREADS", "4"),
                       ("PYTHONUNBUFFERED", "1")):
        env.setdefault(key, value)
    return env


def launcher_args(device):
    args = ["--headless", "--device", device]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        args.append("--kit_args=--allow-root")
    return args


def run_artifact_command(command, directory, label):
    log = directory / (label + ".log")
    with log.open("w") as output:
        subprocess.run(command, cwd=REPO, env=runtime_environment(), stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    if any(marker in log.read_text(errors="replace") for marker in FATAL):
        raise RuntimeError(f"{label} reported an error; inspect {log}")


def motion_arguments(options):
    if options.get("motion_config"):
        return ["--motion_config", str(Path(options["motion_config"]).resolve())]
    return ["--motion_dir", str(Path(options["motion_dir"]).resolve())]


def package_milestone(run_dir, iteration):
    import torch

    torch.set_num_threads(2)
    target = run_dir / "milestones" / f"{iteration:06d}"
    if (target / "ready.json").exists():
        print(f"Milestone {iteration} is already complete: {target}", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    save(target / "job_status.json", {"phase": "evaluating", "iteration": iteration, "updated_at": now()})
    try:
        configuration = load(run_dir / "args.json")
        training = configuration["arguments"]
        commands = load(run_dir / "commands.json")
        final = iteration == configuration["arguments"]["iterations"]
        source = run_dir / ("last.pt" if final else f"ckpt_{iteration:06d}.pt")
        shutil.copy2(source, target / "checkpoint.pt")
        checkpoint = torch.load(target / "checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["iteration"] != iteration or checkpoint["args"].get("resume") is not None:
            raise ValueError("Checkpoint lineage differs from the fresh experiment")
        for model in ("pi", "wm"):
            if not all(bool(torch.isfinite(value).all()) for value in checkpoint[model].values()):
                raise FloatingPointError(f"Nonfinite {model} checkpoint")
        checkpoint_meta = {
            "iteration": iteration, "sim_steps": checkpoint["sim_steps"],
            "training_wall_hours": checkpoint["wall_time"] / 3600,
            "resumable": all(key in checkpoint for key in ("critic", "opt_pi", "opt_wm", "opt_v")),
        }
        del checkpoint
        for name in ("args.json", "reward_config.json", "motion_library.json", "motion_config.json",
                     "data_manifest.json", "source_sha256.json", "commands.json", "resources.lock.json"):
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, target / name)
        rows = [row for row in read_rows(run_dir / "train_log.csv") if row["iteration"] <= iteration]
        with (target / "train_log.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        run_artifact_command([
            sys.executable, str(SCRIPTS / "evaluate.py"), *launcher_args(training["device"]),
            "--run_dir", str(target), "--ckpt", str(target / "checkpoint.pt"),
            *motion_arguments(training), "--record", str(target / "trajectories.npz"),
        ], target, "evaluation")
        shutil.copy2(target / "checkpoint_eval.json", target / "last_eval.json")
        result = load(target / "last_eval.json")
        expected_names = load(run_dir / "motion_library.json")["names"]
        motion_count = len(expected_names)
        if result["iteration"] != iteration or set(result["per_motion"]) != set(expected_names):
            raise ValueError("Incomplete motion evaluation")
        run_artifact_command([
            sys.executable, str(SCRIPTS / "export_policy.py"),
            "--checkpoint", str(target / "checkpoint.pt"), "--output_dir", str(target / "policy"),
        ], target, "export")
        shutil.make_archive(str(target / "policy"), "zip", target, "policy")
        render_command = [
            sys.executable, str(SCRIPTS / "render_demo.py"),
            "--trajectory", str(target / "trajectories.npz"),
            "--evaluation", str(target / "last_eval.json"), "--output", str(target / "demo.mp4"),
            "--seconds", str(commands["render_seconds"]),
        ]
        if commands.get("render_clips"):
            render_command.extend(["--clips", *commands["render_clips"]])
        run_artifact_command(render_command, target, "render")
        import imageio_ffmpeg

        run_artifact_command([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(target / "demo.mp4"),
            "-f", "null", "-",
        ], target, "video_decode")
        with (target / "per_motion_metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["motion", "normalized_return", "failed", *TERMS])
            for name, metrics in result["per_motion"].items():
                writer.writerow([name, metrics["normalized_return"], metrics["failed"],
                                 *(metrics["reward_components"][key] for key in TERMS)])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for axis, field in zip(axes.flat[:3], ("wm_loss", "sim_rew", "ep_ret_norm")):
            axis.plot([r["iteration"] for r in rows], [r[field] for r in rows])
            axis.set(xlabel="Iteration", title=field)
            axis.grid(alpha=0.2)
        axes[1, 1].barh(list(TERMS), [result["reward_components"][key] for key in TERMS])
        axes[1, 1].set_title("Full-clip normalized reward contributions")
        fig.savefig(target / "training_curves.png", dpi=160)
        plt.close(fig)
        component_lines = "\n".join(
            f"| {name} | {result['reward_components'][name]:.8f} |" for name in TERMS
        )
        report = (
            f"# SuperMimic iteration {iteration}\n\n"
            f"Fresh training with {training['num_envs']} environments, "
            f"{training['obs_mode']} observations, reset noise {training['reset_noise']}, "
            f"seed {training['seed']}.\n\n"
            f"- Full-clip macro return: {result['norm_return']:.8f}.\n"
            f"- Frame-weighted return: {result['norm_return_framew']:.8f}.\n"
            f"- Failed motions: {result['n_failed']}/{motion_count}.\n"
            f"- Cumulative training time: {checkpoint_meta['training_wall_hours']:.3f} h.\n"
            f"- Simulator transitions: {checkpoint_meta['sim_steps']}.\n"
            f"- Full optimizer checkpoint included: {checkpoint_meta['resumable']}.\n\n"
            "| Reward contribution | Full-clip macro value |\n| --- | ---: |\n"
            f"{component_lines}\n\n"
            "Evaluation covers the loaded training library, starting at each clip's first frame. "
            "Reward after early failure is zero in the full-clip denominator. "
            "This PhysX port has a different physics backend from vibed's Newton/MuJoCo-Warp. "
            "Held-out generalization is unmeasured. See docs/results.md for recorded comparisons.\n\n"
            "## Failed motions\n\n" + "\n".join(f"- {name}" for name in result["failed"]) + "\n"
        )
        if training.get("task", "g1") == "go2":
            provenance = result["protocol"].get("data_provenance", {})
            report += (
                "\n## Go2 gait measurements\n\n"
                f"Reference source: {provenance.get('source', 'See data_manifest.json')}. "
                "This is reference-conditioned gait imitation, not a free velocity-command controller.\n\n"
                "| Motion | World-X speed (m/s) | Reference (m/s) | Root position RMSE (m) | Joint RMSE (rad) | Alive (s) |\n"
                "| --- | ---: | ---: | ---: | ---: | ---: |\n"
            )
            for name, metrics in result["per_motion"].items():
                track = metrics["tracking"]
                report += (
                    f"| {name} | {track['mean_forward_speed_m_s']:.4f} | "
                    f"{track['reference_forward_speed_m_s']:.4f} | {track['root_position_rmse_m']:.4f} | "
                    f"{track['joint_position_rmse_rad']:.4f} | {track['measured_seconds']:.2f} |\n"
                )
        (target / "REPORT.md").write_text(report)
        save(target / "experiment_summary.json", {
            **checkpoint_meta, "evaluation": result, "configuration": configuration,
            "video_decode": "passed", "fresh_initialization": True,
        })
        files = sorted(
            path for path in target.rglob("*") if path.is_file()
            and path.name not in ("job_status.json", "ready.json", "artifact_manifest.json")
            and path.suffix not in (".log", ".npz")
        )
        manifest = {str(path.relative_to(target)): digest(path) for path in files}
        save(target / "artifact_manifest.json", {"sha256": manifest})
        save(target / "ready.json", {
            "iteration": iteration, "ready_at": now(),
            "files": [*manifest, "artifact_manifest.json"],
            "norm_return": result["norm_return"], "n_failed": result["n_failed"],
        })
        save(target / "job_status.json", {"phase": "completed", "iteration": iteration, "updated_at": now()})
    except Exception as error:
        save(target / "job_status.json", {"phase": "failed", "error": str(error), "updated_at": now()})
        raise


def supervise(args):
    run_dir = args.run_dir.resolve()
    config = load_config(args.config)
    if args.preflight:
        preflight = args.preflight.resolve()
        summary = load(preflight / "run_summary.json")
        check_args = load(preflight / "args.json")["arguments"]
        check_rows = read_rows(preflight / "train_log.csv")
        if summary["completed_iterations"] <= check_args["wm_only_iters"] or summary["pi_skipped_updates"]:
            raise RuntimeError("Preflight must exercise policy updates with zero skipped updates")
        if check_args["num_envs"] != args.num_envs:
            raise ValueError("Preflight environment count differs")
        if not check_rows or not all(math.isfinite(v) for row in check_rows for v in row.values()):
            raise ValueError("Preflight metrics must be finite")
        if (preflight / "stdout.log").exists():
            if any(marker in (preflight / "stdout.log").read_text() for marker in FATAL):
                raise RuntimeError("Preflight simulator reported an error")
    run_dir.mkdir(parents=True, exist_ok=False)
    stop = run_dir / "STOP"
    state = {
        "phase": "starting", "started_at": now(), "supervisor_pid": os.getpid(),
        "task": config.get("task", "g1"),
        "run_dir": str(run_dir), "num_envs": args.num_envs, "target_iteration": args.iterations,
        "start_iteration": 0, "fresh_initialization": True, "resume": None,
        "observation_mode": config.get("obs_mode", "relative_ori99"),
        "reset_noise": config.get("reset_noise", True),
        "milestones": args.milestones,
        "evaluation_errors": {},
    }

    def status(**fields):
        state.update(fields, updated_at=now())
        save(run_dir / "job_status.json", state)

    def request_stop(signum, _frame):
        stop.touch()
        status(stop_requested_by_signal=signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    command = [
        sys.executable, str(SCRIPTS / "train.py"), "--config", str(args.config.resolve()),
        *launcher_args(args.device),
        "--num_envs", str(args.num_envs), "--iterations", str(args.iterations),
        "--ckpt_every", str(args.ckpt_every), "--stop_file", str(stop),
        *motion_arguments(vars(args)), "--log_dir", str(run_dir),
    ]
    save(run_dir / "commands.json", {
        "training": command, "preflight": str(args.preflight) if args.preflight else None,
        "render_seconds": args.render_seconds, "render_clips": args.render_clips,
    })
    sources = [
        *SCRIPTS.glob("*.py"),
        *REPO.glob("source/trackerLab/trackerLab/**/supermimic/*.py"),
        *REPO.glob("source/trackerLab/trackerLab/**/supermimic.py"),
        *REPO.glob("source/trackerTask/trackerTask/supermimic/*.py"),
        *REPO.glob("source/**/__init__.py"),
        *REPO.glob("source/**/_supermimic_original_init.py"),
        *SCRIPTS.glob("configs/*.json"),
        SCRIPTS / "resources.lock.json", REPO / "pyproject.toml",
        REPO / "scripts/install_trackerlab.py", REPO / ".supermimic-install.json",
    ]
    sources = sorted({path for path in sources if path.is_file()})
    save(run_dir / "source_sha256.json", {str(p.relative_to(REPO)): digest(p) for p in sources})
    with tarfile.open(run_dir / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(REPO)))
    if args.motion_config:
        shutil.copy2(args.motion_config, run_dir / "motion_config.json")
        motion_config = load(args.motion_config)
        data_root = (args.motion_config.resolve().parent / motion_config["root"]).resolve()
        if (data_root / "manifest.json").exists():
            shutil.copy2(data_root / "manifest.json", run_dir / "data_manifest.json")
    shutil.copy2(SCRIPTS / "resources.lock.json", run_dir / "resources.lock.json")
    try:
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except subprocess.CalledProcessError:
        base_commit = None
    save(run_dir / "initialization.json", {
        "fresh": True, "start_iteration": 0, "seed": config.get("seed", 0), "resume": None,
        "preflight_weights_loaded": False, "created_at": now(),
        "base_git_commit": base_commit,
    })
    env = runtime_environment()
    train_output = (run_dir / "stdout.log").open("w")
    gpu_output = (run_dir / "gpu.csv").open("w")
    child = subprocess.Popen(command, cwd=REPO, env=env, stdout=train_output, stderr=subprocess.STDOUT)
    gpu = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw", "--format=csv", "-l", "10",
    ], stdout=gpu_output, stderr=subprocess.STDOUT)
    artifact_child, artifact_output, artifact_iteration = None, None, None
    attempted = set()
    detected_error = None
    status(phase="training", child_pid=child.pid)
    try:
        with (run_dir / "stdout.log").open() as log_reader:
            while True:
                output = log_reader.read()
                for marker in FATAL:
                    if marker in output:
                        detected_error = marker
                        stop.touch()
                rows = read_rows(run_dir / "train_log.csv")
                if rows:
                    latest = rows[-1]
                    if not all(math.isfinite(value) for value in latest.values()):
                        detected_error = "nonfinite training metric"
                        stop.touch()
                    if config.get("bounded_critic") and (
                        latest.get("v_loss", 0) > 1e4 or latest.get("imag_rew", 0) < -100
                        or latest.get("sim_rew", 0) < -5
                    ):
                        detected_error = "bounded-critic run exceeded divergence thresholds; preserving checkpoint"
                        stop.touch()
                    state["latest_metrics"] = latest
                    # Early world-model-only timing cannot predict the later policy-update cost.
                    if latest["iteration"] > config.get("wm_only_iters", 500) + 50 and latest["it_per_s"] > 0:
                        duration = max(0, args.iterations - latest["iteration"]) / latest["it_per_s"]
                        state["estimated_training_finish_at"] = (datetime.now().astimezone() + timedelta(seconds=duration)).isoformat()
                if artifact_child is not None and artifact_child.poll() is not None:
                    if artifact_child.returncode:
                        state["evaluation_errors"][str(artifact_iteration)] = artifact_child.returncode
                    artifact_output.close()
                    artifact_child, artifact_output = None, None
                    state["active_evaluation_iteration"] = None
                code = child.poll()
                if code is not None:
                    state["training_exit_code"] = code
                    if code or detected_error:
                        status(phase="failed", error=detected_error or f"training exit {code}")
                        break
                    if not (run_dir / "run_summary.json").is_file():
                        status(phase="failed", error="Training exited without run_summary.json; inspect stdout.log")
                        raise RuntimeError("Training exited without run_summary.json; inspect stdout.log")
                    runtime = load(run_dir / "run_summary.json")
                    if runtime["completed_iterations"] != args.iterations:
                        status(phase="stopped", completed_iteration=runtime["completed_iterations"], reason=runtime["stop_reason"])
                        break
                    state["phase"] = "evaluating"
                for iteration in state["milestones"]:
                    checkpoint = run_dir / f"ckpt_{iteration:06d}.pt"
                    eligible = checkpoint.exists() and (iteration < args.iterations or code == 0)
                    if artifact_child is None and iteration not in attempted and eligible:
                        attempted.add(iteration)
                        artifact_iteration = iteration
                        artifact_output = (run_dir / f"milestone_{iteration:06d}.log").open("w")
                        artifact_child = subprocess.Popen([
                            sys.executable, str(Path(__file__).resolve()), "--run_dir", str(run_dir),
                            "--package_iteration", str(iteration),
                        ], cwd=REPO, env=env, stdout=artifact_output, stderr=subprocess.STDOUT)
                        state["active_evaluation_iteration"] = iteration
                ready = sorted(int(p.parent.name) for p in (run_dir / "milestones").glob("*/ready.json"))
                status(completed_milestones=ready, error=detected_error)
                if code == 0 and artifact_child is None and len(attempted) == len(state["milestones"]):
                    status(phase="completed" if not state["evaluation_errors"] else "completed_with_evaluation_errors",
                           completed_iteration=args.iterations)
                    break
                time.sleep(15)
    finally:
        if child.poll() is None:
            stop.touch()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=30)
        if artifact_child is not None:
            artifact_child.wait()
            artifact_output.close()
        gpu.terminate()
        gpu.wait(timeout=10)
        train_output.close()
        gpu_output.close()
    if state["phase"] in ("failed", "completed_with_evaluation_errors"):
        raise RuntimeError(f"Run ended with {state['phase']}; inspect {run_dir / 'job_status.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--config", type=Path, default=SCRIPTS / "configs/lafan1_8096.json")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--ckpt_every", type=int)
    motions = parser.add_mutually_exclusive_group()
    motions.add_argument("--motion_dir", type=Path, default=REPO / "data/retargeted/lafan1_g1")
    motions.add_argument("--motion_config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--milestones", type=int, nargs="+")
    parser.add_argument("--render_seconds", type=float, default=12.0)
    parser.add_argument("--render_clips", nargs="+")
    parser.add_argument("--package_iteration", type=int)
    args = parser.parse_args()
    if args.package_iteration:
        package_milestone(args.run_dir.resolve(), args.package_iteration)
    else:
        config = load_config(args.config)
        args.num_envs = args.num_envs if args.num_envs is not None else config.get("num_envs", 4096)
        args.iterations = args.iterations if args.iterations is not None else config.get("iterations", 320000)
        args.ckpt_every = args.ckpt_every if args.ckpt_every is not None else config.get("ckpt_every", 2000)
        if min(args.num_envs, args.iterations, args.ckpt_every, args.render_seconds) <= 0:
            parser.error("environment count, iterations, checkpoint interval and video duration must be positive")
        milestones = args.milestones if args.milestones else [
            i for i in (2000, 10000, 20000, 50000) if i < args.iterations
        ]
        args.milestones = sorted(set([*milestones, args.iterations]))
        if any(i <= 0 or i > args.iterations or (i < args.iterations and i % args.ckpt_every)
               for i in args.milestones):
            parser.error("milestones must match saved checkpoint iterations within the training target")
        supervise(args)


if __name__ == "__main__":
    main()

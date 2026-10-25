"""Package a velocity checkpoint with continuous evaluation and hashed artifacts."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys


def retain_checkpoint(source, target, digest):
    """Retain an immutable milestone without duplicating same-filesystem bytes."""
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Existing milestone checkpoint differs: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise IOError("Copied checkpoint SHA-256 differs from source")
        return "copy"


def package(run_dir, iteration, trials=16):
    from supervise import load, save, digest, now, launcher_args, run_artifact_command, SCRIPTS, REPO

    run_dir = Path(run_dir)
    if trials not in (8, 16):
        raise ValueError("Use 16 trials for milestones or 8 for early screening")
    target = run_dir / "milestones" / f"{iteration:06d}"
    if (target / "ready.json").exists():
        ready = load(target / "ready.json")
        if ready.get("trials") != trials:
            raise ValueError("Existing milestone uses a different trial count")
        if ready.get("artifact_manifest_sha256") != digest(target / "artifact_manifest.json"):
            raise ValueError("Ready milestone manifest failed SHA-256")
        manifest = load(target / "artifact_manifest.json")["sha256"]
        for name, expected in manifest.items():
            if not (target / name).is_file() or digest(target / name) != expected:
                raise ValueError(f"Ready milestone artifact failed SHA-256: {name}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    save(target / "job_status.json", {"phase": "evaluating", "iteration": iteration, "updated_at": now()})
    try:
        training = load(run_dir / "args.json")["arguments"]
        if training.get("task") != "go2_velocity_dodge":
            raise ValueError("Expected a velocity/dodge run")
        difficulty = 1.0
        if training.get("task_profile") in ("jump", "adaptive"):
            rows = []
            with (run_dir / "train_log.csv").open(newline="") as stream:
                rows = [row for row in csv.DictReader(stream)
                        if int(float(row["iteration"])) <= iteration]
            difficulty = float(rows[-1]["difficulty"]) if rows else 0.0
        else:
            profile = training.get("task_profile", "standard")
            delay, duration = (100, 4000) if profile == "rapid" else (300, 6000)
            difficulty = min(1.0, max(0.0, (iteration - delay) / duration))
        checkpoint = target / "checkpoint.pt"
        storage = retain_checkpoint(run_dir / f"ckpt_{iteration:06d}.pt", checkpoint, digest)
        for name in (
            "args.json", "reward_config.json", "motion_library.json", "source_sha256.json",
            "commands.json", "transfer.json", "train_log.csv",
        ):
            if (run_dir / name).is_file():
                shutil.copy2(run_dir / name, target / name)
        source_manifest = run_dir / "source_sha256.json"
        if source_manifest.is_file():
            training_sources = load(source_manifest)
            actual_sources = {name: digest(REPO / name) for name in training_sources}
            changes = {name: {"training": training_sources[name], "evaluation": actual_sources[name]}
                       for name in training_sources if training_sources[name] != actual_sources[name]}
            reporting_only = {
                "scripts/rls/supermimic/evaluate_velocity.py",
                "scripts/rls/supermimic/package_velocity.py",
                "scripts/rls/supermimic/render_velocity.py",
            }
            if set(changes) - reporting_only:
                raise ValueError("Task/training source differs from the saved experiment snapshot")
            save(target / "evaluation_source_sha256.json", {
                "training_manifest_sha256": digest(source_manifest),
                "sha256": actual_sources, "reporting_changes_from_training": changes,
                "scope": "Training/task source is unchanged; reporting revisions are listed explicitly.",
            })
        run_artifact_command([
            sys.executable, str(SCRIPTS / "evaluate_velocity.py"),
            "--checkpoint", str(checkpoint), "--output", str(target),
            "--difficulty", str(difficulty), "--trials", str(trials),
            *launcher_args(training["device"]),
        ], target, "evaluation")
        result = load(target / "evaluation.json")
        if result["iteration"] != iteration or result["overall"]["trials"] != trials:
            raise ValueError("Evaluated checkpoint iteration or trial count differs")
        if result["checkpoint_sha256"] != digest(checkpoint):
            raise ValueError("Evaluated checkpoint SHA-256 differs")
        if result["task_profile"] != training.get("task_profile", "standard"):
            raise ValueError("Evaluated profile differs from training arguments")
        run_artifact_command([
            sys.executable, str(SCRIPTS / "render_velocity.py"),
            "--directory", str(target), "--trial", "0",
        ], target, "render")
        run_artifact_command([
            sys.executable, str(SCRIPTS / "export_policy.py"),
            "--checkpoint", str(checkpoint), "--output_dir", str(target / "policy"),
        ], target, "export")
        shutil.make_archive(str(target / "policy"), "zip", target, "policy")
        report = [
            f"# Go2 velocity/dodge {result['task_profile']} iteration {iteration}", "",
            f"{trials} predetermined continuous trials, 16 seconds each, difficulty {difficulty:.3f}.",
            "Trial 0 is selected for the video before outcomes. All trials are stored in trajectories.npz.",
            "Episodes end on a fall. Videos hold and label the terminal pose through the 16-second display.",
            "Command errors use the full root-body frame and all active transitions.", "",
            "| Mode | Attempts | Successes | Contacts | Falls | Velocity RMSE (m/s) | Yaw RMSE (rad/s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, metrics in result["per_mode"].items():
            velocity = metrics["velocity_rmse_m_s"]
            yaw = metrics["yaw_rmse_rad_s"]
            report.append(
                f"| {name} | {metrics['attempts']} | {metrics['successes']} | "
                f"{metrics['contacts']} | {metrics['falls']} | "
                f"{velocity if velocity is not None else 'unobserved'} | "
                f"{yaw if yaw is not None else 'unobserved'} |")
        overall = result["overall"]
        report += [
            "", f"Completed episodes: {overall['completed_episodes']}/{trials}.",
            f"Successful sequences: {overall['sequence_successes']}/{trials}.",
            f"Survived full episode: {overall['survived_full_episode_trials']}/{trials}.",
            f"All naturally completed events successful: {overall['completed_events_all_success_trials']}/"
            f"{overall['trials_with_completed_events']} trials with completed events.",
            f"Horizon-censored attempts: {overall['censored_attempts']}; retained as failed attempts.",
            f"Deadline misses: {overall['deadline_misses']}; terminal misses: {overall['terminal_deadline_misses']}.",
            f"Expired ahead: {overall['expired_ahead']}; lateral bypasses: {overall['lateral_bypasses']}.",
            f"Failure reasons: {overall['failure_reason_counts']}.",
            f"Raw critic theoretical upper bound: {result['theoretical_value_bound']:.8g}.",
            f"Raw critic fraction above bound: {result['raw_value_above_theoretical_bound_fraction']:.8g}.",
            "evaluation.json includes direction/mode counts, individual attempts, foot/calf minimum "
            "clearance near crossing, actual encounter "
            "intervals, valid root-crossing intervals, and raw-value quantiles.",
            "Attempt success follows the environment's natural-end, crossing, contact and fall rules. "
            "Terminal incomplete attempts count as failures. Jump profile uses a jump-only task category.",
            f"Checkpoint retention: {storage}.", "",
        ]
        if result["task_profile"] == "adaptive":
            report += [
                "## Autonomous passage", "",
                "High/low labels describe obstacle geometry only. The actor and learned dynamics "
                "receive zero category features; there is no jump/duck command or target jump height.",
                f"Successful observed maneuvers: {overall['successful_observed_maneuvers']}.",
                f"Success among naturally completed attempts: "
                f"{overall['natural_completion_success_rate']}.",
                result["protocol"]["maneuver_measurement"], "",
                "The checkpoint's current curriculum difficulty is evaluated. Full-range "
                "generalization requires a separate difficulty=1 evaluation.", "",
            ]
        (target / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
        if (target / "train_log.csv").is_file():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            with (target / "train_log.csv").open(newline="") as stream:
                rows = []
                for row in csv.DictReader(stream):
                    try:
                        values = [float(row[key]) for key in ("iteration", "sim_rew", "wm_loss", "v_loss", "difficulty")]
                    except (ValueError, TypeError, KeyError):
                        continue
                    if values[0] <= iteration:
                        rows.append(values)
            if rows:
                fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
                for index, (ax, key) in enumerate(zip(axes.flat, ("sim_rew", "wm_loss", "v_loss", "difficulty")), 1):
                    ax.plot([row[0] for row in rows], [row[index] for row in rows])
                    ax.set(xlabel="Task iteration", title=key)
                fig.savefig(target / "training_curves.png", dpi=140)
                plt.close(fig)
        if result["task_profile"] == "jump":
            clearance = result["overall"]["minimum_foot_calf_clearance_m"]
            with (target / "REPORT.md").open("a", encoding="utf-8") as report_file:
                report_file.write(
                    "\n## Jump clearance\n\n"
                    f"Events with clearance samples: {result['overall']['events_with_clearance_samples']}. "
                    f"Minimum signed foot/calf clearance distribution: {clearance}.\n"
                )
        required = (
            "checkpoint.pt", "evaluation.json", "trial_manifest.json", "trajectories.npz",
            "demo.mp4", "demo.png", "demo_contact_sheet.png", "demo.json",
            "policy/policy.pt", "policy/policy.ts", "policy/metadata.json", "policy.zip",
        )
        for name in required:
            if not (target / name).is_file() or (target / name).stat().st_size == 0:
                raise FileNotFoundError(f"Required artifact missing or empty: {name}")
        files = sorted(path for path in target.rglob("*") if path.is_file() and
                       path.suffix != ".log" and
                       path.name not in ("ready.json", "artifact_manifest.json", "job_status.json"))
        manifest = {path.relative_to(target).as_posix(): digest(path) for path in files}
        save(target / "artifact_manifest.json", {"sha256": manifest})
        save(target / "ready.json", {
            "task": "go2_velocity_dodge", "task_profile": result["task_profile"],
            "iteration": iteration, "ready_at": now(), "trials": trials, "episode_seconds": 16.0,
            "difficulty": difficulty,
            "checkpoint_storage": storage, "checkpoint_sha256": manifest["checkpoint.pt"],
            "artifact_manifest_sha256": digest(target / "artifact_manifest.json"),
            "files": [*manifest, "artifact_manifest.json"], "evaluation": overall,
        })
        save(target / "job_status.json", {"phase": "completed", "iteration": iteration, "updated_at": now()})
        return target
    except Exception as error:
        save(target / "job_status.json", {
            "phase": "failed", "iteration": iteration, "error": str(error), "updated_at": now(),
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--trials", type=int, choices=(8, 16), default=16)
    args = parser.parse_args()
    package(args.run_dir, args.iteration, args.trials)


if __name__ == "__main__":
    main()

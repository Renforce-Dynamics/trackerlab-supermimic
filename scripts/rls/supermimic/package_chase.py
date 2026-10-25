"""Package and hash one evaluated downstream-task checkpoint."""

import csv
from pathlib import Path
import shutil
import sys


def package(run_dir, iteration):
    from supervise import load, save, digest, now, launcher_args, run_artifact_command, SCRIPTS
    target = run_dir / "milestones" / f"{iteration:06d}"
    if (target / "ready.json").exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    save(target / "job_status.json", {"phase": "evaluating", "iteration": iteration, "updated_at": now()})
    try:
        training = load(run_dir / "args.json")["arguments"]
        shutil.copy2(run_dir / f"ckpt_{iteration:06d}.pt", target / "checkpoint.pt")
        for name in ("args.json", "reward_config.json", "motion_library.json", "source_sha256.json",
                     "commands.json", "transfer.json", "train_log.csv"):
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, target / name)
        # Evaluate identical full-difficulty tasks at every checkpoint.
        run_artifact_command([
            sys.executable, str(SCRIPTS / "evaluate_chase.py"),
            "--checkpoint", str(target / "checkpoint.pt"), "--output", str(target),
            "--difficulty", "1.0", "--trials_per_mode", "16",
            *launcher_args(training["device"]),
        ], target, "evaluation")
        run_artifact_command([
            sys.executable, str(SCRIPTS / "render_chase.py"), "--directory", str(target),
        ], target, "render")
        run_artifact_command([
            sys.executable, str(SCRIPTS / "export_policy.py"),
            "--checkpoint", str(target / "checkpoint.pt"), "--output_dir", str(target / "policy"),
        ], target, "export")
        shutil.make_archive(str(target / "policy"), "zip", target, "policy")
        result = load(target / "evaluation.json")
        lines = [
            f"# Go2 chase/dodge task iteration {iteration}", "",
            "Initialized from the 10,000-iteration Kine2Go gait. This is a local downstream-task extension.",
            "Original video training code and hyperparameters are unpublished.", "",
            "| Mode | Success | Collision | Fall | Velocity RMSE (m/s) | Mean distance (m) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, value in result["per_mode"].items():
            lines.append(
                f"| {name} | {value['successes']}/{value['trials']} | {value['collision_rate']:.3f} | "
                f"{value['fall_rate']:.3f} | {value['velocity_rmse_m_s']:.3f} | {value['mean_target_distance_m']:.3f} |")
        lines += [
            "", "Difficulty 1.0; 16 deterministic trials per mode; 8 seconds per trial.",
            "Videos show trial 0 of each mode, selected before observing outcomes.",
            "Physical contact >1 N counts as collision. Reward uses a differentiable geometric collision proxy.",
            f"Raw critic fraction above 10: {result['raw_value_above_10_fraction']:.6f}.", "",
        ]
        (target / "REPORT.md").write_text("\n".join(lines))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with (target / "train_log.csv").open() as stream:
            rows = [row for row in csv.DictReader(stream) if int(row["iteration"]) <= iteration]
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for ax, key in zip(axes.flat, ("sim_rew", "wm_loss", "v_loss", "difficulty")):
            ax.plot([int(r["iteration"]) for r in rows], [float(r[key]) for r in rows])
            ax.set(xlabel="Task iteration", title=key)
        fig.savefig(target / "training_curves.png", dpi=140)
        plt.close(fig)
        files = sorted(p for p in target.rglob("*") if p.is_file() and
                       p.suffix != ".log" and p.name not in ("ready.json", "artifact_manifest.json", "job_status.json"))
        manifest = {p.relative_to(target).as_posix(): digest(p) for p in files}
        save(target / "artifact_manifest.json", {"sha256": manifest})
        save(target / "ready.json", {"iteration": iteration, "ready_at": now(),
                                     "files": [*manifest, "artifact_manifest.json"], "evaluation": result["per_mode"]})
        save(target / "job_status.json", {"phase": "completed", "iteration": iteration, "updated_at": now()})
    except Exception as error:
        save(target / "job_status.json", {"phase": "failed", "error": str(error), "updated_at": now()})
        raise

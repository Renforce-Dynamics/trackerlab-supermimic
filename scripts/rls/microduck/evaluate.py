"""Evaluate native command tracking, render env zero, and export normalized ONNX."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess


def tracking_squared_errors(state, command):
    """Use the base-link frame, matching the native velocity reward terms."""
    linear = (state.root_link_lin_vel_b[:, :2] - command[:, :2]) ** 2
    angular = (state.root_link_ang_vel_b[:, 2] - command[:, 2]) ** 2
    return linear, angular


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.steps <= 0 or args.num_envs <= 0:
        parser.error("steps and num-envs must be positive")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["WANDB_MODE"] = "disabled"
    import torch
    import imageio_ffmpeg
    import mediapy
    import mjlab.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.wrappers import VideoRecorder
    from mjlab_microduck.export import ExportConfig, run_export

    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["iter"] + 1 == args.expected_updates
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ready.json").exists():
        raise RuntimeError("This evaluation is already complete")
    task = "Mjlab-Velocity-Flat-MicroDuck"
    run_export(task, ExportConfig(checkpoint_file=str(checkpoint), num_envs=1,
                                 onnx_file=str(output / "policy.onnx"), device="cuda:0"))
    cfg, agent = load_env_cfg(task, play=True), load_rl_cfg(task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 123
    cfg.viewer.width, cfg.viewer.height = 960, 540
    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
    cfg.viewer.entity_name, cfg.viewer.body_name = "robot", "trunk_base"
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 1.1, -20.0, 135.0
    cfg.viewer.env_idx, cfg.viewer.max_extra_envs = 0, 0
    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    recorder = VideoRecorder(raw, video_folder=output / "videos",
                             step_trigger=lambda step: step == 0, video_length=args.steps,
                             name_prefix="demo", disable_logger=True)
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    env = RslRlVecEnvWrapper(recorder, clip_actions=agent.clip_actions)
    runner = load_runner_cls(task)(env, asdict(agent), device="cuda:0")
    runner.load(str(checkpoint), map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    observations = env.get_observations()
    linear_sse = torch.zeros(2, device="cuda:0")
    angular_sse = torch.zeros((), device="cuda:0")
    reward_sum = torch.zeros((), device="cuda:0")
    failures = torch.zeros(args.num_envs, dtype=torch.long, device="cuda:0")
    first_failure = torch.full((args.num_envs,), -1, dtype=torch.long, device="cuda:0")
    try:
        with torch.inference_mode():
            for step in range(args.steps):
                command = raw.command_manager.get_command("twist")
                state = raw.scene["robot"].data
                linear_error, angular_error = tracking_squared_errors(state, command)
                linear_sse += linear_error.sum(0)
                angular_sse += angular_error.sum()
                actions = policy(observations)
                assert torch.isfinite(actions).all()
                observations, reward, done, extras = env.step(actions)
                assert torch.isfinite(reward).all()
                reward_sum += reward.sum()
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
                ended = done.bool() & ~timeout
                first_failure[(first_failure < 0) & ended] = step + 1
                failures += ended.long()
    finally:
        env.close()
    samples = args.steps * args.num_envs
    video = output / "videos/demo-step-0.mp4"
    assert video.exists() and video.stat().st_size > 1000
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror",
                    "-i", str(video), "-f", "null", "-"], check=True)
    local_checkpoint = output / "checkpoint.pt"
    if not local_checkpoint.exists():
        os.link(checkpoint, local_checkpoint)
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with local_checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == checkpoint_hash
    result = {
        "task": task, "algorithm": "PPO", "backend": "mjlab / MuJoCo Warp",
        "completed_updates": args.expected_updates, "checkpoint_index": saved["iter"],
        "checkpoint_sha256": checkpoint_hash,
        "num_envs": args.num_envs, "steps_per_env": args.steps, "seed": 123,
        "horizon_seconds": args.steps * raw.step_dt,
        "linear_velocity_rmse_xy_m_s": (linear_sse / samples).sqrt().cpu().tolist(),
        "yaw_rate_rmse_rad_s": float((angular_sse / samples).sqrt()),
        "mean_step_reward_unnormalized": float(reward_sum / samples),
        "non_timeout_terminations": int(failures.sum()),
        "environments_without_termination": int((failures == 0).sum()),
        "first_failure_step": first_failure.cpu().tolist(),
        "video_environment": 0, "video_decode": "passed",
        "video_camera": {"body": "trunk_base", "distance_m": 1.1,
                         "elevation_degrees": -20.0, "azimuth_degrees": 135.0,
                         "other_environments_visible": False},
        "protocol": "Official play configuration and command distribution; randomization and push events retained; one fixed seed; auto-resets included",
        "limitations": "No held-out terrain or hardware claim; reward differs from SuperMimic normalized returns",
        "visual_review": "pending",
    }
    assert all(torch.isfinite(value).all() for value in (linear_sse, angular_sse, reward_sum))
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "REPORT.md").write_text(
        f"# MicroDuck native PPO: {args.expected_updates} updates\n\n"
        f"- Velocity RMSE x/y (m/s): {result['linear_velocity_rmse_xy_m_s']}\n"
        f"- Yaw-rate RMSE (rad/s): {result['yaw_rate_rmse_rad_s']:.6f}\n"
        f"- Non-timeout terminations: {result['non_timeout_terminations']}\n"
        f"- Environments without termination: {result['environments_without_termination']}/{args.num_envs}\n"
        f"- Evaluation horizon: {result['horizon_seconds']:.2f} seconds per environment\n"
        "- Video shows environment zero, without outcome-based selection.\n"
        "- Official play randomization and push events remain enabled; auto-resets are included.\n"
        "- Native PPO reward is not comparable with SuperMimic imitation return.\n"
        "- Numerical checks and video decode passed; human visual review is pending.\n")
    hashes = {}
    for name in ("checkpoint.pt", "policy.onnx", "evaluation.json", "REPORT.md", "videos/demo-step-0.mp4"):
        with (output / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "artifact_manifest.json").write_text(json.dumps({"sha256": hashes}, indent=2) + "\n")
    (output / "ready.json").write_text(json.dumps({
        "completed_updates": args.expected_updates, "evaluation_complete": True,
        "visual_review": "pending", "files": [*hashes, "artifact_manifest.json"],
    }, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

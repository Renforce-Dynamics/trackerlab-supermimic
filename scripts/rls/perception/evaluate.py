"""Evaluate first episodes on held-out terrain seeds, with perception ablations."""

import argparse
from copy import deepcopy
from pathlib import Path

from common import digest, load, now, save
from trackerTask.native.locomotion.perception.registry import ppo_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--modes", nargs="+", choices=("normal", "zero", "shuffle"),
                        default=("normal", "zero", "shuffle"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if min(args.num_envs, args.steps) < 1:
        parser.error("num-envs and steps must be positive")
    if args.output.exists():
        parser.error("Evaluation output already exists; preserve its receipt")
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from trackerLab.tracker_env.perception.env import PerceptionEnv

    checkpoint = args.checkpoint.resolve()
    config = load(checkpoint.parent / "run_config.json")
    task = config["arguments"]["task"]
    solver = config["arguments"]["solver_iterations"]
    env = PerceptionEnv(task, args.num_envs, args.device, args.seed,
                        eval_mode=True, solver_iterations=solver)
    runner = OnPolicyRunner(env, deepcopy(ppo_config()), device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    levels = torch.arange(args.num_envs, device=args.device) % 6
    results = []
    renderer = writer = None
    if args.video:
        import mujoco
        import imageio_ffmpeg
        import numpy as np
        render_data = mujoco.MjData(env.cpu_model)
        renderer = mujoco.Renderer(env.cpu_model, height=540, width=960)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.5 * env.scale, 130, -20
        args.output.mkdir(parents=True)
        writer = imageio_ffmpeg.write_frames(str(args.output / "env-zero.mp4"), (960, 540),
                                             fps=25, macro_block_size=2)
        writer.send(None)
    try:
        with torch.inference_mode():
            for mode in args.modes:
                torch.manual_seed(args.seed)
                env.levels.copy_(levels)
                env.perception_mode = mode
                obs, _ = env.reset()
                active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
                duration = torch.zeros(args.num_envs, device=args.device)
                reward_sum = torch.zeros_like(duration)
                progress = torch.zeros_like(duration)
                velocity_sse = torch.zeros_like(duration)
                successes = torch.zeros_like(active)
                falls = torch.zeros_like(active)
                for step in range(args.steps):
                    velocity_sse += active * ((env.qvel[:, 0] - env.command) ** 2)
                    actions = policy(obs)
                    if not bool(torch.isfinite(actions).all()):
                        raise FloatingPointError("Nonfinite evaluation actions")
                    obs, reward, done, extra = env.step(actions)
                    terminal = extra["terminal"]
                    duration += active
                    reward_sum += active * reward
                    progress = torch.where(active, terminal["progress"], progress)
                    successes |= active & terminal["success"]
                    falls |= active & terminal["fall"]
                    if writer and mode == "normal" and step % 2 == 0:
                        render_data.qpos[:] = terminal["qpos"][0].cpu().numpy()
                        mujoco.mj_forward(env.cpu_model, render_data)
                        camera.lookat[:] = render_data.qpos[:3]
                        renderer.update_scene(render_data, camera=camera)
                        writer.send(np.asarray(renderer.render()))
                    active &= ~done.bool()
                    if writer and mode == "normal" and not bool(active[0]):
                        writer.close()
                        renderer.close()
                        writer = renderer = None
                    if not bool(active.any()) and not (writer and mode == "normal"):
                        break
                for level in range(6):
                    mask = levels == level
                    if not bool(mask.any()):
                        continue
                    results.append({
                        "mode": mode, "level": level, "episodes": int(mask.sum()),
                        "successes": int(successes[mask].sum()), "falls": int(falls[mask].sum()),
                        "censored_or_timeout": int((~successes & ~falls & mask).sum()),
                        "mean_progress_m": float(progress[mask].mean()),
                        "mean_first_episode_seconds": float(duration[mask].mean()) * env.step_dt,
                        "mean_return": float(reward_sum[mask].mean()),
                        "velocity_rmse_m_s": float((velocity_sse[mask].sum() /
                                                    duration[mask].sum().clamp_min(1)).sqrt()),
                    })
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    save(args.output / "evaluation.json", {
        "task": task, "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "completed_updates": runner.current_learning_iteration + 1,
        "training_seed": config["arguments"]["seed"], "evaluation_seed": args.seed,
        "terrain_seed_held_out": config["arguments"]["seed"] != args.seed,
        "seed_changes_geometry": env.task.profile == "mountain",
        "protocol": "One first episode per environment and mode; levels stratified equally; identical initial conditions; unfinished horizons remain censored.",
        "perception_limit": "Idealized terrain rays, no RGB/depth encoder. Shuffle is spatial reversal; ablations are diagnostics, not proof of causal sensor use.",
        "results": results, "finished_at": now(), "video_visual_review": "pending" if args.video else "not_recorded",
    })
    print(results, flush=True)


if __name__ == "__main__":
    main()

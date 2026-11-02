"""Compare GPU terrain rays with CPU MuJoCo and verify real physics transitions."""

import argparse
from pathlib import Path

from common import save
from trackerTask.native.locomotion.perception.registry import TASKS


def validate(task, device, seed):
    import mujoco
    import numpy as np
    import torch
    from trackerLab.tracker_env.perception.env import PerceptionEnv

    torch.set_num_threads(8)
    torch.manual_seed(seed)
    env = PerceptionEnv(task, num_envs=6, device=device, seed=seed, eval_mode=True)
    env.levels.copy_(torch.arange(6, device=device))
    env.reset()
    # Position the sensor above varied terrain; compare identical physical rays on both backends.
    env.qpos[:, 0] = torch.linspace(1, 7, 6, device=device) * env.scale
    env.qpos[:, 2] = 3 * env.height
    env._update_scan()
    qpos = env.qpos.cpu().numpy()
    origins = env.ray_origin.cpu().numpy()
    distances = env.ray_dist.cpu().numpy()
    geoms = env.ray_geom.cpu().numpy()
    data = mujoco.MjData(env.cpu_model)
    error, samples = 0., 0
    for world in range(6):
        data.qpos[:] = qpos[world]
        mujoco.mj_forward(env.cpu_model, data)
        for ray in range(0, env.num_rays, 7):
            geom = np.array([-1], dtype=np.int32)
            expected = mujoco.mj_ray(
                env.cpu_model, data, origins[world, ray].astype(np.float64),
                np.array([0., 0., -1.]), np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8),
                1, -1, geom,
            )
            if (expected < 0) != (geoms[world, ray] < 0):
                raise AssertionError("CPU/GPU ray visibility differs")
            error = max(error, abs(expected - float(distances[world, ray])))
            samples += 1
    if error > 2e-4:
        raise AssertionError(f"CPU/GPU ray distance error {error}")
    env.reset()
    before = env.qpos.clone()
    with env._warp_context():
        env._set_reset_state(torch.tensor([1], device=device))
        env.mw.forward(env.model, env.data)
    torch.testing.assert_close(env.qpos[[0, 2, 3, 4, 5]], before[[0, 2, 3, 4, 5]])
    env._update_scan()
    normal, _ = env.get_observations()
    env.perception_mode = "zero"
    zero, _ = env.get_observations()
    torch.testing.assert_close(normal[:, :-187], zero[:, :-187])
    if not bool((normal[:, -187:] != zero[:, -187:]).any()):
        raise AssertionError("Actor observation does not contain measured terrain heights")
    env.perception_mode = "normal"
    nreset = 0
    for _ in range(64):
        obs, rewards, done, extras = env.step(torch.randn_like(env.actions) * .1)
        assert torch.isfinite(obs).all() and torch.isfinite(rewards).all()
        assert extras["time_outs"].shape == (6,)
        nreset += int(done.sum())
    return {"task": task, "passed": True, "ray_samples": samples, "ray_max_error_m": error,
            "partial_reset_isolation": True, "actor_perception_ablation": True,
            "physics_steps_per_world": 256, "automatic_resets": nreset,
            "observation_dim": normal.shape[-1], "contract": env.contract}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = []
    for task in args.tasks:
        row = validate(task, args.device, args.seed)
        rows.append(row)
        save(args.output, {"passed": True, "results": rows})
        print({k: v for k, v in row.items() if k != "contract"}, flush=True)


if __name__ == "__main__":
    main()

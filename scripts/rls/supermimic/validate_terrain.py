"""Qualify terrain physics, measured perception, resets and all 18 learner jobs."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import torch

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.terrain import TASKS, generate, robot_parameters
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField, TerrainMDP
from trackerLab.tracker_env.supermimic.terrain import TerrainSim


@torch.no_grad()
def qualify(task):
    device = "cuda:0"
    spec, height, _ = robot_parameters(task.robot)
    data = generate(task.profile, height, 42)
    field = HeightField(data, height, device)
    model, *_ = build_model(task.robot, data)
    joint_ids = [model.joint(name).id for name in spec.joint_names]
    limits = torch.tensor(model.jnt_range[joint_ids], device=device, dtype=torch.float32)
    mdp = TerrainMDP(field, torch.tensor(spec.default_q, device=device), limits[:, 0], limits[:, 1])
    env = TerrainSim(task, mdp, 24, device, 42)
    levels = torch.arange(24, device=device) % 6
    env.reset(levels=levels)
    env.qpos[:, env._root_qpos] = (torch.rand(24, device=device) * 12 - 1) * field.scale
    env.qpos[:, env._root_qpos + 2] = field.sample(env.qpos[:, env._root_qpos:env._root_qpos + 2]) + height
    with env._warp_context():
        env.mw.forward(env.model, env.data)
    env._update_rays()
    predicted = field.features(env.state())
    measured = env.observe()[:, -187:]
    scan_error = float((predicted - measured).abs().max().item())
    if scan_error > 2e-3:
        raise AssertionError(f"{task.id}: physical ray/heightfield difference {scan_error}")
    if not (env.ray_geom >= 0).all().item():
        raise AssertionError(f"{task.id}: terrain rays missed the ground")
    before_q, before_v = env.qpos.clone(), env.qvel.clone()
    before_xi = env.xi[[0, 2]].clone()
    env.reset(torch.tensor([0, 2], device=device))
    reset_ids = torch.tensor([0, 2], device=device)
    if torch.equal(env.xi[reset_ids], before_xi):
        raise AssertionError(f"{task.id}: episode randomization did not refresh")
    expected_com = env.base_body_ipos[reset_ids, env.com_id, :3] + env.xi[reset_ids, -3:]
    torch.testing.assert_close(env.body_ipos[reset_ids, env.com_id, :3], expected_com)
    torch.testing.assert_close(env.geom_friction[reset_ids, :, 0],
                               env.xi[reset_ids, :1].expand(-1, env.geom_friction.shape[1]))
    keep = torch.ones(24, dtype=torch.bool, device=device)
    keep[[0, 2]] = False
    torch.testing.assert_close(env.qpos[keep], before_q[keep], atol=0, rtol=0)
    torch.testing.assert_close(env.qvel[keep], before_v[keep], atol=0, rtol=0)
    snapshot = env.state()
    copies = tuple(value.clone() for value in snapshot)
    for _ in range(16):
        tr = env.step(torch.zeros(24, spec.num_joints, device=device))
        for name in ("s", "s1", "r"):
            if not torch.isfinite(tr[name]).all().item():
                raise AssertionError(f"{task.id}: nonfinite {name}")
    for value, copy in zip(snapshot, copies):
        torch.testing.assert_close(value, copy, atol=0, rtol=0)
    result = {"task": task.id, "physics": "passed", "partial_reset": "passed",
              "state_snapshot_storage": "passed", "episode_randomization": "passed",
              "height_ray_max_abs_error": scan_error,
              "num_joints": spec.num_joints, "simulator_contract": env.contract}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    rows = []
    for name in args.tasks:
        row = qualify(TASKS[name])
        gc.collect()
        torch.cuda.empty_cache()
        if not args.skip_training:
            task = TASKS[name]
            run = args.run_dir / task.robot / task.profile
            command = [sys.executable, "-u", str(Path(__file__).with_name("train_terrain.py")),
                       "--task", name, "--run-dir", str(run), "--num-envs", "24",
                       "--iterations", "4", "--wm-only-iters", "1", "--wm-batches", "1",
                       "--wm-batch", "8", "--pi-batch", "8", "--horizon", "4", "--hist", "2",
                       "--wm-window", "2", "--buffer-steps", "16", "--warmup-steps", "8",
                       "--critic-epochs", "1", "--critic-minibatches", "1",
                       "--log-every", "1", "--no-compile"]
            subprocess.run(command, check=True, cwd=REPO)
            row["learner"] = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
        rows.append(row)
        (args.run_dir / "validation.json").write_text(
            json.dumps({"passed": len(rows), "requested": len(args.tasks), "jobs": rows}, indent=2) + "\n",
            encoding="utf-8")
        print("QUALIFIED " + name, flush=True)


if __name__ == "__main__":
    main()

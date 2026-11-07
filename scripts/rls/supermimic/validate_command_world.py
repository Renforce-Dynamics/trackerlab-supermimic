"""Compare frozen command WMs on the same recorded physical trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bootstrap import configure_paths
configure_paths()

import trackerLab.algorithms.supermimic.networks as networks
from trackerLab.algorithms.supermimic.terrain_networks import TerrainWorldModel
from trackerLab.tracker_env.mdp.supermimic import CONTROL_DT, normalize_xi, unflatten_state
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import generate, get_task


HORIZONS = (1, 8, 32)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_bank(bank, num_joints):
    """Reject incompatible shapes and nonfinite values before selecting windows."""
    required = {"s", "s1", "a", "xi", "done", "stealth_mode_id", "c"}
    if missing := required - bank.keys():
        raise ValueError(f"Bank missing fields: {sorted(missing)}")
    if bank["s"].ndim != 3:
        raise ValueError("Bank s must have shape [T, N, state_dim]")
    steps, envs = bank["s"].shape[:2]
    if steps <= 48 or envs < 1:
        raise ValueError("Bank needs T > 48 and N > 0 for the fixed sampling")
    widths = {"s": 13 + 2 * num_joints, "s1": 13 + 2 * num_joints,
              "a": num_joints, "xi": num_joints + 4, "c": 7, "c1": 7}
    for name, value in bank.items():
        expected = (steps, envs, widths[name]) if name in widths else (steps, envs)
        if tuple(value.shape) != expected:
            raise ValueError(f"Bank {name} shape {tuple(value.shape)}; expected {expected}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Bank {name} contains nonfinite values")
    if not bool(((bank["done"] == 0) | (bank["done"] == 1)).all()):
        raise ValueError("Bank done must contain boolean or 0/1 values")


@torch.inference_mode()
def evaluate_bank(reference, candidate, bank, robot="g1"):
    """Replay fixed starts without changing either model or its normalizers.

    Starts match the audit: range(16, T-32, 16), with all 32 done flags
    false. Cumulative MSE is the sum of per-step MSEs, averaged over starts.
    Physical RMSEs describe the endpoint, averaged over samples and channels.
    """
    n = reference.num_joints
    validate_bank(bank, n)
    if candidate.num_joints != n or (robot == "g1" and n != 29):
        raise ValueError("World model joint counts must match the bank and robot")
    device = bank["s"].device
    steps, envs = bank["s"].shape[:2]
    starts = torch.arange(16, steps - 32, 16, device=device)
    t0 = starts[:, None].expand(-1, envs).reshape(-1)
    e0 = torch.arange(envs, device=device).repeat(len(starts))
    valid = ~bank["done"][t0[:, None] + torch.arange(32, device=device),
                          e0[:, None]].bool().any(-1)
    excluded = int((~valid).sum())
    t0, e0 = t0[valid], e0[valid]
    if not len(t0):
        raise ValueError("Bank has no reset-free validation windows")
    common_std = reference.out_norm.std().detach().clamp_min(.01)
    if common_std.shape != (12 + 2 * n,) or not bool(torch.isfinite(common_std).all()):
        raise FloatingPointError("Reference output std must have finite state-error channels")
    channels = {
        "root_position_m": slice(0, 3), "root_rotation_rad": slice(3, 6),
        "linear_velocity_m_s": slice(6, 9), "angular_velocity_rad_s": slice(9, 12),
    }
    joints = {"leg": (0, 12), "waist": (12, 15), "arm": (15, 29)} if robot == "g1" else {
        "joint": (0, n)}
    for label, (lo, hi) in joints.items():
        channels[label + "_position_rad"] = slice(12 + lo, 12 + hi)
        channels[label + "_velocity_rad_s"] = slice(12 + n + lo, 12 + n + hi)
    results = []
    for name, world in (("reference", reference), ("candidate", candidate)):
        predicted = unflatten_state(bank["s"][t0, e0])
        cumulative = torch.zeros(len(t0), device=device, dtype=torch.float64)
        finite = torch.ones(len(t0), device=device, dtype=torch.bool)
        for k in range(32):
            predicted = world.step(predicted, bank["a"][t0 + k, e0],
                                   normalize_xi(bank["xi"][t0 + k, e0]))
            truth = unflatten_state(bank["s1"][t0 + k, e0])
            error = world.state_error(predicted, truth).double()
            for value in predicted:
                finite &= torch.isfinite(value).flatten(1).all(-1)
            finite &= torch.isfinite(error).all(-1)
            endpoint = (error / common_std.double()).square().mean(-1)
            cumulative += endpoint
            finite &= torch.isfinite(cumulative)
            if not bool(finite.all()):
                raise FloatingPointError(
                    f"{name} horizon {k + 1}: nonfinite_samples={int((~finite).sum())}")
            if k + 1 in HORIZONS:
                results.append({
                    "world_model": name, "group": "all", "horizon_steps": k + 1,
                    "samples": len(t0), "nonfinite_samples": int((~finite).sum()),
                    "common_normalized_endpoint_mse": float(endpoint.mean()),
                    "common_normalized_cumulative_mse": float(cumulative.mean()),
                    **{label + "_rmse": float(error[:, group].square().mean().sqrt())
                       for label, group in channels.items()},
                })
    return {
        "sampling": {"start": 16, "stop_exclusive": steps - 32, "stride": 16,
                     "horizons": list(HORIZONS), "samples": len(t0),
                     "excluded_reset_windows": excluded,
                     "done_rule": "all done[t:t+32] must be false"},
        "common_output_std": common_std.cpu().tolist(),
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "reference-checkpoint", "bank", "bank-protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {"bank": args.bank, "bank_protocol": args.bank_protocol,
             "reference_checkpoint": args.reference_checkpoint, "checkpoint": args.checkpoint}
    hashes = {name: digest(path) for name, path in paths.items()}
    protocol = json.loads(args.bank_protocol.read_text(encoding="utf-8"))
    task = get_task(protocol["task"])
    dt = protocol.get("control_dt_s", protocol.get("control_dt"))
    if dt != CONTROL_DT or protocol.get("robot", task.robot) != task.robot:
        raise ValueError("Bank protocol robot/control_dt differs from the task runtime")
    if ("control_dt" in protocol and protocol["control_dt"] != CONTROL_DT
            or type(protocol.get("seed")) is not int):
        raise ValueError("Bank protocol needs the runtime control_dt and an integer seed")
    if not all(isinstance(protocol.get("hashes", {}).get(key), str)
               and len(protocol["hashes"][key]) == 64 for key in ("old", "new")):
        raise ValueError("Bank protocol needs old/new collection checkpoint SHA256 hashes")
    device = torch.device(args.device)
    networks.AMP_DTYPE = torch.bfloat16 if device.type == "cuda" else None
    spec = get_robot(task.robot)
    field = HeightField(generate(task.profile, task.height, protocol["seed"]), task.height, device)
    models = []
    for path in (args.reference_checkpoint, args.checkpoint):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["task"] != task.id or checkpoint["robot"] != task.robot:
            raise ValueError(f"Checkpoint task/robot differs from bank protocol: {path}")
        for metadata in (checkpoint, checkpoint.get("arguments", {})):
            for key in ("control_dt", "control_dt_s"):
                if key in metadata and metadata[key] != CONTROL_DT:
                    raise ValueError(f"Checkpoint {key} differs from bank protocol: {path}")
        world = TerrainWorldModel(field, spec.num_joints).to(device).eval()
        world.load_state_dict(checkpoint["world"])
        if any(not bool(torch.isfinite(value).all()) for value in world.state_dict().values()):
            raise FloatingPointError(f"Checkpoint world contains nonfinite values: {path}")
        models.append(world)
        del checkpoint
    with np.load(args.bank, allow_pickle=False) as archive:
        bank = {name: torch.as_tensor(archive[name], device=device) for name in archive.files}
    for key, actual in (("steps", bank["s"].shape[0]), ("num_envs", bank["s"].shape[1])):
        if key in protocol and protocol[key] != actual:
            raise ValueError(f"Bank {key} differs from protocol")
    report = evaluate_bank(*models, bank, robot=task.robot)
    if any(digest(path) != hashes[name] for name, path in paths.items()):
        raise RuntimeError("A validation input changed during replay")
    report.update(
        task=task.id, robot=task.robot, seed=protocol["seed"], control_dt_s=dt,
        device=str(device), bank_protocol=protocol,
        inputs={name: {"path": str(path.resolve()), "sha256": hashes[name]}
                for name, path in paths.items()},
        inputs_unchanged=True,
        normalization="reference out_norm std clamped to 0.01",
        cumulative_mse="sum of per-step channel MSE, averaged over sampled starts",
        inference_precision="bfloat16 MLP / float32 state" if device.type == "cuda" else "float32",
    )
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"output": str(args.output), "samples": report["sampling"]["samples"]}))


if __name__ == "__main__":
    main()

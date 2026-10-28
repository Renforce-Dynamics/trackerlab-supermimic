"""Bounded exact FP32 CUDA checks for the actual tracker normalizer widths."""
from __future__ import annotations

import torch


def verify_actual_widths(networks, wm, original_update):
    from trackerTask.supermimic.robots import get_robot
    spec = get_robot("g1")
    goal_dim = spec.goal_frame_dim * 10
    widths = {
        "pi.obs_norm": 99,  # Explicit relative_ori99 training interface.
        "pi.goal_norm": goal_dim,
        "wm.in_norm": wm.in_norm.mean.numel(),
        "wm.out_norm": wm.out_norm.mean.numel(),
        "critic.norm": spec.state_dim + spec.critic_extra + goal_dim,
    }
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    rows = []
    try:
        for name, width in widths.items():
            epsilon = 1e-4 if name == "wm.out_norm" else 1e-2
            reference = networks.RunningNorm(width, eps=epsilon).cuda()
            candidate = networks.RunningNorm(width, eps=epsilon).cuda()
            initial_pointers = {key: getattr(candidate, key).data_ptr()
                                for key in ("mean", "var", "count")}
            base = torch.arange(64 * width, dtype=torch.float32, device="cuda").reshape(64, width)
            weights = (base.remainder(31) - 15) / 31
            for step in range(3):
                values = (base.remainder(127) - 63) / 64 + step / 8
                original_update(reference, values)
                candidate.update(values)
                left = (values / 2).clone().requires_grad_(True)
                right = left.detach().clone().requires_grad_(True)
                out_left, out_right = reference(left), candidate(right)
                grad_left, = torch.autograd.grad((out_left * weights).sum(), left)
                grad_right, = torch.autograd.grad((out_right * weights).sum(), right)
                pairs = {
                    **{field: (getattr(reference, field), getattr(candidate, field))
                       for field in ("mean", "var", "count")},
                    "std": (reference.std(), candidate.std()),
                    "forward": (out_left, out_right),
                    "input_gradient": (grad_left, grad_right),
                }
                checks = {
                    key: {
                        "exact": torch.equal(a, b),
                        "both_finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                        "both_fp32": a.dtype == b.dtype == torch.float32,
                        "max_abs": float((a - b).abs().max()),
                    }
                    for key, (a, b) in pairs.items()
                }
                pointers = all(getattr(candidate, key).data_ptr() == value
                               for key, value in initial_pointers.items())
                rows.append({
                    "normalizer": name, "width": width, "update": step + 1,
                    "count": float(candidate.count),
                    "candidate_storage_preserved": pointers, "comparisons": checks,
                })
            del reference, candidate, base, weights, values, left, right
            del out_left, out_right, grad_left, grad_right, pairs
        cpu_unchanged = torch.equal(cpu_rng, torch.get_rng_state())
        cuda_unchanged = all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    passed = (
        cpu_unchanged and cuda_unchanged and len(rows) == 15
        and all(row["candidate_storage_preserved"] for row in rows)
        and all(all(value[key] is True for key in ("exact", "both_finite", "both_fp32"))
                and value["max_abs"] == 0 for row in rows for value in row["comparisons"].values())
    )
    return {
        "schema": "actual-width-normalizer-fp32-cuda-parity-v1",
        "passed": passed, "widths": widths, "batch_rows": 64,
        "updates_per_normalizer": 3, "comparisons": 90, "rows": rows,
        "cpu_rng_unchanged": cpu_unchanged, "cuda_rng_unchanged": cuda_unchanged,
        "rng_restored": True, "optimizer_created": False,
        "actual_training_model_mutated": False,
        "source_widths": "relative_ori99 plus actual WorldModel norms and G1 RobotSpec formulas",
        "scope": "Deterministic FP32 actual-width fixture; actual compiled WM batches retain separate gates.",
    }

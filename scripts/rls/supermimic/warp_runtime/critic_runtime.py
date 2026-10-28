"""Optional unchanged-state frozen norm guards and a pure compiled critic loss."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch


def frozen_normalizer_plan(named_norms, *, enabled: bool):
    """Sample counts once; active normalizers keep their original update forever.

    The returned names are safe only while registered normalization state and
    max_count stay fixed. The training runner loads the checkpoint before calling
    this function and does not subsequently reload or unfreeze normalizers.
    """
    frozen = set()
    rows = []
    for name, norm in named_norms.items():
        count = float(norm.count.detach().cpu())
        max_count = float(norm.max_count)
        skip = enabled and count >= max_count
        if skip:
            frozen.add(name)
        rows.append({"name": name, "count": count, "max_count": max_count,
                     "skip_update_and_arguments": skip})
    return frozenset(frozen), {
        "schema": "startup-frozen-normalizer-plan-v1",
        "enabled": enabled, "normalizers": rows,
        "normalization_arithmetic_changed": False,
        "registered_state_changed": False,
        "active_normalizers_keep_original_update": True,
    }


def make_critic_minibatch_loss(
    critic, *, compile_enabled: bool, backend: str = "inductor"
) -> Callable:
    """Compile only indexing, critic forward and scalar MSE.

    Permutations, zero_grad, backward, finite-gradient checks and Adam steps stay
    at their existing call sites. The policy bootstrap continues using the
    original critic path. No dtype or matmul flag is changed.
    """
    def minibatch_loss(features, targets, indices):
        return ((critic(features[indices]) - targets[indices]) ** 2).mean()

    if not compile_enabled:
        return minibatch_loss
    kwargs = {"backend": backend, "fullgraph": True, "dynamic": False}
    if backend == "inductor":
        # Keep this arm separate from the other agent's CUDA Graph experiments.
        kwargs["options"] = {"triton.cudagraphs": False}
    return torch.compile(minibatch_loss, **kwargs)


def write_contract(log_dir, norm_plan, *, compile_enabled: bool):
    path = Path(log_dir) / "critic_norm_candidate_contract.json"
    path.write_text(json.dumps({
        "schema": "isolated-critic-normalizer-candidate-v1",
        "norm_plan": norm_plan,
        "compile_critic_minibatch_loss": compile_enabled,
        "critic_backend": "inductor" if compile_enabled else "eager",
        "critic_cuda_graphs_enabled": False,
        "precision_flags_changed": False,
        "optimizer_changed": False,
        "finite_checks_preserved": True,
        "minibatch_permutations_preserved": True,
        "policy_bootstrap_path_changed": False,
    }, indent=2) + "\n", encoding="utf-8")

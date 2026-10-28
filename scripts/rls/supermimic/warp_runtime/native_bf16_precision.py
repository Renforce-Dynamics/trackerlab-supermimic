"""Verify native BF16 MLPs, shared repaired sources and fresh CUDA numerics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if __name__ == "__main__":
    import sys
    from runtime_settings import configure_environment

    configure_environment()
    repository = Path(os.environ.get(
        "TRACKERLAB_BASE_REPO", Path(__file__).resolve().parents[4])).expanduser().resolve()
    for package in ("trackerLab", "trackerTask"):
        sys.path.insert(0, str(repository / "source" / package))
    os.environ.setdefault("SUPERMIMIC_REPO_ROOT", str(repository))

import torch
import torch.nn as nn

from canonical_source_contract import canonical_contracts


def contract(networks, args, modules, *, normalizer_source):
    if args.amp != "bf16" or networks.AMP_DTYPE is not torch.bfloat16:
        raise RuntimeError("The restored native MLP path requires BF16 autocast.")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("The restored native path requires NVIDIA_TF32_OVERRIDE=0.")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") != "0":
        raise RuntimeError("Global cuBLAS TF32 override is forbidden.")
    if os.environ.get("SM_FROZEN_NORM_GUARD", "0") != "0":
        raise RuntimeError("Scratch normalizer updates must remain active.")
    if (torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"
            or torch.is_autocast_enabled("cuda")):
        raise RuntimeError("Geometry and the outer context require IEEE FP32.")
    fresh_sources = canonical_contracts(networks)
    if normalizer_source != fresh_sources["normalizer"]:
        raise RuntimeError("Canonical RunningNorm source receipt differs from the loaded code.")
    network = normalizer_source["network"]
    # Recompute the existing actual-width fixture on this machine. A private
    # deployment receipt cannot qualify another checkout or CUDA installation.
    from legacy_normalizer_reference import LegacyRunningNorm
    from norm_actual_width_fixture import verify_actual_widths
    norm_fixture = verify_actual_widths(networks, modules["wm"], LegacyRunningNorm.update)
    if not norm_fixture["passed"]:
        raise RuntimeError("Fresh CUDA normalizer value/storage/RNG checks failed.")
    linear = {}
    for name, module in modules.items():
        layers = [(key, layer) for key, layer in module.named_modules()
                  if isinstance(layer, nn.Linear)]
        if not layers or any(type(layer) is not nn.Linear for _, layer in layers):
            raise RuntimeError("Expected original nn.Linear modules: " + name)
        if any(value.dtype != torch.float32 for value in module.state_dict().values()):
            raise RuntimeError("Parameters and normalizers must remain FP32: " + name)
        linear[name] = [key for key, _ in layers]
    return {
        "schema": "scratch-canonical-native-bf16-precision-v2",
        "implementation": "Original networks.run_mlp with CUDA BF16 autocast; native nn.Linear",
        "bf16_enabled": True,
        "bf16_scope": "World-model, policy and critic MLP forward GEMMs; FP32 MLP outputs",
        "strict_fp32_scope": ["geometry", "SO3", "state integration", "rewards",
                             "normalizers", "parameters", "Adam states"],
        "global_matmul_allow_tf32": False,
        "global_cudnn_allow_tf32": False,
        "NVIDIA_TF32_OVERRIDE": "0",
        "custom_tf32_kernel_used": False,
        "networks_source": network["source"],
        "networks_source_sha256_lf": network["sha256_lf"],
        "source_hash_rule": network["hash_rule"],
        "native_linear_modules": linear,
        "normalizer_update_method_replaced_at_runtime": False,
        "normalizer_storage_fix_verified": True,
        "normalizer_storage_fix": {
            "implementation": "Canonical networks.RunningNorm.update",
            "canonical_source": normalizer_source,
            "qualification": "Fresh actual-width FP32 CUDA fixture on this machine",
            "arithmetic_formula_changed": False,
            "variance_count_storage_preserved": True,
            "actual_width_cuda_numeric_parity_exact": True,
            "scope": "Retain FP32 buffer storage; original active statistics formula and schedule"},
        "normalizer_guard": "disabled; active updates with verified storage preservation",
        "actual_width_normalizer_fixture": norm_fixture,
        "runtime_rng_bitwise_parity_verified": False,
        "model_or_optimizer_mutation": False,
        "qualification": "Fresh actual WM batches and actor32 geometry remain mandatory.",
    }


def main():
    """Small native-runtime admission; actual learner batches own parity gates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior native BF16 admission.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 support is required.")
    import trackerLab.algorithms.supermimic.networks as networks
    from trackerLab.kinematics.supermimic.rotations import so3_exp
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    networks.AMP_DTYPE = torch.bfloat16
    torch.manual_seed(20261026)
    torch.set_float32_matmul_precision("highest")
    norm_storage_contract = canonical_contracts(networks)["normalizer"]
    module = networks.WorldModel().cuda()
    config = argparse.Namespace(amp="bf16")
    report = contract(networks, config, {"wm": module},
                      normalizer_source=norm_storage_contract)
    norm_width_fixture = report["actual_width_normalizer_fixture"]
    observed_dtypes = []
    hook = module.net[0].register_forward_hook(
        lambda _module, _inputs, output: observed_dtypes.append(str(output.dtype)))
    try:
        x = torch.zeros(32, module.in_dim, device="cuda")
        output = networks.run_mlp(module.net, x)
        output.square().mean().backward()
    finally:
        hook.remove()
    angles = torch.linspace(-0.4, 0.4, 384, device="cuda").reshape(128, 3)
    rotation = so3_exp(angles)
    orthogonality = float(
        (rotation.transpose(-1, -2) @ rotation
         - torch.eye(3, device="cuda")).abs().max())
    determinant = float((torch.linalg.det(rotation) - 1).abs().max())
    checks = {
        "actual_width_normalizer_parity": norm_width_fixture["passed"],
        "native_linear_observed_bf16": observed_dtypes == ["torch.bfloat16"],
        "mlp_output_fp32": output.dtype is torch.float32,
        "mlp_output_finite": bool(torch.isfinite(output).all()),
        "native_gradients_finite": all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in module.net.parameters()),
        "geometry_fp32": rotation.dtype is torch.float32,
        "geometry_finite": bool(torch.isfinite(rotation).all()),
        "geometry_SO3_unchanged_5e_5_threshold": max(orthogonality, determinant) <= 5e-5,
    }
    report.update(
        schema="native-bf16-runtime-admission-v1", checks=checks,
        passed=all(checks.values()), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(),
        geometry_orthogonality_max_abs=orthogonality,
        geometry_determinant_max_abs=determinant,
        scope="Small native MLP/geometry preflight; no actual learner batch or throughput claim.",
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise RuntimeError("Native BF16 runtime admission failed.")


if __name__ == "__main__":
    main()

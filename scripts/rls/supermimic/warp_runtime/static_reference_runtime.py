"""Pre-compilation root layout and guards for full-corpus CUDA graph inputs."""
from __future__ import annotations

import torch


ROOT_FIELDS = ("root_p", "root_R", "root_v", "root_w", "q", "qd",
               "root_v_local", "root_w_local", "root_gz")
BODY_FIELDS = ("body_p", "body_R", "body_v", "body_w")
METADATA_FIELDS = ("clip_len", "clip_start", "clip_end", "_clip_bin_start",
                   "bin_clip", "bin_start")


def install_static_root_fields(library, *, chunk_rows=32768, allow_cpu=False):
    """Convert packed roots before body allocation; verify all finite bits."""
    if hasattr(library, "_static_root_contract"):
        raise ValueError("Static root layout already installed")
    if hasattr(library, "_resident_body_reference"):
        raise ValueError("Convert roots before allocating body references")
    root = getattr(library, "_resident_root", None)
    if not isinstance(root, torch.Tensor) or root.ndim != 2 or root.shape[1] != 83:
        raise ValueError("Expected accepted packed 83-column root reference")
    if root.dtype != torch.float32 or root.requires_grad:
        raise ValueError("Root reference must be non-gradient float32")
    if root.device.type != "cuda" and not allow_cpu:
        raise ValueError("Production reference conversion requires CUDA")
    provider = library._gpu_reference_bodies
    if provider.library is not library or provider.cache or provider.forward_calls:
        raise ValueError("Expected unused body provider referencing the current library")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(root.device).clone() if root.is_cuda else None
    replacements, details = {}, {}
    all_finite = torch.ones((), dtype=torch.bool, device=root.device)
    all_bits_equal = torch.ones_like(all_finite)
    root_storage = root.untyped_storage().data_ptr()
    for name in ROOT_FIELDS:
        source = getattr(library, name)
        if source.untyped_storage().data_ptr() != root_storage:
            raise ValueError(f"{name} no longer aliases accepted packed roots")
        target = source.clone(memory_format=torch.contiguous_format)
        if not target.is_contiguous() or target.data_ptr() % 16:
            raise RuntimeError(f"Failed contiguous/aligned root allocation: {name}")
        checked = 0
        for start in range(0, source.shape[0], chunk_rows):
            old = source[start:start+chunk_rows].contiguous()
            new = target[start:start+chunk_rows]
            all_finite &= torch.isfinite(new).all()
            all_bits_equal &= (old.view(torch.int32) == new.view(torch.int32)).all()
            checked += new.numel()
        replacements[name] = target
        details[name] = {
            "shape": list(source.shape), "old_stride": list(source.stride()),
            "new_stride": list(target.stride()), "new_data_ptr_mod16": 0,
            "values_checked": checked,
            "bytes": target.numel() * target.element_size(),
        }
    if not bool(all_finite) or not bool(all_bits_equal):
        raise FloatingPointError("Root conversion failed full finite/bitwise check")
    if not torch.equal(cpu_rng, torch.get_rng_state()) or (
            cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(root.device))):
        raise RuntimeError("Root conversion changed RNG state")
    for name, value in replacements.items():
        torch._dynamo.mark_static_address(value, guard=True)
        setattr(library, name, value)
    library._resident_root = None
    library._resident_root_fields = tuple(replacements.values())
    contract = {
        "schema": "static-root-fields-v2",
        "fields": details, "frames": root.shape[0],
        "all_values_finite": True, "all_values_bitwise_equal": True,
        "root_reference_bytes": sum(row["bytes"] for row in details.values()),
        "temporary_root_copy_bytes": root.untyped_storage().nbytes(),
        "root_conversion_precedes_body_allocation": True,
        "packed_root_owner_released": True,
        "provider_refers_to_current_library": True,
        "provider_cache_empty": True, "rng_unchanged": True,
    }
    library._static_root_contract = contract
    return contract


def finalize_static_reference_fields(library):
    """Assert bodies were built separately; mark every immutable field/metadata."""
    if not hasattr(library, "_static_root_contract"):
        raise ValueError("Root layout acceptance is missing")
    if not library._resident_body_reference_report.get("separate_contiguous_body_fields"):
        raise ValueError("Body installer must allocate four separate contiguous tensors")
    identities, tensors, details = {}, [], {}
    for name in (*ROOT_FIELDS, *BODY_FIELDS, *METADATA_FIELDS):
        value = getattr(library, name)
        if not isinstance(value, torch.Tensor) or not value.is_contiguous() or value.data_ptr() % 16:
            raise ValueError(f"Unaligned/noncontiguous static reference: {name}")
        if value.requires_grad:
            raise ValueError(f"Reference field unexpectedly requires gradients: {name}")
        torch._dynamo.mark_static_address(value, guard=True)
        identities[name] = (id(value), value.data_ptr(), value._version)
        tensors.append(value)
        details[name] = {
            "shape": list(value.shape), "stride": list(value.stride()),
            "bytes": value.numel() * value.element_size(), "data_ptr_mod16": 0,
            "address_guarded": True,
        }
    library._static_reference_keepalive = tuple(tensors)
    library._static_reference_identity = identities
    provider = library._gpu_reference_bodies
    if provider.library is not library or provider.cache or provider.forward_calls:
        raise ValueError("Original reference-provider provenance changed")
    if isinstance(library._resident_body_reference, torch.Tensor):
        raise ValueError("Packed body allocation is unexpectedly retained")
    return {
        "schema": "static-all-reference-fields-v2", "fields": details,
        "root_contract": library._static_root_contract,
        "full_body_finite_bitwise_copy_check": {
            key: library._resident_body_reference_report[key]
            for key in ("all_values_finite", "full_corpus_copy_bitwise_equal",
                        "body_values_checked", "frames")},
        "packed_root_owner_released": library._resident_root is None,
        "packed_body_allocation_created": False,
        "root_and_body_reference_bytes": sum(details[name]["bytes"]
                                           for name in (*ROOT_FIELDS, *BODY_FIELDS)),
        "cuda_static_input_classification_verified": False,
    }


def assert_static_reference_identity(library):
    for name, expected in library._static_reference_identity.items():
        value = getattr(library, name)
        actual = (id(value), value.data_ptr(), value._version)
        if actual != expected:
            raise RuntimeError(f"Static reference identity/version changed: {name}")

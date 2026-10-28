"""Initialization-only strict-FP32 reference-body residency candidate.

This is an isolated performance candidate, not an installed production change.
It computes the accepted eager FK once in bounded GPU chunks and replaces the
four compatibility field wrappers with ordinary tensor views. Root references,
predicted-state FK, observations, losses, random draws and training membership
are unchanged. The historical body.f32 is deliberately not consumed.
"""
from __future__ import annotations

import math
import os
import time

import torch


GiB = 1024**3
ROOT_NAMES = ("root_p", "root_R", "root_v", "root_w", "q", "qd")
BODY_NAMES = ("body_p", "body_R", "body_v", "body_w")


def _device(library):
    device = torch.device(library.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _precision_gate(device, allow_cpu_for_tests):
    if device.type == "cpu" and allow_cpu_for_tests:
        return
    if device.type != "cuda":
        raise ValueError("Production body residency requires CUDA")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise ValueError("Launch must pin NVIDIA_TF32_OVERRIDE=0 before importing torch")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("TF32 must remain disabled")
    if torch.get_float32_matmul_precision() != "highest":
        raise ValueError("Reference FK requires highest float32 matmul precision")


def install_resident_reference_bodies(
    library, kinematics, *, chunk_rows=32768, reserve_bytes=32 * GiB,
    max_reference_bytes=64 * GiB, allow_cpu_for_tests=False, progress=None,
):
    """Compute and install immutable full-corpus body tensors before training.

    ``reserve_bytes`` is headroom AFTER the body allocation. At initialization
    it must cover not-yet-created learner, replay, simulator and compile peaks.
    The root owner must review the measured workload before choosing this value.
    No allocation is evicted, no CUDA allocator cache is cleared, and no host
    file cache is flushed. The function fails rather than changing settings.

    All allocations/copies/validation finish before committing field changes.
    Use only before any concurrent reader or compiled training invocation.
    ``progress`` receives CPU metadata only; it must not mutate the library.
    """
    if (not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool)
            or chunk_rows <= 0):
        raise ValueError("chunk_rows must be a positive integer")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in (reserve_bytes, max_reference_bytes)):
        raise ValueError("Memory budgets must be nonnegative integers")
    if hasattr(library, "_resident_body_reference"):
        raise ValueError("Resident body references are already installed")
    provider = getattr(library, "_gpu_reference_bodies", None)
    if provider is None or provider.kinematics is not kinematics:
        raise ValueError("Expected the current computed-body provider and exact kinematics")
    if provider.cache or provider.forward_calls:
        raise ValueError("Install immediately after library construction, before reference queries")
    device = _device(library)
    _precision_gate(device, allow_cpu_for_tests)
    frames = library.num_frames
    if not isinstance(frames, int) or frames < 1:
        raise ValueError("Reference corpus must contain at least one frame")
    for name in ROOT_NAMES:
        value = getattr(library, name)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.device != device or value.shape[0] != frames
                or value.requires_grad):
            raise ValueError(f"{name} must be a non-gradient FP32 resident root tensor")
    shapes = [(len(kinematics.tracked_idx), 3), (len(kinematics.tracked_idx), 3, 3),
              (len(kinematics.tracked_idx), 3), (len(kinematics.tracked_idx), 3)]
    for name, shape in zip(BODY_NAMES, shapes):
        if tuple(getattr(library, name).shape) != (frames, *shape):
            raise ValueError(f"Unexpected tracked-body layout: {name}")
    widths = [math.prod(shape) for shape in shapes]
    width = sum(widths)
    required = frames * width * 4
    root = getattr(library, "_resident_root", None)
    if isinstance(root, torch.Tensor):
        root_bytes = root.numel() * root.element_size()
    elif getattr(library, "_static_root_contract", {}).get("all_values_bitwise_equal"):
        root_bytes = library._static_root_contract["root_reference_bytes"]
    else:
        raise ValueError("Accepted packed or separately verified roots are required")
    if root_bytes + required > max_reference_bytes:
        raise MemoryError("Root plus body references exceed max_reference_bytes")
    if device.type == "cuda":
        free_before, total = torch.cuda.mem_get_info(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        reusable_before = max(0, reserved_before - allocated_before)
        available_before = free_before + reusable_before
        # Conservative transient budget: full per-link FK tensors plus inputs,
        # results, and intermediates. No duplicate full-corpus body allocation.
        transient = chunk_rows * (kinematics.num_bodies * 160 + 512) * 4
        if available_before < required + reserve_bytes + transient:
            raise MemoryError("Insufficient device memory for body fields, FK chunks and reserve")
    else:
        total, transient, free_before = None, None, None
        allocated_before = reserved_before = available_before = None
    started = time.perf_counter()
    # Allocate final fields directly; never create a packed 38-GiB body copy.
    replacements = {
        name: torch.empty((frames, *shape), dtype=torch.float32, device=device)
        for name, shape in zip(BODY_NAMES, shapes)
    }
    finite = torch.ones((), dtype=torch.bool, device=device)
    copy_equal = torch.ones_like(finite)
    body_values_checked = 0
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    chunks = 0
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, frames, chunk_rows):
            stop = min(start + chunk_rows, frames)
            # Query-time advanced indexing creates contiguous inputs. Recreate
            # those strides rather than passing strided root-cache slices.
            states = tuple(getattr(library, name)[start:stop].contiguous()
                           for name in ROOT_NAMES)
            outputs = tuple(kinematics.forward(*states))
            if len(outputs) != 4:
                raise ValueError("Kinematics must return four tracked-body fields")
            for name, shape, value in zip(BODY_NAMES, shapes, outputs):
                if (value.dtype != torch.float32 or value.device != device
                        or tuple(value.shape) != (stop - start, *shape)):
                    raise ValueError(f"Unexpected current-FK output: {name}")
                finite &= torch.isfinite(value).all()
                replacements[name][start:stop].copy_(value)
                copied = replacements[name][start:stop]
                copy_equal &= (copied.view(torch.int32)
                               == value.contiguous().view(torch.int32)).all()
                finite &= torch.isfinite(copied).all()
                body_values_checked += copied.numel()
            chunks += 1
            if progress is not None:
                progress({"completed_frames": stop, "total_frames": frames,
                          "chunks": chunks, "elapsed_seconds": time.perf_counter() - started})
            del states, outputs
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        allocated_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        available_after = free_after + max(0, reserved_after - allocated_after)
        if available_after < reserve_bytes:
            raise MemoryError("Device free memory fell below the reviewed reserve")
    else:
        free_after = None
        allocated_after = reserved_after = available_after = None
    if not bool(finite):
        raise FloatingPointError("Current FK generated nonfinite body references")
    if not bool(copy_equal) or body_values_checked != frames * width:
        raise RuntimeError("Body full-corpus destination copy differs from eager FK")
    for name in BODY_NAMES:
        value = replacements[name]
        if not value.is_contiguous() or value.data_ptr() % 16:
            raise RuntimeError("Body field is not independently contiguous/aligned")
        torch._dynamo.mark_static_address(value, guard=True)
    if not torch.equal(rng_cpu, torch.get_rng_state()):
        raise RuntimeError("Body construction changed CPU torch RNG state")
    if rng_cuda is not None and not torch.equal(rng_cuda, torch.cuda.get_rng_state(device)):
        raise RuntimeError("Body construction changed CUDA torch RNG state")
    report = {
        "candidate": "strict-fp32-body-residency-separate-static-v2",
        "separate_contiguous_body_fields": True,
        "full_corpus_copy_bitwise_equal": True,
        "body_values_checked": body_values_checked,
        "packed_body_allocation_created": False,
        "source": "current eager FK over accepted resident roots",
        "historical_body_cache_consumed": False,
        "frames": frames, "tracked_bodies": len(kinematics.tracked_idx),
        "root_reference_bytes": root_bytes, "body_reference_bytes": required,
        "resident_reference_bytes": root_bytes + required,
        "reserve_bytes": reserve_bytes, "estimated_chunk_transient_bytes": transient,
        "free_device_bytes_before": free_before, "free_device_bytes_after": free_after,
        "allocator_accounting": {
            "free_before": free_before, "allocated_before": allocated_before,
            "reserved_before": reserved_before, "available_before": available_before,
            "free_after": free_after, "allocated_after": allocated_after,
            "reserved_after": reserved_after, "available_after": available_after,
            "reserve_bytes_unchanged": reserve_bytes,
            "cuda_cache_purge": False, "host_cache_change": False,
            "allocation_failure_still_fatal": True,
        },
        "device_total_bytes": total, "device": str(device),
        "chunk_rows": chunk_rows, "chunks": chunks,
        "elapsed_seconds": time.perf_counter() - started,
        "reference_fk_at_runtime": False, "reference_field_graph_breaks": False,
        "runtime_host_reference_gathers": False, "all_values_finite": True,
        "rng_cpu_unchanged": True, "rng_cuda_unchanged": True,
        "predicted_state_fk_unchanged": True, "numerical_acceptance_pending": True,
    }
    info = dict(getattr(library, "storage_info", {}))
    info.update(
        backend="gpu-resident-strict-fp32-root-and-body",
        body_reference_backend="current eager FK computed once before training",
        resident_reference_bytes=root_bytes + required,
        reference_host_gather_required=False,
        compile_note="Root and body fields are ordinary tensor views; no reference-FK wrapper",
    )
    replacements.update(
        _resident_body_reference=tuple(replacements[name] for name in BODY_NAMES),
        storage_info=info,
                        _resident_body_reference_report=report)
    # The old empty provider is retained only as immutable audit provenance.
    # Its query/field wrappers are no longer used by the library.
    library.__dict__.update(replacements)
    return report

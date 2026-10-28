"""Initialization-only GPU reference kinematics for root-resident motion data.

The source MotionLib creates all four body fields with kin.forward applied to
its root state, q, and qd. This candidate computes that same function for the
requested GPU rows and reuses its outputs within one query. It avoids storing
38 GiB of redundant body fields and performs no host reference-data gather.
It intentionally retains graph breaks at this compatibility boundary.
"""
from __future__ import annotations

from collections import OrderedDict
import weakref

import torch


BODY_NAMES = ("body_p", "body_R", "body_v", "body_w")
ROOT_NAMES = ("root_p", "root_R", "root_v", "root_w", "q", "qd")


class GPUReferenceBodies:
    """Bounded GPU query cache; resident root tensors remain authoritative."""

    def __init__(self, library, kinematics, max_cache_bytes=256 * 1024**2):
        if not isinstance(max_cache_bytes, int) or max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be a nonnegative integer")
        self.library = library
        self.kinematics = kinematics
        self.max_cache_bytes = max_cache_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.forward_calls = 0
        self.queried_frames = 0
        self.cache_hits = 0
        self.device = torch.device(library.device)

    @torch.compiler.disable
    @torch.no_grad()
    def query(self, original):
        version = None
        if isinstance(original, torch.Tensor):
            try:
                version = original._version
            except RuntimeError:
                # Inference tensors have no version counter. Recompute rather
                # than cache a potentially mutable tensor under an unsafe key.
                pass
        key = (id(original), version)
        if version is not None and key in self.cache:
            ref, outputs, index_shape, nbytes = self.cache[key]
            if ref() is original:
                self.cache.move_to_end(key)
                self.cache_hits += 1
                return outputs, index_shape
            self.cache_bytes -= nbytes
            del self.cache[key]
        index = original
        if isinstance(index, slice):
            index = torch.arange(
                *index.indices(self.library.num_frames), device=self.device
            )
        index = torch.as_tensor(index, dtype=torch.long, device=self.device)
        index_shape = tuple(index.shape)
        flat = index.reshape(-1)
        states = tuple(getattr(self.library, name)[flat] for name in ROOT_NAMES)
        outputs = tuple(self.kinematics.forward(*states))
        if len(outputs) != 4 or any(value.shape[0] != flat.numel() for value in outputs):
            raise ValueError("Kinematics must return four tracked-body tensors")
        if any(value.dtype != torch.float32 or value.device != self.device for value in outputs):
            raise ValueError("Reference body fields must remain float32 on the root device")
        self.forward_calls += 1
        self.queried_frames += flat.numel()
        nbytes = sum(value.numel() * value.element_size() for value in outputs)
        if version is not None and nbytes <= self.max_cache_bytes:
            # Purge dead query objects and earlier versions of this tensor.
            for previous, (ref, _, _, size) in list(self.cache.items()):
                if ref() is None or previous[0] == id(original):
                    self.cache_bytes -= size
                    del self.cache[previous]
            while self.cache and self.cache_bytes + nbytes > self.max_cache_bytes:
                _, (_, _, _, size) = self.cache.popitem(last=False)
                self.cache_bytes -= size
            self.cache[key] = (weakref.ref(original), outputs, index_shape, nbytes)
            self.cache_bytes += nbytes
        return outputs, index_shape


class GPUComputedBodyField:
    def __init__(self, provider, field_index, feature_shape):
        self.provider = provider
        self.field_index = field_index
        self.feature_shape = tuple(feature_shape)
        self.shape = (provider.library.num_frames, *self.feature_shape)
        self.ndim = len(self.shape)
        self.dtype = torch.float32
        self.device = provider.device

    @torch.compiler.disable
    def __getitem__(self, index):
        extra = ()
        if isinstance(index, tuple):
            index, extra = index[0], index[1:]
        outputs, index_shape = self.provider.query(index)
        value = outputs[self.field_index].reshape(*index_shape, *self.feature_shape)
        return value[(..., *extra)] if extra else value


def install_gpu_reference_bodies(
    library, kinematics, *, max_cache_bytes=256 * 1024**2, allow_cpu_for_tests=False
):
    """Replace only body-field readers after root-group GPU promotion.

    Installation occurs before simulation, compiled training, and concurrent
    reads. The caller must compare returned values with the original body cache
    on actual data before accepting the candidate.
    """
    if hasattr(library, "_gpu_reference_bodies"):
        raise ValueError("Computed body fields are already installed")
    device = torch.device(library.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda" and not allow_cpu_for_tests:
        raise ValueError("Production reference-body computation requires CUDA")
    for name in ROOT_NAMES:
        value = getattr(library, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a resident tensor")
        if value.device != device or value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 on {device}")
        if value.shape[0] != library.num_frames:
            raise ValueError(f"{name} frame count differs from the corpus")
    shapes = [tuple(getattr(library, name).shape[1:]) for name in BODY_NAMES]
    provider = GPUReferenceBodies(library, kinematics, max_cache_bytes)
    provider.device = device
    replacements = {
        name: GPUComputedBodyField(provider, index, shape)
        for index, (name, shape) in enumerate(zip(BODY_NAMES, shapes))
    }
    info = dict(getattr(library, "storage_info", {}))
    info.update(
        backend="gpu-root-resident-computed-bodies-float32",
        body_reference_backend="same kinematics as source MotionLib, evaluated on requested GPU rows",
        body_query_cache_max_bytes=max_cache_bytes,
        reference_host_gather_required=False,
        compile_note="Root tensor indexing; compatibility graph breaks around cached GPU reference FK",
    )
    replacements.update(_gpu_reference_bodies=provider, storage_info=info)
    library.__dict__.update(replacements)
    return {
        "installed_fields": list(BODY_NAMES),
        "root_fields": list(ROOT_NAMES),
        "body_query_cache_max_bytes": max_cache_bytes,
        "reference_host_gather_required": False,
        "precision": "float32",
        "numerical_validation_required": True,
        "graph_breaks_eliminated": False,
    }

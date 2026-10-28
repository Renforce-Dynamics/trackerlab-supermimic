"""Float32 GPU root corpus with reference-body FK computed on the GPU.

Accepts the root-only cache with its selected-configuration hash. Loading
performs one integrity-checked copy. Training never reads CPU reference rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from trackerLab.motion_buffer.motion_lib.supermimic import BIN_FRAMES, MotionLib
from reference_gpu_fk import install_gpu_reference_bodies


ROOT_LAYOUT = (
    ("root_p", (3,)), ("root_R", (3, 3)), ("root_v", (3,)), ("root_w", (3,)),
    ("q", (29,)), ("qd", (29,)), ("root_v_local", (3,)), ("root_w_local", (3,)),
    ("root_gz", ()),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class _FrameMap:
    """Device-side clip/bin lookup without a dense integer array per frame."""

    def __init__(self, owner, bins=False):
        self.owner, self.bins = owner, bins
        self.shape, self.dtype, self.device = (
            (owner.num_frames,), torch.long, torch.device(owner.device))

    def __getitem__(self, index):
        motion = self.owner
        if isinstance(index, slice):
            index = torch.arange(*index.indices(motion.num_frames), device=motion.device)
        index = torch.as_tensor(index, dtype=torch.long, device=motion.device)
        index = torch.where(index < 0, index + motion.num_frames, index)
        clip = torch.searchsorted(motion.clip_end, index.contiguous())
        if not self.bins:
            return clip
        offset = index - motion.clip_start[clip]
        result = motion._clip_bin_start[clip] + torch.div(
            offset, BIN_FRAMES, rounding_mode="floor")
        # Match MotionLib.frame_bin exactly: at lengths 51, 101, ... the
        # inclusive final frame retains the dense array's initial zero.
        tail = ((offset == motion.clip_len[clip] - 1)
                & (offset.remainder(BIN_FRAMES) == 0))
        return torch.where(tail, 0, result)


class CompactGpuMotionLib(MotionLib):
    def __init__(
        self, cache_dir, kinematics, device, motion_config, *,
        reserve_bytes=24 * 1024**3, chunk_rows=131072, allow_cpu_for_tests=False,
    ):
        cache = Path(cache_dir).resolve()
        meta_path = cache / "manifest.json"
        meta = json.loads(meta_path.read_text())
        if meta.get("schema") != "supermimic-root-cache-v1":
            raise ValueError("Unsupported compact reference manifest")
        config_path = Path(motion_config).resolve()
        config = json.loads(config_path.read_text())
        expected_names = [Path(name).stem for name in config["files"]]
        if expected_names != meta["names"] or len(set(expected_names)) != len(expected_names):
            raise ValueError("Cache motion order must equal the selected configuration")
        if config["joint_names"] != list(kinematics.joint_names):
            raise ValueError("Reference CSV joint order differs from kinematics")
        if config["source_fps"] != 50 or config["quaternion_order"] != "xyzw":
            raise ValueError("This root cache represents canonical 50Hz XYZW input")
        if sha256(config_path) != meta.get("source_config_sha256"):
            raise ValueError("Source configuration hash changed")
        lengths = meta["clip_lengths"]
        if len(lengths) != len(expected_names) or any(n < 2 for n in lengths):
            raise ValueError("Every clip requires at least two frames")
        self.num_frames = sum(lengths)
        if "num_frames" in meta and self.num_frames != meta["num_frames"]:
            raise ValueError("Frame count differs from clip lengths")
        self.names = expected_names
        self.num_joints = 29
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        if dev.type != "cuda" and not allow_cpu_for_tests:
            raise ValueError("Production compact references require CUDA")
        self.device = str(dev)
        group = meta["groups"]["root"]
        offset = 0
        for field, (expected_name, expected_shape) in zip(group["fields"], ROOT_LAYOUT):
            if (field["name"] != expected_name or tuple(field["shape"]) != expected_shape
                    or field["offset"] != offset):
                raise ValueError("Unexpected root field layout")
            offset += int(np.prod(expected_shape, dtype=np.int64)) if expected_shape else 1
        if len(group["fields"]) != len(ROOT_LAYOUT) or group["width"] != offset or offset != 83:
            raise ValueError("Reference rows must contain exactly the nine float32 root fields")
        source = (cache / group["file"]).resolve()
        if not source.is_relative_to(cache):
            raise ValueError("Reference binary must remain inside its cache directory")
        required = self.num_frames * 83 * 4
        if source.stat().st_size != required or group["bytes"] != required:
            raise ValueError("Reference binary size differs from its manifest")
        if sha256(source) != group["sha256"]:
            raise ValueError("Reference binary hash differs from its manifest")
        if dev.type == "cuda":
            free, _ = torch.cuda.mem_get_info(dev)
            if free < required + reserve_bytes:
                raise MemoryError("Insufficient free device memory for references and learner reserve")
        mapping = np.memmap(source, mode="c", dtype=np.float32, shape=(self.num_frames, 83))
        root = torch.empty((self.num_frames, 83), dtype=torch.float32, device=dev)
        for start in range(0, self.num_frames, chunk_rows):
            stop = min(self.num_frames, start + chunk_rows)
            root[start:stop].copy_(torch.from_numpy(mapping[start:stop]), non_blocking=False)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        del mapping
        self._resident_root = root
        for field in group["fields"]:
            shape = tuple(field["shape"])
            width = int(np.prod(shape)) if shape else 1
            setattr(self, field["name"], root[:, field["offset"]:field["offset"] + width].view(
                self.num_frames, *shape))
        self.clip_len = torch.tensor(lengths, dtype=torch.long, device=dev)
        self.clip_start = torch.cat((torch.zeros(1, dtype=torch.long, device=dev),
                                     self.clip_len.cumsum(0)[:-1]))
        self.clip_end = self.clip_start + self.clip_len - 1
        counts = ((self.clip_len - 1 + BIN_FRAMES - 1) // BIN_FRAMES).clamp_min(1)
        self._clip_bin_start = torch.cat((torch.zeros(1, dtype=torch.long, device=dev),
                                          counts.cumsum(0)[:-1]))
        self.bin_clip = torch.repeat_interleave(
            torch.arange(len(lengths), device=dev), counts)
        self.num_bins = int(counts.sum().item())
        local_bin = torch.arange(self.num_bins, device=dev) - self._clip_bin_start[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + local_bin * BIN_FRAMES
        self.clip_id, self.frame_bin = _FrameMap(self), _FrameMap(self, bins=True)
        nbody = len(kinematics.tracked_idx)
        for name, shape in (
            ("body_p", (nbody, 3)), ("body_R", (nbody, 3, 3)),
            ("body_v", (nbody, 3)), ("body_w", (nbody, 3)),
        ):
            setattr(self, name, SimpleNamespace(shape=(self.num_frames, *shape)))
        self.provenance = {
            "source": str(config_path), "source_config_sha256": sha256(config_path),
            "cache_manifest_sha256": sha256(meta_path),
            "source_filter_sha256": meta.get("source_filter_sha256"),
            "cache_root_sha256": group["sha256"],
        }
        self.storage_info = {
            "backend": "gpu-root-resident-computed-bodies-float32",
            "cache_dir": str(cache), "full_corpus": True,
            "reference_file_bytes": required, "root_resident_bytes": required,
            "runtime_reference_host_to_device_bytes": 0,
            "body_reference": "Current kinematics from float32 root state; old body cache is unused",
            "terminal_goal": "Canonical MotionLib: recorded endpoint velocity retained; synthetic tail velocity zero",
        }
        install_gpu_reference_bodies(self, kinematics, allow_cpu_for_tests=allow_cpu_for_tests)

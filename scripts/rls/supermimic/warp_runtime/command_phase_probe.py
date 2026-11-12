"""Optional boundary-synchronized timing for command/AMP training.

The caller owns opt-in, exclusion from headline throughput measurements, and
calling ``end`` after normal logging/checkpoint I/O. This probe never resets
allocator peaks or changes model, optimizer, compilation, normalizer or RNG
state. ``finish`` also runs on normal process exit; signal handling remains
caller-owned.
"""
from __future__ import annotations

import atexit
from collections.abc import Mapping
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import torch


_OPTIMIZERS = ("world_model", "actor", "critic", "amp")


def _index(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer (zero-based)")
    return int(value)


def _optimizer_snapshot(optimizer):
    """Use .get(), not defaultdict indexing: absent state must stay absent."""
    if optimizer is None:
        return None, ()
    steps, parameters = [], []
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            parameters.append(parameter)
            state = optimizer.state.get(parameter, {})
            record = {
                "parameter": f"{group_index}:{parameter_index}",
                "step": None, "status": "missing",
            }
            if "step" in state:
                value = state["step"]
                record["storage"] = str(value.device) if torch.is_tensor(value) else "python"
                try:
                    if torch.is_tensor(value):
                        if value.numel() != 1:
                            raise ValueError("Step must be scalar")
                        value = value.detach().item()
                    if (isinstance(value, bool) or not isinstance(value, Real)
                            or not math.isfinite(value) or value < 0 or int(value) != value):
                        raise ValueError("Step must be finite, integral and nonnegative")
                    record.update(step=int(value), status="ok")
                except (TypeError, ValueError, RuntimeError, OverflowError):
                    record["status"] = "invalid"
            steps.append(record)
    values = [entry["step"] for entry in steps]
    uniform = (values[0] if values and None not in values
               and len(set(values)) == 1 else None)
    return {"steps": steps, "uniform_step": uniform}, tuple(parameters)


def _optimizer_delta(source, end, source_parameters, end_parameters):
    if source is None or end is None:
        return None, "unavailable"
    if (len(source_parameters) != len(end_parameters)
            or any(a is not b for a, b in zip(source_parameters, end_parameters))
            or [s["parameter"] for s in source["steps"]]
            != [s["parameter"] for s in end["steps"]]):
        return None, "parameter_layout_changed"
    pairs = list(zip(source["steps"], end["steps"]))
    if not pairs or any(a["step"] is None or b["step"] is None for a, b in pairs):
        return None, "missing_or_invalid_steps"
    deltas = [b["step"] - a["step"] for a, b in pairs]
    if any(delta < 0 for delta in deltas):
        return None, "steps_decreased"
    if len(set(deltas)) != 1:
        return None, "nonuniform_delta"
    return deltas[0], "valid"


class CommandPhaseProbe:
    """Measure at most 1,000 requested, zero-based iteration indices.

    Optimizer entries may be omitted when a learner is absent. Missing/lazy
    step state remains unknown, never an inferred zero. Metadata must be a
    finite JSON mapping; initialization provenance is supplied by the caller.
    Repeated phase names are supported sequentially, with distinct names for
    simultaneously open phases.
    """

    def __init__(self, output_directory, *, first_iteration, updates, device,
                 optimizers, metadata=None):
        self.first_iteration = _index(first_iteration, "first_iteration")
        self.updates = _index(updates, "updates")
        if not 1 <= self.updates <= 1000:
            raise ValueError("updates must be an integer in 1..1000")
        if not isinstance(device, (str, torch.device)):
            raise ValueError("device must specify cpu or cuda")
        try:
            self.device = torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise ValueError("device must specify cpu or cuda") from error
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must specify cpu or cuda")
        if not isinstance(optimizers, Mapping) or set(optimizers) - set(_OPTIMIZERS):
            raise ValueError(f"optimizers must map names from {_OPTIMIZERS}")
        if any(not isinstance(optimizer, torch.optim.Optimizer)
               for optimizer in optimizers.values()):
            raise TypeError("optimizers must contain existing torch optimizers")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a finite JSON mapping")
        self.metadata = json.loads(json.dumps(
            dict(metadata) if metadata is not None else {}, allow_nan=False))
        self.output_directory = Path(output_directory)
        self._optimizers = dict(optimizers)
        self._end_exclusive = self.first_iteration + self.updates
        self._rows = []
        self._row = None
        self._pending = {}
        self._last_started = None
        self._wall_start = None
        self._source = None
        self._memory_start = None
        self._report = None
        self._finished = False
        self._device_info = {"requested": str(self.device), "resolved": None}
        atexit.register(self.finish)

    @property
    def active(self):
        return self._row is not None

    @property
    def finished(self):
        return self._finished

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _stamp(self):
        host = time.perf_counter()
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
        return event, host

    def _snapshot(self):
        return {name: _optimizer_snapshot(self._optimizers.get(name))
                for name in _OPTIMIZERS}

    def _memory(self):
        if self.device.type == "cpu":
            return None
        return {
            "allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }

    def begin(self, zero_based_iteration):
        index = _index(zero_based_iteration, "zero_based_iteration")
        if self._finished or self._report is not None:
            return
        if self.active:
            raise RuntimeError("end the active iteration before begin")
        if index >= self._end_exclusive:
            self.finish()
            return
        if index < self.first_iteration:
            if self._last_started is not None:
                raise ValueError("iteration indices must increase")
            return
        if self._last_started is not None and index <= self._last_started:
            raise ValueError("iteration indices must increase")
        if self._wall_start is None:
            if self.device.type == "cuda" and self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            self._synchronize()
            self._source = self._snapshot()
            self._memory_start = self._memory()
            self._device_info["resolved"] = str(self.device)
            if self.device.type == "cuda":
                properties = torch.cuda.get_device_properties(self.device)
                self._device_info.update(
                    name=properties.name, total_memory_bytes=properties.total_memory,
                    capability=[properties.major, properties.minor])
        marker = self._stamp()
        if self._wall_start is None:
            self._wall_start = marker[1]
        self._row = {"iteration": index, "whole_start": marker, "phases": []}
        self._last_started = index

    def start_phase(self, name):
        if not self.active:
            return
        if not isinstance(name, str) or not name.strip() or name == "whole":
            raise ValueError("phase name must be nonempty; 'whole' is reserved")
        if name in self._pending:
            raise ValueError(f"phase already active: {name}")
        self._pending[name] = self._stamp()

    def end_phase(self, name):
        if not self.active:
            return
        if name not in self._pending:
            raise ValueError(f"phase is not active: {name}")
        marker = self._stamp()
        self._row["phases"].append((name, self._pending.pop(name), marker, True))

    def end(self, zero_based_iteration):
        index = _index(zero_based_iteration, "zero_based_iteration")
        if not self.active:
            return
        if index != self._row["iteration"]:
            raise ValueError("end iteration must match begin")
        if self._pending:
            raise RuntimeError("close named phases before ending an iteration")
        self._row["whole_end"] = self._stamp()
        self._rows.append(self._row)
        self._row = None
        if index + 1 == self._end_exclusive:
            self.finish()

    def _build_report(self):
        partial = self._row
        if partial is not None:
            marker = self._stamp()
            partial["whole_end"] = marker
            partial["phases"].extend(
                (name, start, marker, False) for name, start in self._pending.items())
            self._pending.clear()
            self._row = None
        wall, end, memory_end = None, None, None
        if self._wall_start is not None:
            self._synchronize()
            wall = time.perf_counter() - self._wall_start
            end = self._snapshot()
            memory_end = self._memory()
        issues = []

        def finite(value, label):
            if value is None:
                return None
            if not math.isfinite(value) or value < 0:
                issues.append(label)
                return None
            return value

        def resolve(row, completed):
            phases = [("whole", row["whole_start"], row["whole_end"], completed)]
            phases.extend(row["phases"])
            result = {"iteration": row["iteration"], "completed": completed, "phases": []}
            for name, start, stop, phase_completed in phases:
                label = f'{row["iteration"]}:{name}'
                cuda_ms = None
                if start[0] is not None:
                    try:
                        cuda_ms = start[0].elapsed_time(stop[0])
                    except RuntimeError:
                        issues.append(label + ":cuda_event_unavailable")
                result["phases"].append({
                    "name": name, "completed": phase_completed,
                    "host_ms": finite((stop[1] - start[1]) * 1000, label + ":host_ms"),
                    "cuda_ms": finite(cuda_ms, label + ":cuda_ms"),
                })
            return result

        rows = [resolve(row, True) for row in self._rows]
        partial_row = resolve(partial, False) if partial is not None else None
        by_phase = {}
        for row in rows:
            for phase in row["phases"]:
                by_phase.setdefault(phase["name"], []).append(phase)
        summaries = {}
        for name, calls in by_phase.items():
            summary = {"calls": len(calls)}
            for metric in ("host_ms", "cuda_ms"):
                values = [call[metric] for call in calls if call[metric] is not None]
                summary[metric] = {
                    "count": len(values),
                    "mean": finite(statistics.mean(values), f"{name}:{metric}:mean")
                    if values else None,
                    "median": finite(statistics.median(values), f"{name}:{metric}:median")
                    if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
            summaries[name] = summary
        counters = {}
        for name in _OPTIMIZERS:
            source, source_parameters = self._source[name] if self._source else (None, ())
            last, end_parameters = end[name] if end else (None, ())
            delta, status = _optimizer_delta(source, last, source_parameters, end_parameters)
            counters[name] = {
                "source": source, "end": last, "uniform_delta": delta, "delta_status": status,
            }
        complete = [r["iteration"] for r in rows] == list(
            range(self.first_iteration, self._end_exclusive))
        state = ("not-started" if self._wall_start is None else
                 "full" if complete and partial is None else "partial")
        wall = finite(wall, "wall_seconds")
        rate = (finite(len(rows) / wall, "completed_updates_per_second")
                if rows and partial is None and wall is not None and wall > 0 else None)
        return {
            "schema": "command-phase-profile-v1",
            "iteration_indexing": "zero-based",
            "requested_first_iteration": self.first_iteration,
            "requested_last_iteration": self._end_exclusive - 1,
            "requested_updates": self.updates,
            "first_iteration": rows[0]["iteration"] if rows else (
                partial["iteration"] if partial is not None else None),
            "last_iteration": rows[-1]["iteration"] if rows else None,
            "last_started_iteration": self._last_started,
            "state": state, "window_complete": state == "full",
            "completed_updates": len(rows), "rows": rows, "partial_iteration": partial_row,
            "wall_seconds": wall, "completed_updates_per_second": rate,
            "phase_statistics": summaries, "timing_valid": not issues, "timing_issues": issues,
            "optimizer_counters": counters,
            "optimizer_counter_scope": {
                "completed_rows": len(rows), "includes_partial_iteration": partial is not None,
                "partial_iteration": partial["iteration"] if partial is not None else None,
                "per_completed_row_attribution_available": False,
            },
            "device": self._device_info,
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": str(torch.__version__), "cuda_build": torch.version.cuda},
            "metadata": self.metadata,
            "memory_peaks": {
                "scope": "Allocator lifetime since external reset; probe never resets peaks.",
                "source": self._memory_start, "end": memory_end,
            },
            "notes": [
                "Caller owns opt-in and exclusion from the default headline speed window.",
                "Initialization/resume/warm-start provenance is caller metadata, never inferred.",
                "All iteration indices are zero-based; last_iteration is the last completed row.",
                "Whole wall starts after the initial drain/counter snapshot and ends after the final drain; end must follow normal logs/checkpoint I/O.",
                "Phase CUDA spans use the current stream and include stream gaps; unrelated streams are not measured.",
                "Named nested phases may overlap: sums must not be interpreted as total kernel time.",
                "Statistics use completed rows only; repeated names are separate calls.",
                "Optimizer values are actual registered-parameter step states at two boundaries; missing steps are unknown, not zero.",
                "Counter deltas include incomplete active work when present and cannot be attributed to completed rows alone.",
                "Memory peaks have process/allocator scope, not an isolated window or GPU-exclusivity claim.",
            ],
        }

    def finish(self):
        """Flush once, including partial work, on owner stop or normal exit."""
        if self._finished:
            return
        if self._report is None:
            self._report = self._build_report()
        payload = json.dumps(self._report, indent=2, allow_nan=False) + "\n"
        self.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.output_directory / "command_phase_profile.json"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.output_directory,
                    prefix=".command_phase_profile-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._finished = True
        atexit.unregister(self.finish)
        self._rows.clear()
        self._source = None

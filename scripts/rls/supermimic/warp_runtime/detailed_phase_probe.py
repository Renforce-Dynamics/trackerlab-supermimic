"""Bounded observational timing, outside the complete-update speed window."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch


class DetailedPhaseProbe:
    """Observe labels20..30; no compiled function, norm or RNG changes."""

    def __init__(self, output, first=20, last=30):
        self.output = Path(output)
        self.first, self.last = first, last
        self.active = False
        self.finished = False
        self.pending = {}
        self.rows = []
        self.graph_record_calls = 0
        self.original_record = None
        self.record_hook = None
        self.record_class = None
        self.graph_counter_supported = False

    def _install_counter(self):
        # Count actual Python graph-record entries without changing arguments,
        # results or the graph manager's state. Remove the hook after label30.
        import torch._inductor.cudagraph_trees as trees
        self.record_class = trees.CUDAGraphTreeManager
        self.original_record = self.record_class.record_function
        probe = self

        def observe(manager, *args, **kwargs):
            probe.graph_record_calls += 1
            return probe.original_record(manager, *args, **kwargs)

        self.record_hook = observe
        self.record_class.record_function = observe
        self.graph_counter_supported = True

    def begin(self, zero_based_iteration, normalizers):
        label = zero_based_iteration + 1
        self.active = not self.finished and self.first <= label <= self.last
        if not self.active:
            return
        if self.original_record is None:
            self._install_counter()
        self.row = {
            "iteration": label,
            "normalizer_storage_before": {
                name: {field: getattr(norm, field).data_ptr()
                       for field in ("mean", "var", "count")}
                for name, norm in normalizers.items()
            },
            "calls": [],
        }

    def start(self, phase):
        if self.active:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.pending[phase] = (event, time.perf_counter(), self.graph_record_calls)

    def end(self, phase):
        if self.active:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            start, host, graph_calls = self.pending.pop(phase)
            self.row["calls"].append({
                "phase": phase, "events": (start, event),
                "host_ms": 1000 * (time.perf_counter() - host),
                "actual_graph_record_calls": self.graph_record_calls - graph_calls,
            })

    def end_iteration(self, zero_based_iteration, normalizers):
        if not self.active:
            return
        self.row["normalizer_storage_after"] = {
            name: {field: getattr(norm, field).data_ptr()
                   for field in ("mean", "var", "count")}
            for name, norm in normalizers.items()
        }
        self.rows.append(self.row)
        if zero_based_iteration + 1 == self.last:
            self.finish()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.active = False
        if self.original_record is not None:
            if self.record_class.record_function is not self.record_hook:
                raise RuntimeError("Graph-record observer ownership changed.")
            self.record_class.record_function = self.original_record
        if not self.rows:
            return
        torch.cuda.synchronize()
        for row in self.rows:
            for call in row["calls"]:
                start, end = call.pop("events")
                call["cuda_span_ms"] = start.elapsed_time(end)
                if not all(math.isfinite(call[key]) for key in ("host_ms", "cuda_span_ms")):
                    raise RuntimeError("Nonfinite bounded phase timing.")
        report = {
            "schema": "scratch-bounded-actual-phase-observation-v1",
            "first_iteration": self.rows[0]["iteration"],
            "last_iteration": self.rows[-1]["iteration"],
            "requested_window": [self.first, self.last],
            "window_complete": len(self.rows) == self.last - self.first + 1,
            "graph_counter_supported": self.graph_counter_supported,
            "actual_graph_record_calls": self.graph_record_calls,
            "graph_record_hook_removed": self.original_record is not None,
            "rows": self.rows,
            "qualification": [
                "Per-call host and CUDA stream-span times are observations.",
                "Collection and ten WM updates retain their original ordering.",
                "No synchronization inside individual measured calls.",
                "The one final synchronization is outside the541..1540 speed window.",
                "No RNG equivalence, convergence or heldout claim is inferred.",
            ],
        }
        path = self.output / "bounded_phase_observation.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

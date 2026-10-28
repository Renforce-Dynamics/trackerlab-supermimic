"""Bounded live timing of complete scratch-training updates.

The initial WM-only phase is explicitly excluded. Timing is observational;
optimizer, sampling and checkpoint logic remain owned by the training runner.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import time

import torch


class ScratchSpeedProbe:
    def __init__(self, output, *, wm_only_iters, actor_warmup=40, measured_updates=1000):
        self.output = Path(output)
        self.start = int(wm_only_iters) + int(actor_warmup)
        self.end_exclusive = self.start + int(measured_updates)
        self.wm_only = int(wm_only_iters)
        self.actor_warmup = int(actor_warmup)
        self.target_count = int(measured_updates)
        self.rows = []
        self.pending = {}
        self.row = None
        self.active = False
        self.finished = False
        self.wall_start = None
        self.counters_start = None
        self.counters_end = None

    def begin(self, iteration, counters=None):
        self.active = not self.finished and self.start <= iteration < self.end_exclusive
        if not self.active:
            return
        if self.wall_start is None:
            torch.cuda.synchronize()
            self.wall_start = time.perf_counter()
            self.counters_start = dict(counters) if counters is not None else None
        self.row = {"iteration": iteration + 1}
        self.start_phase("whole")

    def start_phase(self, name):
        if self.active:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.pending[name] = (event, time.perf_counter())

    def end_phase(self, name):
        if self.active:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            event0, host0 = self.pending.pop(name)
            self.row[name] = (event0, event, time.perf_counter() - host0)

    def end(self, iteration, counters=None):
        if self.active:
            self.end_phase("whole")
            self.rows.append(self.row)
            self.counters_end = dict(counters) if counters is not None else None
            if iteration + 1 == self.end_exclusive:
                self.finish()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.active = False
        if self.wall_start is None:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.wall_start
        rows = []
        for row in self.rows:
            result = {"iteration": row["iteration"]}
            for phase, values in row.items():
                if phase == "iteration":
                    continue
                first, last, host_seconds = values
                result[phase + "_cuda_ms"] = first.elapsed_time(last)
                result[phase + "_host_ms"] = host_seconds * 1000
            rows.append(result)
        assert all(math.isfinite(v) for row in rows for v in row.values())
        statistics_by_phase = {}
        for key in rows[0]:
            if key != "iteration":
                values = [r[key] for r in rows]
                statistics_by_phase[key] = {
                    "mean": statistics.mean(values), "median": statistics.median(values),
                    "min": min(values), "max": max(values),
                }
        report = {
            "schema": "scratch-complete-update-timing-v1",
            "initialization": "random",
            "wm_only_iterations_excluded": self.wm_only,
            "actor_warmup_iterations_excluded": self.actor_warmup,
            "requested_measured_updates": self.target_count,
            "measured_updates": len(rows),
            "window_complete": len(rows) == self.target_count,
            "first_iteration": rows[0]["iteration"],
            "last_iteration": rows[-1]["iteration"],
            "wall_seconds": elapsed,
            "updates_per_second": len(rows) / elapsed,
            "milliseconds_per_update": elapsed * 1000 / len(rows),
            "phase_statistics": statistics_by_phase,
            "rows": rows,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "optimizer_counters_start": self.counters_start,
            "optimizer_counters_end": self.counters_end,
            "actual_optimizer_counter_deltas": (
                {k: self.counters_end[k] - self.counters_start[k] for k in self.counters_start}
                if self.counters_start is not None and self.counters_end is not None else None
            ),
            "notes": [
                "Whole-iteration timing includes collection and configured WM, actor and critic updates; actual counter deltas are recorded.",
                "The WM-only phase and actor compilation warmup are excluded.",
                "CUDA event spans include stream gaps.",
                "GPU exclusivity is caller-managed and is not verified by this portable timing probe.",
                "This is a throughput observation, not a heldout quality result.",
            ],
        }
        path = self.output / "scratch_speed_summary.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
        self.rows.clear()
        print("SCRATCH_SPEED_SUMMARY " + json.dumps({
            k: report[k] for k in ("measured_updates", "updates_per_second",
                                  "milliseconds_per_update", "window_complete")}), flush=True)

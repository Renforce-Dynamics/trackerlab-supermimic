"""Training and per-motion evaluation summaries for SuperMimic."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Any


class MetricWindow:
    """Accumulate finite scalar observations and return arithmetic means."""

    def __init__(self):
        self._sum: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)

    def update(self, **values: Any) -> None:
        for name, value in values.items():
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach().float().mean().item()
            value = float(value)
            if math.isfinite(value):
                self._sum[name] += value
                self._count[name] += 1

    def means(self) -> dict[str, float]:
        return {name: self._sum[name] / self._count[name] for name in self._sum if self._count[name]}

    def clear(self) -> None:
        self._sum.clear()
        self._count.clear()


def summarize_motion_returns(
    names: list[str],
    returns: list[float],
    frame_counts: list[int],
    failures: list[bool],
    max_normalized_return: float = 1.0,
) -> dict[str, Any]:
    """Summarize per-clip returns with clip-weighted and frame-weighted aggregates."""
    n = len(names)
    if not (len(returns) == len(frame_counts) == len(failures) == n):
        raise ValueError("names, returns, frame_counts, and failures must have equal lengths")
    if n == 0:
        raise ValueError("at least one motion is required")
    if any(count < 0 for count in frame_counts):
        raise ValueError("frame counts must be non-negative")

    normalized = [float(r) / max(max_normalized_return * int(t), 1e-12) for r, t in zip(returns, frame_counts)]
    total_frames = sum(frame_counts)
    frame_weighted = (
        sum(value * count for value, count in zip(normalized, frame_counts)) / total_frames if total_frames else 0.0
    )
    failed_names = [name for name, failed in zip(names, failures) if failed]
    return {
        "clip_weighted_return": sum(normalized) / n,
        "frame_weighted_return": frame_weighted,
        "failed_count": len(failed_names),
        "motion_count": n,
        "coverage": 1.0 - len(failed_names) / n,
        "failed_motions": failed_names,
        "per_motion": {
            name: {
                "normalized_return": normalized[i],
                "reward_sum": float(returns[i]),
                "frames": int(frame_counts[i]),
                "failed": bool(failures[i]),
            }
            for i, name in enumerate(names)
        },
    }


class CsvMetricWriter:
    """Append metric rows while preserving a stable CSV header."""

    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path):
            with open(path, newline="", encoding="utf-8") as stream:
                existing = next(csv.reader(stream), [])
            if existing != self.fieldnames:
                raise ValueError(f"CSV header mismatch at {path}: {existing}")
        else:
            with open(path, "w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=self.fieldnames).writeheader()

    def write(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames, extrasaction="raise")
            writer.writerow({name: row.get(name, "") for name in self.fieldnames})


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temp_path, path)

"""Integrity admission for shared repaired sources; never install replacements."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

NETWORKS_SHA256 = "c5b335145fc906f3189cf76003592571ba8b8d7fd29b5c19b30f683a53fad203"
MOTION_SHA256 = "80b5aa7a7f35c1802eb7caf43a1305f70f28f2a6c0f1248865dda84b6ede953f"
LEGACY_REFERENCE_SHA256 = "40ca6131b136723cca1d08d888c3cab2a7af22f10da4a3c1687235ad5c2e2720"


def source_sha256(path):
    """Canonical LF digest permits ordinary Windows/Linux Git line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _verify_source(obj, expected):
    path = Path(inspect.getsourcefile(inspect.unwrap(obj))).resolve()
    digest = source_sha256(path)
    if digest != expected:
        raise RuntimeError(f"Qualified canonical source differs: {path} ({digest})")
    return {"source": str(path), "sha256_lf": digest, "hash_rule": "SHA256 with CRLF normalized to LF"}


def canonical_contracts(networks):
    from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
    from legacy_normalizer_reference import LegacyRunningNorm

    network = _verify_source(networks, NETWORKS_SHA256)
    update = _verify_source(networks.RunningNorm.update, NETWORKS_SHA256)
    motion = _verify_source(MotionLib.goal, MOTION_SHA256)
    legacy = _verify_source(LegacyRunningNorm.update, LEGACY_REFERENCE_SHA256)
    if network["source"] != update["source"]:
        raise RuntimeError("RunningNorm.update must use the canonical networks module")
    return {
        "normalizer": {
            "schema": "canonical-running-norm-storage-v1",
            "implementation": "Shared networks.RunningNorm.update",
            "network": network, "update": update, "legacy_fixture": legacy,
            "method_replaced_at_runtime": False,
            "arithmetic_formula_changed": False,
            "active_update_schedule_retained": True,
            "fresh_cuda_fixture_required": True,
        },
        "goal": {
            "schema": "canonical-reference-goal-tail-v1",
            "implementation": "Shared MotionLib.goal", "source": motion,
            "synthetic_tail_velocity": "zero",
            "real_final_frame": "preserved", "pose": "preserved",
            "method_replaced_at_runtime": False,
            "fresh_reference_fixture_required": True,
        },
    }

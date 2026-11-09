"""Explicit AMP-only feature changes while retaining the learned control system."""

import copy
import hashlib
import json
from pathlib import Path


def restore_resume_style(style, checkpoint, motion, *, allow_feature_change=False, run_dir=None):
    saved = checkpoint["amp_style"]
    source = saved.get("feature_mode")
    target = style.feature_mode
    if source == target:
        style.restore(saved)
        return None
    if not allow_feature_change:
        raise ValueError("Resume style feature change requires --allow-style-feature-change")
    if (checkpoint.get("task") != "locomotion/g1/stealth"
            or source != "postural_end_effectors" or target != "postural_end_effectors_vx"):
        raise ValueError("Style adaptation only supports G1 stealth posture AMP to signed-vx AMP")
    if run_dir is None or (Path(run_dir) / "last.pt").exists():
        raise ValueError("A style adaptation requires a fresh run directory")
    expected_dimension = saved.get("feature_dim", 0) + style.history_steps
    if style.feature_dim != expected_dimension:
        raise ValueError("Style adaptation must add exactly one scalar per history frame")
    layout = copy.deepcopy(style.feature_layout)
    if layout["frame"].pop() != ["root_local_forward_velocity/robot_height", 1]:
        raise ValueError("Signed-vx feature must preserve the original layout prefix")
    layout["feature_mode"] = source
    if layout != saved.get("feature_layout"):
        raise ValueError("Style adaptation changed more than the signed-vx feature")
    # Reproduce the original expert identity using the unchanged feature prefix.
    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps(layout, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    fingerprint.update(json.dumps(
        [motion.clip_start.tolist(), motion.clip_end.tolist()],
        separators=(",", ":")).encode("ascii"))
    for chunk in style._expert_frames[:, :-1].split(8192):
        fingerprint.update(chunk.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes())
    if fingerprint.hexdigest() != saved.get("expert_hash"):
        raise ValueError("Style adaptation requires identical expert clips and prefix features")
    if style.checkpoint()["updates"] != 0:
        raise ValueError("Style adaptation requires a fresh discriminator and optimizer")
    return {
        "source_feature_mode": source, "target_feature_mode": target,
        "source_expert_hash": saved["expert_hash"], "target_expert_hash": style.expert_hash,
        "expert_prefix_identical": True,
        "actor_world_critic_and_optimizers_preserved": True,
        "discriminator_normalizer_optimizer": "fresh; real-data bootstrap before actor updates",
    }


def bootstrap_adapted_style(style, updates, batch_size, collect, stop_requested):
    """Fit only the new prior against expert and fresh real policy histories."""
    if type(updates) is not int or updates < 1:
        raise ValueError("Style bootstrap updates must be positive")
    before = style.checkpoint()["updates"]
    last = {}
    for _ in range(updates):
        if stop_requested():
            raise InterruptedError("Stop requested during AMP-only bootstrap")
        collect(1)
        last = style.update(batch_size=batch_size, updates=1, tensor_metrics=True)
        if last["updates"] != 1:
            raise RuntimeError("Style bootstrap requires complete real-policy history windows")
    return {
        "optimizer_updates": style.checkpoint()["updates"] - before,
        "collected_steps": updates,
        "actor_optimizer_updates": 0,
        "expert_score": float(last["expert_score"]),
        "policy_score": float(last["policy_score"]),
    }

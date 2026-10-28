"""Observational startup checks for the new random-initialized experiment."""
import hashlib
import json
from pathlib import Path

import torch


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def norms(pi, wm, critic):
    return {"pi.obs_norm": pi.obs_norm, "pi.goal_norm": pi.goal_norm,
            "wm.in_norm": wm.in_norm, "wm.out_norm": wm.out_norm,
            "critic.norm": critic.norm}


def record_initialization(output, args, pi, wm, critic, opt_pi, opt_wm, opt_v):
    if args.resume is not None:
        raise RuntimeError("Scratch experiment forbids a source checkpoint")
    modules = {"pi": pi, "wm": wm, "critic": critic}
    digests = {}
    for name, module in modules.items():
        digest = hashlib.sha256()
        for key, tensor in module.state_dict().items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Nonfinite scratch initialization: " + name + "." + key)
            raw = tensor.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(raw.dtype).encode())
            digest.update(str(list(raw.shape)).encode())
            digest.update(raw.numpy().tobytes())
        digests[name] = digest.hexdigest()
    counts = {name: float(norm.count) for name, norm in norms(pi, wm, critic).items()}
    optimizers_empty = all(not optimizer.state for optimizer in (opt_pi, opt_wm, opt_v))
    passed = all(value == 0 for value in counts.values()) and optimizers_empty
    receipt = {
        "schema": "scratch-random-initialization-v1", "passed": passed,
        "seed": args.seed, "resume": None, "iteration": 0,
        "model_state_sha256": digests, "normalizer_counts": counts,
        "optimizer_states_empty": optimizers_empty,
        "bf16_enabled": args.amp == "bf16", "amp_argument": args.amp,
    }
    write(Path(output) / "scratch_initialization.json", receipt)
    if not passed:
        raise RuntimeError("Scratch initialization state failed")


def check_goal_tail(output, motion, goal_frames):
    from trackerLab.kinematics.supermimic.rotations import mat_to_6d, yaw_mat
    indices = torch.linspace(0, len(motion.names) - 1, 128,
                             device=motion.clip_end.device).long().unique()
    end = motion.clip_end[indices]
    frame = end - 1
    fixed = motion.goal(frame, goal_frames).reshape(len(indices), goal_frames, -1)
    future = motion.future_frames(frame, goal_frames)
    heading_t = yaw_mat(motion.root_R[frame]).transpose(-1, -2).unsqueeze(1)
    position = (heading_t @ (
        motion.root_p[future] - motion.root_p[frame].unsqueeze(1)
    ).unsqueeze(-1)).squeeze(-1)
    prior = torch.cat((
        position, mat_to_6d(heading_t @ motion.root_R[future]),
        motion.root_v_local[future], motion.root_w_local[future],
        motion.q[future], motion.qd[future] * 0.1,
    ), -1)
    joints = motion.num_joints
    velocity_columns = list(range(9, 15)) + list(range(15 + joints, 15 + 2 * joints))
    passed = (bool(torch.isfinite(fixed).all())
              and torch.equal(fixed[:, 0], prior[:, 0])
              and bool((fixed[:, 1:, velocity_columns] == 0).all())
              and torch.equal(fixed[:, :, :9], prior[:, :, :9])
              and torch.equal(fixed[:, :, 15:15 + joints], prior[:, :, 15:15 + joints]))
    write(Path(output) / "goal_tail_acceptance.json", {
        "schema": "scratch-real-reference-goal-tail-v1", "passed": passed,
        "sampled_clips": len(indices), "goal_frames": goal_frames,
        "real_final_frame_preserved": torch.equal(fixed[:, 0], prior[:, 0]),
        "synthetic_velocity_max_abs": float(fixed[:, 1:, velocity_columns].abs().max()),
        "pose_and_joint_targets_preserved": (
            torch.equal(fixed[:, :, :9], prior[:, :, :9])
            and torch.equal(fixed[:, :, 15:15 + joints], prior[:, :, 15:15 + joints])),
        "implementation": "Canonical MotionLib.goal; no process-local method replacement",
        "reference_samples_mutated": False,
    })
    if not passed:
        raise RuntimeError("Real reference terminal-goal checks failed")


def check_actor_rollout(output, iteration, pi, states, actions, rewards, dones,
                        survival, objective, loss, raw_gradient_norm, horizon, batch):
    tensors = {"state_" + str(i): value for i, value in enumerate(states)}
    tensors.update(actions=actions, rewards=rewards, dones=dones, survival=survival,
                   objective=objective, loss=loss, raw_gradient_norm=raw_gradient_norm)
    finite = {name: bool(torch.isfinite(value).all()) for name, value in tensors.items()}
    gradients = {name: p.grad is not None and bool(torch.isfinite(p.grad).all())
                 for name, p in pi.named_parameters()}
    rotation = states[1]
    eye = torch.eye(3, device=rotation.device)
    orthogonality = float((rotation.transpose(-1, -2) @ rotation - eye).abs().max())
    determinant_error = float((torch.linalg.det(rotation) - 1).abs().max())
    passed = (all(finite.values()) and all(gradients.values())
              and max(orthogonality, determinant_error) <= 5e-5)
    receipt = {
        "schema": "scratch-actual-actor32-finite-geometry-v1", "passed": passed,
        "iteration": iteration, "horizon": horizon, "batch": batch,
        "finite_tensors": finite, "finite_policy_gradients": gradients,
        "SO3_max_error_threshold": 5e-5,
        "orthogonality_max_abs": orthogonality, "determinant_max_abs_error": determinant_error,
        "raw_action_max_abs": float(actions.detach().abs().max()),
        "objective_mean": float(objective.detach().mean()),
        "raw_gradient_norm": float(raw_gradient_norm.detach()),
        "optimizer_steps_during_check": 0,
        "scope": "Actual compiled production actor-through-WM path; finite/geometry check, no paper-quality claim.",
    }
    write(Path(output) / "scratch_actor32_acceptance.json", receipt)
    if not passed:
        raise FloatingPointError("Actual scratch actor32 finite/geometry gate failed")


def record_progress(output, iteration, counters, pi, wm, critic):
    counts = {name: float(norm.count) for name, norm in norms(pi, wm, critic).items()}
    write(Path(output) / "scratch_progress.json", {
        "schema": "scratch-tracker-progress-v1", "iteration": iteration,
        "optimizer_counters": dict(counters), "normalizer_counts": counts,
        "normalizer_guard_disabled": True,
    })


def record_optimizer_health(output, iteration, modules, optimizers):
    """Observe actual post-update parameters and Adam states at startup."""
    checks = {}
    counts = {}
    for name, module in modules.items():
        tensors = list(module.state_dict().values())
        checks[name + "_model_finite"] = all(
            bool(torch.isfinite(tensor).all()) for tensor in tensors
        )
        states = [value for state in optimizers[name].state.values()
                  for value in state.values() if isinstance(value, torch.Tensor)]
        checks[name + "_adam_finite"] = bool(states) and all(
            bool(torch.isfinite(tensor).all()) for tensor in states
        )
        counts[name] = {"model_tensors": len(tensors), "adam_tensors": len(states)}
    receipt = {
        "schema": "scratch-post-update-model-adam-health-v1",
        "iteration": iteration, "passed": all(checks.values()),
        "checks": checks, "tensor_counts": counts, "state_mutation": False,
    }
    write(Path(output) / "scratch_optimizer_health.json", receipt)
    if not receipt["passed"]:
        raise FloatingPointError("Scratch post-update model/Adam health failed")

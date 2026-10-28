"""Bounded actual-corpus acceptance reusing the learner's resident references."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from reference_gpu_fk import GPUComputedBodyField
from resident_reference_bodies import BODY_NAMES, ROOT_NAMES


def validate_resident_reference(
    library, kinematics, mdp, output, *, compile_reward=True, sample_count=4096,
    gradient_rows=256,
):
    """Validate without rebuilding references or consuming training RNG.

    Only temporary selected tensors and a small compiled reward graph are
    created. The original provider's empty cache/counters and CPU/CUDA RNG are
    restored in ``finally``, including rejection. The caller owns launch locks
    and resource admission. A fresh receipt is required; failed acceptance
    raises before simulator/network construction or training.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("Resident acceptance requires a fresh receipt")
    if sample_count < 64 or not 1 <= gradient_rows <= sample_count:
        raise ValueError("Invalid bounded acceptance sizes")
    if mdp.motion is not library or mdp.kin is not kinematics:
        raise ValueError("Acceptance must use the learner's actual MDP and kinematics")
    if not hasattr(library, "_resident_body_reference"):
        raise ValueError("Resident body installation must precede acceptance")
    device = torch.device(library.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    provider = library._gpu_reference_bodies
    if provider.kinematics is not kinematics:
        raise ValueError("Expected the accepted source FK provider")
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    saved_provider = {
        name: getattr(provider, name)
        for name in ("cache", "cache_bytes", "forward_calls", "queried_frames", "cache_hits")
    }
    # Isolate temporary cache entries so restore retains the original object.
    provider.cache = copy.copy(provider.cache)
    started = time.perf_counter()
    receipt = {
        "schema": "resident-reference-startup-acceptance-v1",
        "status": "running", "device": str(device),
        "corpus_frames": library.num_frames,
        "corpus_clips": len(getattr(library, "names", [])),
        "sample_count": sample_count, "gradient_rows": gradient_rows,
        "body_module_sha256": hashlib.sha256(
            Path(__file__).with_name("resident_reference_bodies.py").read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": [], "field_parity": [],
        "training_launched_by_validator": False,
        "full_reference_builds_by_validator": 0,
    }
    error = None
    try:
        baseline = copy.copy(library)
        for i, name in enumerate(BODY_NAMES):
            setattr(baseline, name, GPUComputedBodyField(
                provider, i, tuple(getattr(library, name).shape[1:])))
        old_mdp = copy.copy(mdp)
        old_mdp.motion = baseline
        _check_values(library, baseline, mdp, old_mdp, device, receipt,
                      sample_count, gradient_rows, compile_reward)
        receipt["status"] = (
            "passed" if all(row["passed"] for row in receipt["checks"]) else "rejected")
    except Exception as exc:
        receipt.update(status="error", error=f"{type(exc).__name__}: {exc}")
        error = exc
    finally:
        for name, value in saved_provider.items():
            setattr(provider, name, value)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        receipt.update(
            elapsed_seconds=time.perf_counter()-started,
            cpu_rng_restored=torch.equal(cpu_rng, torch.get_rng_state()),
            cuda_rng_restored=(cuda_rng is None or torch.equal(
                cuda_rng, torch.cuda.get_rng_state(device))),
            original_provider_cache_and_counters_restored=all(
                getattr(provider, name) is value if name == "cache"
                else getattr(provider, name) == value
                for name, value in saved_provider.items()),
        )
        if not (receipt["cpu_rng_restored"] and receipt["cuda_rng_restored"]
                and receipt["original_provider_cache_and_counters_restored"]):
            receipt["status"] = "rejected"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    if error is not None:
        raise RuntimeError(f"Resident reference startup acceptance errored: {output}") from error
    if receipt["status"] != "passed":
        failed = [row["name"] for row in receipt["checks"] if not row["passed"]]
        raise RuntimeError(f"Resident reference startup acceptance rejected {failed}; {output}")
    return receipt


def _check_values(motion, baseline, mdp, old_mdp, device, receipt,
                  sample_count, gradient_rows, compile_reward):
    from trackerLab.kinematics.supermimic.rotations import so3_exp

    checks = receipt["checks"]

    def compare(label, actual, expected, atol=2e-6, rtol=2e-6):
        delta = (actual - expected).abs()
        passed = (actual.shape == expected.shape and actual.dtype == torch.float32
                  and bool(torch.isfinite(actual).all())
                  and torch.allclose(actual, expected, atol=atol, rtol=rtol))
        row = {"name": label, "passed": bool(passed),
               "max_abs": delta.max().item() if delta.numel() else 0.,
               "rmse": delta.square().mean().sqrt().item() if delta.numel() else 0.,
               "atol": atol, "rtol": rtol, "shape": list(actual.shape)}
        checks.append({"name": label, "passed": bool(passed)})
        return row

    n = motion.num_frames
    sample = (torch.arange(sample_count, device=device) * 104729 + 17) % n
    sample[:2] = torch.tensor([0, n - 1], device=device)
    if hasattr(motion, "clip_start"):
        step = max(1, len(motion.clip_start)//max(1, sample_count//4))
        boundaries = torch.cat((motion.clip_start[::step], motion.clip_end[::step]))
        count = min(len(boundaries), len(sample) - 2)
        sample[2:2+count] = boundaries[:count]
    cases = {
        "corpus_sample": sample, "matrix": sample[:64].view(8, 8),
        "scalar_last": torch.tensor(n-1, device=device),
        "empty": sample[:0], "negative": torch.tensor([-1, -2, 0], device=device),
        "slice": slice(0, min(n, 96), 3),
    }
    for name, index in cases.items():
        rows = {}
        for field in BODY_NAMES:
            rows[field] = compare(
                f"{name}/{field}", getattr(motion, field)[index],
                getattr(baseline, field)[index],
                atol=2e-5 if field in ("body_v", "body_w") else 2e-6,
            )
        receipt["field_parity"].append({"case": name, "fields": rows})
    rotations = motion.body_R[sample].double()
    ortho = ((rotations.transpose(-1, -2) @ rotations)
             - torch.eye(3, device=device, dtype=torch.float64)).abs().max().item()
    determinant = (torch.linalg.det(rotations) - 1).abs().max().item()
    receipt["SO3"] = {"max_orthogonality_error": ortho, "max_det_error": determinant,
                      "threshold": 5e-5}
    checks.append({"name": "SO3", "passed": ortho < 5e-5 and determinant < 5e-5})
    f = sample[:gradient_rows]
    reference_states = [getattr(motion, name)[f] for name in ROOT_NAMES]
    states = [value.detach().clone() + .017 for value in reference_states]
    states[1] = reference_states[1] @ so3_exp(torch.full((len(f), 3), .021, device=device))
    state_old = tuple(value.detach().clone().requires_grad_() for value in states)
    state_new = tuple(value.detach().clone().requires_grad_() for value in states)
    a = torch.full((len(f), mdp.num_joints), .021, device=device)
    ap = torch.full_like(a, .008)
    xi = torch.zeros((len(f), mdp.num_joints+4), device=device)
    xi[:, 0] = .75
    reward_old, terms_old = old_mdp.reward(state_old, f, a, ap, terms=True)
    reward_new, terms_new = mdp.reward(state_new, f, a, ap, terms=True)
    receipt["reward"] = compare("reward", reward_new, reward_old, atol=2e-6, rtol=2e-5)
    receipt["reward_terms"] = {
        name: compare(f"reward_term/{name}", value, terms_old[name], atol=2e-5, rtol=2e-5)
        for name, value in terms_new.items()
    }
    grad_old = torch.autograd.grad(reward_old.mean(), state_old)
    grad_new = torch.autograd.grad(reward_new.mean(), state_new)
    receipt["reward_gradients"] = _gradient_checks(grad_old, grad_new, checks, "reward_gradient")
    receipt["observation"] = compare(
        "observation99", mdp.obs(state_new, a, xi, f), old_mdp.obs(state_old, a, xi, f),
        atol=0, rtol=0,
    )
    receipt["critic_features"] = compare(
        "critic_features", mdp.critic_features(state_new, f, xi),
        old_mdp.critic_features(state_old, f, xi), atol=2e-6, rtol=2e-6,
    )
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    noisy_old = old_mdp.add_noise(old_mdp.obs(state_old, a, xi, f))
    end_cpu = torch.get_rng_state()
    end_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state(rng_cuda, device)
    noisy_new = mdp.add_noise(mdp.obs(state_new, a, xi, f))
    receipt["noise"] = compare("observation_noise_same_rng", noisy_new, noisy_old, atol=0, rtol=0)
    checks.append({"name": "noise_rng_advance_equal",
                   "passed": torch.equal(end_cpu, torch.get_rng_state()) and (
                       end_cuda is None or torch.equal(end_cuda, torch.cuda.get_rng_state(device)))})
    if compile_reward:
        def pure_reward(S, frame, action, previous):
            return mdp.reward(S, frame, action, previous), mdp.soft_termination(S, frame)
        compiled = torch.compile(pure_reward, fullgraph=True)
        graph_states = tuple(value.detach().clone().requires_grad_() for value in states)
        compiled_value, compiled_done = compiled(graph_states, f, a, ap)
        receipt["compiled_reward"] = compare(
            "compiled_resident_reward", compiled_value, reward_old, atol=2e-6, rtol=2e-5)
        receipt["compiled_soft_termination"] = compare(
            "compiled_resident_soft_termination", compiled_done,
            old_mdp.soft_termination(state_old, f), atol=2e-6, rtol=2e-5)
        compiled_grad = torch.autograd.grad(compiled_value.mean(), graph_states)
        receipt["compiled_reward_gradients"] = _gradient_checks(
            grad_old, compiled_grad, checks, "compiled_reward_gradient")
        receipt["compiled_reward_fullgraph"] = True
    else:
        receipt["compiled_reward_fullgraph"] = False
    receipt["provider_calls_for_acceptance"] = motion._gpu_reference_bodies.forward_calls


def _gradient_checks(expected_values, actual_values, checks, prefix):
    reports = {}
    for name, expected, actual in zip(ROOT_NAMES, expected_values, actual_values):
        old_flat, new_flat = expected.flatten().double(), actual.flatten().double()
        error = torch.linalg.vector_norm(new_flat-old_flat).item()
        denominator = max(torch.linalg.vector_norm(old_flat).item(), 1e-20)
        relative = error / denominator
        cosine = torch.nn.functional.cosine_similarity(old_flat, new_flat, dim=0).item()
        passed = bool(torch.isfinite(actual).all()) and (relative <= 1e-3)
        if denominator > 1e-12:
            passed = passed and cosine >= .999999
        checks.append({"name": f"{prefix}/{name}", "passed": bool(passed)})
        reports[name] = {"relative_l2": relative, "cosine": cosine, "passed": bool(passed),
                         "relative_l2_threshold": 1e-3, "cosine_threshold": .999999}
    return reports

"""Actual learner batch numerical gates; no optimizer steps or production RNG changes."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import torch


THRESHOLDS = {
    "tensor_atol": 2e-5,
    "tensor_rtol": 2e-3,
    "wm_gradient_relative_l2": 1e-3,
    "wm_gradient_cosine": .999999,
    "actor_gradient_relative_l2": 3e-3,
    "actor_gradient_cosine": .99999,
    "near_zero_gradient_absolute_l2": 1e-8,
    "SO3_max_error": 5e-5,
}


class PreserveLearner:
    def __init__(self, modules):
        self.modules = dict(modules)

    def __enter__(self):
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        self.states = {
            name: {key: value.detach().cpu().clone()
                   for key, value in module.state_dict().items()}
            for name, module in self.modules.items()
        }
        self.flags = {
            name: [p.requires_grad for p in module.parameters()]
            for name, module in self.modules.items()
        }
        self.grads = {
            name: [p.grad for p in module.parameters()]
            for name, module in self.modules.items()
        }
        return self

    def __exit__(self, kind, error, tb):
        changed = []
        for name, module in self.modules.items():
            state = module.state_dict()
            for key, expected in self.states[name].items():
                actual = state[key]
                if not torch.equal(actual.detach().cpu(), expected):
                    changed.append(f"{name}.{key}")
                    with torch.no_grad():
                        actual.copy_(expected.to(device=actual.device))
            for parameter, flag, gradient in zip(
                    module.parameters(), self.flags[name], self.grads[name]):
                parameter.requires_grad_(flag)
                if parameter.grad is not gradient:
                    changed.append(f"{name}.grad_object")
                    parameter.grad = gradient
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        self.receipt = {
            "model_and_buffer_values_unchanged": not changed,
            "changed_then_restored": changed,
            "cpu_rng_restored": torch.equal(self.cpu_rng, torch.get_rng_state()),
            "cuda_rng_restored": all(torch.equal(a, b) for a, b in zip(
                self.cuda_rng, torch.cuda.get_rng_state_all())) if self.cuda_rng else True,
            "parameter_requires_grad_restored": True,
            "original_grad_objects_retained": not any(x.endswith("grad_object") for x in changed),
            "optimizer_steps": 0,
        }
        if changed and error is None:
            raise RuntimeError("Validation mutated learner state: " + ", ".join(changed))
        return False


def _snapshot(tensors):
    return {name: value.detach().cpu().clone() for name, value in tensors.items()}


def _gradients(loss, module):
    named = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    values = torch.autograd.grad(loss, [p for _, p in named], allow_unused=False)
    return {name: value.detach().cpu().clone() for (name, _), value in zip(named, values)}


def _compare(expected, actual, *, kind):
    checks, tensors = [], {}
    for name, old in expected["tensors"].items():
        new = actual["tensors"][name]
        finite = bool(torch.isfinite(new).all())
        passed = (new.shape == old.shape and new.dtype == old.dtype and finite
                  and torch.allclose(new, old, atol=THRESHOLDS["tensor_atol"],
                                     rtol=THRESHOLDS["tensor_rtol"]))
        maximum = float((new.double() - old.double()).abs().max()) if new.numel() else 0.
        tensors[name] = {"passed": bool(passed), "max_abs": maximum,
                         "shape": list(new.shape)}
        checks.append(bool(passed))
    sum_error, sum_old, sum_new, dot = 0., 0., 0., 0.
    per_parameter = {}
    for name, old in expected["gradients"].items():
        new = actual["gradients"][name]
        a, b = old.double().flatten(), new.double().flatten()
        error2 = float((a-b).square().sum())
        old2, new2 = float(a.square().sum()), float(b.square().sum())
        inner = float((a*b).sum())
        rel = (error2 / max(old2, 1e-30)) ** .5
        cosine = inner / max((old2*new2)**.5, 1e-30)
        near_zero = old2**.5 < 1e-10
        passed = bool(torch.isfinite(b).all()) and (
            error2**.5 <= THRESHOLDS["near_zero_gradient_absolute_l2"] if near_zero
            else rel <= THRESHOLDS[f"{kind}_gradient_relative_l2"])
        per_parameter[name] = {
            "passed": passed, "relative_l2": rel, "cosine": cosine,
            "absolute_error_l2": error2**.5, "reference_l2": old2**.5}
        checks.append(passed)
        sum_error += error2
        sum_old += old2
        sum_new += new2
        dot += inner
    relative = (sum_error / max(sum_old, 1e-30))**.5
    cosine = dot / max((sum_old*sum_new)**.5, 1e-30)
    global_pass = (relative <= THRESHOLDS[f"{kind}_gradient_relative_l2"]
                   and cosine >= THRESHOLDS[f"{kind}_gradient_cosine"])
    checks.append(global_pass)
    return {
        "passed": all(checks), "tensors": tensors,
        "gradient_global": {"passed": global_pass, "relative_l2": relative,
                            "cosine": cosine},
        "gradients_per_parameter": per_parameter,
    }


def _execute_gate(path, name, modules, action):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Use a fresh acceptance receipt: {path}")
    report = {"schema": "actual-batch-graph-parity-v1", "name": name,
              "thresholds": THRESHOLDS, "status": "running", "optimizer_steps": 0}
    started = time.perf_counter()
    guard = PreserveLearner(modules)
    caught = None
    try:
        with guard:
            action(report)
        report["status"] = "passed" if all(
            row["passed"] for row in report["comparisons"]) else "rejected"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        caught = exc
    finally:
        report["preservation"] = getattr(guard, "receipt", {})
        report["elapsed_seconds"] = time.perf_counter() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    if report["status"] != "passed":
        raise RuntimeError(f"{name} rejected; inspect {path}") from caught
    return report


def validate_world_model_batch(path, wm, batch, raw_rollout, raw_error,
                               candidate_rollout, candidate_error):
    """Full actual B2048/H8 batch, three candidate graph generations."""
    def action(report):
        report.update(batch=list(batch["s"].shape), window=batch["a"].shape[1],
                      deterministic=True, candidate_uses_actual_training_callables=True,
                      reference_compile_mode="default", candidate_generations=3)
        default_rollout = torch.compile(raw_rollout, mode="default")
        default_error = torch.compile(raw_error, mode="default")

        def evaluate(rollout, error):
            states = rollout(batch["s"], batch["a"], batch["xi"])
            loss, tail = error(states, batch["s1"], batch["mask"])
            tensors = _snapshot({"loss": loss, "tail": tail,
                                 **{f"state_{i}": value for i, value in enumerate(states)}})
            gradients = _gradients(loss, wm)
            del states, loss, tail
            return {"tensors": tensors, "gradients": gradients}

        expected = evaluate(default_rollout, default_error)
        comparisons = []
        for index in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            got = evaluate(candidate_rollout, candidate_error)
            check = _compare(expected, got, kind="wm")
            check["generation"] = index + 1
            comparisons.append(check)
            del got
        report["comparisons"] = comparisons
        del expected, default_rollout, default_error
        gc.collect()

    return _execute_gate(path, "world-model", {"wm": wm}, action)


def validate_actor_start(path, pi, wm, critic, mdp, motion, state, history,
                         previous_action, frame, xi, xi_normalized, goal,
                         *, goal_frames, horizon=4, gamma=.99, bounded_critic=False,
                         max_step_reward=.1, no_critic=False):
    """Actual B4096 start; deterministic H1/H4 recurrence, production H32 unchanged."""
    if horizon not in (1, 4):
        raise ValueError("Bounded actor gate supports H1 or H4")

    def action(report):
        report.update(batch=history.shape[0], horizon=horizon,
                      full_production_H32_validated=False,
                      production_noise_rng_bitwise_equivalence_claimed=False,
                      noise="one explicit fixed additive tensor used by both modes",
                      reference_compile_mode="default",
                      candidate_compile_mode="reduce-overhead",
                      candidate_generations=3)
        noise = (2 * torch.rand((horizon, history.shape[0], mdp.obs_dim),
                               device=history.device, dtype=history.dtype) - 1) * mdp.obs_noise

        def deterministic_step(S, hist, ap, f, context, normalized, g, fixed_noise):
            a = pi(hist, g)
            S1 = wm.step(S, a, normalized)
            f1 = f + 1
            obs = mdp.obs(S1, a, context, f1) + fixed_noise
            hist1 = torch.cat((hist[:, 1:], obs.unsqueeze(1)), 1)
            return S1, a, hist1, f1, motion.goal(f1, goal_frames)

        baseline = torch.compile(deterministic_step, mode="default")
        candidate = torch.compile(deterministic_step, mode="reduce-overhead")

        def evaluate(step):
            S, hist, ap, f, g = state, history, previous_action, frame, goal
            rewards, dones, states, actions, snapshots = [], [], [], [], {}
            for t in range(horizon):
                prev = ap
                S, ap, hist, f, g = step(S, hist, ap, f, xi, xi_normalized, g, noise[t])
                rewards.append(mdp.reward(S, f, ap, prev))
                dones.append(mdp.soft_termination(S, f))
                states.append(S)
                actions.append(ap)
            rewards, dones = torch.stack(rewards), torch.stack(dones)
            alive = torch.cumprod(1-dones, 0)
            ct = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
            discount = gamma ** torch.arange(horizon, device=history.device,
                                             dtype=torch.float32).unsqueeze(-1)
            objective = (discount * ct * rewards).sum(0)
            if not no_critic:
                critic_input = torch.cat((mdp.critic_features(S, f, xi_normalized), g), -1)
                bootstrap = critic(critic_input)
                if bounded_critic:
                    bootstrap = bootstrap.clamp(max=max_step_reward/(1-gamma))
                objective = objective + gamma**horizon * alive[-1] * bootstrap
            loss = -objective.mean()
            snapshots.update(loss=loss, rewards=rewards, dones=dones,
                             actions=torch.stack(actions), history=hist, goal=g)
            snapshots.update({f"state_{i}": torch.stack([value[i] for value in states])
                              for i in range(6)})
            tensors = _snapshot(snapshots)
            gradients = _gradients(loss, pi)
            R = tensors["state_1"].double()
            ortho = float((R.transpose(-1, -2) @ R - torch.eye(3)).abs().max())
            determinant = float((torch.linalg.det(R) - 1).abs().max())
            return {"tensors": tensors, "gradients": gradients,
                    "SO3": {"orthogonality": ortho, "determinant": determinant,
                            "passed": max(ortho, determinant) <= THRESHOLDS["SO3_max_error"]}}

        expected = evaluate(baseline)
        comparisons = []
        for index in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            got = evaluate(candidate)
            check = _compare(expected, got, kind="actor")
            check.update(generation=index+1, SO3=got["SO3"])
            check["passed"] = check["passed"] and got["SO3"]["passed"]
            comparisons.append(check)
            del got
        report["comparisons"] = comparisons
        report["baseline_SO3"] = expected["SO3"]
        del expected, baseline, candidate, noise
        gc.collect()

    return _execute_gate(path, "actor-start", {"pi": pi, "wm": wm, "critic": critic}, action)

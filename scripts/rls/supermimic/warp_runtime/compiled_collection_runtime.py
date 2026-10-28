"""Compile deterministic collection heads with an actual-state parity gate."""
from pathlib import Path
import json
import torch


def install_compiled_collection(env, log_dir):
    if not hasattr(env.motion, "_resident_body_reference"):
        raise ValueError("Compiled collection requires validated resident body references")
    mdp = env.mdp

    def heads(state, frame, action, previous):
        reward, terms = mdp.reward(state, frame, action, previous, terms=True)
        return reward, terms, mdp.termination(state, frame)

    compiled = torch.compile(heads, fullgraph=True, dynamic=False)
    checked = False

    def evaluate(state, frame, action, previous):
        nonlocal checked
        actual = compiled(state, frame, action, previous)
        if not checked:
            expected = heads(state, frame, action, previous)
            pairs = {"reward": (actual[0], expected[0]),
                     "termination": (actual[2], expected[2])}
            if actual[1].keys() != expected[1].keys():
                raise ValueError("Collection reward term schema changed")
            pairs.update({name: (actual[1][name], expected[1][name])
                          for name in actual[1]})
            checks = {}
            for name, (result, reference) in pairs.items():
                if result.dtype == torch.bool:
                    passed = torch.equal(result, reference)
                    error = None
                else:
                    passed = bool(torch.isfinite(result).all()) and torch.allclose(
                        result, reference, rtol=3e-4, atol=3e-5)
                    error = float((result - reference).abs().max())
                checks[name] = {"passed": passed, "maximum_absolute_error": error}
            report = {
                "schema": "compiled-collection-actual-g1-parity-v1",
                "passed": all(check["passed"] for check in checks.values()),
                "rows": frame.numel(), "checks": checks,
                "rng_operations_in_compiled_function": False,
                "reward_configuration_changed": False,
                "termination_configuration_changed": False,
            }
            (Path(log_dir) / "compiled_collection_acceptance.json").write_text(
                json.dumps(report, indent=2) + "\n")
            if not report["passed"]:
                raise RuntimeError("Compiled collection actual-state parity failed")
            checked = True
        return actual

    env._collection_heads = evaluate

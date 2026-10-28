"""WM split/combined boundary diagnostic with unchanged acceptance thresholds."""
from __future__ import annotations

import gc
import os

import torch

from actual_batch_validation import _compare, _execute_gate, _gradients, _snapshot


def validate_combined_world_model_batch(path, wm, batch, raw_rollout, raw_error,
                                       raw_window, candidate_window):
    def action(report):
        report.update(
            batch=list(batch["s"].shape), window=batch["a"].shape[1],
            deterministic=True, reference_compile_mode="split/default",
            candidate_compile_mode="combined/reduce-overhead",
            baseline_repeats=3, candidate_generations=3,
            thresholds_changed=False, optimizer_steps=0)
        baseline_rollout = torch.compile(raw_rollout, mode="default")
        baseline_error = torch.compile(raw_error, mode="default")
        combined_default = torch.compile(raw_window, mode="default")

        def split(s, a, xi, s1, mask):
            states = baseline_rollout(s, a, xi)
            loss, tail = baseline_error(states, s1, mask)
            return loss, tail, states

        def evaluate(fn):
            loss, tail, states = fn(
                batch["s"], batch["a"], batch["xi"], batch["s1"], batch["mask"])
            tensors = _snapshot({"loss": loss, "tail": tail,
                                 **{f"state_{i}": v for i, v in enumerate(states)}})
            gradients = _gradients(loss, wm)
            return {"tensors": tensors, "gradients": gradients}

        reference = evaluate(split)
        baseline_checks = []
        for index in range(2):
            repeat = evaluate(split)
            result = _compare(reference, repeat, kind="wm")
            result["repeat"] = index + 2
            result["all_gradients_bitwise_equal"] = all(
                torch.equal(reference["gradients"][k], repeat["gradients"][k])
                for k in reference["gradients"])
            baseline_checks.append(result)
            del repeat
        report["baseline_repeat_comparisons"] = baseline_checks
        combined_reference = evaluate(combined_default)
        report["combined_default_vs_split_default"] = _compare(
            reference, combined_reference, kind="wm")
        candidate_checks, against_combined, candidate_snapshot = [], [], None
        for index in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            actual = evaluate(candidate_window)
            result = _compare(reference, actual, kind="wm")
            result["generation"] = index + 1
            candidate_checks.append(result)
            local = _compare(combined_reference, actual, kind="wm")
            local["generation"] = index + 1
            against_combined.append(local)
            if index == 2:
                candidate_snapshot = actual
            else:
                del actual
        report["comparisons"] = [*baseline_checks, *candidate_checks]
        report["candidate_vs_combined_default"] = against_combined
        # A-B-A verifies that the reference itself did not move after replay.
        again = evaluate(split)
        last_reference = _compare(reference, again, kind="wm")
        last_reference["all_gradients_bitwise_equal"] = all(
            torch.equal(reference["gradients"][k], again["gradients"][k])
            for k in reference["gradients"])
        report["baseline_after_candidate"] = last_reference
        report["comparisons"].append(last_reference)
        del again
        # Optional higher-precision diagnostic. It never replaces the gate's
        # split/default reference and never expands any pass threshold.
        if os.environ.get("SM_WM_FP32_REFERENCE", "0") == "1":
            import trackerLab.algorithms.supermimic.networks as networks
            original_amp = networks.AMP_DTYPE
            try:
                networks.AMP_DTYPE = None
                fp32 = evaluate(torch.compile(raw_window, mode="default"))
                report["fp32_reference_diagnostic"] = {
                    "baseline_vs_fp32": _compare(fp32, reference, kind="wm"),
                    "candidate_vs_fp32": _compare(fp32, candidate_snapshot, kind="wm"),
                    "used_for_acceptance": False,
                    "training_amp_restored": True,
                }
                del fp32
            finally:
                networks.AMP_DTYPE = original_amp
        del reference, combined_reference, candidate_snapshot
        del baseline_rollout, baseline_error, combined_default
        gc.collect()

    return _execute_gate(path, "world-model-combined-window", {"wm": wm}, action)

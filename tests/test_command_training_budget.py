"""Standard-library checks for minimum-review training decisions."""

import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from command_training_budget import decide


SCRIPT = REPO / "scripts/rls/supermimic/command_training_budget.py"


def gates(forward_passed=True, active_passed=True):
    protocol = {"evaluation_seed": 101, "num_envs": 24, "physics_steps": 1000,
                "observation_noise": True, "entry_duration_s": 4.84}
    forward = {
        "scope": "forward_gait_preservation_only", "passed": forward_passed,
        "reasons": [] if forward_passed else ["candidate.body_vx_m_s: 0.1 < 0.15"],
        "observed": {
            "baseline": {"protocol": copy.deepcopy(protocol), "body_vx_m_s": .21},
            "candidate": {"protocol": copy.deepcopy(protocol),
                          "body_vx_m_s": .21 if forward_passed else .1},
        },
    }
    active = {
        "scope": "active_mode_fall_guard_only", "passed": active_passed,
        "reasons": [] if active_passed else ["forward_right.fall_fraction: 0.16666666666666666 > 0.1"],
        "observed": {"per_mode": [{"name": "forward_right", "episodes": 12,
                                  "falls": 0 if active_passed else 2}]},
        "all_commands_accepted": False, "visual_review_required": True,
    }
    return forward, active


class CommandTrainingBudgetTests(unittest.TestCase):
    def assert_invalid(self, forward, active, iteration=13000, floor=17000, reason=None):
        result = decide(forward, active, iteration, floor)
        self.assertFalse(result["passed"])
        self.assertFalse(result["conclusion_deferred"])
        self.assertFalse(result["valid_evidence"])
        self.assertTrue(result["reasons"])
        if reason:
            self.assertIn(reason, "\n".join(result["reasons"]))
        json.dumps(result, allow_nan=False)
        return result

    def test_review_floor_truth_table_and_exact_boundary(self):
        for forward_passed, active_passed, iteration in itertools.product(
                (True, False), (True, False), (16999, 17000, 17001)):
            with self.subTest(forward=forward_passed, active=active_passed, iteration=iteration):
                forward, active = gates(forward_passed, active_passed)
                result = decide(forward, active, iteration, 17000)
                deferred = iteration < 17000
                self.assertEqual(result["passed"], deferred or (forward_passed and active_passed))
                self.assertEqual(result["conclusion_deferred"], deferred)
                self.assertTrue(result["valid_evidence"])
                self.assertEqual(result["iteration"], iteration)
                self.assertEqual(result["minimum_review_iteration"], 17000)
                self.assertEqual(result["gate_failures"], forward["reasons"] + active["reasons"])
                json.dumps(result, allow_nan=False)

    def test_default_zero_floor_enforces_both_gates_immediately(self):
        for forward_passed, active_passed in itertools.product((True, False), repeat=2):
            with self.subTest(forward=forward_passed, active=active_passed):
                result = decide(*gates(forward_passed, active_passed), iteration=1)
                self.assertEqual(result["passed"], forward_passed and active_passed)
                self.assertEqual(result["minimum_review_iteration"], 0)
                self.assertFalse(result["conclusion_deferred"])
                self.assertTrue(result["valid_evidence"])

    def test_original_failure_reasons_and_input_receipts_are_preserved(self):
        forward, active = gates(False, False)
        forward["reasons"] += ["candidate.fall_fraction: 0.2 > 0.1", "shared diagnostic"]
        active["reasons"] += ["shared diagnostic"]
        original = copy.deepcopy((forward, active))
        result = decide(forward, active, 13000, 17000)
        self.assertTrue(result["passed"])
        self.assertTrue(result["conclusion_deferred"])
        self.assertEqual(result["gate_failures"], forward["reasons"] + active["reasons"])
        self.assertEqual((forward, active), original)
        result["gate_failures"].append("decision-only note")
        self.assertEqual((forward, active), original)
        result = decide(forward, active, 17000, 17000)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reasons"], forward["reasons"] + active["reasons"])

    def test_iteration_and_floor_reject_invalid_types_bounds_and_nonfinite_numbers(self):
        for value in (None, True, False, "13000", 13000., -1, float("nan"),
                      float("inf"), -float("inf"), 10**400):
            with self.subTest(iteration=value):
                self.assert_invalid(*gates(), iteration=value, reason="iteration")
            with self.subTest(floor=value):
                self.assert_invalid(*gates(), floor=value, reason="minimum_review_iteration")
        self.assert_invalid(*gates(), iteration=0, reason="iteration")
        result = decide(*gates(), iteration=1, minimum_review_iteration=0)
        self.assertTrue(result["passed"])

    def test_missing_receipts_wrong_scopes_and_nonboolean_pass_flags_reject(self):
        for index in (0, 1):
            for value in (None, [], True, "receipt"):
                with self.subTest(index=index, receipt=value):
                    pair = list(gates())
                    pair[index] = value
                    self.assert_invalid(*pair, reason="receipt object")
            for scope in (None, "", "wrong", "active_mode_fall_guard_only"
                          if index == 0 else "forward_gait_preservation_only"):
                with self.subTest(index=index, scope=scope):
                    pair = gates()
                    pair[index]["scope"] = scope
                    self.assert_invalid(*pair, reason="scope")
            for value in (None, 0, 1, "true", [], {}):
                with self.subTest(index=index, passed=value):
                    pair = gates()
                    pair[index]["passed"] = value
                    self.assert_invalid(*pair, reason="passed")
            for field in ("scope", "passed"):
                with self.subTest(index=index, missing=field):
                    pair = gates()
                    del pair[index][field]
                    self.assert_invalid(*pair, reason=field)

    def test_observed_evidence_is_required_even_for_passing_receipts_below_floor(self):
        for index in (0, 1):
            for value in (None, {}, [], True, "observed"):
                with self.subTest(index=index, observed=value):
                    pair = gates()
                    pair[index]["observed"] = value
                    self.assert_invalid(*pair, reason="observed")
            pair = gates()
            del pair[index]["observed"]
            self.assert_invalid(*pair, reason="observed")
        for label in ("baseline", "candidate"):
            for value in (None, {}, [], True, "observed"):
                with self.subTest(label=label, observed=value):
                    forward, active = gates()
                    forward["observed"][label] = value
                    self.assert_invalid(forward, active, reason=label)
            forward, active = gates()
            del forward["observed"][label]
            self.assert_invalid(forward, active, reason=label)

    def test_reasons_must_agree_with_boolean_gate_outcomes(self):
        for index in (0, 1):
            for value in (None, "failed", [None], [1], [""], [" "]):
                with self.subTest(index=index, reasons=value):
                    pair = gates(False, False)
                    pair[index]["reasons"] = value
                    self.assert_invalid(*pair, reason="reasons")
            pair = gates()
            pair[index]["reasons"] = ["contradictory failure"]
            self.assert_invalid(*pair, reason="passed and reasons disagree")
            pair = gates(False, False)
            pair[index]["reasons"] = []
            self.assert_invalid(*pair, reason="passed and reasons disagree")
            pair = gates()
            del pair[index]["reasons"]
            self.assert_invalid(*pair, reason="reasons")

    def test_nonfinite_values_reject_recursively_anywhere_in_either_receipt(self):
        for index, value in itertools.product(
                (0, 1), (float("nan"), float("inf"), -float("inf"), 10**400)):
            with self.subTest(index=index, value=value):
                pair = gates()
                pair[index]["extra"] = {"nested": [{"metric": value}]}
                self.assert_invalid(*pair, reason="finite number")
        forward, active = gates(False, False)
        forward["observed"]["candidate"]["body_vx_m_s"] = float("nan")
        self.assert_invalid(forward, active, reason="body_vx_m_s")
        forward, active = gates()
        active["observed"]["per_mode"][0]["falls"] = float("inf")
        self.assert_invalid(forward, active, reason="falls")

    def test_recursive_and_non_json_evidence_is_rejected_with_safe_decisions(self):
        for value in ({1: "numeric key"}, {"value": object()}, (1, 2)):
            pair = gates()
            pair[1]["extra"] = value
            self.assert_invalid(*pair, reason="expected")
        forward, active = gates()
        active["extra"] = active
        self.assert_invalid(forward, active, reason="recursion")

    def test_invalid_evidence_in_one_gate_keeps_other_gate_failures_and_blocks_deferral(self):
        forward, active = gates(False, False)
        original_reasons = forward["reasons"] + active["reasons"]
        active["observed"] = None
        result = self.assert_invalid(forward, active, reason="active_gate.observed")
        self.assertEqual(result["gate_failures"], original_reasons)

    def test_forward_protocol_mismatch_is_invalid_evidence_even_with_both_observations(self):
        forward, active = gates(False, True)
        forward["observed"]["candidate"]["protocol"]["evaluation_seed"] = 103
        forward["reasons"] = ["protocol.evaluation_seed: baseline and candidate differ"]
        result = self.assert_invalid(forward, active, reason="protocols must match")
        self.assertEqual(result["gate_failures"], forward["reasons"])
        forward, active = gates()
        forward["observed"]["candidate"]["protocol"]["observation_noise"] = 1
        self.assert_invalid(forward, active, reason="protocols must match")
        forward, active = gates()
        del forward["observed"]["candidate"]["protocol"]
        self.assert_invalid(forward, active, reason="protocols must match")
        forward, active = gates()
        forward["observed"]["baseline"]["protocol"]["entry_duration_s"] = 4.
        forward["observed"]["candidate"]["protocol"]["entry_duration_s"] = 4
        self.assertTrue(decide(forward, active, 13000, 17000)["passed"])

    def test_existing_gate_receipts_preserve_failures_through_deferred_review(self):
        from check_command_curriculum import evaluate_curriculum
        from check_wasd_recovery import evaluate_recovery
        from test_command_curriculum_gate import MODE_NAMES, evaluation as cycle_evaluation
        from test_wasd_recovery_gate import evaluation as forward_evaluation

        forward = evaluate_recovery(forward_evaluation(), forward_evaluation(.1))
        active = evaluate_curriculum(
            cycle_evaluation(MODE_NAMES, falls={"forward_right": 2}), MODE_NAMES)
        self.assertFalse(forward["passed"])
        self.assertFalse(active["passed"])
        original = copy.deepcopy((forward, active))
        for iteration, expected in ((13000, True), (16999, True), (17000, False)):
            with self.subTest(iteration=iteration):
                result = decide(forward, active, iteration, 17000)
                self.assertEqual(result["passed"], expected)
                self.assertEqual(result["conclusion_deferred"], expected)
                self.assertTrue(result["valid_evidence"])
                self.assertEqual(result["gate_failures"], forward["reasons"] + active["reasons"])
        self.assertEqual((forward, active), original)

    def run_cli(self, forward, active, output, iteration=13000, floor=17000):
        return subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(SCRIPT),
             "--forward-gate", str(forward), "--active-mode-gate", str(active),
             "--iteration", str(iteration), "--minimum-review-iteration", str(floor),
             "--output", str(output)],
            capture_output=True, text=True, timeout=15)

    def test_cli_writes_decisions_with_exit_codes_and_preserves_failed_input_receipts(self):
        with tempfile.TemporaryDirectory(prefix="command-training-budget-") as temporary:
            root = Path(temporary)
            forward_path, active_path = root / "forward.json", root / "active.json"
            output = root / "decision" / "budget.json"
            for forward_passed, active_passed, iteration in itertools.product(
                    (True, False), (True, False), (13000, 17000)):
                with self.subTest(forward=forward_passed, active=active_passed, iteration=iteration):
                    forward, active = gates(forward_passed, active_passed)
                    before = json.dumps(forward), json.dumps(active)
                    forward_path.write_text(before[0], encoding="utf-8")
                    active_path.write_text(before[1], encoding="utf-8")
                    process = self.run_cli(forward_path, active_path, output, iteration)
                    expected = iteration < 17000 or (forward_passed and active_passed)
                    self.assertEqual(process.returncode, 0 if expected else 2, process.stderr)
                    decision = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(decision, json.loads(process.stdout))
                    self.assertEqual(decision["passed"], expected)
                    self.assertEqual(decision["gate_failures"], forward["reasons"] + active["reasons"])
                    self.assertEqual(forward_path.read_text(encoding="utf-8"), before[0])
                    self.assertEqual(active_path.read_text(encoding="utf-8"), before[1])

    def test_cli_invalid_and_missing_inputs_never_receive_budget_deferral(self):
        with tempfile.TemporaryDirectory(prefix="command-training-invalid-") as temporary:
            root = Path(temporary)
            forward_path, active_path, output = root / "forward.json", root / "active.json", root / "budget.json"
            forward, active = gates()
            for index in (0, 1):
                for payload, reason in (
                        ("{", "could not read gate receipt"),
                        ('{"passed":true,"passed":false}', "duplicate JSON field"),
                        (json.dumps({**active, "extra": [float("nan")]}), "finite number"),
                        ("[]", "receipt object"), (None, "could not read gate receipt")):
                    with self.subTest(index=index, payload=payload):
                        forward_path.write_text(json.dumps(forward), encoding="utf-8")
                        active_path.write_text(json.dumps(active), encoding="utf-8")
                        target = (forward_path, active_path)[index]
                        if payload is None:
                            target.unlink()
                        else:
                            target.write_text(payload, encoding="utf-8")
                        process = self.run_cli(forward_path, active_path, output)
                        self.assertEqual(process.returncode, 2, process.stderr)
                        decision = json.loads(process.stdout)
                        self.assertEqual(decision, json.loads(output.read_text(encoding="utf-8")))
                        self.assertFalse(decision["passed"])
                        self.assertFalse(decision["valid_evidence"])
                        self.assertFalse(decision["conclusion_deferred"])
                        self.assertIn(reason, "\n".join(decision["reasons"]))

    def test_cli_invalid_iteration_and_floor_write_json_safe_rejections(self):
        with tempfile.TemporaryDirectory(prefix="command-training-iteration-") as temporary:
            root = Path(temporary)
            forward_path, active_path, output = root / "forward.json", root / "active.json", root / "budget.json"
            forward, active = gates()
            forward_path.write_text(json.dumps(forward), encoding="utf-8")
            active_path.write_text(json.dumps(active), encoding="utf-8")
            for iteration, floor in ((0, 17000), (-1, 17000), (13000, -1)):
                with self.subTest(iteration=iteration, floor=floor):
                    process = self.run_cli(forward_path, active_path, output, iteration, floor)
                    self.assertEqual(process.returncode, 2, process.stderr)
                    decision = json.loads(process.stdout)
                    self.assertEqual(decision, json.loads(output.read_text(encoding="utf-8")))
                    self.assertFalse(decision["valid_evidence"])
                    self.assertFalse(decision["conclusion_deferred"])

    def test_cli_aliases_and_output_errors_preserve_both_input_receipts(self):
        with tempfile.TemporaryDirectory(prefix="command-training-alias-") as temporary:
            root = Path(temporary)
            forward_path, active_path = root / "forward.json", root / "active.json"
            forward, active = gates()
            forward_bytes, active_bytes = json.dumps(forward), json.dumps(active)
            forward_path.write_text(forward_bytes, encoding="utf-8")
            active_path.write_text(active_bytes, encoding="utf-8")
            outputs = [forward_path, active_path, forward_path / "budget.json"]
            for path in (forward_path, active_path):
                alias = root / f"{path.stem}-hardlink.json"
                os.link(path, alias)
                outputs.append(alias)
            for output in outputs:
                with self.subTest(output=output):
                    process = self.run_cli(forward_path, active_path, output)
                    self.assertEqual(process.returncode, 2, process.stderr)
                    decision = json.loads(process.stdout)
                    self.assertFalse(decision["passed"])
                    self.assertIn("output: could not write decision receipt", "\n".join(decision["reasons"]))
                    self.assertEqual(forward_path.read_text(encoding="utf-8"), forward_bytes)
                    self.assertEqual(active_path.read_text(encoding="utf-8"), active_bytes)


if __name__ == "__main__":
    unittest.main()

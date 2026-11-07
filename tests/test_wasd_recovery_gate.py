"""CPU-only forward-gait recovery checks against evaluator-shaped reports."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from check_wasd_recovery import evaluate_recovery


def evaluation(vx=.25, *, episodes=(10,) * 6, falls=(0,) * 6):
    """Mirror evaluate_terrain's lane counters and held-command metric rows."""
    lanes = []
    for level, (count, fallen) in enumerate(zip(episodes, falls)):
        lanes.append({
            "level": level, "episodes": float(count), "falls": float(fallen),
            "successes": 0., "timeouts": float(count - fallen),
            "return_sum": 0., "progress_sum": 0., "success_fraction": 0.,
            "fall_fraction": fallen / max(count, 1), "mean_return": 0.,
            "mean_max_progress_scale_m": 0.,
        })
    modes = []
    for name, command in (
            ("forward", [.25, 0., 0.]), ("backward", [-.20, 0., 0.]),
            ("turn_left", [0., 0., .55]), ("turn_right", [0., 0., -.55]),
            ("forward_left", [.25, 0., .45]), ("forward_right", [.25, 0., -.45]),
            ("backward_left", [-.20, 0., .45]), ("backward_right", [-.20, 0., -.45]),
            ("idle", [0., 0., 0.])):
        active = name == "forward"
        modes.append({
            "name": name, "command": command, "samples": 88800 if active else 0,
            "tracking_fraction": .9 if active else 0.,
            "body_vx_m_s": vx if active else 0., "body_vy_m_s": 0.,
            "body_yaw_rate_rad_s": 0., "mean_planar_speed_m_s": abs(vx) if active else 0.,
            "mean_planar_drift_m_s": 0., "desired_planar_speed_m_s": .25 if active else 0.,
            "mean_projected_velocity_m_s": vx if active else 0.,
            "mean_clearance_m": .7 if active else 0.,
            "mean_support_sole_slip_m_s": .01 if active else 0.,
            "supported_fraction": 1. if active else 0.,
            "velocity_rmse_m_s": abs(vx - .25) if active else 0.,
            "yaw_rate_rmse_rad_s": 0., "episodes": sum(episodes) if active else 0,
            "falls": sum(falls) if active else 0, "successes": 0,
        })
    return {
        "task": "locomotion/g1/stealth", "fixed_command_mode": "forward",
        "evaluation_seed": 101, "num_envs": 60, "physics_steps": 1500,
        "entry_duration_s": 4.84, "observation_noise": True,
        "physical_domain_randomization": {
            "enabled": True, "sampling": "once per simulator world; retained across episode resets",
            "source": "four fixed joint-offset channels in privileged context xi",
            "actor_observes_physical_scales_directly": False,
            "scale_ranges": {"body_mass": [.9, 1.1], "actuator_kp": [.85, 1.15],
                             "actuator_kd": [.85, 1.15], "actuator_effort": [.9, 1.1]},
            "mass_and_inertia": "scaled together per world",
            "motor_parameters": "kp, kd and symmetric effort limits scaled per world",
        },
        "command_mode_layout": "random", "command_sequence": None,
        "command_metric_samples": 88800, "keyboard_mode_metrics": modes, "lanes": lanes,
        "keyboard_episode_metrics": {
            "attribution": "fixed_command_mode_per_environment",
            "total": {"episodes": sum(episodes), "falls": sum(falls), "successes": 0,
                      "timeouts": sum(episodes) - sum(falls)},
            "per_keyboard_mode": [
                {key: row[key] for key in ("name", "episodes", "falls", "successes")}
                | {"timeouts": row["episodes"] - row["falls"]}
                for row in modes
            ],
        },
    }


class WasdRecoveryGateTests(unittest.TestCase):
    def assert_rejected(self, baseline, candidate, reason):
        receipt = evaluate_recovery(baseline, candidate)
        self.assertFalse(receipt["passed"])
        self.assertIn(reason, "\n".join(receipt["reasons"]))
        json.dumps(receipt, allow_nan=False)
        return receipt

    def test_valid_report_preserves_forward_only_and_reports_evidence(self):
        receipt = evaluate_recovery(evaluation(), evaluation(.20))
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["reasons"], [])
        self.assertEqual(receipt["scope"], "forward_gait_preservation_only")
        self.assertEqual(receipt["thresholds"]["candidate_min_body_vx_m_s"], .1875)
        self.assertEqual(receipt["thresholds"]["candidate_max_body_vx_m_s"], .40)
        self.assertEqual(receipt["observed"]["candidate"]["forward_samples"], 88800)
        self.assertEqual(receipt["observed"]["candidate"]["episodes"], 60)
        self.assertEqual(receipt["observed"]["candidate"]["body_vx_m_s"], .20)
        self.assertEqual(receipt["observed"]["candidate"]["fall_fraction"], 0.)

    def test_velocity_boundaries_and_stopped_actor(self):
        for baseline, candidate, passed in (
                (.15, .15, True), (.25, .1875, True), (.25, .40, True),
                (.25, 0., False), (.25, -.05, False), (.15, .1499, False),
                (.25, .1874, False), (.25, .4001, False)):
            with self.subTest(baseline=baseline, candidate=candidate):
                receipt = evaluate_recovery(evaluation(baseline), evaluation(candidate))
                self.assertEqual(receipt["passed"], passed)
                if not passed:
                    self.assertIn("candidate.body_vx_m_s", "\n".join(receipt["reasons"]))

    def test_fall_limit_is_inclusive_and_uses_pooled_episode_counts(self):
        limit = evaluation(falls=(1,) * 6)
        self.assertTrue(evaluate_recovery(limit, limit)["passed"])
        bad = evaluation(falls=(2, 1, 1, 1, 1, 1))
        self.assert_rejected(evaluation(), bad, "candidate.fall_fraction")
        uneven = evaluation(episodes=(1, 100, 100, 100, 100, 100), falls=(1, 0, 0, 0, 0, 0))
        receipt = evaluate_recovery(evaluation(), uneven)
        self.assertTrue(receipt["passed"])
        self.assertAlmostEqual(receipt["observed"]["candidate"]["fall_fraction"], 1 / 501)
        sparse = evaluation(episodes=(10, 0, 0, 0, 0, 0))
        self.assertTrue(evaluate_recovery(sparse, sparse)["passed"])

    def test_invalid_baseline_cannot_approve_an_acceptable_candidate(self):
        for baseline, reason in (
                (evaluation(0.), "baseline.body_vx_m_s"),
                (evaluation(.1499), "baseline.body_vx_m_s"),
                (evaluation(falls=(2,) * 6), "baseline.fall_fraction")):
            with self.subTest(reason=reason):
                self.assert_rejected(baseline, evaluation(), reason)

    def test_every_protocol_field_must_match(self):
        changes = {
            "evaluation_seed": 102, "num_envs": 61, "physics_steps": 1501,
            "entry_duration_s": 0., "observation_noise": False,
        }
        changed_randomization = copy.deepcopy(evaluation()["physical_domain_randomization"])
        changed_randomization["scale_ranges"]["body_mass"][0] = .8
        changes["physical_domain_randomization"] = changed_randomization
        for field, value in changes.items():
            with self.subTest(field=field):
                candidate = evaluation()
                candidate[field] = value
                self.assert_rejected(evaluation(), candidate, f"protocol.{field}")

    def test_protocol_compares_numeric_values_and_preserves_nested_boolean_types(self):
        baseline, candidate = evaluation(), evaluation()
        baseline["entry_duration_s"], candidate["entry_duration_s"] = 4., 4
        randomization = candidate["physical_domain_randomization"]
        candidate["physical_domain_randomization"] = dict(reversed(list(randomization.items())))
        self.assertTrue(evaluate_recovery(baseline, candidate)["passed"])
        candidate["physical_domain_randomization"]["actor_observes_physical_scales_directly"] = 0
        self.assert_rejected(baseline, candidate, "protocol.physical_domain_randomization")

    def test_missing_protocol_fields_in_both_reports_cannot_match(self):
        for field in ("task", "fixed_command_mode", "evaluation_seed", "num_envs",
                      "physics_steps", "entry_duration_s", "observation_noise",
                      "physical_domain_randomization"):
            with self.subTest(field=field):
                report = evaluation()
                del report[field]
                receipt = self.assert_rejected(report, report, f"{field}: missing required field")
                self.assertEqual(len(receipt["reasons"]), 2)

    def test_task_mode_command_and_protocol_types_are_validated_on_both_sides(self):
        changes = (
            ("task", "locomotion/g1/walk"), ("fixed_command_mode", "idle"),
            ("evaluation_seed", True), ("num_envs", 0), ("physics_steps", 1.5),
            ("entry_duration_s", -1.), ("observation_noise", 1),
            ("physical_domain_randomization", None),
            ("physical_domain_randomization", {}),
        )
        for field, value in changes:
            for label in ("baseline", "candidate"):
                with self.subTest(field=field, value=value, label=label):
                    reports = {"baseline": evaluation(), "candidate": evaluation()}
                    reports[label][field] = value
                    self.assert_rejected(**reports, reason=f"{label}.{field}")
        for command in ([.2, 0, 0], [.25, .1, 0], [.25, 0, .1], [.25, 0],
                        [.25, False, 0], [.25, "0", 0], None):
            with self.subTest(command=command):
                report = evaluation()
                report["keyboard_mode_metrics"][0]["command"] = command
                self.assert_rejected(report, report, "forward.command")

    def test_nonfinite_and_nonnumeric_required_metrics_reject_with_json_safe_receipts(self):
        paths = (
            ("keyboard_mode_metrics", 0, "body_vx_m_s"),
            ("keyboard_mode_metrics", 0, "samples"),
            ("lanes", 0, "episodes"), ("lanes", 0, "falls"),
            ("lanes", 0, "fall_fraction"), ("entry_duration_s",),
        )
        for path in paths:
            for value in (float("nan"), float("inf"), -float("inf"), None, "0.25", True):
                with self.subTest(path=path, value=value):
                    report = evaluation()
                    row = report
                    for field in path[:-1]:
                        row = row[field]
                    row[path[-1]] = value
                    self.assert_rejected(report, report, str(path[-1]))
        report = evaluation()
        report["keyboard_mode_metrics"][1]["mean_clearance_m"] = float("nan")
        self.assert_rejected(evaluation(), report, "mean_clearance_m")

    def test_missing_metrics_and_invalid_counts_reject(self):
        for path in (
                ("keyboard_mode_metrics",), ("keyboard_mode_metrics", 0, "body_vx_m_s"),
                ("keyboard_mode_metrics", 0, "samples"),
                ("keyboard_mode_metrics", 0, "command"), ("lanes",),
                ("lanes", 0, "episodes"), ("lanes", 0, "falls")):
            with self.subTest(missing=path):
                report = evaluation()
                row = report
                for field in path[:-1]:
                    row = row[field]
                del row[path[-1]]
                self.assert_rejected(report, report, f"{path[-1]}: missing required field")
        for field, value in (("episodes", -1), ("episodes", .5), ("falls", -1), ("falls", 11)):
            with self.subTest(field=field, value=value):
                report = evaluation()
                report["lanes"][0][field] = value
                self.assert_rejected(evaluation(), report, f"lanes[0].{field}")

    def test_absent_samples_episodes_and_ambiguous_forward_rows_reject(self):
        empty = evaluation(episodes=(0,) * 6)
        self.assert_rejected(empty, empty, "positive completed episodes")
        for field, value in (("samples", 0), ("episodes", 0), ("episodes", None), ("falls", 1)):
            with self.subTest(field=field, value=value):
                report = evaluation()
                report["keyboard_mode_metrics"][0][field] = value
                self.assert_rejected(report, report, f"forward.{field}")
        for rows in ([], evaluation()["keyboard_mode_metrics"][1:],
                     evaluation()["keyboard_mode_metrics"] * 2, None, [None]):
            with self.subTest(rows=rows):
                report = evaluation()
                report["keyboard_mode_metrics"] = rows
                self.assert_rejected(report, report, "keyboard_mode_metrics")
        report = evaluation()
        report["lanes"] = []
        self.assert_rejected(report, report, "lanes")
        report = evaluation()
        report["command_metric_samples"] = 0
        self.assert_rejected(report, report, "command_metric_samples")

    def test_cli_exit_codes_and_receipts_for_pass_reject_and_malformed_input(self):
        with tempfile.TemporaryDirectory(prefix="wasd-recovery-gate-") as temporary:
            root = Path(temporary)
            baseline, candidate = root / "baseline.json", root / "candidate.json"
            output = root / "receipts" / "gate.json"
            baseline.write_text(json.dumps(evaluation()), encoding="utf-8")
            cases = (
                (json.dumps(evaluation()), 0, None),
                (json.dumps(evaluation(0.)), 2, "candidate.body_vx_m_s"),
                (json.dumps(evaluation(float("nan"))), 2, "finite number"),
                ("{", 2, "could not read evaluation report"),
                ('{"task": 1, "task": 2}', 2, "duplicate JSON field"),
                ("[]", 2, "expected a JSON object"),
                (None, 2, "could not read evaluation report"),
            )
            for payload, code, reason in cases:
                with self.subTest(code=code, reason=reason):
                    if payload is None:
                        candidate.unlink()
                    else:
                        candidate.write_text(payload, encoding="utf-8")
                    process = subprocess.run(
                        [sys.executable, "-B",
                         str(REPO / "scripts/rls/supermimic/check_wasd_recovery.py"),
                         "--baseline", str(baseline), "--candidate", str(candidate),
                         "--output", str(output)],
                        capture_output=True, text=True, timeout=15,
                    )
                    self.assertEqual(process.returncode, code, process.stderr)
                    receipt = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(json.loads(process.stdout), receipt)
                    self.assertEqual(receipt["passed"], code == 0)
                    self.assertTrue({"passed", "reasons", "observed", "thresholds"} <= receipt.keys())
                    json.dumps(receipt, allow_nan=False)
                    if reason:
                        self.assertIn(reason, "\n".join(receipt["reasons"]))

    def test_cli_output_error_returns_rejection_receipt_on_stdout(self):
        with tempfile.TemporaryDirectory(prefix="wasd-recovery-output-") as temporary:
            baseline = Path(temporary) / "baseline.json"
            payload = json.dumps(evaluation())
            baseline.write_text(payload, encoding="utf-8")
            process = subprocess.run(
                [sys.executable, "-B",
                 str(REPO / "scripts/rls/supermimic/check_wasd_recovery.py"),
                 "--baseline", str(baseline), "--candidate", str(baseline),
                 "--output", str(baseline / "gate.json")],
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(process.returncode, 2, process.stderr)
            receipt = json.loads(process.stdout)
            self.assertFalse(receipt["passed"])
            self.assertIn("output: could not write receipt", "\n".join(receipt["reasons"]))
            self.assertEqual(baseline.read_text(encoding="utf-8"), payload)


if __name__ == "__main__":
    unittest.main()

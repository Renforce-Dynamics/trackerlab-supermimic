"""Standard-library checks for per-active-mode curriculum fall guards."""

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from check_command_curriculum import COMMAND_MODES, MODE_NAMES, evaluate_curriculum


ORIGINAL_FIVE = ["forward", "turn_left", "turn_right", "forward_left", "forward_right"]
SIX_MODES = [*ORIGINAL_FIVE, "idle"]
SCRIPT = REPO / "scripts/rls/supermimic/check_command_curriculum.py"


def evaluation(training_modes=SIX_MODES, *, falls=None, successes=None):
    """Mirror the evaluator's distinct keyboard and flat-task lane timeouts."""
    falls, successes = falls or {}, successes or {}
    rows, outcomes = [], []
    totals = dict.fromkeys(("episodes", "falls", "successes", "timeouts"), 0)
    lanes = [{"level": i, "episodes": 0., "falls": 0., "successes": 0., "timeouts": 0.}
             for i in range(6)]
    for index, (name, command) in enumerate(COMMAND_MODES):
        fallen, succeeded = falls.get(name, 0), successes.get(name, 0)
        counts = {"episodes": 10, "falls": fallen, "successes": succeeded,
                  "timeouts": 10 - fallen - succeeded}
        outcomes.append({"name": name, **counts})
        for key, count in counts.items():
            totals[key] += count
            if key != "timeouts":
                lanes[index % 6][key] += count
        lanes[index % 6]["timeouts"] += counts["timeouts"] + counts["successes"]
        rows.append({
            "name": name, "command": list(command), "samples": 14000,
            "body_vx_m_s": command[0], "body_vy_m_s": command[1],
            "body_yaw_rate_rad_s": command[2], "tracking_fraction": .8,
            "mean_planar_speed_m_s": abs(command[0]), "mean_planar_drift_m_s": 0.,
            "desired_planar_speed_m_s": abs(command[0]),
            "mean_projected_velocity_m_s": abs(command[0]), "mean_clearance_m": .70,
            "mean_support_sole_slip_m_s": .05, "supported_fraction": .9,
            "velocity_rmse_m_s": .02, "yaw_rate_rmse_rad_s": .03,
            **{key: counts[key] for key in ("episodes", "falls", "successes")},
        })
    for lane in lanes:
        lane["fall_fraction"] = lane["falls"] / max(lane["episodes"], 1)
        lane["success_fraction"] = lane["successes"] / max(lane["episodes"], 1)
    return {
        "task": "locomotion/g1/stealth", "stealth_control_mode": "wasd",
        "command_reward_mode": "legacy", "training_command_modes": list(training_modes),
        "fixed_command_mode": None, "command_mode_layout": "cycle", "command_sequence": None,
        "num_envs": 90, "physics_steps": 1500, "evaluation_seed": 101,
        "entry_duration_s": 4.84, "policy_exploration_noise": False, "observation_noise": True,
        "physical_domain_randomization": {"enabled": True},
        "command_metric_samples": sum(row["samples"] for row in rows),
        "keyboard_mode_metrics": rows, "lanes": lanes,
        "keyboard_episode_metrics": {
            "attribution": "environment_id_modulo_cycle_mode; stable across resets",
            "total": totals, "per_keyboard_mode": outcomes,
        },
    }


class CommandCurriculumGateTests(unittest.TestCase):
    def assert_rejected(self, report, reason, training_modes=SIX_MODES):
        receipt = evaluate_curriculum(report, training_modes)
        self.assertFalse(receipt["passed"])
        self.assertIn(reason, "\n".join(receipt["reasons"]))
        self.assertFalse(receipt["all_commands_accepted"])
        self.assertTrue(receipt["visual_review_required"])
        json.dumps(receipt, allow_nan=False)
        return receipt

    def test_valid_guard_reports_all_modes_and_limits_its_claim(self):
        report = evaluation()
        original = copy.deepcopy(report)
        receipt = evaluate_curriculum(report, SIX_MODES)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["reasons"], [])
        self.assertEqual(receipt["scope"], "active_mode_fall_guard_only")
        self.assertFalse(receipt["all_commands_accepted"])
        self.assertTrue(receipt["visual_review_required"])
        self.assertEqual(receipt["thresholds"], {"max_active_mode_fall_fraction": .10})
        rows = receipt["observed"]["per_mode"]
        self.assertEqual([row["name"] for row in rows], list(MODE_NAMES))
        self.assertEqual(receipt["observed"]["total"], {
            "episodes": 90, "falls": 0, "successes": 0, "timeouts": 90,
        })
        self.assertEqual(receipt["observed"]["samples"], 126000)
        for row in rows:
            self.assertEqual(row["active_training_mode"], row["name"] in SIX_MODES)
            self.assertEqual(row["diagnostics_only"], row["name"] not in SIX_MODES)
            self.assertEqual(row["fall_guard_passed"], True if row["name"] in SIX_MODES else None)
            self.assertEqual(row["mean_support_sole_slip_m_s"], .05)
            self.assertEqual(row["tracking_fraction"], .8)
            self.assertFalse(row["directional_capability"]["readiness_assessed"])
        self.assertEqual(report, original)
        json.dumps(receipt, allow_nan=False)

    def test_active_mode_failure_cannot_be_diluted_by_pooled_average(self):
        report = evaluation(falls={"idle": 2})
        self.assertLess(report["keyboard_episode_metrics"]["total"]["falls"] / 90, .10)
        receipt = self.assert_rejected(report, "idle.fall_fraction: 0.2 > 0.1")
        self.assertIsNotNone(receipt["observed"])
        for mode in SIX_MODES:
            with self.subTest(mode=mode):
                self.assert_rejected(evaluation(falls={mode: 2}), f"{mode}.fall_fraction")

    def test_fall_limit_is_inclusive_and_excluded_modes_are_diagnostics(self):
        falls = {name: 1 if name in SIX_MODES else 10 for name in MODE_NAMES}
        receipt = evaluate_curriculum(evaluation(falls=falls), SIX_MODES)
        self.assertTrue(receipt["passed"])
        for row in receipt["observed"]["per_mode"]:
            self.assertEqual(row["fall_fraction"], .1 if row["active_training_mode"] else 1.)
            if row["diagnostics_only"]:
                self.assertIsNone(row["fall_guard_passed"])
        seven = [*SIX_MODES, "backward"]
        self.assert_rejected(evaluation(seven, falls=falls), "backward.fall_fraction", seven)
        self.assert_rejected(evaluation(MODE_NAMES, falls=falls), "backward_left.fall_fraction", MODE_NAMES)

    def test_directional_signs_are_diagnostic_even_when_fall_guard_passes(self):
        report = evaluation(MODE_NAMES)
        for row in report["keyboard_mode_metrics"]:
            row["body_vx_m_s"] = -row["command"][0]
            row["body_yaw_rate_rad_s"] = -row["command"][2]
            row["tracking_fraction"] = 0.
            row["mean_support_sole_slip_m_s"] = 1.
        receipt = evaluate_curriculum(report, MODE_NAMES)
        self.assertTrue(receipt["passed"])
        self.assertFalse(receipt["all_commands_accepted"])
        for row in receipt["observed"]["per_mode"]:
            direction = row["directional_capability"]
            self.assertEqual(direction["scope"], "signs_of_mean_motion_only")
            self.assertFalse(direction["readiness_assessed"])
            self.assertEqual(direction["translation_sign_matches"], False if row["command"][0] else None)
            self.assertEqual(direction["yaw_sign_matches"], False if row["command"][2] else None)
        report = evaluation(MODE_NAMES)
        receipt = evaluate_curriculum(report, MODE_NAMES)
        reverse = next(row for row in receipt["observed"]["per_mode"] if row["name"] == "backward_left")
        self.assertTrue(reverse["directional_capability"]["translation_sign_matches"])
        self.assertTrue(reverse["directional_capability"]["yaw_sign_matches"])

    def test_cycle_protocol_requires_full_wasd_legacy_reward_and_fixed_dimensions(self):
        changes = {
            "task": "locomotion/g1/velocity", "stealth_control_mode": "keyboard",
            "command_reward_mode": "wasd", "command_mode_layout": "random",
            "fixed_command_mode": "forward", "command_sequence": {},
            "num_envs": 9, "physics_steps": 1000, "evaluation_seed": True,
            "entry_duration_s": -1., "policy_exploration_noise": True,
            "observation_noise": False, "physical_domain_randomization": {"enabled": False},
        }
        for field, value in changes.items():
            with self.subTest(field=field, value=value):
                report = evaluation()
                report[field] = value
                self.assert_rejected(report, f"cycle.{field}")
            with self.subTest(missing=field):
                report = evaluation()
                del report[field]
                self.assert_rejected(report, f"{field}: missing required field")
        for field, value in (("num_envs", 90.), ("physics_steps", 1500.),
                             ("observation_noise", 1), ("policy_exploration_noise", 0),
                             ("physical_domain_randomization", {"enabled": 1})):
            with self.subTest(field=field, value=value):
                report = evaluation()
                report[field] = value
                self.assert_rejected(report, f"cycle.{field}")

    def test_training_subset_matches_recorded_names_and_null_means_all_modes(self):
        for modes in ([], ["unknown"], ["idle", "idle"], "idle", None, [True], [[]]):
            with self.subTest(modes=modes):
                self.assert_rejected(evaluation(), "training_modes", modes)
        for recorded in ([], ["idle"], ["idle", "idle"], "idle", None):
            with self.subTest(recorded=recorded):
                report = evaluation()
                report["training_command_modes"] = recorded
                self.assert_rejected(report, "cycle.training_command_modes")
        report = evaluation(MODE_NAMES)
        report["training_command_modes"] = None
        self.assertTrue(evaluate_curriculum(report, MODE_NAMES)["passed"])
        report = evaluation()
        report["training_command_modes"].reverse()
        self.assertTrue(evaluate_curriculum(report, SIX_MODES)["passed"])
        del report["training_command_modes"]
        self.assert_rejected(report, "training_command_modes: missing required field")

    def test_all_nine_rows_are_required_even_when_modes_are_excluded(self):
        for field, container in (("keyboard_mode_metrics", ()),
                                 ("per_keyboard_mode", ("keyboard_episode_metrics",))):
            for change in ("missing", "duplicate", "unknown", "invalid"):
                with self.subTest(field=field, change=change):
                    report = evaluation()
                    parent = report if not container else report[container[0]]
                    rows = parent[field]
                    if change == "missing":
                        rows.pop(MODE_NAMES.index("backward"))
                    elif change == "duplicate":
                        rows[1] = copy.deepcopy(rows[0])
                    elif change == "unknown":
                        rows[1]["name"] = "sideways"
                    else:
                        rows[1] = None
                    self.assert_rejected(report, field)
        report = evaluation()
        report["keyboard_mode_metrics"].reverse()
        report["keyboard_episode_metrics"]["per_keyboard_mode"].reverse()
        self.assertTrue(evaluate_curriculum(report, SIX_MODES)["passed"])

    def test_canonical_twists_reject_aliases_nonfinite_and_boolean_components(self):
        for command in ([.2, 0., 0.], [.25, False, 0.], [.25, "0", 0.], [0., 0.],
                        [float("nan"), 0., 0.], None):
            with self.subTest(command=command):
                report = evaluation()
                report["keyboard_mode_metrics"][0]["command"] = command
                self.assert_rejected(report, "command")
        report = evaluation()
        report["keyboard_mode_metrics"][6]["command"][2] = -.45
        self.assert_rejected(report, "backward_left.command")

    def test_every_metric_must_be_finite_numeric_and_present_on_excluded_rows(self):
        fields = ("body_vx_m_s", "body_vy_m_s", "body_yaw_rate_rad_s", "tracking_fraction",
                  "mean_planar_speed_m_s", "mean_planar_drift_m_s", "desired_planar_speed_m_s",
                  "mean_projected_velocity_m_s", "mean_clearance_m", "mean_support_sole_slip_m_s",
                  "supported_fraction", "velocity_rmse_m_s", "yaw_rate_rmse_rad_s")
        for field in fields:
            for value in (float("nan"), float("inf"), -float("inf"), None, True, "0.1", 10**400):
                with self.subTest(field=field, value=value):
                    report = evaluation()
                    report["keyboard_mode_metrics"][1][field] = value
                    self.assert_rejected(report, field)
            with self.subTest(missing=field):
                report = evaluation()
                del report["keyboard_mode_metrics"][1][field]
                self.assert_rejected(report, f"{field}: missing required field")
        report = evaluation()
        report["unused_diagnostic"] = [float("inf")]
        self.assert_rejected(report, "unused_diagnostic")

    def test_fraction_and_nonnegative_metric_ranges_are_checked(self):
        for field in ("tracking_fraction", "supported_fraction"):
            for value in (-.01, 1.01):
                with self.subTest(field=field, value=value):
                    report = evaluation()
                    report["keyboard_mode_metrics"][1][field] = value
                    self.assert_rejected(report, field)
        for field in ("mean_planar_speed_m_s", "mean_planar_drift_m_s", "desired_planar_speed_m_s",
                      "mean_support_sole_slip_m_s", "velocity_rmse_m_s", "yaw_rate_rmse_rad_s"):
            with self.subTest(field=field):
                report = evaluation()
                report["keyboard_mode_metrics"][1][field] = -.1
                self.assert_rejected(report, field)

    def test_positive_samples_and_completed_episodes_are_required_for_every_mode(self):
        for field in ("samples", "episodes", "falls", "successes"):
            for value in (None, True, "10", -.1, .5, float("nan")):
                with self.subTest(field=field, value=value):
                    report = evaluation()
                    report["keyboard_mode_metrics"][1][field] = value
                    self.assert_rejected(report, field)
        for value in (0, 15001):
            report = evaluation()
            report["keyboard_mode_metrics"][1]["samples"] = value
            self.assert_rejected(report, "backward.samples")
        report = evaluation()
        report["keyboard_episode_metrics"]["per_keyboard_mode"][1]["episodes"] = 0
        self.assert_rejected(report, "backward.episodes")
        for key in ("samples", "episodes", "falls", "successes"):
            report = evaluation()
            del report["keyboard_mode_metrics"][1][key]
            self.assert_rejected(report, f"{key}: missing required field")

    def test_episode_attribution_and_per_mode_counts_are_consistent(self):
        report = evaluation()
        report["keyboard_episode_metrics"]["attribution"] = "random_mode_resampling"
        self.assert_rejected(report, "attribution")
        for field in ("episodes", "falls", "successes"):
            with self.subTest(field=field):
                report = evaluation()
                report["keyboard_mode_metrics"][1][field] += 1
                self.assert_rejected(report, f"backward.{field}")
        for field in ("episodes", "falls", "successes", "timeouts"):
            report = evaluation()
            report["keyboard_episode_metrics"]["per_keyboard_mode"][1][field] += 1
            self.assert_rejected(report, "backward.episodes")
            report = evaluation()
            report["keyboard_episode_metrics"]["total"][field] += 1
            self.assert_rejected(report, "keyboard_episode_metrics.total")
        report = evaluation()
        report["command_metric_samples"] += 1
        self.assert_rejected(report, "command_metric_samples")

    def test_successful_flat_timeouts_have_distinct_lane_and_keyboard_counts(self):
        report = evaluation(successes={"forward": 8, "idle": 6}, falls={"forward": 1})
        receipt = evaluate_curriculum(report, SIX_MODES)
        self.assertTrue(receipt["passed"])
        totals = receipt["observed"]["total"]
        self.assertEqual(totals, {"episodes": 90, "falls": 1, "successes": 14, "timeouts": 75})
        self.assertEqual(sum(lane["timeouts"] for lane in report["lanes"]), 89)
        report["lanes"][0]["timeouts"] -= 8
        self.assert_rejected(report, "inconsistent flat-task episode counts")

    def test_lane_counts_fractions_and_levels_must_agree(self):
        for value in ([], None, [{}] * 5):
            report = evaluation()
            report["lanes"] = value
            self.assert_rejected(report, "cycle.lanes")
        for field, value in (("level", 1), ("episodes", 100), ("falls", -1), ("successes", 100),
                             ("timeouts", .5), ("fall_fraction", .5), ("success_fraction", .5)):
            with self.subTest(field=field, value=value):
                report = evaluation()
                report["lanes"][0][field] = value
                self.assert_rejected(report, "cycle.lanes")
        report = evaluation()
        report["lanes"][0]["episodes"] += 1
        report["lanes"][0]["timeouts"] += 1
        self.assert_rejected(report, "totals differ from per-mode episode sums")

    def test_invalid_objects_return_json_safe_rejection_receipts(self):
        for report in (None, [], True, "cycle", {}):
            with self.subTest(report=report):
                self.assert_rejected(report, "cycle")
        report = evaluation()
        report["self"] = report
        self.assert_rejected(report, "recursion")

    def test_stdlib_mode_table_matches_the_production_versioned_contract(self):
        path = REPO / "source/trackerLab/trackerLab/tracker_env/mdp/supermimic_command.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        table = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "STEALTH_WASD_MODES_V1"
                             for target in node.targets))
        expected = tuple((ast.literal_eval(row.args[0]), tuple(ast.literal_eval(arg) for arg in row.args[1:4]))
                         for row in table.elts)
        self.assertEqual(COMMAND_MODES, expected)

    def run_cli(self, cycle, output, training_modes=SIX_MODES):
        # Disable site packages so the command proves its standard-library boundary.
        return subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(SCRIPT), "--cycle", str(cycle),
             "--training-modes", *training_modes, "--output", str(output)],
            capture_output=True, text=True, timeout=15)

    def test_cli_pass_reject_malformed_and_missing_input_receipts(self):
        with tempfile.TemporaryDirectory(prefix="command-curriculum-gate-") as temporary:
            root = Path(temporary)
            cycle, output = root / "cycle.json", root / "receipts" / "gate.json"
            cases = (
                (json.dumps(evaluation()), 0, None),
                (json.dumps(evaluation(falls={"idle": 2})), 2, "idle.fall_fraction"),
                ("{", 2, "could not read evaluation report"),
                ('{"task":1,"task":2}', 2, "duplicate JSON field"),
                ('{"unused":NaN}', 2, "finite number"),
                ("[]", 2, "expected a JSON object"),
                (None, 2, "could not read evaluation report"),
            )
            for payload, code, reason in cases:
                with self.subTest(code=code, reason=reason):
                    if payload is None:
                        cycle.unlink()
                    else:
                        cycle.write_text(payload, encoding="utf-8")
                    process = self.run_cli(cycle, output)
                    self.assertEqual(process.returncode, code, process.stderr)
                    receipt = json.loads(process.stdout)
                    self.assertEqual(receipt, json.loads(output.read_text(encoding="utf-8")))
                    self.assertEqual(receipt["passed"], code == 0)
                    self.assertFalse(receipt["all_commands_accepted"])
                    self.assertTrue(receipt["visual_review_required"])
                    json.dumps(receipt, allow_nan=False)
                    if reason:
                        self.assertIn(reason, "\n".join(receipt["reasons"]))

    def test_cli_output_failure_and_input_alias_preserve_evidence(self):
        with tempfile.TemporaryDirectory(prefix="command-curriculum-output-") as temporary:
            cycle = Path(temporary) / "cycle.json"
            payload = json.dumps(evaluation())
            cycle.write_text(payload, encoding="utf-8")
            for output in (cycle / "gate.json", cycle):
                with self.subTest(output=output):
                    process = self.run_cli(cycle, output)
                    self.assertEqual(process.returncode, 2, process.stderr)
                    receipt = json.loads(process.stdout)
                    self.assertFalse(receipt["passed"])
                    self.assertIn("output: could not write receipt", "\n".join(receipt["reasons"]))
                    self.assertEqual(cycle.read_text(encoding="utf-8"), payload)


if __name__ == "__main__":
    unittest.main()

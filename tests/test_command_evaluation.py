"""CPU checks for checkpoint-calibrated evaluation rewards and motion evidence."""

import ast
import numpy as np
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from evaluate_terrain import (
    WASD_RELEASE_REGRESSION, attach_locomotion_quality, command_evaluation_summary,
    command_runtime_options, command_sequence_plan, evaluation_episode_limit, freeze_recorded_commands,
    keyboard_episode_sums, keyboard_transition_sums, validate_command_selection,
)
from trackerLab.algorithms.supermimic.locomotion_quality import LocomotionQuality
from trackerLab.tracker_env.mdp.supermimic import unflatten_state
from test_command_locomotion import CpuCommandSim


class EvaluationKinematics:
    device = "cpu"
    robot = SimpleNamespace(
        name="g1", default_q=(0.,) * 4, action_scale=(.25,) * 4)
    body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    tracked_idx = (0, 1)
    q_lo, q_hi = torch.full((4,), -10.), torch.full((4,), 10.)

    def forward(self, p, R, v, w, q, qd):
        local = p.new_tensor([[0., .1, -.74], [0., -.1, -.74]])
        position = p[..., None, :] + (R[..., None, :, :] @ local[..., None]).squeeze(-1)
        velocity = v[..., None, :].expand_as(position)
        return (position, R[..., None, :, :].expand(*R.shape[:-2], 2, 3, 3),
                velocity, w[..., None, :].expand_as(position))


class CommandEvaluationTests(unittest.TestCase):
    def test_runtime_options_restore_checkpoint_command_resampling(self):
        for interval in ([1000, 1000], (1000, 1000), [100, 250], (100, 250)):
            with self.subTest(interval=interval):
                arguments = {
                    "command_resample_steps": interval, "jump_height": .2,
                    "jump_probability": .3, "action_limit": 20.,
                }
                self.assertEqual(command_runtime_options({"arguments": arguments}), {
                    "command_resample_steps": tuple(interval), "jump_height": .2,
                    "jump_probability": .3, "action_limit": 20.,
                    "stealth_keyboard_mode": False, "self_collisions": False,
                    "collision_mask_version": 1,
                    "physical_domain_randomization": False,
                    "grounded_start": False, "standing_joint_noise": 0.,
                })
                self.assertIs(arguments["command_resample_steps"], interval)

    def test_runtime_options_preserve_legacy_checkpoint_defaults(self):
        self.assertEqual(command_runtime_options({"arguments": {}}), {
            "command_resample_steps": (100, 250), "jump_height": 0.,
            "jump_probability": .15, "action_limit": 4.,
            "stealth_keyboard_mode": False, "self_collisions": False,
            "collision_mask_version": 1,
            "physical_domain_randomization": False,
            "grounded_start": False, "standing_joint_noise": 0.,
        })

    def test_runtime_options_restore_physical_domain_randomization(self):
        options = command_runtime_options({
            "arguments": {"physical_domain_randomization": True},
        })
        self.assertTrue(options["physical_domain_randomization"])

    def test_runtime_options_restore_repaired_collision_masks(self):
        options = command_runtime_options({
            "arguments": {"self_collisions": True, "collision_mask_version": 2},
        })
        self.assertTrue(options["self_collisions"])
        self.assertEqual(options["collision_mask_version"], 2)

    def test_wasd_runtime_selects_nine_mode_sampling(self):
        options = command_runtime_options({"arguments": {"stealth_control_mode": "wasd"}})
        self.assertTrue(options["stealth_keyboard_mode"])
        self.assertEqual(options["command_mode_set"], "wasd")

    def test_evaluation_uses_saved_calibration_and_includes_penalty_in_episode_return(self):
        sim = CpuCommandSim(max_episode_steps=1)
        kin = EvaluationKinematics()
        motors = [(20., 12., 25., .5, .01)] * 4
        saved_height = torch.tensor([.01, .03])
        with patch.object(LocomotionQuality, "calibrate", side_effect=AssertionError("reference read")):
            quality = attach_locomotion_quality(
                sim, kin, sim.mdp.field, motors, {
                    "quality_sole_height": saved_height,
                    "arguments": {"foot_slide_weight": .04},
                })
        torch.testing.assert_close(quality.sole_height, saved_height)
        self.assertEqual(quality.foot_slide_weight, .04)
        self.assertIs(sim.locomotion_reward_fn.__self__, quality)
        saved_height.fill_(.5)
        torch.testing.assert_close(quality.sole_height, torch.tensor([.01, .03]))
        action = torch.full((1, 4), .2)
        with torch.no_grad():
            transition = sim.step(action)
            previous, following = (unflatten_state(transition[key]) for key in ("s", "s1"))
            task_reward = sim.mdp.reward(
                following, transition["a"], transition["a_prev"], transition["c1"])
            penalty = quality.reward(previous, following, transition["a"], transition["xi"])
            expected = task_reward + penalty + transition["reward_terms"]["terminal"]
        self.assertLess(penalty.item(), 0.)
        torch.testing.assert_close(transition["reward_terms"]["locomotion_regularization"], penalty)
        torch.testing.assert_close(transition["r"], expected)
        torch.testing.assert_close(transition["episode_return"], expected)

    def test_invalid_saved_sole_calibration_is_rejected(self):
        for heights in (torch.zeros(1), torch.tensor([0., float("nan")])):
            with self.subTest(heights=heights):
                sim = CpuCommandSim()
                with self.assertRaisesRegex(ValueError, "sole calibration"):
                    attach_locomotion_quality(
                        sim, EvaluationKinematics(), sim.mdp.field,
                        [(20., 12., 25., .5, .01)] * 4, {"quality_sole_height": heights})
                self.assertFalse(hasattr(sim, "locomotion_reward_fn"))

    def test_command_quality_receives_poststep_command_without_double_regularization(self):
        sim = CpuCommandSim(max_episode_steps=1)
        seen = []
        def command_quality(previous, state, action, xi, command):
            seen.append(command.clone())
            return -.02 * torch.ones_like(command[:, 0])
        def old_quality(*args):
            raise AssertionError("command quality already contains base regularization")
        sim.command_locomotion_reward_fn = command_quality
        sim.locomotion_reward_fn = old_quality
        transition = sim.step(torch.zeros(1, 4))
        torch.testing.assert_close(seen[0], transition["c1"])
        torch.testing.assert_close(transition["reward_terms"]["locomotion_regularization"],
                                   torch.tensor([-.02]))

    def test_report_preserves_motion_evidence_and_separates_amp_from_task_return(self):
        result = command_evaluation_summary(
            torch.tensor([.08, .18, .02]), torch.tensor(2.),
            torch.tensor([.6, .1]), torch.tensor([1.5, 2.]),
            torch.tensor([.16, .04, 2000., 0., 2.]), {"version": 1})
        motion = result["motion_metrics"]
        self.assertAlmostEqual(motion["mean_requested_planar_speed_mps"], .3)
        self.assertAlmostEqual(motion["mean_projected_velocity_along_command_mps"], .05)
        self.assertEqual(motion["samples"], 2)
        self.assertAlmostEqual(result["command_metrics"]["velocity_rmse_scale_m_s"], .2)
        self.assertAlmostEqual(result["style_metrics"]["mean_reward_rate"], .75)
        definition = result["task_return_definition"]
        self.assertIn("locomotion_regularization", definition["components"])
        self.assertIn("terminal_reward", definition["components"])
        self.assertFalse(definition["includes_amp_style_reward"])
        self.assertFalse(definition["matches_combined_training_objective"])

    def test_report_retains_zero_or_negative_progress_without_dividing_by_zero(self):
        for progress, count in ((0., 2.), (-.4, 2.), (0., 0.)):
            with self.subTest(progress=progress, count=count):
                result = command_evaluation_summary(
                    torch.zeros(3), torch.tensor(count),
                    torch.tensor([.6 if count else 0., progress]),
                    torch.zeros(2), torch.zeros(5), None)
                motion = result["motion_metrics"]
                self.assertAlmostEqual(
                    motion["mean_projected_velocity_along_command_mps"], progress / max(count, 1.))
                self.assertEqual(motion["samples"], int(count))

    def test_report_summarizes_keyboard_modes_independently(self):
        modes = [
            {"name": "forward", "command": [.25, 0., 0.]},
            {"name": "turn_left", "command": [0., 0., .55]},
        ]
        per_mode = {
            "modes": modes,
            "values": [
                torch.tensor([100., 75., .1, .02, .01, .12, .03, .2, 24., .69, .04, 90., 1., .04]),
                torch.tensor([40., 20., 0., 0., .2, .1, .1, 0., 0., .69, .08, 30., 2.25, .16]),
            ],
        }
        episodes, per_mode_episodes = keyboard_episode_sums(
            torch.tensor([0, 0, 1, 1]), torch.tensor([True, True, True, False]),
            torch.tensor([False, True, False, False]),
            torch.tensor([True, False, False, False]), 2)
        result = command_evaluation_summary(
            torch.zeros(3), torch.tensor(140.), torch.tensor([24., 24.]),
            torch.zeros(2), torch.zeros(5), {"command_modes": modes}, per_mode,
            per_mode_episodes, episodes, "cycle_environment_id_modulo_mode")
        rows = result["keyboard_mode_metrics"]
        self.assertEqual([row["name"] for row in rows], ["forward", "turn_left"])
        self.assertEqual([row["samples"] for row in rows], [100, 40])
        self.assertAlmostEqual(rows[0]["tracking_fraction"], .75)
        self.assertAlmostEqual(rows[1]["tracking_fraction"], .5)
        self.assertAlmostEqual(rows[0]["mean_projected_velocity_m_s"], .24)
        self.assertEqual(rows[1]["mean_projected_velocity_m_s"], 0.)
        self.assertAlmostEqual(rows[0]["body_vx_m_s"], .001)
        self.assertAlmostEqual(rows[1]["body_yaw_rate_rad_s"], .005)
        self.assertAlmostEqual(rows[0]["mean_planar_drift_m_s"], .0003)
        self.assertAlmostEqual(rows[0]["mean_clearance_m"], .0069)
        self.assertAlmostEqual(rows[1]["mean_support_sole_slip_m_s"], .002)
        self.assertAlmostEqual(rows[0]["supported_fraction"], .9)
        self.assertAlmostEqual(rows[0]["velocity_rmse_m_s"], .1)
        self.assertAlmostEqual(rows[1]["yaw_rate_rmse_rad_s"], (.16 / 40) ** .5)
        self.assertEqual((rows[0]["episodes"], rows[0]["falls"], rows[0]["successes"]),
                         (2, 1, 1))
        self.assertEqual((rows[1]["episodes"], rows[1]["falls"], rows[1]["successes"]),
                         (1, 0, 0))
        self.assertEqual(result["keyboard_episode_metrics"]["total"], {
            "episodes": 3, "falls": 1, "successes": 1, "timeouts": 1,
        })

    def test_keyboard_episode_stats_pool_random_and_sequence_without_mode_attribution(self):
        done = torch.tensor([True, True, True, False])
        falls = torch.tensor([False, True, False, False])
        success = torch.tensor([True, False, False, False])
        total, by_mode = keyboard_episode_sums(None, done, falls, success, 3)
        self.assertIsNone(by_mode)
        modes = [{"name": name, "command": [0., 0., 0.]}
                 for name in ("forward", "idle", "backward")]
        transition = {"modes": modes, "values": [torch.zeros(14) for _ in modes]}
        result = command_evaluation_summary(
            torch.zeros(3), torch.zeros(()), torch.zeros(2), torch.zeros(2), torch.zeros(5),
            {"command_modes": modes}, transition, None, total,
            "random_mode_resampling; episodes pooled without mode attribution")
        episode_report = result["keyboard_episode_metrics"]
        self.assertIsNone(episode_report["per_keyboard_mode"])
        self.assertEqual(episode_report["total"], {
            "episodes": 3, "falls": 1, "successes": 1, "timeouts": 1,
        })
        self.assertTrue(all(row["episodes"] is None for row in result["keyboard_mode_metrics"]))
        self.assertIn("pooled", episode_report["attribution"])

    def test_wasd_release_sequence_order_duration_and_validation(self):
        modes = [{"name": name} for name in (
            "forward", "backward", "turn_left", "turn_right", "forward_left",
            "forward_right", "backward_left", "backward_right", "idle")]
        plan = command_sequence_plan(
            "wasd-release-regression", modes, dt=.02, stage_seconds=3.)
        self.assertEqual([row["mode"] for row in plan["stages"]],
                         ["forward", "idle", "backward", "forward_left",
                          "idle", "backward_right", "idle"])
        self.assertEqual([row[0] for row in WASD_RELEASE_REGRESSION],
                         [row["mode"] for row in plan["stages"]])
        self.assertTrue(all(row["steps"] == 150 for row in plan["stages"]))
        self.assertEqual(plan["total_steps"], 1050)
        self.assertIn("key changes preserve the clock", plan["entry_clock_behavior"])
        self.assertIsNone(command_sequence_plan(None, modes))
        self.assertEqual(evaluation_episode_limit(1000, plan["total_steps"], plan), 1051)
        self.assertEqual(evaluation_episode_limit(1200, plan["total_steps"], plan), 1200)
        self.assertEqual(evaluation_episode_limit(1000, 1500, None), 1000)
        for value in (False, 0, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation_episode_limit(value, plan["total_steps"], plan)
        for name, selected_modes, duration, dt in (
            ("unknown", modes, 3., .02),
            ("wasd-release-regression", modes[:-1], 3., .02),
            ("wasd-release-regression", modes, 2.99, .02),
            ("wasd-release-regression", modes, float("nan"), .02),
            ("wasd-release-regression", modes, 3., 0.),
        ):
            with self.subTest(name=name, duration=duration, dt=dt), self.assertRaises(ValueError):
                command_sequence_plan(name, selected_modes, dt=dt, stage_seconds=duration)

    def test_evaluation_selection_allows_only_supported_nonconflicting_layouts(self):
        wasd = ("forward", "idle", "backward", "forward_left", "backward_right")
        keyboard = ("forward", "turn_left", "turn_right")
        self.assertIsNone(validate_command_selection(None, "random", None, "wasd", wasd))
        self.assertIsNone(validate_command_selection(None, "cycle", None, "wasd", wasd))
        self.assertIsNone(validate_command_selection("idle", "random", None, "wasd", wasd))
        plan = validate_command_selection(
            None, "random", "wasd-release-regression", "wasd", wasd)
        self.assertEqual(plan["total_steps"], 1050)
        self.assertIsNone(validate_command_selection(None, "random", None, "keyboard", keyboard))
        self.assertIsNone(validate_command_selection(None, "cycle", None, "keyboard", keyboard))
        self.assertIsNone(validate_command_selection("forward", "random", None, "keyboard", keyboard))
        invalid = (
            ("idle", "cycle", None, "wasd", wasd),
            ("idle", "random", "wasd-release-regression", "wasd", wasd),
            (None, "cycle", "wasd-release-regression", "wasd", wasd),
            (None, "cycle", None, "fixed", ()),
            ("backward", "random", None, "keyboard", keyboard),
            (None, "random", "wasd-release-regression", "keyboard", keyboard),
            (None, "fixed", None, "wasd", wasd),
        )
        for selection in invalid:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                validate_command_selection(*selection)

    def test_sequence_mode_changes_preserve_entry_clock_and_stealth_clearance(self):
        from trackerLab.tracker_env.mdp.supermimic_command import CommandMDP

        sim = CpuCommandSim("stealth", batch=1, clearance_ratio=.92)
        sim.mdp = CommandMDP(
            sim.mdp.field, sim.default_q, sim.qlo, sim.qhi, profile="stealth",
            clearance_ratio=.92, entry_duration_s=4.84, control_mode="wasd")
        sim.set_wasd_keyboard_mode()
        sim.set_fixed_keyboard_mode("forward")
        sim.c[:, 6] = 2.4
        clearance = sim.c[:, 4].clone()
        for mode in ("idle", "backward", "forward_left", "idle",
                     "backward_right", "idle"):
            clock = sim.c[:, 6].clone()
            sim.set_fixed_keyboard_mode(mode)
            torch.testing.assert_close(sim.c[:, 6], clock)
            torch.testing.assert_close(sim.c[:, 4], clearance)
            expected = torch.tensor([
                row[1:4] for row in sim._keyboard_modes() if row[0] == mode])
            torch.testing.assert_close(sim.c[:, 1:4], expected)
            self.assertEqual(sim._fixed_keyboard_mode_name, mode)

    def test_first_attempt_command_history_freezes_with_terminal_pose(self):
        previous = np.array([[.1, 0., 0.], [.2, .1, .4], [-.2, 0., -.4]])
        current = np.array([[.3, 0., 0.], [0., 0., 0.], [.2, .1, .2]])
        active = np.array([True, False, True])
        actual = freeze_recorded_commands(previous, current, active)
        np.testing.assert_array_equal(actual, np.array([
            [.3, 0., 0.], [.2, .1, .4], [.2, .1, .2],
        ]))
        with self.assertRaisesRegex(ValueError, "align"):
            freeze_recorded_commands(previous, current[:2], active)

    def test_keyboard_physical_transition_aggregation_includes_drift_clearance_and_slip(self):
        mode_ids = torch.tensor([0, 0, 1, 2])
        valid = torch.tensor([True, True, True, False])
        tracking = torch.tensor([True, False, True, False])
        velocity = torch.tensor([[.2, .1, 0.], [.4, -.1, 0.], [0., 0., 0.], [9., 9., 0.]])
        angular = torch.tensor([[0., 0., .1], [0., 0., -.1], [0., 0., .4], [9., 0., 0.]])
        command = torch.tensor([
            [0., .25, 0., 0., .7, 0., 0.], [0., .25, 0., 0., .7, 0., 0.],
            [0., 0., 0., .55, .7, 0., 0.], [0., 0., 0., 0., .7, 0., 0.],
        ])
        clearance, slip = torch.tensor([.69, .71, .68, 9.]), torch.tensor([.03, .05, .07, 9.])
        support = torch.tensor([True, False, True, False])
        sums = keyboard_transition_sums(
            mode_ids, valid, tracking, velocity, angular, command,
            clearance, slip, support, 3)
        torch.testing.assert_close(sums[0, :2], torch.tensor([2., 1.]))
        torch.testing.assert_close(sums[0, 2:5], torch.tensor([.6, 0., 0.]))
        torch.testing.assert_close(sums[0, 8:12], torch.tensor([.6, 1.4, .08, 1.]))
        torch.testing.assert_close(sums[1, 2:5], torch.tensor([0., 0., .4]))
        torch.testing.assert_close(sums[1, 6:12], torch.tensor([0., 0., 0., .68, .07, 1.]))
        torch.testing.assert_close(sums[0, 12:], torch.tensor([.045, .02]))
        torch.testing.assert_close(sums[1, 12:], torch.tensor([0., .0225]))
        self.assertEqual(sums[2].sum().item(), 0.)
        for invalid_ids in (torch.tensor([0, 0, 1, 3]), torch.tensor([-1, 0, 1, 0])):
            with self.subTest(invalid_ids=invalid_ids), self.assertRaisesRegex(ValueError, "mode ID"):
                keyboard_transition_sums(
                    invalid_ids, valid, tracking, velocity, angular, command,
                    clearance, slip, support, 3)

    def test_evaluator_aggregates_transition_motion_metrics_with_the_validity_mask(self):
        path = REPO / "scripts/rls/supermimic/evaluate_terrain.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        accumulation = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
            and node.target.id == "motion_sum"
        ]
        self.assertEqual(len(accumulation), 1)
        namespace = {
            "torch": torch, "motion_sum": torch.zeros(2),
            "valid": torch.tensor([True, False, True]),
            "tr": {"command_metrics": {
                "command_speed_mps": torch.tensor([.3, 99., .5]),
                "commanded_progress_mps": torch.tensor([0., 99., -.2]),
            }},
        }
        exec(compile(ast.Module(body=accumulation, type_ignores=[]), str(path), "exec"), namespace)
        torch.testing.assert_close(namespace["motion_sum"], torch.tensor([.8, -.2]))
        result = command_evaluation_summary(
            torch.zeros(3), torch.tensor(2.), namespace["motion_sum"],
            torch.zeros(2), torch.zeros(5), None)
        self.assertAlmostEqual(result["motion_metrics"]["mean_requested_planar_speed_mps"], .4)
        self.assertAlmostEqual(result["motion_metrics"]["mean_projected_velocity_along_command_mps"], -.1)


if __name__ == "__main__":
    unittest.main()

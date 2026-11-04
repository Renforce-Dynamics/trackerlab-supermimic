"""CPU checks for checkpoint-calibrated evaluation rewards and motion evidence."""

import ast
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

from evaluate_terrain import attach_locomotion_quality, command_evaluation_summary
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
    def test_evaluation_uses_saved_calibration_and_includes_penalty_in_episode_return(self):
        sim = CpuCommandSim(max_episode_steps=1)
        kin = EvaluationKinematics()
        motors = [(20., 12., 25., .5, .01)] * 4
        saved_height = torch.tensor([.01, .03])
        with patch.object(LocomotionQuality, "calibrate", side_effect=AssertionError("reference read")):
            quality = attach_locomotion_quality(
                sim, kin, sim.mdp.field, motors, {"quality_sole_height": saved_height})
        torch.testing.assert_close(quality.sole_height, saved_height)
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

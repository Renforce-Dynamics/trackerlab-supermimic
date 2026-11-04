"""Contracts for the reference-free, motion-style training campaign."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from supervise_style import build_plan, command_quality, resolve_styles, recover_jobs, checkpoint_iteration
from train_terrain import parse_args
from trackerTask.supermimic.registry import get_task


class StyleQueueTests(unittest.TestCase):
    def test_plan_is_twenty_nine_jobs_and_ten_thousand_iterations(self):
        plan = build_plan()
        self.assertEqual(len(plan), 29)
        self.assertEqual(len({row["task"] for row in plan}), 29)
        self.assertTrue(all(row["iterations"] == 10000 for row in plan))
        self.assertEqual([row["profile"] for row in plan[:6]], ["velocity"] * 6)
        self.assertEqual(sum(row["profile"] == "stealth" for row in plan), 5)
        self.assertFalse(any(row["robot"] == "go2" and row["style"] == "stealth" for row in plan))
        for row in plan[6:]:
            self.assertEqual(row["depends_on"], f"locomotion/{row['robot']}/velocity")

    def report(self):
        return {"lanes": [{"episodes": 10, "success_fraction": .95, "fall_fraction": .05}] * 6,
                "command_metrics": {"velocity_rmse_scale_m_s": .15,
                                    "yaw_rate_rmse_rad_s": .2},
                "style_metrics": {"mean_reward_rate": .8}}

    def test_gate_requires_both_held_out_physics_and_command_errors(self):
        good = self.report()
        self.assertTrue(command_quality([good, good], "velocity")["passed"])
        self.assertFalse(command_quality([good], "velocity")["passed"])
        bad = copy.deepcopy(good)
        bad["command_metrics"]["velocity_rmse_scale_m_s"] = 1.
        self.assertFalse(command_quality([good, bad], "velocity")["passed"])
        bad = copy.deepcopy(good)
        bad["lanes"][0] = dict(bad["lanes"][0], episodes=0)
        self.assertFalse(command_quality([bad, good], "parkour")["passed"])
        bad = copy.deepcopy(good)
        bad["style_metrics"]["mean_reward_rate"] = .1
        self.assertFalse(command_quality([bad, good], "velocity")["passed"])
        for invalid in (float("nan"), float("inf"), -1., 2., None):
            bad = copy.deepcopy(good)
            bad["style_metrics"]["mean_reward_rate"] = invalid
            for reports in ([good, bad], [bad, good]):
                self.assertFalse(command_quality(reports, "velocity")["passed"])

    def test_style_map_resolves_relative_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "style-map.json"
            path.write_text(json.dumps({"version": 1, "robots": {
                "g1": {"walk": "g1/walk.json", "stealth": None},
            }}))
            actual = resolve_styles(path)
            self.assertEqual(actual["g1"]["walk"], root / "g1/walk.json")
            self.assertNotIn("stealth", actual["g1"])
            path.write_text('{"version": 2, "robots": {}}')
            with self.assertRaises(ValueError):
                resolve_styles(path)

    def test_command_presets_parse_without_requiring_trajectory_goal(self):
        for profile in ("velocity", "stealth"):
            args = parse_args([
                "--task", f"locomotion/g1/{profile}", "--run-dir", "unused-style-probe",
                "--config", str(REPO / f"scripts/rls/supermimic/configs/command_{profile}.json"),
                "--amp-motion-config", "data/style/train.json",
            ])
            self.assertEqual(args.iterations, 10000)
            self.assertEqual(args.amp_weight, .5)
            self.assertIsNone(args.gait_warm_start)
            self.assertIsNone(args.gait_quality)

    def test_every_new_task_factory_is_reference_free_and_simulator_free(self):
        for job in build_plan():
            cfg = get_task(job["task"]).make_env_cfg()
            self.assertEqual(cfg.task_id, job["task"])
            self.assertEqual(cfg.robot_name, job["robot"])
            self.assertEqual(cfg.profile, job["profile"])
            self.assertFalse(cfg.runtime_reference_motion)
            self.assertTrue(cfg.style_reward_enabled)
            self.assertEqual(cfg.goal_dim, 12)
            self.assertEqual(cfg.simulator, "mujoco-warp")

    def test_restart_keeps_completed_work_and_recovers_failed_export(self):
        jobs = [
            {"task": "a", "phase": "failed"},
            {"task": "b", "phase": "completed", "quality_passed": True},
            {"task": "c", "phase": "completed", "quality_passed": True},
            {"task": "d", "phase": "completed", "quality_passed": False},
        ]
        status = {"jobs": jobs, "qualified": {"c": {}}}
        recover_jobs(status)
        self.assertEqual([job["phase"] for job in jobs],
                         ["pending", "exporting", "completed", "completed"])

    def test_atomic_checkpoint_progress_survives_missing_summary(self):
        import torch
        from trackerLab.tracker_env.mdp.supermimic_command import COMMAND_VERSION
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save({"training_stage": "command-style", "objective_version": COMMAND_VERSION,
                        "task": "locomotion/g1/velocity", "iteration": 10000}, path)
            self.assertEqual(checkpoint_iteration(path, "locomotion/g1/velocity"), 10000)
            with self.assertRaises(ValueError):
                checkpoint_iteration(path, "locomotion/go2/velocity")


if __name__ == "__main__":
    unittest.main()

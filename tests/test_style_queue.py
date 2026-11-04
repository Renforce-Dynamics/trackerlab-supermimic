"""Contracts for the reference-free, motion-style training campaign."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from supervise_style import (
    build_plan, command_quality, resolve_styles, recover_jobs, checkpoint_iteration,
    visual_review_passed, visual_review_binding, revalidate_qualifications, dispatch_ready,
)
import supervise_style
from train_terrain import parse_args, validate_resume_command_resampling
from trackerTask.supermimic.registry import get_task


class StyleQueueTests(unittest.TestCase):
    def reviewed_run(self, root, profile="velocity", dependency=None):
        run = root / "g1" / profile / "round_01"
        run.mkdir(parents=True)
        checkpoint, trajectory = run / "last.pt", run / "trajectories.npz"
        checkpoint.write_bytes(b"reviewed actor " + profile.encode())
        trajectory.write_bytes(b"reviewed first attempt " + profile.encode())
        review = {
            "passed": True, "reviewer": "test reviewer", "notes": "reviewed recorded gait",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        }
        (run / "visual_review.json").write_text(json.dumps(review), encoding="utf-8")
        binding = visual_review_binding(run, checkpoint)
        self.assertIsNotNone(binding)
        job = {
            "task": f"locomotion/g1/{profile}", "robot": "g1", "profile": profile,
            "style": "walk" if profile == "velocity" else "parkour",
            "depends_on": dependency, "phase": "completed", "quality_passed": True,
            "round": 1, "iterations": 10000, "run_dir": str(run),
            "checkpoint": str(checkpoint), "visual_review_binding": binding,
        }
        return run, job, {**binding, "quality": str(run / "quality.json")}

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
                "style_metrics": {"mean_reward_rate": .8},
                "gait_metrics": {"support_foot_slip_scale_m_s": .08,
                                 "close_feet_fraction": .02,
                                 "joint_acceleration_squared": 1000.}}

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
        for name in good["gait_metrics"]:
            bad = copy.deepcopy(good)
            bad["gait_metrics"][name] = 1e9
            self.assertFalse(command_quality([good, bad], "velocity")["passed"])

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
            self.assertEqual(args.amp_weight, .25)
            self.assertEqual(args.amp_history, 10)
            self.assertEqual(args.command_resample_steps, (100, 250))
            self.assertIsNone(args.gait_warm_start)
            self.assertIsNone(args.gait_quality)
            fixed = parse_args([
                "--task", "locomotion/g1/velocity", "--run-dir", "fixed-command-probe",
                "--amp-motion-config", "data/style/train.json", "--max-episode-steps", "1000",
                "--command-resample-steps", "1000", "1000",
            ])
            self.assertEqual(fixed.command_resample_steps, [1000, 1000])

    def test_resume_command_resampling_accepts_equal_lists_and_tuples(self):
        for interval in ((100, 250), (1000, 1000)):
            for saved in (interval, list(interval)):
                for requested in (interval, list(interval)):
                    with self.subTest(saved=saved, requested=requested):
                        validate_resume_command_resampling(
                            {"arguments": {"command_resample_steps": saved}},
                            SimpleNamespace(command_resample_steps=requested))

    def test_resume_command_resampling_accepts_legacy_default_only(self):
        checkpoint = {"arguments": {}}
        for interval in ((100, 250), [100, 250]):
            validate_resume_command_resampling(
                checkpoint, SimpleNamespace(command_resample_steps=interval))
        with self.assertRaisesRegex(ValueError, "Resume command_resample_steps differs"):
            validate_resume_command_resampling(
                checkpoint, SimpleNamespace(command_resample_steps=[1000, 1000]))

    def test_resume_command_resampling_rejects_changed_intervals(self):
        for saved, requested in (([1000, 1000], (100, 250)),
                                 ((100, 250), [1000, 1000]),
                                 ([100, 250], (100, 251))):
            with self.subTest(saved=saved, requested=requested):
                with self.assertRaisesRegex(ValueError, "Resume command_resample_steps differs"):
                    validate_resume_command_resampling(
                        {"arguments": {"command_resample_steps": saved}},
                        SimpleNamespace(command_resample_steps=requested))

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
                         ["pending", "awaiting_visual_review", "awaiting_visual_review", "completed"])
        self.assertEqual(status["qualified"], {})

    def test_visual_qualification_binds_review_to_exact_checkpoint_and_trajectory(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint, trajectory = run / "last.pt", run / "trajectories.npz"
            checkpoint.write_bytes(b"current actor")
            trajectory.write_bytes(b"first physical attempt")
            self.assertFalse(visual_review_passed(run, checkpoint))
            review = {"passed": True, "reviewer": "test", "notes": "reviewed recorded gait",
                      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest()}
            (run / "visual_review.json").write_text(json.dumps(review))
            self.assertTrue(visual_review_passed(run, checkpoint))
            checkpoint.write_bytes(b"another actor")
            self.assertFalse(visual_review_passed(run, checkpoint))

    def test_malformed_or_incomplete_visual_review_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, job, entry = self.reviewed_run(root)
            for raw in (b"{", b"[]", b"null", b"\xff", b'{"passed":true}'):
                with self.subTest(raw=raw):
                    (run / "visual_review.json").write_bytes(raw)
                    self.assertFalse(visual_review_passed(run, run / "last.pt"))
                    status = {"jobs": [copy.deepcopy(job)], "qualified": {job["task"]: entry}}
                    self.assertEqual(revalidate_qualifications(status), {job["task"]})
                    self.assertEqual(status["qualified"], {})
                    self.assertEqual(status["jobs"][0]["phase"], "awaiting_visual_review")

    def test_recovery_preserves_current_hash_bound_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            _, job, entry = self.reviewed_run(Path(directory))
            status = {"jobs": [job], "qualified": {job["task"]: copy.deepcopy(entry)}}
            recover_jobs(status)
            self.assertEqual(job["phase"], "completed")
            self.assertEqual(status["qualified"][job["task"]], entry)

    def test_checkpoint_trajectory_or_receipt_change_revokes_qualification(self):
        for artifact in ("last.pt", "trajectories.npz", "visual_review.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                run, job, entry = self.reviewed_run(Path(directory))
                if artifact == "visual_review.json":
                    review = json.loads((run / artifact).read_text())
                    review["notes"] = "a replacement approval receipt"
                    (run / artifact).write_text(json.dumps(review))
                    self.assertTrue(visual_review_passed(run, run / "last.pt"))
                else:
                    (run / artifact).write_bytes(b"changed after approval")
                child = {
                    "task": "locomotion/g1/stealth", "phase": "pending",
                    "depends_on": job["task"],
                }
                status = {"jobs": [job, child], "qualified": {job["task"]: entry}}
                self.assertEqual(revalidate_qualifications(status), {job["task"]})
                self.assertEqual(status["qualified"], {})
                self.assertEqual(job["phase"], "awaiting_visual_review")
                self.assertEqual(child["phase"], "held_for_velocity_quality")
                self.assertTrue((run / "last.pt").is_file())
                self.assertTrue((run / "trajectories.npz").is_file())

    def test_legacy_qualification_requires_a_new_hash_bound_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            _, job, entry = self.reviewed_run(Path(directory))
            entry.pop("visual_review_sha256")
            status = {"jobs": [job], "qualified": {job["task"]: entry}}
            self.assertEqual(revalidate_qualifications(status), {job["task"]})
            self.assertEqual(job["phase"], "awaiting_visual_review")

    def test_revocation_propagates_to_qualified_dependent_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, parent, parent_entry = self.reviewed_run(root)
            _, child, child_entry = self.reviewed_run(root, "stairs", parent["task"])
            status = {"jobs": [parent, child], "qualified": {
                parent["task"]: parent_entry, child["task"]: child_entry,
            }}
            (run / "visual_review.json").unlink()
            self.assertEqual(revalidate_qualifications(status), {parent["task"], child["task"]})
            self.assertEqual(status["qualified"], {})
            self.assertEqual(child["phase"], "held_for_velocity_quality")

    def test_dispatch_rechecks_dependency_and_exact_export_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_run, parent, parent_entry = self.reviewed_run(root)
            child_run, child, _ = self.reviewed_run(root, "stairs", parent["task"])
            status = {"jobs": [parent, child], "qualified": {parent["task"]: parent_entry}}
            self.assertTrue(dispatch_ready(status, child, child_run, "train"))
            self.assertTrue(dispatch_ready(status, child, child_run, "export"))
            review_path = child_run / "visual_review.json"
            review = json.loads(review_path.read_text())
            review["notes"] = "updated while export was being prepared"
            review_path.write_text(json.dumps(review))
            self.assertFalse(dispatch_ready(status, child, child_run, "export"))
            self.assertEqual(child["phase"], "awaiting_visual_review")
            (parent_run / "last.pt").write_bytes(b"unreviewed parent")
            self.assertFalse(dispatch_ready(status, child, child_run, "train"))
            self.assertEqual(child["phase"], "held_for_velocity_quality")
            self.assertNotIn(parent["task"], status["qualified"])

    def test_supervisor_holds_stale_dependency_without_launching_any_child(self):
        for timing in ("recovery", "dispatch"):
            with self.subTest(timing=timing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, parent, entry = self.reviewed_run(root)
                child = {
                    "task": "locomotion/g1/stairs", "robot": "g1", "profile": "stairs",
                    "style": "parkour", "depends_on": parent["task"], "phase": "pending",
                    "round": 1, "iterations": 10000,
                }
                plan = [copy.deepcopy(parent), copy.deepcopy(child)]
                if timing == "recovery":
                    (run / "last.pt").write_bytes(b"unreviewed replacement")
                receipt = root / "campaign.json"
                receipt.write_text(json.dumps({
                    "stage": "supermimic-command-amp-v4", "jobs": [parent, child],
                    "qualified": {parent["task"]: entry},
                }))
                style = root / "train.json"
                style.write_text("{}")
                argv = ["supervise_style.py", "--run-dir", str(root), "--style-map", "unused"]
                fcntl = SimpleNamespace(flock=Mock(), LOCK_EX=1, LOCK_NB=2)

                def check_dispatch(status, selected, directory, kind):
                    if timing == "dispatch":
                        (run / "last.pt").write_bytes(b"replaced immediately before dispatch")
                    return dispatch_ready(status, selected, directory, kind)

                with patch.object(sys, "argv", argv), \
                        patch.object(supervise_style, "REPO", root), \
                        patch.object(supervise_style, "build_plan", return_value=plan), \
                        patch.object(supervise_style, "resolve_styles",
                                     return_value={"g1": {"walk": style, "parkour": style}}), \
                        patch.object(supervise_style, "dispatch_ready", side_effect=check_dispatch), \
                        patch.object(supervise_style.subprocess, "Popen") as launch, \
                        patch.dict(sys.modules, {"fcntl": fcntl}):
                    supervise_style.main()
                launch.assert_not_called()
                saved = json.loads(receipt.read_text())
                self.assertEqual(saved["qualified"], {})
                self.assertEqual(saved["jobs"][1]["phase"], "held_for_velocity_quality")
                self.assertEqual(saved["phase"], "quality_review_required")

    def test_supervisor_rechecks_artifacts_after_export_before_qualification(self):
        for changed_during_export in (False, True):
            with self.subTest(changed=changed_during_export), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run, job, _ = self.reviewed_run(root)
                job["phase"] = "awaiting_visual_review"
                (run / "run_summary.json").write_text(json.dumps({
                    "phase": "completed", "completed_iterations": 10000,
                }))
                for seed in (44, 45):
                    (run / f"evaluation_seed_{seed}.json").write_text(json.dumps(self.report()))
                receipt = root / "campaign.json"
                receipt.write_text(json.dumps({
                    "stage": "supermimic-command-amp-v4", "jobs": [job], "qualified": {},
                }))
                style = root / "train.json"
                style.write_text("{}")
                argv = ["supervise_style.py", "--run-dir", str(root), "--style-map", "unused"]
                fcntl = SimpleNamespace(flock=Mock(), LOCK_EX=1, LOCK_NB=2)

                def export(*args, **kwargs):
                    if changed_during_export:
                        (run / "last.pt").write_bytes(b"actor replaced during export")
                    return SimpleNamespace(pid=123, poll=lambda: 0, returncode=0)

                with patch.object(sys, "argv", argv), \
                        patch.object(supervise_style, "REPO", root), \
                        patch.object(supervise_style, "build_plan", return_value=[copy.deepcopy(job)]), \
                        patch.object(supervise_style, "resolve_styles", return_value={"g1": {"walk": style}}), \
                        patch.object(supervise_style.subprocess, "Popen", side_effect=export) as launch, \
                        patch.dict(sys.modules, {"fcntl": fcntl}):
                    supervise_style.main()
                launch.assert_called_once()
                self.assertIn("export_command_policy.py", launch.call_args.args[0][2])
                saved = json.loads(receipt.read_text())
                if changed_during_export:
                    self.assertEqual(saved["qualified"], {})
                    self.assertEqual(saved["jobs"][0]["phase"], "awaiting_visual_review")
                else:
                    expected = visual_review_binding(run, run / "last.pt")
                    for name, value in expected.items():
                        self.assertEqual(saved["qualified"][job["task"]][name], value)
                    self.assertEqual(saved["phase"], "completed")

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

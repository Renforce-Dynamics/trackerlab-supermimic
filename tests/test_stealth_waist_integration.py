"""Portable waist control and source-bound preparation without a GPU runtime."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from export_command_policy import ConstrainedCommandPolicy, export
import prepare_stealth_waist_entry as prepare
from train_terrain import parse_args
from trackerLab.algorithms.supermimic import networks
from trackerLab.algorithms.supermimic.stealth_upper_body import UpperBodyConstraint
from trackerLab.algorithms.supermimic.stealth_waist_entry import (
    WaistEntryConstraint, compose_entry_waist_constraint,
)
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerLab.tracker_env.mdp.supermimic_command import COMMAND_VERSION
from trackerTask.supermimic.robots import get_robot


class WaistIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        cls.addClassCleanup(setattr, networks, "AMP_DTYPE", networks.AMP_DTYPE)
        torch.set_num_threads(2)
        networks.AMP_DTYPE = None

    def controls(self):
        spec = get_robot("g1")
        arm_names = [name for name in spec.joint_names
                     if any(part in name for part in ("shoulder", "elbow", "wrist"))]
        initial = {name: spec.default_q[spec.joint_names.index(name)] for name in arm_names}
        target = {name: value + .1 for name, value in initial.items()}
        arms = UpperBodyConstraint(
            spec.joint_names, spec.default_q, spec.action_scale, target,
            residual_radius_rad=0., max_joint_acceleration_rad_s2=3.,
            entry_times_s=[0., 4.82], entry_reference_q=[initial, target],
            posture_tolerance_rad=.25)
        waist = WaistEntryConstraint(
            spec.joint_names, spec.default_q, spec.action_scale,
            dict(zip(prepare.WAIST_NAMES, (-.03, -.02, .52))), 1.02)
        return spec, arms, waist

    def test_export_embeds_waist_for_dynamic_times_and_preserves_legs_and_arms(self):
        spec, arms, waist = self.controls()
        actor = TerrainPolicy(283, 12, hist=2, num_joints=29, action_limit=20.).eval()
        with torch.no_grad():
            actor.dec[-1].bias[12:15] = torch.tensor([-10., -5., 4.])
        composite = compose_entry_waist_constraint(arms, waist.to_dict())
        eager = ConstrainedCommandPolicy(actor, composite, 29, entry_duration_s=4.84)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoint.pt"
            torch.save({
                "training_stage": "command-style", "objective_version": COMMAND_VERSION,
                "task": "locomotion/g1/stealth", "robot": "g1", "iteration": 1500,
                "arguments": {"hist": 2, "action_limit": 20., "entry_duration_s": 4.84,
                              "stealth_control_mode": "wasd"},
                "actor": actor.state_dict(),
                "stealth_upper_body_constraint": arms.to_dict(),
                "stealth_waist_constraint": waist.to_dict(),
            }, source)
            before = prepare.digest(source)
            metadata = export(source, root / "export")
            self.assertEqual(before, prepare.digest(source))
            portable = torch.jit.load(str(root / "export/policy.ts"))
            saved = torch.load(root / "export/actor.pt", weights_only=True)
            self.assertEqual(saved["stealth_waist_constraint"], waist.to_dict())
            self.assertEqual(metadata["stealth_waist_constraint"], waist.to_dict())
            self.assertFalse(metadata["runtime_reference_motion"])
            times = torch.tensor([0., .1, .4, .8, 1.0, 1.02, 4.84])
            history = torch.randn(7, 2, 283)
            goals = torch.randn(7, 12)
            goals[:, 6] = times / 4.84
            with torch.no_grad():
                actual = portable(history, goals)
                torch.testing.assert_close(actual, eager(history, goals))
                raw = actor(history, goals)
                arms_only = ConstrainedCommandPolicy(actor, arms, 29, 4.84)(history, goals)
                torch.testing.assert_close(actual[:, :12], raw[:, :12])
                torch.testing.assert_close(actual[:, 15:], arms_only[:, 15:])
                target = waist.reference_at(times + .02)
                indices = waist.waist_indices
                torch.testing.assert_close(
                    waist.default_q[indices] + actual[:, indices] * waist.action_scale[indices],
                    target)

    def test_forward_preparation_uses_an_observed_waist_pose_and_derived_ramp(self):
        spec = get_robot("g1")
        rows = np.tile(np.r_[np.zeros(3), (0., 0., 0., 1.), spec.default_q], (5, 1))
        columns = 7 + np.array([spec.joint_names.index(name) for name in prepare.WAIST_NAMES])
        poses = np.array([[-.2, -.1, .52], [.02, .01, .52], [.03, .02, .52],
                          [.04, .03, .52], [.2, .1, .52]])
        rows[:, columns] = poses
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            clip = root / "forward.csv"
            np.savetxt(clip, rows, delimiter=",")
            record = {"clip_role": "forward", "split": "train", "file": clip.name,
                      "sha256": prepare.digest(clip), "frames": 5,
                      "source_file": "aiming2_subject5.bvh", "source_frame_range": [2830, 2891]}
            with patch.object(prepare, "verify_pool", return_value={"files": [record]}):
                document = prepare.prepare(root)
            frame = document["source"]["converted_frame"]
            self.assertEqual(frame, 2)
            np.testing.assert_array_equal(list(document["constraint"]["reference_q"].values()), poses[frame])
            duration = document["constraint"]["ramp_duration_s"]
            self.assertEqual(duration, 1.02)
            required = max(1.875 * .52 / 1.2, math.sqrt((10 / math.sqrt(3)) * .52 / 3))
            self.assertGreaterEqual(duration, required)
            self.assertLess(duration - required, .02)
            with patch.object(prepare, "verify_pool", return_value={"files": [record, record]}):
                with self.assertRaisesRegex(ValueError, "one explicitly"):
                    prepare.prepare(root)
            record["sha256"] = "0" * 64
            with patch.object(prepare, "verify_pool", return_value={"files": [record]}):
                with self.assertRaisesRegex(ValueError, "changed"):
                    prepare.prepare(root)

    def test_waist_cli_is_opt_in_and_visual_pilot_keeps_legacy_style_and_rewards(self):
        base = ["--task", "locomotion/g1/stealth", "--run-dir", "unused"]
        self.assertIsNone(parse_args(base).waist_entry_reference)
        self.assertEqual(parse_args([*base, "--waist-entry-reference", "source.json"]).waist_entry_reference,
                         Path("source.json"))
        config = REPO / "scripts/rls/supermimic/configs/command_stealth_visual_waist.json"
        args = parse_args([*base, "--config", str(config)])
        self.assertEqual(args.style_feature_mode, "postural_end_effectors")
        self.assertEqual(args.wasd_reward_version, 2)
        self.assertEqual(args.iterations, 1500)
        self.assertEqual(args.wm_only_iters, 500)
        self.assertEqual(args.upper_body_residual, 0.)
        self.assertTrue(args.freeze_actor_normalization)


if __name__ == "__main__":
    unittest.main()

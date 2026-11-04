"""Gait qualification contracts independent of GPU simulation."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from trackerTask.supermimic.gait import FOOT_BODIES, QUALITY_LIMITS, quality_passes
from trackerTask.supermimic.robots import get_robot
from trackerLab.tracker_env.mdp.supermimic_gait import GaitMDP
from supervise_gait import candidate_score
from train_terrain import load_gait_foundation
from trackerTask.supermimic.registry import TASK_REGISTRY


class GaitTests(unittest.TestCase):
    def test_gait_tasks_registered_for_all_six_robots(self):
        for robot in ("g1", "go2", "a3", "t1", "bumi", "pi_plus"):
            entry = TASK_REGISTRY[f"gait/{robot}"]
            self.assertEqual(entry.robot, robot)
            self.assertEqual(entry.runtime_task, "gait-foundation")

    def test_feet_belong_to_every_robot_tracking_contract(self):
        for robot, feet in FOOT_BODIES.items():
            spec = get_robot(robot)
            self.assertTrue(set(feet).issubset(spec.tracked_bodies), robot)

    def test_every_quality_measure_is_required(self):
        good = dict(QUALITY_LIMITS)
        good["full_clip_survival"] = 1.
        self.assertTrue(quality_passes(good))
        for name in good:
            missing = good.copy()
            missing.pop(name)
            self.assertFalse(quality_passes(missing), name)
            bad = good.copy()
            bad[name] = .5 if name == "full_clip_survival" else good[name] * 2
            self.assertFalse(quality_passes(bad), name)
            bad[name] = float("nan")
            self.assertFalse(quality_passes(bad), name)

    def test_quality_selection_prefers_motion_quality_over_survival_alone(self):
        qualified = {"quality_passed": True, "metrics": {
            "full_clip_survival": .99, "foot_position_rmse_scale_m": .03, "joint_rmse_rad": .1}}
        ugly = {"quality_passed": False, "metrics": {
            "full_clip_survival": 1., "foot_position_rmse_scale_m": .3, "joint_rmse_rad": 1.}}
        self.assertGreater(candidate_score(qualified), candidate_score(ugly))

    def test_foot_point_velocity_includes_the_rotating_calf_offset(self):
        mdp = object.__new__(GaitMDP)
        mdp.feet = torch.tensor([0])
        mdp.foot_offset = torch.tensor([0., 0., -1.])
        p = torch.zeros(1, 1, 3)
        R = torch.eye(3).reshape(1, 1, 3, 3)
        velocity = torch.zeros_like(p)
        angular = torch.tensor([[[0., 1., 0.]]])
        position, speed = mdp.foot_state((p, R, velocity, angular))
        torch.testing.assert_close(position, torch.tensor([[[0., 0., -1.]]]))
        torch.testing.assert_close(speed, torch.tensor([[[-1., 0., 0.]]]))

    def test_support_slip_and_joint_error_keep_policy_gradients(self):
        mdp = object.__new__(GaitMDP)
        mdp.scale = 1.
        mdp.feet = torch.tensor([0])
        mdp.foot_offset = torch.zeros(3)
        mdp.reference_stance = torch.ones(1, 1, dtype=torch.bool)
        mdp.reference_feet = torch.tensor([[[0., 0., 0.]]])
        mdp.motion = SimpleNamespace(root_p=torch.tensor([[0., 0., 1.]]),
                                     root_R=torch.eye(3)[None], q=torch.zeros(1, 2))
        root = torch.tensor([[2., 3., 1.]])
        velocity = torch.tensor([[[.2, 0., 0.]]], requires_grad=True)
        joint = torch.ones(1, 2, requires_grad=True)
        body = (torch.tensor([[[2., 3., 0.]]]), torch.eye(3).reshape(1, 1, 3, 3),
                velocity, torch.zeros(1, 1, 3))
        state = (root, torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3),
                 joint, torch.zeros(1, 2))
        errors = mdp.gait_errors(state, torch.zeros(1, dtype=torch.long), body)
        self.assertEqual(errors["foot_tracking"].item(), 0)
        (errors["stance_slip"] + errors["joint_tracking"]).sum().backward()
        self.assertGreater(velocity.grad[0, 0, 0].item(), 0)
        self.assertTrue((joint.grad > 0).all())

    def test_only_qualified_same_robot_gait_transfers_motion_policy_and_dynamics(self):
        class Actor:
            hist, num_joints, scan_dim, obs_dim = 2, 2, 4, 25
            def __init__(self):
                self.state = {
                    "obs_norm.mean": torch.zeros(25), "obs_norm.var": torch.ones(25),
                    "obs_norm.count": torch.zeros(()),
                    "dec.0.weight": torch.zeros(3, 32 + 53 * 2),
                    "dec.0.bias": torch.zeros(3), "dec.2.weight": torch.zeros(2, 3),
                    "enc.0.weight": torch.zeros(4, 2),
                }
            def state_dict(self):
                return self.state
            def load_state_dict(self, state):
                self.state = state

        class World:
            def __init__(self):
                self.field = SimpleNamespace(model_indices=torch.arange(54))
                self.net = SimpleNamespace(state_dict=lambda: {"layer": None})
                self.state = {"net.layer": torch.zeros(2), "terrain_norm.mean": torch.zeros(54),
                              "terrain_adapter.0.weight": torch.zeros(2, 54)}
            def state_dict(self):
                return self.state
            def load_state_dict(self, state, strict=False):
                self.loaded = state
                return (["terrain_norm.mean", "terrain_adapter.0.weight"], [])

        actor, world = Actor(), World()
        source_actor = {
            "obs_norm.mean": torch.arange(18.).clone(), "obs_norm.var": torch.ones(18) * 2,
            "obs_norm.count": torch.tensor(99.),
            "dec.0.weight": torch.arange(3 * (32 + 18 * 2.)).reshape(3, 68),
            "dec.0.bias": torch.ones(3), "dec.2.weight": torch.ones(2, 3),
            "enc.0.weight": torch.ones(4, 2),
        }
        source_world = {"net.layer": torch.ones(2)}
        checkpoint = {"args": {"task": "g1", "gait_foundation_version": 1, "hist": 2},
                      "iteration": 2000, "pi": source_actor, "wm": source_world}
        quality = {"quality_passed": True, "selection": {"quality_passed": True}, "held_out_tests": [
            {"quality_passed": True}, {"quality_passed": True}]}
        receipt = load_gait_foundation(actor, world, checkpoint, quality, "g1")
        self.assertEqual(receipt["source_iteration"], 2000)
        self.assertTrue(receipt["critic_reset"])
        self.assertTrue(receipt["optimizers_reset"])
        self.assertTrue(torch.equal(actor.state["dec.0.weight"][:, 32:32 + 12],
                                    source_actor["dec.0.weight"][:, 32:32 + 12]))
        self.assertTrue(torch.all(actor.state["dec.0.weight"][:, 32 + 12:32 + 15] == 0))
        self.assertTrue(torch.equal(actor.state["dec.0.weight"][:, 32 + 15:32 + 21],
                                    source_actor["dec.0.weight"][:, 32 + 12:32 + 18]))
        self.assertEqual(actor.state["obs_norm.mean"][12:15].tolist(), [0., 0., 0.])
        self.assertEqual(actor.state["obs_norm.mean"][15:21].tolist(), [12., 13., 14., 15., 16., 17.])
        self.assertTrue(torch.all(actor.state["dec.0.weight"][:, 32 + 21:32 + 53] == 0))
        self.assertTrue(torch.equal(
            actor.state["dec.0.weight"][:, 32 + 53 + 15:32 + 53 + 21],
            source_actor["dec.0.weight"][:, 32 + 18 + 12:32 + 18 + 18],
        ))
        self.assertTrue(torch.all(actor.state["dec.0.weight"][:, 32 + 53 + 12:32 + 53 + 15] == 0))
        self.assertTrue(torch.equal(world.loaded["net.layer"], source_world["net.layer"]))
        quality["held_out_tests"][1]["quality_passed"] = False
        with self.assertRaisesRegex(ValueError, "quality gate"):
            load_gait_foundation(Actor(), World(), checkpoint, quality, "g1")
        with self.assertRaisesRegex(ValueError, "different robot"):
            load_gait_foundation(Actor(), World(), checkpoint,
                                 {"quality_passed": True, "held_out_tests": [{"quality_passed": True}]},
                                 "go2")


if __name__ == "__main__":
    unittest.main()

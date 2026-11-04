"""CPU contracts for the imagined actor's real-data-only motion prior."""

import copy
from contextlib import chdir
from dataclasses import replace
from io import BytesIO
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.amp_style import AMPStyle
from trackerLab.kinematics.supermimic.robot import RobotKinematics
from trackerLab.kinematics.supermimic.rotations import so3_exp
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import generate


class ArticulatedKinematics:
    num_joints = 3
    joint_names = ("hip", "knee", "ankle")
    tracked_idx = (0, 1)
    body_names = ("left_foot", "right_foot")

    def forward(self, p, R, v, w, q, qd):
        left = torch.stack((q[..., 0].sin(), 0.2 + q[..., 1] * 0.1,
                            -0.5 + q[..., 2].cos() * 0.1), -1)
        right = torch.stack((q[..., 1].sin(), -0.2 + q[..., 0] * 0.1,
                             -0.5 + q[..., 2].sin() * 0.1), -1)
        offsets = torch.stack((left, right), -2)
        body_p = p.unsqueeze(-2) + (R.unsqueeze(-3) @ offsets.unsqueeze(-1)).squeeze(-1)
        return (body_p, R.unsqueeze(-3).expand(*R.shape[:-2], 2, 3, 3),
                v.unsqueeze(-2).expand(*v.shape[:-1], 2, 3),
                w.unsqueeze(-2).expand(*w.shape[:-1], 2, 3))


def make_motion(lengths=(5, 4), requires_grad=False):
    n = sum(lengths)
    t = torch.arange(n, dtype=torch.float32)
    states = (
        torch.stack((0.02 * t, -0.01 * t, 0.8 + 0.02 * t.sin()), -1),
        so3_exp(torch.stack((0.05 * t.sin(), 0.1 * t.cos(), 0.03 * t), -1)),
        torch.stack((0.3 + 0.02 * t, 0.1 * t.sin(), 0.02 * t.cos()), -1),
        torch.stack((0.02 * t.sin(), 0.03 * t.cos(), 0.1 + 0.01 * t), -1),
        torch.stack((0.2 * t.sin(), 0.3 * t.cos(), 0.1 * (t + 0.2).sin()), -1),
        torch.stack((0.1 * t.cos(), 0.2 * t.sin(), 0.3 * (t + 0.2).cos()), -1),
    )
    states = tuple(value.detach().requires_grad_(requires_grad) for value in states)
    lens = torch.tensor(lengths, dtype=torch.long)
    starts = lens.cumsum(0) - lens
    data = dict(zip(("root_p", "root_R", "root_v", "root_w", "q", "qd"), states))
    data.update(clip_start=starts, clip_end=starts + lens - 1, clip_len=lens,
                clip_id=torch.repeat_interleave(torch.arange(len(lens)), lens),
                names=[f"clip_{i}" for i in range(len(lens))],
                provenance={"source": "synthetic articulated test fixture"},
                num_frames=n, num_joints=3)
    motion = SimpleNamespace(**data)
    motion.ref_state = lambda frames: tuple(value[frames] for value in states)
    return motion


class AMPStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(73)
        self.kin = ArticulatedKinematics()
        self.motion = make_motion()

    def learner(self, motion=None, **kwargs):
        return AMPStyle(self.kin, motion or self.motion, 0.8,
                        hidden=kwargs.pop("hidden", (32, 16)),
                        replay_capacity=kwargs.pop("replay_capacity", 32), **kwargs)

    def states(self, motion=None):
        motion = motion or self.motion
        return tuple(value.detach().clone() for value in motion.ref_state(torch.tensor([0, 1, 5]))), \
            tuple(value.detach().clone() for value in motion.ref_state(torch.tensor([1, 2, 6])))

    def assert_nonzero_finite(self, value):
        self.assertIsNotNone(value)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertGreater(value.abs().sum().item(), 0)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    @staticmethod
    def transform(state, rotation, translation):
        p, R, v, w, q, qd = state
        rotate = lambda value: (rotation @ value.unsqueeze(-1)).squeeze(-1)
        return rotate(p) + translation, rotation @ R, rotate(v), rotate(w), q, qd

    def test_features_and_reward_are_world_xy_and_yaw_invariant(self):
        learner = self.learner()
        S0, S1 = self.states()
        yaw = so3_exp(torch.tensor([[0., 0., 1.2], [0., 0., -2.7], [0., 0., 0.8]]))
        translation = torch.tensor([[9., -8., 0.], [-20., 30., 0.], [4., 5., 0.]])
        T0, T1 = (self.transform(state, yaw, translation) for state in (S0, S1))
        torch.testing.assert_close(learner.features(S0, S1), learner.features(T0, T1),
                                   atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(learner.reward(S0, S1), learner.reward(T0, T1),
                                   atol=2e-6, rtol=2e-6)
        self.assertEqual(learner.feature_dim, 2 * (13 + 2 * 3 + 3 * 2))
        self.assertEqual(learner.features(S0, S1).shape, (3, learner.feature_dim))

    def test_canonical_fk_matches_root_local_world_fk(self):
        learner = self.learner()
        S0, _ = self.states()
        p, R, *_ = S0
        body_p = self.kin.forward(*S0)[0]
        expected = (R.transpose(-1, -2).unsqueeze(-3)
                    @ (body_p - p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1) / 0.8
        torch.testing.assert_close(learner.frame_features(S0)[..., -6:], expected.flatten(-2))

    def test_pair_features_batch_endpoints_into_one_fk(self):
        learner = self.learner()
        S0, S1 = self.states()
        expected = torch.cat((learner.frame_features(S0), learner.frame_features(S1)), -1)
        original = self.kin.forward
        self.kin.forward = Mock(wraps=original)
        actual = learner.features(S0, S1)
        self.assertEqual(self.kin.forward.call_count, 1)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(learner.pair_features(S0, S1).shape, actual.shape)

    def test_reference_ground_is_zero_and_policy_ground_is_differentiable(self):
        sampler = Mock(side_effect=lambda xy: 0.2 + 0.1 * xy[..., 0])
        learner = self.learner(ground_sampler=sampler)
        sampler.assert_not_called()
        S0, S1 = self.states()
        p = S0[0].clone().requires_grad_()
        state = (p, *S0[1:])
        actual = learner.frame_features(state)
        torch.testing.assert_close(actual[..., 0], (p[..., 2] - 0.2 - 0.1 * p[..., 0]) / 0.8)
        actual[..., 0].sum().backward()
        torch.testing.assert_close(p.grad[:, 0], torch.full((3,), -0.125))
        torch.testing.assert_close(p.grad[:, 2], torch.full((3,), 1.25))
        sampler.reset_mock()
        reference = learner.features(S0, S1, reference=True)
        sampler.assert_not_called()
        torch.testing.assert_close(reference[..., 0], S0[0][..., 2] / 0.8)
        self.assertTrue(bool(torch.isfinite(reference).all()))

    def test_flat_terrain_vertical_shift_preserves_features(self):
        flat = self.learner()
        elevated = self.learner(ground_sampler=lambda xy: xy[..., 0] * 0 + 4)
        S0, S1 = self.states()
        shifted = tuple((state[0] + torch.tensor([0., 0., 4.]), *state[1:]) for state in (S0, S1))
        torch.testing.assert_close(flat.features(S0, S1), elevated.features(*shifted))

    def test_observe_filters_terminals_raw_nonfinite_and_sampler_nonfinite(self):
        S0, S1 = self.states()
        S0 = tuple(value.repeat(2, *([1] * (value.ndim - 1))) for value in S0)
        S1 = tuple(value.repeat(2, *([1] * (value.ndim - 1))) for value in S1)
        S1[4][2, 0] = float("nan")
        S0[0][3, 0] = float("inf")
        sampler = Mock(side_effect=lambda xy: torch.where(
            xy[..., 0] == self.motion.root_p[1, 0], float("nan"), 0.))
        learner = self.learner(ground_sampler=sampler)
        count_before = learner.norm.count.item()
        accepted = learner.observe(S0, S1, valid=torch.tensor([True, False, True, True, False, True]))
        self.assertEqual(accepted, 1)
        self.assertEqual(learner.replay_size, 1)
        self.assertEqual(learner.norm.count.item(), count_before + 2)
        self.assertTrue(bool(torch.isfinite(sampler.call_args.args[0]).all()))
        self.assertTrue(bool(torch.isfinite(learner._replay[:learner.replay_size]).all()))

    def test_empty_observations_do_not_change_normalization(self):
        learner = self.learner()
        S0, S1 = self.states()
        before = copy.deepcopy(learner.norm.state_dict())
        self.assertEqual(learner.observe(S0, S1, valid=torch.zeros(3, dtype=torch.bool)), 0)
        empty = tuple(value[:0] for value in S0)
        self.assertEqual(learner.observe(empty, empty), 0)
        self.assert_tree_equal(before, learner.norm.state_dict())
        metrics = learner.update()
        self.assertEqual(metrics["updates"], 0)
        self.assertEqual(metrics["loss"], 0.)

    def test_observe_rejects_imagined_autograd_graphs(self):
        learner = self.learner()
        S0, S1 = self.states()
        action = torch.zeros(3, 3, requires_grad=True)
        imagined = (*S1[:4], S1[4] + action, S1[5])
        with self.assertRaisesRegex(ValueError, "detached REAL"):
            learner.observe(S0, imagined)
        self.assertEqual(learner.replay_size, 0)

    def test_replay_wrap_and_large_insert_preserve_latest_real_pairs(self):
        learner = self.learner(replay_capacity=4)
        S0, S1 = self.states()
        features = learner.features(S0, S1)
        learner.observe(S0, S1)
        learner.observe(S0, S1)
        self.assertEqual(learner.replay_size, 4)
        self.assertEqual(learner._replay_cursor, 2)
        torch.testing.assert_close(learner._replay,
                                   torch.stack((features[1], features[2], features[2], features[0])))
        long0 = tuple(value.repeat(3, *([1] * (value.ndim - 1))) for value in S0)
        long1 = tuple(value.repeat(3, *([1] * (value.ndim - 1))) for value in S1)
        self.assertEqual(learner.observe(long0, long1), 9)
        self.assertEqual(learner._replay_cursor, 0)
        torch.testing.assert_close(learner._replay, learner.features(long0, long1)[-4:])
        snapshot = learner._replay.clone()
        long0[4].add_(100.)
        torch.testing.assert_close(learner._replay, snapshot, atol=0, rtol=0)
        self.assertIsNone(learner._replay.grad_fn)

    def test_expert_pairs_never_cross_clips_including_single_frame_clips(self):
        motion = make_motion((1, 2, 1, 4))
        learner = self.learner(motion)
        expected = torch.tensor([1, 4, 5, 6])
        torch.testing.assert_close(learner._expert_starts, expected)
        features, f0, f1 = learner.sample_expert(4096, return_indices=True)
        torch.testing.assert_close(f1, f0 + 1)
        torch.testing.assert_close(motion.clip_id[f0], motion.clip_id[f1])
        self.assertEqual(set(f0.tolist()), set(expected.tolist()))
        torch.testing.assert_close(features, learner.features(motion.ref_state(f0),
                                                             motion.ref_state(f1), reference=True))
        self.assertFalse(features.requires_grad)

    def test_rejects_empty_nonfinite_and_invalid_clip_experts(self):
        with self.assertRaisesRegex(ValueError, "Empty experts"):
            self.learner(make_motion((1, 1)))
        empty = make_motion((0,))
        with self.assertRaisesRegex(ValueError, "Empty experts"):
            self.learner(empty)
        for field in ("root_p", "root_R", "root_v", "root_w", "q", "qd"):
            with self.subTest(field=field):
                motion = make_motion()
                getattr(motion, field)[-1].fill_(float("nan"))
                with self.assertRaisesRegex(ValueError, "Nonfinite expert"):
                    self.learner(motion)
        motion = make_motion()
        motion.clip_id[4] = 1
        with self.assertRaisesRegex(ValueError, "clip IDs"):
            self.learner(motion)
        motion = make_motion()
        motion.clip_start[1] -= 1
        with self.assertRaisesRegex(ValueError, "clip bounds"):
            self.learner(motion)

    def test_full_history_windows_keep_clip_identity_and_chronological_gradients(self):
        learner = self.learner(history_steps=3)
        features, starts, ends = learner.sample_expert(128, return_indices=True)
        self.assertEqual(features.shape, (128, 3 * learner.frame_dim))
        self.assertTrue(torch.equal(self.motion.clip_id[starts], self.motion.clip_id[ends]))
        self.assertTrue(torch.all(ends - starts == 2))

        history = torch.zeros(2, 3, learner.frame_dim)
        following = torch.arange(4 * 2 * learner.frame_dim, dtype=torch.float32)
        following = following.reshape(4, 2, learner.frame_dim).requires_grad_()
        windows = learner.rollout_features(history, following)
        self.assertEqual(windows.shape, (4, 2, 3 * learner.frame_dim))
        expected = torch.cat((history[0, 1:], following[:1, 0]), 0)
        torch.testing.assert_close(windows[0, 0], expected.flatten())
        windows.sum().backward()
        self.assertTrue(bool(torch.isfinite(following.grad).all()))
        self.assertGreater(following.grad.abs().sum().item(), 0)

    def test_real_history_observation_filters_invalid_windows(self):
        learner = self.learner(history_steps=3, replay_capacity=4)
        frames = torch.randn(3, 3, learner.frame_dim)
        accepted = learner.observe_window(frames, torch.tensor([True, False, True]))
        self.assertEqual(accepted, 2)
        self.assertEqual(learner.replay_size, 2)
        self.assertEqual(learner._observed, 2)
        with self.assertRaisesRegex(ValueError, "boolean validity"):
            learner.observe_window(frames, torch.ones(3))

    def test_motion_prior_can_exclude_standing_windows_without_mutating_source(self):
        motion = make_motion((20, 20))
        motion.root_v[:10, 0] = 0.
        motion.root_v[10:20, 0] = .3
        original = motion.root_v.clone()
        learner = self.learner(motion, history_steps=10, minimum_forward_speed=.15)
        _, starts, ends = learner.sample_expert(128, return_indices=True)
        self.assertTrue(bool((starts >= 8).all()))
        self.assertTrue(bool((ends - starts == 9).all()))
        torch.testing.assert_close(motion.root_v, original)

    def test_update_trains_discriminator_without_expert_or_state_gradients(self):
        motion = make_motion(requires_grad=True)
        learner = self.learner(motion)
        S0, S1 = self.states(motion)
        S1[4].add_(0.6)
        learner.observe(S0, S1)
        before = copy.deepcopy(learner.discriminator.state_dict())
        norm_before = copy.deepcopy(learner.norm.state_dict())
        with torch.no_grad():
            metrics = learner.update(batch_size=16, updates=3)
        self.assertEqual(metrics["updates"], 3)
        for name in ("loss", "expert_score", "policy_score", "gradient_penalty"):
            self.assertTrue(math.isfinite(metrics[name]), name)
        self.assertGreater(metrics["gradient_penalty"], 0)
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in learner.discriminator.state_dict().items()))
        for value in motion.ref_state(torch.arange(motion.num_frames)):
            self.assertFalse(value.is_leaf)
        for name in ("root_p", "root_R", "root_v", "root_w", "q", "qd"):
            self.assertIsNone(getattr(motion, name).grad)
        self.assertIsNone(learner._expert_frames.grad_fn)
        self.assertFalse(learner._expert_frames.requires_grad)
        self.assert_tree_equal(norm_before, learner.norm.state_dict())
        self.assertTrue(learner.optimizer.state)

    def test_frozen_discriminator_preserves_reward_to_joint_and_action_gradients(self):
        learner = self.learner()
        S0, S1 = self.states()
        q = S0[4].clone().requires_grad_()
        action = torch.full_like(q, 0.05, requires_grad=True)
        current, following = (*S0[:4], q, S0[5]), (*S1[:4], q + action * 0.3, S1[5])
        norm_before = copy.deepcopy(learner.norm.state_dict())
        count_before = learner.replay_size
        with learner.freeze():
            self.assertTrue(all(not parameter.requires_grad for parameter in learner.parameters()))
            reward = learner.reward(current, following)
            self.assertTrue(bool(((reward > 0) & (reward <= 1)).all()))
            reward.sum().backward()
            self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))
        self.assert_nonzero_finite(q.grad)
        self.assert_nonzero_finite(action.grad)
        self.assertTrue(all(parameter.requires_grad for parameter in learner.parameters()))
        self.assertEqual(count_before, learner.replay_size)
        self.assert_tree_equal(norm_before, learner.norm.state_dict())
        learner.requires_grad_(False)
        q2 = q.detach().requires_grad_()
        learner.reward((*S0[:4], q2, S0[5]), S1).sum().backward()
        self.assert_nonzero_finite(q2.grad)
        with self.assertRaisesRegex(RuntimeError, "Unfreeze"):
            learner._replay_size = 1
            learner.update(batch_size=4)

    def test_freeze_restores_flags_after_nested_context_and_exception(self):
        learner = self.learner()
        first = next(learner.discriminator.parameters())
        first.requires_grad_(False)
        flags = [parameter.requires_grad for parameter in learner.parameters()]
        with self.assertRaisesRegex(RuntimeError, "test"):
            with learner.freeze():
                with learner.freeze():
                    self.assertTrue(all(not p.requires_grad for p in learner.parameters()))
                self.assertTrue(all(not p.requires_grad for p in learner.parameters()))
                raise RuntimeError("test")
        self.assertEqual([parameter.requires_grad for parameter in learner.parameters()], flags)

    def test_reward_is_lsgan_style_rate_without_control_dt(self):
        learner = self.learner(hidden=())
        layer = learner.discriminator[-1]
        S0, S1 = self.states()
        with torch.no_grad():
            layer.weight.zero_()
            for score, expected in ((1., 1.), (-1., 0.), (0., 0.75), (4., 0.)):
                layer.bias.fill_(score)
                torch.testing.assert_close(learner.reward(S0, S1), torch.full((3,), expected))

    def test_lsgan_targets_and_expert_gradient_penalty(self):
        learner = self.learner(hidden=(), lr=1e-8, gradient_penalty_weight=3.)
        S0, S1 = self.states()
        learner.observe(S0, S1)
        layer = learner.discriminator[-1]
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.fill_(0.25)
        metrics = learner.update(batch_size=4)
        self.assertAlmostEqual(metrics["loss"], 0.5 * ((0.25 - 1) ** 2 + (0.25 + 1) ** 2))
        self.assertEqual(metrics["gradient_penalty"], 0)
        with torch.no_grad():
            layer.weight.fill_(0.02)
        metrics = learner.update(batch_size=4)
        self.assertAlmostEqual(metrics["gradient_penalty"], learner.feature_dim * 0.02 ** 2, places=6)

    def test_ten_frame_relu_prior_roundtrip_and_zero_gradient_penalty(self):
        motion = make_motion((12, 13))
        learner = self.learner(
            motion, history_steps=10, activation="ReLU", optimizer_name="AdamW",
            gradient_penalty_scope="both", gradient_penalty_weight=5.)
        features, start, end = learner.sample_expert(8, return_indices=True)
        self.assertTrue(torch.all(end - start == 9))
        self.assertTrue(torch.equal(motion.clip_id[start], motion.clip_id[end]))
        learner.observe_window(features.reshape(8, 10, learner.frame_dim), torch.ones(8, dtype=torch.bool))
        with torch.no_grad():
            for parameter in learner.discriminator.parameters():
                parameter.zero_()
        result = learner.update(batch_size=4)
        self.assertEqual(result["gradient_penalty"], 0.)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in learner.parameters()))
        saved = learner.checkpoint()
        self.kin.device = "cpu"
        scorer = AMPStyle.for_scoring(self.kin, saved)
        torch.testing.assert_close(scorer.reward_from_features(features),
                                   learner.reward_from_features(features))
        for name in ("activation", "hidden", "optimizer_name", "history_steps",
                     "gradient_penalty_scope", "weight_decay_coefficient"):
            changed = copy.deepcopy(saved)
            changed.pop(name)
            with self.assertRaisesRegex(ValueError, "metadata|history|weight_decay|penalty"):
                AMPStyle.for_scoring(self.kin, changed)
        changed = copy.deepcopy(saved)
        changed["hidden"] = [99]
        with self.assertRaisesRegex(ValueError, "hidden"):
            AMPStyle.for_scoring(self.kin, changed)

    def test_combined_gradient_penalty_reuses_score_forward_for_lsgan_terms(self):
        motion = make_motion((12, 13))
        learner = self.learner(
            motion, history_steps=10, gradient_penalty_scope="both",
            gradient_penalty_weight=5.)
        features = learner.sample_expert(8).reshape(8, 10, learner.frame_dim)
        learner.observe_window(features, torch.ones(8, dtype=torch.bool))
        with patch.object(learner, "_score", wraps=learner._score) as score:
            metrics = learner.update(batch_size=4)
        self.assertEqual(score.call_count, 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))

    def test_normalization_is_bounded_shared_and_differentiable(self):
        learner = self.learner()
        features = torch.full((2, learner.feature_dim), 1e30)
        normalized = learner._normalize(features)
        self.assertTrue(bool(torch.isfinite(normalized).all()))
        self.assertLessEqual(normalized.abs().max().item(), learner.NORM_LIMIT)
        torch.testing.assert_close(normalized[:, :learner.frame_dim], normalized[:, learner.frame_dim:])
        features = learner.sample_expert(3).requires_grad_()
        learner._normalize(features).sum().backward()
        self.assert_nonzero_finite(features.grad)
        S0, S1 = self.states()
        S1[4].fill_(1e30)
        self.assertEqual(learner.observe(S0, S1), 3)
        self.assertTrue(bool(torch.isfinite(learner.norm.var).all()))
        self.assertTrue(bool(torch.isfinite(learner.reward(S0, S1)).all()))
        self.assertTrue(math.isfinite(learner.update(batch_size=4)["loss"]))

    def test_checkpoint_roundtrip_restores_adam_norm_reward_and_continuation(self):
        learner = self.learner(lr=3e-4)
        S0, S1 = self.states()
        learner.observe(S0, S1)
        learner.update(batch_size=8, updates=2)
        checkpoint = learner.checkpoint()
        stream = BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True, map_location="cpu")
        restored = self.learner(lr=9e-3)
        restored.observe(S0, S1)
        receipt = restored.restore(saved)
        self.assertEqual(receipt, {"replay_restored": False, "replay_refill_required": True})
        self.assertFalse(saved["replay_restored"])
        self.assertEqual(restored.replay_size, 0)
        self.assertEqual(restored.update()["updates"], 0)
        self.assert_tree_equal(restored.discriminator.state_dict(), learner.discriminator.state_dict())
        self.assert_tree_equal(restored.norm.state_dict(), learner.norm.state_dict())
        self.assert_tree_equal(restored.optimizer.state_dict(), learner.optimizer.state_dict())
        self.assertEqual(restored.optimizer.param_groups[0]["lr"], 3e-4)
        for parameter, state in restored.optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(state[key].device, parameter.device)
        torch.testing.assert_close(restored.reward(S0, S1), learner.reward(S0, S1), atol=0, rtol=0)
        learner.restore(saved)
        for model in (learner, restored):
            model.observe(S0, S1)
        torch.manual_seed(991)
        expected = learner.update(batch_size=8, updates=2)
        torch.manual_seed(991)
        actual = restored.update(batch_size=8, updates=2)
        self.assertEqual(actual, expected)
        self.assert_tree_equal(restored.discriminator.state_dict(), learner.discriminator.state_dict())
        self.assert_tree_equal(restored.optimizer.state_dict(), learner.optimizer.state_dict())
        self.assertNotIn("_replay", restored.state_dict())
        self.assertNotIn("_expert_frames", restored.state_dict())

    def test_expert_hash_covers_features_layout_and_clip_pairing(self):
        learner = self.learner()
        self.assertEqual(learner.expert_hash, self.learner().expert_hash)
        checkpoint = learner.checkpoint()
        changed = make_motion()
        changed.q[-1, 0] += 0.001
        different = self.learner(changed)
        self.assertNotEqual(learner.expert_hash, different.expert_hash)
        with self.assertRaisesRegex(ValueError, "expert identity"):
            different.restore(checkpoint)
        repartitioned = self.learner(make_motion((4, 5)))
        self.assertNotEqual(learner.expert_hash, repartitioned.expert_hash)
        with self.assertRaisesRegex(ValueError, "expert identity"):
            repartitioned.restore(checkpoint)
        altered = copy.deepcopy(checkpoint)
        altered["feature_layout"]["version"] += 1
        with self.assertRaisesRegex(ValueError, "feature layout"):
            learner.restore(altered)
        self.assertEqual(learner.expert_provenance["feature_dim"], learner.feature_dim)
        self.assertEqual(learner.expert_provenance["expert_hash"], learner.expert_hash)
        self.assertEqual(learner.expert_provenance["provenance"], self.motion.provenance)
        snapshot = learner._expert_frames.clone()
        self.motion.q.add_(0.1)
        torch.testing.assert_close(learner._expert_frames, snapshot, atol=0, rtol=0)

    def test_checkpoint_is_snapshot_and_rejects_nonfinite_state(self):
        learner = self.learner()
        checkpoint = learner.checkpoint()
        before = copy.deepcopy(checkpoint["discriminator"])
        S0, S1 = self.states()
        learner.observe(S0, S1)
        learner.update(batch_size=4)
        self.assert_tree_equal(checkpoint["discriminator"], before)
        invalid = copy.deepcopy(checkpoint)
        invalid["norm"]["var"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            learner.restore(invalid)
        invalid["norm"]["var"][0] = -1.
        with self.assertRaisesRegex(ValueError, "normalization"):
            learner.restore(invalid)

    def test_features_compile_fullgraph_and_support_sequence_and_scalar_batches(self):
        learner = self.learner()
        S0, S1 = self.states()
        compiled = torch.compile(learner.features, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(S0, S1), learner.features(S0, S1))
        sequence0 = tuple(value.unsqueeze(0).expand(2, *value.shape) for value in S0)
        sequence1 = tuple(value.unsqueeze(0).expand(2, *value.shape) for value in S1)
        self.assertEqual(learner.features(sequence0, sequence1).shape, (2, 3, learner.feature_dim))
        scalar0, scalar1 = (tuple(value[0] for value in state) for state in (S0, S1))
        self.assertEqual(learner.features(scalar0, scalar1).shape, (learner.feature_dim,))
        self.assertEqual(learner.observe(scalar0, scalar1, valid=torch.tensor(True)), 1)

    def test_compiled_reward_with_real_terrain_keeps_state_gradients(self):
        field = HeightField(generate("stairs", 0.8, 19), 0.8, "cpu")
        learner = self.learner(ground_sampler=field.sample).requires_grad_(False)
        S0, S1 = self.states()
        q = S1[4].clone().requires_grad_()
        p = S1[0].clone().requires_grad_()
        following = (p, *S1[1:4], q, S1[5])
        compiled = torch.compile(learner.reward, backend="aot_eager", fullgraph=True)
        actual = compiled(S0, following)
        torch.testing.assert_close(actual, learner.reward(S0, following))
        actual.sum().backward()
        self.assert_nonzero_finite(q.grad)
        self.assert_nonzero_finite(p.grad)
        self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))

    def test_real_robot_kinematics_across_all_six_robots(self):
        for name in ("go2", "g1", "a3", "t1", "bumi", "pi_plus"):
            with self.subTest(robot=name):
                spec = get_robot(name)
                if not Path(spec.mjcf_path).is_file():
                    self.skipTest(f"Local {name} MJCF assets unavailable")
                # MuJoCo's Windows path loader requires an ASCII relative path.
                relative_spec = replace(spec, mjcf_path=str(Path(spec.mjcf_path).relative_to(REPO)))
                with chdir(REPO):
                    kin = RobotKinematics("cpu", relative_spec)
                motion = make_motion((4,))
                motion.q = torch.tensor(spec.default_q).repeat(4, 1)
                motion.q += torch.arange(4)[:, None] * 0.002
                motion.qd = torch.full_like(motion.q, 0.1)
                motion.num_joints = spec.num_joints
                motion.ref_state = lambda frames: MotionLib.ref_state(motion, frames)
                learner = AMPStyle(kin, motion, 0.8, hidden=(16,), replay_capacity=8)
                S0, S1 = motion.ref_state(torch.tensor([0, 1])), motion.ref_state(torch.tensor([1, 2]))
                frame = learner.frame_features(S1)
                p, R = S1[:2]
                positions = kin.forward(*S1)[0] - p.unsqueeze(-2)
                expected = (R.transpose(-1, -2).unsqueeze(-3)
                            @ positions.unsqueeze(-1)).squeeze(-1) / 0.8
                torch.testing.assert_close(frame[..., -expected.shape[-2] * 3:],
                                           expected.flatten(-2), atol=2e-6, rtol=2e-6)
                learner.observe(S0, S1)
                self.assertTrue(math.isfinite(learner.update(batch_size=4)["loss"]))
                action = torch.full_like(S1[4], 0.01, requires_grad=True)
                imagined = (*S1[:4], S1[4] + 0.1 * action, S1[5])
                with learner.freeze():
                    learner.reward(S0, imagined).sum().backward()
                self.assert_nonzero_finite(action.grad)

    def test_real_motionlib_ref_state_conventions(self):
        # Use MotionLib's implementation directly on the fixture's stored arrays.
        learner = self.learner()
        indices = torch.tensor([0, 1, 5])
        states = MotionLib.ref_state(self.motion, indices)
        self.assertEqual(len(states), 6)
        torch.testing.assert_close(learner.frame_features(states), learner._expert_frames[indices])

    def test_invalid_hyperparameters_and_masks_are_rejected(self):
        for kwargs in ({"robot_height": 0}, {"robot_height": float("nan")},
                       {"replay_capacity": 0}, {"replay_capacity": 2.5},
                       {"hidden": (0,)}, {"lr": 0}, {"gradient_penalty_weight": -1}):
            options = {"robot_height": 0.8, "hidden": (8,), "replay_capacity": 8, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AMPStyle(self.kin, self.motion, **options)
        learner = self.learner()
        S0, S1 = self.states()
        with self.assertRaisesRegex(ValueError, "boolean"):
            learner.observe(S0, S1, valid=torch.ones(3))
        with self.assertRaisesRegex(ValueError, "matching"):
            learner.observe(S0, S1, valid=torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            learner.update(batch_size=0)
        with self.assertRaises(ValueError):
            learner.update(updates=-1)


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for the opt-in signed-velocity AMP feature experiment."""

import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from test_amp_end_effectors import WristKinematics, motion_fixture, recorded
from trackerLab.algorithms.supermimic.amp_style import AMPStyle
from trackerLab.kinematics.supermimic.rotations import so3_exp
from train_terrain import parse_args


MODE = "postural_end_effectors_vx"
BASE_MODE = "postural_end_effectors"
VX_LABEL = "root_local_forward_velocity/robot_height"
# Captured before this experiment, using the same synthetic CPU fixture.
LEGACY_FINGERPRINTS = {
    (2, "full"): (
        "55997820d81ab3823d2cc7fbf63afed0602254224782cf3950b65f8c3d93fc7f",
        "3390757bdc8c9685d139bf0a2a8a3f02b4ca2f17df837701d264f4b6136d7b2d"),
    (2, "proprioceptive"): (
        "a720b5be6030f0053cdc364f2ac37e5e701d7bacae5163a10f48b7dde57d518c",
        "730e64a0e800e92a24470db654643551913cb778633a04af523b68c16c545411"),
    (2, "postural"): (
        "7627b521cff4d7dabe853745e81fc177f7179411fd29303fd81bad1709d4e196",
        "87a6862d04bcf969c52f63861935e5fbfb335a6c8d5b5e28ae2dd8ba2e869701"),
    (2, BASE_MODE): (
        "7e43f588231a642f6c8224777212f855f3d01538cd0c10e766860915ccc1276f",
        "e2a46a21b37a285eb1c7d26e91737ac189508439911b5c0cc555592ebe2de68d"),
    (10, "full"): (
        "63a0579be83dce881aec37eddc1bebb7755da1a96475be8c3e01376f2692b035",
        "06d054b27e767089a881bba796ffd6cdfa5ff2f8ed7386ee933caa1422551680"),
    (10, "proprioceptive"): (
        "b1271bef4d6db9eeb0fe53ec441be83435bdf6bfb7f3909076847b424184a66b",
        "e76da45fa7e6a5b7b4e3edc1f144751a41149b4f37d64a4d4413c01c54cd9355"),
    (10, "postural"): (
        "dda5906c08e1c0a2c8e3cd1333c6165a288e5a6fe88c0390ee0ff6932ae816fa",
        "a5de6610385859238cc5b0e79df261b520e79dce3bedddba9dc727f9c697af61"),
    (10, BASE_MODE): (
        "059bec6516b09fd803556f1bf05049a0b2b0681e28177a918544b98d64870dc1",
        "adc0add349f0a71b4166923c5c7190b5644f2ff6d64f43111512fe131b4ddc5d"),
}


class AMPBackwardVelocityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(63)
        self.kin, self.motion = WristKinematics(), motion_fixture((12, 13))

    def learner(self, mode=MODE, motion=None, **kwargs):
        return AMPStyle(self.kin, motion or self.motion, .8, replay_capacity=16,
                        hidden=kwargs.pop("hidden", (16, 8)), feature_mode=mode, **kwargs)

    def states(self):
        return recorded(self.motion, [0, 2, 12]), recorded(self.motion, [1, 3, 13])

    @staticmethod
    def with_local_velocity(state, xyz):
        velocity = state[2].new_tensor(xyz).expand_as(state[2])
        return (*state[:2], (state[1] @ velocity[..., None]).squeeze(-1), *state[3:])

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                self.assert_tree_equal(left[name], right[name])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_cli_opt_in_and_existing_defaults(self):
        argv = ["--task", "locomotion/g1/stealth", "--run-dir", "unused-amp-vx-test"]
        self.assertEqual(parse_args(argv).style_feature_mode, "proprioceptive")
        self.assertEqual(parse_args([*argv, "--style-feature-mode", MODE]).style_feature_mode, MODE)
        self.assertEqual(AMPStyle(self.kin, self.motion, .8).feature_mode, "full")

    def test_exact_end_effector_prefix_and_appended_scalar_metadata(self):
        for hands in (None, (), ("right_wrist_yaw_link",),
                      ("right_wrist_yaw_link", "left_wrist_yaw_link")):
            with self.subTest(hands=hands):
                base = self.learner(BASE_MODE, hand_body_names=hands)
                learner = self.learner(hand_body_names=hands)
                layout = copy.deepcopy(base.feature_layout)
                layout["feature_mode"] = MODE
                layout["frame"].append([VX_LABEL, 1])
                self.assertEqual(learner.feature_layout, layout)
                self.assertEqual(learner.frame_dim, base.frame_dim + 1)
                self.assertEqual(learner.feature_dim, 2 * learner.frame_dim)
                self.assertEqual(sum(width for _, width in layout["frame"]), learner.frame_dim)
                torch.testing.assert_close(learner._expert_frames[..., :-1], base._expert_frames,
                                           atol=0, rtol=0)
                torch.testing.assert_close(learner.norm.mean[:-1], base.norm.mean, atol=2e-7, rtol=1e-6)
                for state in self.states():
                    torch.testing.assert_close(learner.frame_features(state)[..., :-1],
                                               base.frame_features(state), atol=0, rtol=0)
                self.assertNotEqual(learner.expert_hash, base.expert_hash)
                scorer = AMPStyle.for_scoring(self.kin, learner.checkpoint())
                self.assertEqual(scorer.feature_layout, layout)
                torch.testing.assert_close(scorer.reward(*self.states()), learner.reward(*self.states()),
                                           atol=0, rtol=0)

    def test_signed_root_velocity_is_invisible_to_existing_mode_and_visible_to_new_mode(self):
        base, learner = self.learner(BASE_MODE), self.learner()
        state, _ = self.states()
        forward = self.with_local_velocity(state, (.2, 0., 0.))
        backward = self.with_local_velocity(state, (-.2, 0., 0.))
        torch.testing.assert_close(base.frame_features(forward), base.frame_features(backward),
                                   atol=0, rtol=0)
        torch.testing.assert_close(base.features(forward, forward), base.features(backward, backward),
                                   atol=0, rtol=0)
        torch.testing.assert_close(base.reward(forward, forward), base.reward(backward, backward),
                                   atol=0, rtol=0)
        positive, negative = learner.frame_features(forward), learner.frame_features(backward)
        torch.testing.assert_close(positive[..., :-1], negative[..., :-1], atol=0, rtol=0)
        torch.testing.assert_close(positive[..., -1], torch.full((3,), .25))
        torch.testing.assert_close(negative[..., -1], torch.full((3,), -.25))
        stopped = self.with_local_velocity(state, (0., 0., 0.))
        torch.testing.assert_close(learner.frame_features(stopped)[..., -1], torch.zeros(3))
        lateral = self.with_local_velocity(state, (.2, .7, -.6))
        torch.testing.assert_close(learner.frame_features(lateral), positive, atol=2e-7, rtol=2e-7)

    def test_expert_identity_distinguishes_signed_velocity_only_in_new_mode(self):
        changed = copy.deepcopy(self.motion)
        changed.root_v.neg_()
        self.assertEqual(self.learner(BASE_MODE).expert_hash,
                         self.learner(BASE_MODE, motion=changed).expert_hash)
        learner, reversed_learner = self.learner(), self.learner(motion=changed)
        self.assertNotEqual(learner.expert_hash, reversed_learner.expert_hash)
        torch.testing.assert_close(learner._expert_frames[..., -1],
                                   -reversed_learner._expert_frames[..., -1], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "expert identity"):
            learner.restore(reversed_learner.checkpoint())

    def test_world_xy_translation_and_yaw_leave_features_and_rewards_invariant(self):
        learner = self.learner()
        states = self.states()
        transformed = []
        for step, (p, R, v, w, q, qd) in enumerate(states):
            yaw = so3_exp(torch.tensor([[0., 0., 1.7 + step], [0., 0., -2.8], [0., 0., .5]]))
            transformed.append((
                (yaw @ p[..., None]).squeeze(-1) + torch.tensor([1e7, -1e7, 0.]),
                yaw @ R, (yaw @ v[..., None]).squeeze(-1),
                (yaw @ w[..., None]).squeeze(-1), q, qd,
            ))
        torch.testing.assert_close(learner.features(*states), learner.features(*transformed),
                                   atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(learner.reward(*states), learner.reward(*transformed),
                                   atol=2e-6, rtol=2e-6)
        for state in states:
            moved = (state[0] + torch.tensor([1e7, -1e7, 0.]), *state[1:])
            torch.testing.assert_close(learner.frame_features(moved), learner.frame_features(state),
                                       atol=0, rtol=0)

    def test_signed_velocity_scaling_and_world_velocity_gradient(self):
        state, _ = self.states()
        for height in (.4, .8, 1.6):
            with self.subTest(height=height):
                learner = AMPStyle(self.kin, self.motion, height, hidden=(16, 8),
                                   replay_capacity=16, feature_mode=MODE)
                v = state[2].clone().requires_grad_()
                features = learner.frame_features((*state[:2], v, *state[3:]))
                expected = (state[1].transpose(-1, -2) @ v[..., None]).squeeze(-1)[..., 0] / height
                torch.testing.assert_close(features[..., -1], expected, atol=0, rtol=0)
                actual = torch.autograd.grad(features[..., -1].sum(), v)[0]
                torch.testing.assert_close(actual, state[1][..., :, 0] / height, atol=1e-7, rtol=1e-6)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                self.assertGreater(actual.abs().sum().item(), 0)

    def test_reward_keeps_root_velocity_gradient_with_frozen_discriminator(self):
        learner = self.learner(hidden=())
        S0, S1 = self.states()
        with torch.no_grad():
            learner.discriminator[-1].weight.zero_()
            learner.discriminator[-1].weight[0, -1] = .02
            learner.discriminator[-1].bias.zero_()
        v = S1[2].clone().requires_grad_()
        with learner.freeze():
            reward = learner.reward(S0, (*S1[:2], v, *S1[3:]))
            reward.sum().backward()
            self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))
        self.assertTrue(bool(torch.isfinite(v.grad).all()))
        self.assertGreater(v.grad.abs().sum().item(), 0)
        self.assertEqual(learner.replay_size, 0)

    def test_compiled_pair_features_support_scalar_and_sequence_batches(self):
        learner = self.learner()
        S0, S1 = self.states()
        compiled = torch.compile(learner.features, backend="eager", fullgraph=True)
        for left, right in ((S0, S1),
                            (tuple(value[0] for value in S0), tuple(value[0] for value in S1)),
                            (tuple(value[None].expand(2, *value.shape) for value in S0),
                             tuple(value[None].expand(2, *value.shape) for value in S1))):
            torch.testing.assert_close(compiled(left, right), learner.features(left, right),
                                       atol=0, rtol=0)
        v = S1[2].clone().requires_grad_()
        following = (*S1[:2], v, *S1[3:])
        actual = compiled(S0, following)
        actual_gradient = torch.autograd.grad(actual[..., -1].sum(), v)[0]
        expected = learner.features(S0, following)
        expected_gradient = torch.autograd.grad(expected[..., -1].sum(), v)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)

    def test_ten_frame_sampling_collection_rollout_and_velocity_gradient(self):
        learner = self.learner(history_steps=10)
        self.assertEqual(learner._expert_starts.tolist(), [0, 1, 2, 12, 13, 14, 15])
        features, starts, ends = learner.sample_expert(8, return_indices=True)
        ids = starts[:, None] + torch.arange(10)
        frames = learner.frame_features(recorded(self.motion, ids), reference=True)
        torch.testing.assert_close(features, frames.flatten(-2), atol=0, rtol=0)
        torch.testing.assert_close(self.motion.clip_id[starts], self.motion.clip_id[ends])
        self.assertTrue(bool((ends - starts == 9).all()))
        self.assertEqual(learner.feature_layout["history_steps"], 10)
        self.assertEqual(learner.feature_layout["endpoints"], "oldest-to-newest full history")
        valid = torch.tensor([True, False, True, True, True, True, True, True])
        self.assertEqual(learner.observe_window(frames, valid), 7)
        torch.testing.assert_close(learner._replay[:7], features[valid], atol=0, rtol=0)
        following = recorded(self.motion, torch.tensor([[1, 13], [2, 14], [3, 15]]))
        v = following[2].requires_grad_()
        next_frames = learner.frame_features((*following[:2], v, *following[3:]))
        windows = learner.rollout_features(frames[:2], next_frames)
        self.assertEqual(windows.shape, (3, 2, 10 * learner.frame_dim))
        for step in range(3):
            sequence = torch.cat((frames[:2, 1:], next_frames[:step + 1].transpose(0, 1)), 1)
            torch.testing.assert_close(windows[step], sequence[:, -10:].flatten(-2), atol=0, rtol=0)
        windows.reshape(3, 2, 10, learner.frame_dim)[..., -1].sum().backward()
        self.assertTrue(bool(torch.isfinite(v.grad).all()))
        self.assertTrue(bool((v.grad.abs().sum(-1) > 0).all()))

    def test_cpu_checkpoint_restore_and_motion_free_scoring_roundtrip(self):
        for history in (2, 10):
            with self.subTest(history=history):
                learner = self.learner(history_steps=history)
                frames = learner.frame_features(recorded(self.motion, torch.arange(history)[None]))
                learner.observe_window(frames, torch.tensor([True]))
                self.assertEqual(learner.update(batch_size=4, updates=2)["updates"], 2)
                features = learner.window_features(frames)
                stream = BytesIO()
                torch.save(learner.checkpoint(), stream)
                stream.seek(0)
                checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
                self.assertEqual(checkpoint["feature_mode"], MODE)
                self.assertEqual(checkpoint["expert_provenance"]["feature_mode"], MODE)
                self.assertEqual(checkpoint["feature_layout"]["frame"][-1], [VX_LABEL, 1])
                self.assertEqual(checkpoint["feature_dim"], history * learner.frame_dim)
                restored = self.learner(history_steps=history)
                receipt = restored.restore(checkpoint)
                self.assertEqual(receipt, {"replay_restored": False, "replay_refill_required": True})
                self.assertEqual(restored.replay_size, 0)
                for field in ("discriminator", "norm", "optimizer", "updates", "observed",
                              "feature_layout", "expert_hash"):
                    self.assert_tree_equal(restored.checkpoint()[field], checkpoint[field])
                torch.testing.assert_close(restored.reward_from_features(features),
                                           learner.reward_from_features(features), atol=0, rtol=0)
                for name in ("optimizer", "updates", "observed"):
                    checkpoint.pop(name)
                with patch.object(AMPStyle, "__init__", side_effect=AssertionError("constructor")), \
                        patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer")), \
                        patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer")):
                    scorer = AMPStyle.for_scoring(self.kin, checkpoint)
                self.assertFalse(scorer.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in scorer.parameters()))
                for name in ("motion", "_expert_frames", "_expert_starts", "_replay", "optimizer"):
                    self.assertFalse(hasattr(scorer, name), name)
                torch.testing.assert_close(scorer.frame_features(self.states()[0]),
                                           learner.frame_features(self.states()[0]), atol=0, rtol=0)
                torch.testing.assert_close(scorer.score(features), learner.score(features), atol=0, rtol=0)
                torch.testing.assert_close(scorer.reward_from_features(features),
                                           learner.reward_from_features(features), atol=0, rtol=0)
                differentiable = features.detach().clone().requires_grad_()
                scorer.reward_from_features(differentiable).sum().backward()
                self.assertGreater(differentiable.grad[..., -1].abs().sum().item(), 0)
                self.assertEqual(scorer.expert_hash, learner.expert_hash)
                with self.assertRaisesRegex(RuntimeError, "Scoring-only"):
                    scorer.checkpoint()

    def test_crossmode_restore_and_scoring_metadata_are_rejected(self):
        learner = self.learner()
        checkpoint = learner.checkpoint()
        for mode in ("full", "proprioceptive", "postural", BASE_MODE):
            with self.subTest(mode=mode):
                legacy = self.learner(mode)
                legacy_checkpoint = legacy.checkpoint()
                with self.assertRaisesRegex(ValueError, "feature layout"):
                    learner.restore(legacy_checkpoint)
                with self.assertRaisesRegex(ValueError, "feature layout"):
                    legacy.restore(checkpoint)
                for original, replacement in ((checkpoint, mode), (legacy_checkpoint, MODE)):
                    altered = copy.deepcopy(original)
                    altered["feature_mode"] = replacement
                    with self.assertRaisesRegex(ValueError, "feature layout"):
                        AMPStyle.for_scoring(self.kin, altered)
        self.assert_tree_equal(learner.checkpoint(), checkpoint)

    def test_restore_and_scorer_reject_tampered_dimensions_and_width_types(self):
        learner = self.learner()
        checkpoint = learner.checkpoint()
        dimension = checkpoint["feature_dim"]
        for value in (True, float(dimension), 0, -1, dimension - 1, dimension + 2, None, str(dimension)):
            with self.subTest(dimension=value):
                altered = copy.deepcopy(checkpoint)
                altered["feature_dim"] = value
                with self.assertRaisesRegex(ValueError, "dimension"):
                    learner.restore(altered)
                with self.assertRaisesRegex(ValueError, "dimension"):
                    AMPStyle.for_scoring(self.kin, altered)
        for width in (True, 1.0, 0, -1, 2, "1"):
            with self.subTest(width=width):
                altered = copy.deepcopy(checkpoint)
                altered["feature_layout"]["frame"][-1][1] = width
                with self.assertRaisesRegex(ValueError, "layout"):
                    learner.restore(altered)
                with self.assertRaisesRegex(ValueError, "layout"):
                    AMPStyle.for_scoring(self.kin, altered)
        self.assert_tree_equal(learner.checkpoint(), checkpoint)

    def test_restore_and_scorer_reject_changed_or_incomplete_layout(self):
        learner = self.learner()
        checkpoint = learner.checkpoint()
        changes = {
            "frame": checkpoint["feature_layout"]["frame"][:-1],
            "feature_mode": BASE_MODE,
            "hand_names": ["right_wrist_yaw_link", "left_wrist_yaw_link"],
            "hand_body_indices": [5, 2],
            "tracked_offsets": [[.01, 0., 0.]] * 5,
            "default_q": [0.] * 6,
        }
        changed_frames = copy.deepcopy(checkpoint["feature_layout"]["frame"])
        changed_frames[-1][0] = "unsigned_root_speed/robot_height"
        changes["renamed_scalar"] = changed_frames
        for field, value in changes.items():
            with self.subTest(field=field):
                altered = copy.deepcopy(checkpoint)
                altered["feature_layout"]["frame" if field == "renamed_scalar" else field] = value
                with self.assertRaisesRegex(ValueError, "layout"):
                    learner.restore(altered)
                with self.assertRaisesRegex(ValueError, "layout"):
                    AMPStyle.for_scoring(self.kin, altered)
        for field in ("feature_mode", "feature_layout", "feature_dim"):
            with self.subTest(missing=field):
                altered = copy.deepcopy(checkpoint)
                altered.pop(field)
                with self.assertRaisesRegex(ValueError, "layout|dimension"):
                    learner.restore(altered)
                with self.assertRaisesRegex(ValueError, "layout|dimension"):
                    AMPStyle.for_scoring(self.kin, altered)

    def test_restore_and_scorer_reject_tampered_network_and_normalizer_shapes(self):
        learner = self.learner()
        checkpoint = learner.checkpoint()
        for kind in ("discriminator", "mean", "var"):
            with self.subTest(kind=kind):
                altered = copy.deepcopy(checkpoint)
                if kind == "discriminator":
                    altered["discriminator"]["0.weight"] = altered["discriminator"]["0.weight"][:, :-1]
                else:
                    altered["norm"][kind] = altered["norm"][kind][:-1]
                with self.assertRaisesRegex(ValueError, "dimension|state"):
                    learner.restore(altered)
                with self.assertRaisesRegex(ValueError, "dimension|state"):
                    AMPStyle.for_scoring(self.kin, altered)
        self.assert_tree_equal(learner.checkpoint(), checkpoint)

    def test_all_legacy_layout_and_expert_fingerprints_and_header_fallback_are_unchanged(self):
        for (history, mode), (expert_hash, layout_hash) in LEGACY_FINGERPRINTS.items():
            with self.subTest(mode=mode, history=history):
                learner = self.learner(mode, history_steps=history)
                layout = json.dumps(learner.feature_layout, sort_keys=True, separators=(",", ":")).encode()
                self.assertEqual(hashlib.sha256(layout).hexdigest(), layout_hash)
                self.assertEqual(learner.expert_hash, expert_hash)
                checkpoint = learner.checkpoint()
                checkpoint.pop("feature_mode")
                scorer = AMPStyle.for_scoring(self.kin, checkpoint)
                self.assertEqual(scorer.feature_mode, mode)
                self.assertEqual(scorer.feature_layout, learner.feature_layout)
                torch.testing.assert_close(scorer.frame_features(self.states()[0]),
                                           learner.frame_features(self.states()[0]), atol=0, rtol=0)
                if mode == "full":
                    self.learner(mode, history_steps=history).restore(checkpoint)
                else:
                    with self.assertRaisesRegex(ValueError, "feature layout"):
                        learner.restore(checkpoint)


if __name__ == "__main__":
    unittest.main()

"""Assetless contracts for root-local AMP wrist/foot geometry."""

import copy
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.amp_style import AMPStyle
from trackerLab.kinematics.supermimic.rotations import axis_angle_to_mat, mat_to_6d, so3_exp, yaw_mat


class WristKinematics:
    """Two articulated arms and legs with distinct body and tracked indices."""

    num_joints, device = 6, "cpu"
    joint_names = ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
                   "left_hip", "right_hip")
    body_names = ("pelvis", "left_shoulder_link", "left_wrist_yaw_link",
                  "left_ankle_roll_link", "right_shoulder_link", "right_wrist_yaw_link",
                  "right_ankle_roll_link")
    tracked_idx = (0, 3, 6, 2, 5)
    robot = SimpleNamespace(default_q=(.1, -.1, 0., 0., .05, -.05))

    def forward(self, p, R, v, w, q, qd):
        def point(xyz):
            return q.new_tensor(xyz).expand(*q.shape[:-1], 3)

        def turn(axis, index):
            return axis_angle_to_mat(q.new_tensor(axis), q[..., index])

        left, right = turn((0., 1., 0.), 0), turn((0., 1., 0.), 1)
        left_leg, right_leg = turn((0., 1., 0.), 4), turn((0., 1., 0.), 5)
        shoulder_l, shoulder_r = point((0., .2, .25)), point((0., -.2, .25))
        arm, leg = point((.45, 0., 0.)), point((0., 0., -.65))
        local_p = torch.stack((
            point((0., 0., 0.)), shoulder_l,
            shoulder_l + (left @ arm[..., None]).squeeze(-1),
            point((0., .12, 0.)) + (left_leg @ leg[..., None]).squeeze(-1),
            shoulder_r, shoulder_r + (right @ arm[..., None]).squeeze(-1),
            point((0., -.12, 0.)) + (right_leg @ leg[..., None]).squeeze(-1),
        ), -2)[..., self.tracked_idx, :]
        local_R = torch.stack((
            torch.eye(3, device=q.device, dtype=q.dtype).expand(*q.shape[:-1], 3, 3),
            left, left @ turn((0., 0., 1.), 2), left_leg,
            right, right @ turn((0., 0., 1.), 3), right_leg,
        ), -3)[..., self.tracked_idx, :, :]
        arm_w = (R[..., None, :, :] @ local_p[..., None]).squeeze(-1)
        return (p[..., None, :] + arm_w, R[..., None, :, :] @ local_R,
                v[..., None, :] + torch.cross(w[..., None, :].expand_as(arm_w), arm_w, dim=-1),
                w[..., None, :].expand_as(arm_w))


def motion_fixture(lengths=(6, 5)):
    lengths = torch.tensor(lengths, dtype=torch.long)
    t = torch.arange(int(lengths.sum()), dtype=torch.float32)
    phase = t[:, None] * .3 + torch.arange(6, dtype=torch.float32)[None] * .2
    starts = lengths.cumsum(0) - lengths
    return SimpleNamespace(
        root_p=torch.stack((t * .02, t * -.03, .8 + .02 * t.sin()), -1),
        root_R=so3_exp(torch.stack((.08 * t.sin(), .06 * t.cos(), .1 * t), -1)),
        root_v=torch.stack((.4 + t * .01, t * .02, t * 0), -1),
        root_w=torch.stack((t * .01, t * 0, .2 + t * .03), -1),
        q=.3 * phase.sin(), qd=.09 * phase.cos(),
        clip_start=starts, clip_end=starts + lengths - 1, clip_len=lengths,
        clip_id=torch.repeat_interleave(torch.arange(len(lengths)), lengths),
        names=["synthetic"] * len(lengths),
        provenance={"source": "analytic arms/legs; no robot assets"},
    )


def recorded(motion, indices):
    return tuple(getattr(motion, name)[indices].detach().clone()
                 for name in ("root_p", "root_R", "root_v", "root_w", "q", "qd"))


class AMPEndEffectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(63)
        self.kin, self.motion = WristKinematics(), motion_fixture()

    def learner(self, **kwargs):
        return AMPStyle(self.kin, self.motion, .8, replay_capacity=16,
                        hidden=kwargs.pop("hidden", (16, 8)),
                        feature_mode=kwargs.pop("feature_mode", "postural_end_effectors"), **kwargs)

    def states(self):
        return recorded(self.motion, [0, 2, 6]), recorded(self.motion, [1, 3, 7])

    @staticmethod
    def block(learner, name):
        start = 0
        for label, width in learner.feature_layout["frame"]:
            if label == name:
                return slice(start, start + width)
            start += width
        raise AssertionError(f"Missing feature block {name}")

    def geometry(self, learner, state):
        features = learner.frame_features(state)
        return (features[..., self.block(learner, "root_local_tracked_position/robot_height")],
                features[..., self.block(learner, "root_local_hand_rotation_6d")])

    def assert_nonzero(self, value):
        self.assertIsNotNone(value)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertGreater(value.abs().sum().item(), 1e-8)

    def assert_norm_equal(self, expected, learner):
        for name, value in learner.norm.state_dict().items():
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)

    def test_layout_declares_origins_hand_axes_and_preserves_postural_prefix(self):
        learner = self.learner()
        postural = self.learner(feature_mode="postural")
        S0, _ = self.states()
        layout = learner.feature_layout
        self.assertEqual(layout["feature_mode"], "postural_end_effectors")
        self.assertEqual(layout["tracked_indices"], [0, 3, 6, 2, 5])
        self.assertEqual(layout["tracked_names"], [self.kin.body_names[i] for i in self.kin.tracked_idx])
        self.assertEqual(layout["tracked_offsets"], [[0., 0., 0.]] * 5)
        self.assertEqual(layout["hand_names"], ["left_wrist_yaw_link", "right_wrist_yaw_link"])
        self.assertEqual(layout["hand_tracked_indices"], [3, 4])
        self.assertEqual(layout["hand_body_indices"], [2, 5])
        self.assertEqual(layout["hand_rotation_frame"], "model_body_axes")
        self.assertEqual(layout["rotation_6d_convention"], "first_two_columns_row_major")
        self.assertEqual(learner.frame_dim, 4 + 2 * 6 + 3 * 5 + 6 * 2)
        self.assertEqual(sum(width for _, width in layout["frame"]), learner.frame_dim)
        self.assertEqual(learner.feature_dim, 2 * learner.frame_dim)
        torch.testing.assert_close(learner.frame_features(S0)[..., :postural.frame_dim],
                                   postural.frame_features(S0), atol=0, rtol=0)
        self.assertNotEqual(learner.expert_hash, postural.expert_hash)

    def test_geometry_matches_world_fk_including_feet_and_rotated_wrist_axes(self):
        learner = self.learner()
        S0, _ = self.states()
        p, R = S0[:2]
        positions, rotations = self.geometry(learner, S0)
        world_p, world_R = self.kin.forward(*S0)[:2]
        Rt = R.transpose(-1, -2)[..., None, :, :]
        expected_p = (Rt @ (world_p - p[..., None, :])[..., None]).squeeze(-1) / .8
        expected_R = Rt @ world_R
        torch.testing.assert_close(positions, expected_p.flatten(-2))
        torch.testing.assert_close(rotations, mat_to_6d(expected_R[..., [3, 4], :, :]).flatten(-2))
        zero = (*S0[:4], torch.zeros_like(S0[4]), S0[5])
        positions, rotations = self.geometry(learner, zero)
        torch.testing.assert_close(positions.reshape(3, 5, 3)[0] * .8, torch.tensor([
            [0., 0., 0.], [0., .12, -.65], [0., -.12, -.65],
            [.45, .2, .25], [.45, -.2, .25],
        ]))
        torch.testing.assert_close(rotations.reshape(3, 2, 6)[0],
                                   torch.tensor([[1., 0., 0., 1., 0., 0.]]).expand(2, -1))

    def test_shoulder_moves_wrist_and_wrist_rotation_changes_facing_at_same_origin(self):
        learner = self.learner()
        S0, _ = self.states()
        positions, rotations = self.geometry(learner, S0)
        q = S0[4].clone()
        q[:, 0] += .4
        shifted_p, shifted_R = self.geometry(learner, (*S0[:4], q, S0[5]))
        self.assertGreater((shifted_p - positions).abs().sum().item(), .1)
        self.assertGreater((shifted_R - rotations).abs().sum().item(), .1)
        torch.testing.assert_close(shifted_p.reshape(3, 5, 3)[:, :3],
                                   positions.reshape(3, 5, 3)[:, :3], atol=0, rtol=0)
        q = S0[4].clone()
        q[:, 2] += .7
        turned_p, turned_R = self.geometry(learner, (*S0[:4], q, S0[5]))
        torch.testing.assert_close(turned_p, positions, atol=0, rtol=0)
        self.assertGreater((turned_R[:, :6] - rotations[:, :6]).abs().sum().item(), .1)
        torch.testing.assert_close(turned_R[:, 6:], rotations[:, 6:], atol=0, rtol=0)

    def test_world_translation_yaw_and_arbitrary_root_twist_leave_features_and_reward_equal(self):
        learner = self.learner()
        states = self.states()
        changed = []
        for step, state in enumerate(states):
            p, R, v, w, q, qd = state
            yaw = so3_exp(torch.tensor([[0., 0., 1.7 + step], [0., 0., -2.8], [0., 0., .5]]))
            changed.append((
                (yaw @ p[..., None]).squeeze(-1) + torch.tensor([1e7, -1e7, 0.]),
                yaw @ R, v + torch.tensor([80., -30., 40.]),
                w + torch.tensor([30., 90., -100.]), q, qd,
            ))
        torch.testing.assert_close(learner.features(*states), learner.features(*changed),
                                   atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(learner.reward(*states), learner.reward(*changed),
                                   atol=2e-6, rtol=2e-6)
        for state in states:
            twisted = (state[0], state[1], -state[2] * 100, state[3] * 200, *state[4:])
            torch.testing.assert_close(learner.frame_features(state),
                                       learner.frame_features(twisted), atol=0, rtol=0)

    def test_local_geometry_ignores_root_roll_height_and_qd_but_postural_channels_keep_them(self):
        learner = self.learner()
        S0, _ = self.states()
        changed = (
            S0[0] + torch.tensor([10., -20., 5.]),
            so3_exp(torch.tensor([.7, -.4, .2])) @ S0[1],
            S0[2], S0[3], S0[4], S0[5] + 20.,
        )
        for original, actual in zip(self.geometry(learner, S0), self.geometry(learner, changed)):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
        before, after = learner.frame_features(S0), learner.frame_features(changed)
        self.assertFalse(torch.equal(before[..., :3], after[..., :3]))
        torch.testing.assert_close(after[..., 3] - before[..., 3], torch.full((3,), 5. / .8))
        velocity = self.block(learner, "joint_velocity*0.05")
        torch.testing.assert_close(after[..., velocity] - before[..., velocity], torch.ones(3, 6))

    def test_reference_zero_ground_and_shifted_policy_ground_share_geometry(self):
        sampler = Mock(side_effect=lambda xy: 4. + xy[..., 0] * 0)
        elevated = self.learner(ground_sampler=sampler)
        sampler.assert_not_called()
        flat = self.learner()
        S0, S1 = self.states()
        shifted = tuple((state[0] + torch.tensor([0., 0., 4.]), *state[1:]) for state in (S0, S1))
        torch.testing.assert_close(elevated.features(*shifted), flat.features(S0, S1))
        sampler.reset_mock()
        torch.testing.assert_close(elevated.features(S0, S1, reference=True), flat.features(S0, S1))
        sampler.assert_not_called()
        learner = self.learner(ground_sampler=lambda xy: .2 + xy[..., 0] * .1)
        p = S0[0].requires_grad_()
        learner.frame_features((p, *S0[1:]))[..., 3].sum().backward()
        torch.testing.assert_close(p.grad, torch.tensor([[-.125, 0., 1.25]]).expand(3, -1))

    def test_reference_real_and_imagined_paths_share_one_canonical_fk(self):
        learner = self.learner()
        S0, S1 = self.states()
        expected = torch.cat((learner._expert_frames[[0, 2, 6]], learner._expert_frames[[1, 3, 7]]), -1)
        self.kin.forward = Mock(wraps=self.kin.forward)
        torch.testing.assert_close(learner.features(S0, S1, reference=True), expected, atol=0, rtol=0)
        self.assertEqual(self.kin.forward.call_count, 1)
        p, R, v, w, q, qd = self.kin.forward.call_args.args
        for value in (p, v, w, qd):
            torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
        torch.testing.assert_close(R, torch.eye(3).expand(6, 3, 3), atol=0, rtol=0)
        torch.testing.assert_close(q, torch.cat((S0[4], S1[4])), atol=0, rtol=0)
        self.kin.forward.reset_mock()
        self.assertEqual(learner.observe(S0, S1, valid=torch.tensor([True, False, True])), 2)
        self.assertEqual(self.kin.forward.call_count, 1)
        torch.testing.assert_close(learner._replay[:2], expected[[0, 2]], atol=0, rtol=0)
        q = S1[4].clone().requires_grad_()
        imagined = (*S1[:4], q, S1[5])
        self.kin.forward.reset_mock()
        actual = learner.features(S0, imagined)
        self.assertEqual(self.kin.forward.call_count, 1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(actual.requires_grad)
        self.assertFalse(learner._expert_frames.requires_grad)
        self.assertFalse(learner._replay.requires_grad)
        with self.assertRaisesRegex(ValueError, "detached REAL"):
            learner.observe(S0, imagined)

    def test_geometry_gradient_reaches_shoulder_and_matches_finite_difference(self):
        learner = self.learner()
        S0, _ = self.states()
        q = S0[4].clone().requires_grad_()

        def objective(q_value):
            positions, rotations = self.geometry(learner, (*S0[:4], q_value, S0[5]))
            return positions.reshape(3, 5, 3)[:, 3, 2].sum() + rotations[:, 0].sum()

        actual = torch.autograd.grad(objective(q), q)[0]
        self.assert_nonzero(actual[:, 0])
        epsilon = .001
        plus, minus = q.detach().clone(), q.detach().clone()
        plus[:, 0] += epsilon
        minus[:, 0] -= epsilon
        numerical = (objective(plus) - objective(minus)) / (2 * epsilon)
        torch.testing.assert_close(actual[:, 0].sum(), numerical, atol=2e-4, rtol=2e-4)
        positions, rotations = self.geometry(learner, (*S0[:4], q, S0[5]))
        position_grad = torch.autograd.grad(positions.reshape(3, 5, 3)[:, 3, 2].sum(),
                                            q, retain_graph=True)[0]
        rotation_grad = torch.autograd.grad(rotations[:, 0].sum(), q)[0]
        self.assert_nonzero(position_grad[:, 0])
        self.assert_nonzero(rotation_grad[:, 0])

    def test_geometry_only_reward_keeps_actor_gradient_and_does_not_read_motion_or_collect(self):
        learner = self.learner(hidden=())
        S0, S1 = self.states()
        column = self.block(learner, "root_local_tracked_position/robot_height").start + 3 * 3 + 2
        with torch.no_grad():
            layer = learner.discriminator[-1]
            layer.weight.zero_()
            layer.weight[0, learner.frame_dim + column] = .02
            layer.bias.zero_()
        norm = copy.deepcopy(learner.norm.state_dict())
        expected = learner.reward(S0, S1).detach()
        for name in ("root_p", "root_R", "root_v", "root_w", "q", "qd"):
            getattr(self.motion, name).fill_(float("nan"))
        action = torch.zeros_like(S1[4], requires_grad=True)
        imagined = (*S1[:4], S1[4] + .2 * action, S1[5])
        with learner.freeze(), patch.object(learner, "sample_expert", side_effect=AssertionError("reference read")):
            reward = learner.reward(S0, imagined)
            torch.testing.assert_close(reward, expected, atol=0, rtol=0)
            self.assertTrue(bool(((reward >= 0) & (reward <= 1)).all()))
            reward.sum().backward()
            self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))
        self.assert_nonzero(action.grad[:, 0])
        torch.testing.assert_close(action.grad[:, 1:], torch.zeros_like(action.grad[:, 1:]), atol=0, rtol=0)
        self.assertEqual(learner.replay_size, 0)
        self.assert_norm_equal(norm, learner)

    def test_serialized_checkpoint_restore_and_motion_free_scorer_roundtrip(self):
        learner = self.learner()
        S0, S1 = self.states()
        learner.observe(S0, S1)
        self.assertEqual(learner.update(batch_size=4, updates=2)["updates"], 2)
        stream = BytesIO()
        torch.save(learner.checkpoint(), stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
        restored = self.learner()
        receipt = restored.restore(checkpoint)
        self.assertTrue(receipt["replay_refill_required"])
        self.assertEqual(restored.replay_size, 0)
        torch.testing.assert_close(restored.reward(S0, S1), learner.reward(S0, S1), atol=0, rtol=0)
        for key in ("optimizer", "updates", "observed"):
            checkpoint.pop(key)
        with patch.object(AMPStyle, "__init__", side_effect=AssertionError("training constructor")), \
                patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer allocation")):
            scorer = AMPStyle.for_scoring(self.kin, checkpoint)
        self.assertEqual(scorer.feature_layout, learner.feature_layout)
        self.assertFalse(scorer.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in scorer.parameters()))
        for name in ("motion", "_expert_frames", "_expert_starts", "_replay", "optimizer"):
            self.assertFalse(hasattr(scorer, name), name)
        torch.testing.assert_close(scorer.features(S0, S1), learner.features(S0, S1), atol=0, rtol=0)
        torch.testing.assert_close(scorer.reward(S0, S1), learner.reward(S0, S1), atol=0, rtol=0)
        q = S1[4].clone().requires_grad_()
        scorer.reward(S0, (*S1[:4], q, S1[5])).sum().backward()
        self.assert_nonzero(q.grad[:, 0])
        self.assertEqual(scorer.expert_hash, learner.expert_hash)
        with self.assertRaisesRegex(RuntimeError, "Scoring-only"):
            scorer.observe(S0, S1)

    def test_explicit_hand_subset_order_empty_subset_and_generic_names_roundtrip(self):
        S0, S1 = self.states()
        default = self.learner()
        default_rotations = self.geometry(default, S0)[1].reshape(3, 2, 6)
        for hands, slots in ((("right_wrist_yaw_link",), [1]),
                             (("right_wrist_yaw_link", "left_wrist_yaw_link"), [1, 0]),
                             ((), [])):
            with self.subTest(hands=hands):
                learner = self.learner(hand_body_names=hands)
                self.assertEqual(learner.frame_dim, 4 + 12 + 15 + 6 * len(hands))
                self.assertEqual(learner.feature_layout["hand_names"], list(hands))
                if hands:
                    torch.testing.assert_close(self.geometry(learner, S0)[1],
                                               default_rotations[:, slots].flatten(-2), atol=0, rtol=0)
                scorer = AMPStyle.for_scoring(self.kin, learner.checkpoint())
                self.assertEqual(scorer.hand_body_names, hands)
                torch.testing.assert_close(scorer.reward(S0, S1), learner.reward(S0, S1), atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, "expert identity"):
                    learner.restore(default.checkpoint())
        self.kin.body_names = tuple(name.replace("wrist_yaw", "tool") for name in self.kin.body_names)
        generic = self.learner()
        self.assertEqual(generic.hand_body_names, ())
        explicit = self.learner(hand_body_names=("left_tool_link",))
        scorer = AMPStyle.for_scoring(self.kin, explicit.checkpoint())
        torch.testing.assert_close(scorer.reward(S0, S1), explicit.reward(S0, S1), atol=0, rtol=0)

    def test_scorer_and_restore_reject_changed_layout_fields(self):
        learner = self.learner()
        checkpoint = learner.checkpoint()
        changes = {
            "tracked_indices": [0, 6, 3, 2, 5],
            "tracked_names": ["changed", *checkpoint["feature_layout"]["tracked_names"][1:]],
            "tracked_offsets": [[.01, 0., 0.]] * 5,
            "hand_names": ["right_wrist_yaw_link", "left_wrist_yaw_link"],
            "hand_tracked_indices": [2, 4],
            "hand_body_indices": [3, 5],
            "hand_rotation_frame": "inferred_palm_axes",
            "rotation_6d_convention": "column_major",
            "default_q": [0.] * 6,
            "joint_names": ["different"] * 6,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                altered = copy.deepcopy(checkpoint)
                altered["feature_layout"][field] = value
                with self.assertRaisesRegex(ValueError, "layout"):
                    AMPStyle.for_scoring(self.kin, altered)
                with self.assertRaisesRegex(ValueError, "layout"):
                    learner.restore(altered)
        for field in ("tracked_offsets", "hand_names", "hand_tracked_indices", "hand_body_indices"):
            with self.subTest(missing=field):
                altered = copy.deepcopy(checkpoint)
                altered["feature_layout"].pop(field)
                with self.assertRaisesRegex(ValueError, "layout"):
                    AMPStyle.for_scoring(self.kin, altered)
        altered = copy.deepcopy(checkpoint)
        altered["feature_dim"] += 2
        with self.assertRaisesRegex(ValueError, "dimension"):
            AMPStyle.for_scoring(self.kin, altered)

    def test_scorer_rejects_kinematics_names_indices_and_fk_count_mismatches(self):
        checkpoint = self.learner().checkpoint()
        kin = WristKinematics()
        kin.tracked_idx = (0, 6, 3, 2, 5)
        with self.assertRaisesRegex(ValueError, "layout"):
            AMPStyle.for_scoring(kin, checkpoint)
        kin = WristKinematics()
        kin.body_names = (*kin.body_names[:2], "changed_wrist", *kin.body_names[3:])
        with self.assertRaisesRegex(ValueError, "tracked body names"):
            AMPStyle.for_scoring(kin, checkpoint)
        kin = WristKinematics()
        original = kin.forward
        kin.forward = lambda *state: tuple(value[..., :-1, :] if value.ndim == 3
                                           else value[..., :-1, :, :] for value in original(*state))
        with self.assertRaisesRegex(ValueError, "FK.*layout"):
            AMPStyle.for_scoring(kin, checkpoint)

    def test_invalid_hand_selection_and_body_metadata_fail_at_construction(self):
        for hands in ("left_wrist_yaw_link", ("unknown",), ("left_shoulder_link",),
                      ("left_wrist_yaw_link",) * 2, (2,), (["left_wrist_yaw_link"],)):
            with self.subTest(hands=hands), self.assertRaisesRegex(ValueError, "hand_body_names"):
                self.learner(hand_body_names=hands)
        with self.assertRaisesRegex(ValueError, "requires postural_end_effectors"):
            self.learner(feature_mode="postural", hand_body_names=())
        for indices in ((), (0, 3, 3), (True,), (-1,), (99,)):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "tracked_idx"):
                self.kin = WristKinematics()
                self.kin.tracked_idx = indices
                self.learner()
        for names in ((), ("duplicate", "duplicate"), (1, "name")):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "body_names"):
                self.kin = WristKinematics()
                self.kin.body_names = names
                self.learner()

    def test_mode_aliases_are_rejected_in_constructor_restore_and_scorer(self):
        learner = self.learner()
        checkpoint = learner.checkpoint()
        for alias in ("postural_end_effector", "postural-end-effectors", "end_effectors",
                      "postural_palms"):
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(ValueError, "feature_mode"):
                    self.learner(feature_mode=alias)
                altered = copy.deepcopy(checkpoint)
                altered["feature_mode"] = alias
                with self.assertRaisesRegex(ValueError, "feature mode"):
                    AMPStyle.for_scoring(self.kin, altered)
                with self.assertRaisesRegex(ValueError, "feature layout"):
                    learner.restore(altered)
                altered = copy.deepcopy(checkpoint)
                altered["feature_layout"]["feature_mode"] = alias
                with self.assertRaisesRegex(ValueError, "feature layout"):
                    AMPStyle.for_scoring(self.kin, altered)
                altered.pop("feature_mode")
                with self.assertRaisesRegex(ValueError, "feature mode"):
                    AMPStyle.for_scoring(self.kin, altered)

    def test_malformed_and_nonfinite_fk_fail_at_construction(self):
        for kind in ("positions", "rotations", "nonfinite"):
            with self.subTest(kind=kind):
                self.kin = WristKinematics()
                original = self.kin.forward

                def invalid(*state):
                    p, R, v, w = original(*state)
                    if kind == "positions":
                        p = p[..., :2]
                    elif kind == "rotations":
                        R = R[..., :2]
                    else:
                        R = R * float("nan")
                    return p, R, v, w

                self.kin.forward = invalid
                with self.assertRaisesRegex(ValueError, "FK"):
                    self.learner()

    def test_history_windows_preserve_chronology_and_imagined_gradients(self):
        self.motion = motion_fixture((12, 13))
        learner = self.learner(history_steps=10)
        features, starts, ends = learner.sample_expert(8, return_indices=True)
        ids = starts[:, None] + torch.arange(10)
        expected = learner.frame_features(recorded(self.motion, ids), reference=True)
        torch.testing.assert_close(features, expected.flatten(-2), atol=0, rtol=0)
        torch.testing.assert_close(self.motion.clip_id[starts], self.motion.clip_id[ends])
        self.assertTrue(bool((ends - starts == 9).all()))
        self.assertEqual(learner.observe_window(expected, torch.ones(8, dtype=torch.bool)), 8)
        history = expected[:2]
        following_state = recorded(self.motion, torch.tensor([[1, 13], [2, 14], [3, 15]]))
        q = following_state[4].requires_grad_()
        following = learner.frame_features((*following_state[:4], q, following_state[5]))
        windows = learner.rollout_features(history, following)
        torch.testing.assert_close(windows[0], torch.cat((history[:, 1:], following[:1].transpose(0, 1)), 1).flatten(-2))
        columns = self.block(learner, "root_local_tracked_position/robot_height")
        windows.reshape(3, 2, 10, learner.frame_dim)[..., columns].sum().backward()
        self.assert_nonzero(q.grad[..., 0])
        scorer = AMPStyle.for_scoring(self.kin, learner.checkpoint())
        torch.testing.assert_close(scorer.reward_from_features(features),
                                   learner.reward_from_features(features), atol=0, rtol=0)

    def test_compiled_features_support_scalar_sequence_and_reward_gradients(self):
        learner = self.learner().requires_grad_(False)
        S0, S1 = self.states()
        compiled = torch.compile(learner.features, backend="eager", fullgraph=True)
        for left, right in ((S0, S1),
                            (tuple(value[0] for value in S0), tuple(value[0] for value in S1)),
                            (tuple(value[None].expand(2, *value.shape) for value in S0),
                             tuple(value[None].expand(2, *value.shape) for value in S1))):
            torch.testing.assert_close(compiled(left, right), learner.features(left, right), atol=0, rtol=0)
        q = S1[4].clone().requires_grad_()
        following = (*S1[:4], q, S1[5])
        reward_fn = torch.compile(learner.reward, backend="aot_eager", fullgraph=True)
        actual = reward_fn(S0, following)
        expected = learner.reward(S0, following)
        torch.testing.assert_close(actual, expected)
        actual_gradient = torch.autograd.grad(actual.sum(), q)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), q)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assert_nonzero(actual_gradient[:, 0])

    def test_legacy_modes_keep_feature_formulas_layouts_and_fk_behavior(self):
        S0, S1 = self.states()
        p, R, v, w, q, qd = S0
        Rt = R.transpose(-1, -2)
        gravity = -R[..., 2, :]
        joint = q - torch.tensor(self.kin.robot.default_q)
        height = (p[..., 2] / .8).unsqueeze(-1)
        velocity = (Rt @ v[..., None]).squeeze(-1)
        angular = (Rt @ w[..., None]).squeeze(-1)
        body_p = self.kin.forward(*S0)[0]
        local = (Rt[..., None, :, :] @ (body_p - p[..., None, :])[..., None]).squeeze(-1)
        expected = {
            "postural": torch.cat((gravity, height, joint, qd * .05), -1),
            "proprioceptive": torch.cat((gravity, joint, qd * .05, velocity, angular), -1),
            "full": torch.cat((height, mat_to_6d(yaw_mat(R).transpose(-1, -2) @ R),
                               velocity / .8, angular, q, qd * .05, (local / .8).flatten(-2)), -1),
        }
        postural_fields = [
            ["projected_gravity", 3], ["terrain_relative_height/robot_height", 1],
            ["joint_position-default_q", 6], ["joint_velocity*0.05", 6],
        ]
        frame_layouts = {
            "postural": postural_fields,
            "proprioceptive": [
                postural_fields[0], *postural_fields[2:],
                ["root_local_linear_velocity", 3], ["root_local_angular_velocity", 3],
            ],
            "full": [
                ["terrain_relative_height/robot_height", 1], ["yaw_removed_root_rotation_6d", 6],
                ["root_local_linear_velocity/robot_height", 3], ["root_local_angular_velocity", 3],
                ["joint_position", 6], ["joint_velocity*0.05", 6],
                ["root_local_tracked_position/robot_height", 15],
            ],
        }
        end_effectors = self.learner()
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                learner = self.learner(feature_mode=mode)
                torch.testing.assert_close(learner.frame_features(S0), values)
                layout = {
                    "version": 2 if mode == "proprioceptive" else 1,
                    "endpoints": ["S0", "S1"], "frame": frame_layouts[mode],
                    "robot_height": .8, "minimum_forward_speed_m_s": 0.,
                    "joint_names": list(self.kin.joint_names),
                }
                if mode == "full":
                    layout.update(tracked_indices=list(self.kin.tracked_idx),
                                  tracked_names=[self.kin.body_names[i] for i in self.kin.tracked_idx])
                else:
                    layout.update(feature_mode=mode, default_q=torch.tensor(self.kin.robot.default_q).tolist())
                self.assertEqual(learner.feature_layout, layout)
                checkpoint = learner.checkpoint()
                restored = self.learner(feature_mode=mode)
                restored.restore(checkpoint)
                torch.testing.assert_close(restored.reward(S0, S1), learner.reward(S0, S1), atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, "feature layout"):
                    learner.restore(end_effectors.checkpoint())
                with self.assertRaisesRegex(ValueError, "feature layout"):
                    end_effectors.restore(checkpoint)
                scorer = AMPStyle.for_scoring(self.kin, learner.checkpoint())
                torch.testing.assert_close(scorer.reward(S0, S1), learner.reward(S0, S1), atol=0, rtol=0)
                if mode != "full":
                    with patch.object(self.kin, "forward", side_effect=AssertionError("legacy FK")):
                        self.learner(feature_mode=mode).frame_features(S0)
                        scorer.frame_features(S0)
        default = AMPStyle(self.kin, self.motion, .8, hidden=(16, 8), replay_capacity=16)
        self.assertEqual(default.feature_mode, "full")
        self.assertEqual(default.expert_hash, self.learner(feature_mode="full").expert_hash)


if __name__ == "__main__":
    unittest.main()

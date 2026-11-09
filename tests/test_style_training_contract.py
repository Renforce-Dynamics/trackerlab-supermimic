"""CPU-only training/scoring contracts for the command actor's motion prior."""

import ast
import copy
from io import BytesIO
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
from trackerLab.algorithms.supermimic.networks import Critic, WorldModel
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy, TerrainWorldModel
from trackerLab.kinematics.supermimic.rotations import so3_exp
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp.supermimic import flatten_state, normalize_xi, unflatten_state
from trackerLab.tracker_env.mdp.supermimic_command import (
    CONTROL_DT, SCENARIO_DIM, COMMAND_MAX_STEP_REWARD, CommandMDP, advance_command,
)
from trackerLab.tracker_env.mdp.supermimic_terrain import (
    FALL_PENALTY, MAX_STEP_REWARD, SUCCESS_BONUS, HeightField,
)
from trackerTask.supermimic.terrain import generate


class LocalKinematics:
    num_joints, device = 3, "cpu"
    joint_names = ("hip", "knee", "ankle")
    tracked_idx, body_names = (0, 1), ("torso", "foot")
    robot = SimpleNamespace(default_q=(0.1, -0.2, 0.3))

    def __init__(self):
        self.frames = 0

    def forward(self, p, R, v, w, q, qd):
        self.frames += p.shape[0]
        local = torch.stack((
            torch.stack((q[..., 0].sin(), q[..., 1] * 0.1, 0.2 + q[..., 2] * 0.1), -1),
            torch.stack((q[..., 1].sin(), q[..., 0] * 0.1, -0.6 + q[..., 2] * 0.1), -1),
        ), -2)
        positions = p.unsqueeze(-2) + (R.unsqueeze(-3) @ local.unsqueeze(-1)).squeeze(-1)
        return positions, R.unsqueeze(-3).expand(-1, 2, -1, -1), v[:, None], w[:, None]


def expert_motion(lengths=(4, 5)):
    lengths = torch.tensor(lengths)
    t = torch.arange(int(lengths.sum()), dtype=torch.float32)
    starts = lengths.cumsum(0) - lengths
    return SimpleNamespace(
        root_p=torch.stack((0.02 * t, t * 0, 0.76 + 0.01 * t.sin()), -1),
        root_R=so3_exp(torch.stack((0.04 * t.sin(), 0.05 * t.cos(), 0.02 * t), -1)),
        root_v=torch.stack((0.3 + 0.02 * t, 0.1 * t.sin(), 0.01 * t.cos()), -1),
        root_w=torch.stack((0.01 * t, 0.01 * t.cos(), 0.1 + 0.01 * t.sin()), -1),
        q=torch.stack((0.1 + 0.05 * t.sin(), -0.2 + 0.03 * t.cos(), 0.3 + 0.01 * t), -1),
        qd=torch.stack((0.1 * t.cos(), 0.2 * t.sin(), 0.1 + 0.03 * t), -1),
        clip_len=lengths, clip_start=starts, clip_end=starts + lengths - 1,
        clip_id=torch.repeat_interleave(torch.arange(2), lengths),
        names=["first", "second"], provenance={"source": "CPU training-contract fixture"},
    )


def recorded_state(motion, indices):
    return tuple(getattr(motion, name)[indices].clone()
                 for name in ("root_p", "root_R", "root_v", "root_w", "q", "qd"))


def make_command_mdp(profile="velocity", gamma=.99):
    field = HeightField(generate(profile, .76, 42), .76, "cpu")
    mdp = CommandMDP(field, torch.tensor(LocalKinematics.robot.default_q),
                     torch.full((3,), -3.), torch.full((3,), 3.),
                     profile=profile, gamma=gamma)
    return mdp


def load_training_functions(namespace, names):
    """Run the checked-in trainer helpers on CPU without its CUDA entry point."""
    path = REPO / "scripts/rls/supermimic/train_terrain.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    functions = [
        copy.deepcopy(node) for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != set(names):
        raise AssertionError("Trainer helper API changed; update the CPU integration fixture")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class StyleTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(184)
        self.kin, self.motion = LocalKinematics(), expert_motion()

    def learner(self, mode="proprioceptive", **kwargs):
        return AMPStyle(self.kin, self.motion, .76, hidden=kwargs.pop("hidden", (16, 8)),
                        replay_capacity=16, feature_mode=mode, **kwargs)

    def states(self):
        return recorded_state(self.motion, torch.tensor([0, 4])), \
            recorded_state(self.motion, torch.tensor([1, 5]))

    def assert_nonzero(self, value):
        self.assertIsNotNone(value)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertGreater(value.abs().sum().item(), 1e-10)

    def assert_state_equal(self, before, model):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0, msg=key)

    def test_proprioceptive_layout_and_values_match_robot_state(self):
        self.kin.forward = Mock(side_effect=AssertionError("proprioceptive FK"))
        sampler = Mock(side_effect=AssertionError("proprioceptive height"))
        learner = self.learner(ground_sampler=sampler)
        S0, S1 = self.states()
        p, R, v, w, q, qd = S0
        Rt = R.transpose(-1, -2)
        expected = torch.cat((
            -R[..., 2, :], q - torch.tensor(self.kin.robot.default_q), qd * .05,
            (Rt @ v[..., None]).squeeze(-1), (Rt @ w[..., None]).squeeze(-1),
        ), -1)
        self.assertEqual(learner.frame_dim, 9 + 2 * self.kin.num_joints)
        self.assertEqual(learner.feature_dim, 2 * expected.shape[-1])
        torch.testing.assert_close(learner.frame_features(S0), expected)
        learner.observe(S0, S1)
        learner.update(batch_size=4)
        learner.reward(S0, S1)
        learner.sample_expert(4)
        self.kin.forward.assert_not_called()
        sampler.assert_not_called()

    def test_proprioceptive_height_xy_yaw_and_tracked_body_independence(self):
        learner = self.learner()
        S0, S1 = self.states()
        yaw = so3_exp(torch.tensor([[0., 0., 1.1], [0., 0., -2.2]]))
        changed = []
        for state in (S0, S1):
            p, R, v, w, q, qd = state
            rotate = lambda value: (yaw @ value[..., None]).squeeze(-1)
            changed.append((rotate(p) + torch.tensor([20., -15., 7.]),
                            yaw @ R, rotate(v), rotate(w), q, qd))
        expected = learner.features(S0, S1)
        self.kin.forward = Mock(side_effect=AssertionError("morphology leakage"))
        torch.testing.assert_close(learner.features(*changed), expected)
        torch.testing.assert_close(learner.reward(*changed), learner.reward(S0, S1))
        self.kin.forward.assert_not_called()

    def test_reward_from_raw_features_and_public_score_share_fixed_norm(self):
        learner = self.learner()
        S0, S1 = self.states()
        raw = learner.features(S0, S1).detach().requires_grad_()
        before = copy.deepcopy(learner.norm.state_dict())
        with learner.freeze():
            score = learner.score(raw)
            expected = (1 - .25 * (score - 1).square()).clamp_min(0)
            actual = learner.reward_from_features(raw)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual, learner.reward(S0, S1))
            actual.sum().backward()
        self.assert_nonzero(raw.grad)
        self.assert_state_equal(before, learner.norm)
        self.assertEqual(learner.replay_size, 0)

    def test_mode_and_default_joint_layout_are_checkpoint_identity(self):
        proprio, full = self.learner(), self.learner("full")
        self.assertNotEqual(proprio.expert_hash, full.expert_hash)
        for source, target in ((proprio, full), (full, proprio)):
            with self.assertRaisesRegex(ValueError, "expert identity"):
                target.restore(source.checkpoint())
        self.kin.robot = SimpleNamespace(default_q=(0.2, -0.2, 0.3))
        changed = self.learner()
        self.assertNotEqual(changed.expert_hash, proprio.expert_hash)
        with self.assertRaisesRegex(ValueError, "expert identity"):
            changed.restore(proprio.checkpoint())
        with self.assertRaisesRegex(ValueError, "feature_mode"):
            self.learner("unknown")

    def test_resume_identity_follows_frozen_experts_not_source_path_or_names(self):
        source = self.learner()
        S0, S1 = self.states()
        source.observe(S0, S1)
        source.update(batch_size=4, updates=2)
        checkpoint = source.checkpoint()
        self.motion.names = ["moved/first", "moved/second"]
        self.motion.provenance = {"source": "a different manifest path"}
        target = self.learner()
        self.assertEqual(target.expert_hash, source.expert_hash)
        receipt = target.restore(checkpoint)
        self.assertTrue(receipt["replay_refill_required"])
        self.assertFalse(receipt["replay_restored"])
        self.assertEqual(target.replay_size, 0)
        self.assert_state_equal(source.norm.state_dict(), target.norm)
        self.assert_state_equal(source.discriminator.state_dict(), target.discriminator)
        self.assertEqual(target.optimizer.state_dict()["param_groups"],
                         source.optimizer.state_dict()["param_groups"])
        for state_id, state in source.optimizer.state_dict()["state"].items():
            for name, expected in state.items():
                torch.testing.assert_close(target.optimizer.state_dict()["state"][state_id][name],
                                           expected, atol=0, rtol=0)
        self.motion.q[0, 1] += .01
        changed = self.learner()
        with self.assertRaisesRegex(ValueError, "expert identity"):
            changed.restore(checkpoint)

    def test_compiled_proprioceptive_reward_keeps_action_gradient_without_fk(self):
        self.kin.forward = Mock(side_effect=AssertionError("compiled FK"))
        learner = self.learner().requires_grad_(False)
        S0, S1 = self.states()
        action = torch.full_like(S1[4], .03, requires_grad=True)
        following = (*S1[:4], S1[4] + .2 * action, S1[5])
        compiled = torch.compile(learner.reward, backend="aot_eager", fullgraph=True)
        expected = learner.reward(S0, following)
        actual = compiled(S0, following)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assert_nonzero(action.grad)
        self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))
        self.kin.forward.assert_not_called()

    def test_scoring_factory_has_reward_parity_without_motion_files_or_adam(self):
        for mode in ("full", "proprioceptive"):
            with self.subTest(mode=mode):
                source = self.learner(mode)
                S0, S1 = self.states()
                source.observe(S0, S1)
                source.update(batch_size=4)
                checkpoint = source.checkpoint()
                # A scorer accepts a slim inference payload too.
                for key in ("optimizer", "observed", "updates", "gradient_penalty_weight"):
                    checkpoint.pop(key)
                with patch.object(MotionLib, "__init__", side_effect=AssertionError("motion load")), \
                        patch.object(MotionLib, "from_config", side_effect=AssertionError("source files")), \
                        patch.object(MotionLib, "ref_state", side_effect=AssertionError("reference read")), \
                        patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer allocation")):
                    scorer = AMPStyle.for_scoring(self.kin, checkpoint)
                    self.assertFalse(scorer.training)
                    self.assertFalse(hasattr(scorer, "optimizer"))
                    self.assertFalse(hasattr(scorer, "_expert_frames"))
                    self.assertFalse(hasattr(scorer, "_expert_starts"))
                    self.assertFalse(hasattr(scorer, "_replay"))
                    torch.testing.assert_close(scorer.reward(S0, S1), source.reward(S0, S1),
                                               atol=0, rtol=0)
                    self.assertEqual(scorer.expert_hash, source.expert_hash)
                    self.assertTrue(all(not p.requires_grad for p in scorer.parameters()))
                    for operation in (lambda: scorer.observe(S0, S1), scorer.update,
                                      scorer.checkpoint, lambda: scorer.restore(checkpoint),
                                      lambda: scorer.sample_expert(1)):
                        with self.assertRaisesRegex(RuntimeError, "Scoring-only"):
                            operation()

    def test_scorer_terrain_and_serialized_checkpoint_reward_parity(self):
        field = make_command_mdp("stairs").field
        learner = self.learner("full", ground_sampler=field.sample)
        S0, S1 = self.states()
        stream = BytesIO()
        torch.save(learner.checkpoint(), stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
        scorer = AMPStyle.for_scoring(self.kin, checkpoint, ground_sampler=field.sample)
        torch.testing.assert_close(scorer.reward(S0, S1), learner.reward(S0, S1), atol=0, rtol=0)
        q = S1[4].clone().requires_grad_()
        scorer.reward(S0, (*S1[:4], q, S1[5])).sum().backward()
        self.assert_nonzero(q.grad)

    def test_scorer_rejects_malformed_metadata_and_weights(self):
        checkpoint = self.learner().checkpoint()
        cases = [
            ("version", 99), ("expert_hash", "wrong"), ("feature_mode", "unknown"),
            ("feature_dim", checkpoint["feature_dim"] + 2), ("feature_layout", {}),
            ("norm", {}), ("discriminator", {"0.weight": torch.ones(2, 2)}),
        ]
        for key, value in cases:
            with self.subTest(field=key):
                invalid = copy.deepcopy(checkpoint)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    AMPStyle.for_scoring(self.kin, invalid)
        invalid = copy.deepcopy(checkpoint)
        invalid["norm"]["mean"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            AMPStyle.for_scoring(self.kin, invalid)
        invalid = copy.deepcopy(checkpoint)
        invalid["feature_layout"]["default_q"][0] += .01
        with self.assertRaisesRegex(ValueError, "layout"):
            AMPStyle.for_scoring(self.kin, invalid)

    def test_evaluator_uses_checkpoint_scorer_without_importing_motion_library(self):
        path = REPO / "scripts/rls/supermimic/evaluate_terrain.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source = ast.unparse(tree)
        self.assertIn("AMPStyle.for_scoring(kin, ck['amp_style'], ground_sampler=field.sample)", source)
        self.assertNotIn("MotionLib", source)
        self.assertNotIn("amp_motion_config", source)

    def test_unique_frame_reuse_matches_pairwise_rewards_and_gradients_for_32_steps(self):
        learner = self.learner("full").requires_grad_(False)
        S0, _ = self.states()
        sequence = tuple(value.unsqueeze(0).repeat(33, *([1] * value.ndim)) for value in S0)
        offsets = torch.arange(33)[:, None, None] * .001
        q = (sequence[4] + offsets).requires_grad_()
        sequence = (*sequence[:4], q, sequence[5])
        previous, following = tuple(value[:-1] for value in sequence), tuple(value[1:] for value in sequence)
        self.kin.frames = 0
        pairwise = learner.reward(previous, following)
        self.assertEqual(self.kin.frames, 64 * len(S0[0]))
        pairwise_gradient = torch.autograd.grad(pairwise.sum(), q)[0]
        self.kin.frames = 0
        flat = tuple(value.flatten(0, 1) for value in sequence)
        frames = learner.frame_features(flat).reshape(33, len(S0[0]), learner.frame_dim)
        reused = learner.reward_from_features(torch.cat((frames[:-1], frames[1:]), -1))
        self.assertEqual(self.kin.frames, 33 * len(S0[0]))
        reused_gradient = torch.autograd.grad(reused.sum(), q)[0]
        torch.testing.assert_close(reused, pairwise, atol=0, rtol=0)
        torch.testing.assert_close(reused_gradient, pairwise_gradient, atol=1e-6, rtol=1e-5)

    def test_numerical_style_gradient_reaches_actor_through_frozen_world_model(self):
        learner = self.learner(hidden=())
        world = WorldModel(hidden=(16, 16), num_joints=3)
        actor = nn.Linear(3, 3)
        with torch.no_grad():
            world.net[-1].weight.normal_(std=.04)
            learner.discriminator[-1].weight.fill_(.01)
        world.requires_grad_(False)
        learner.requires_grad_(False)
        initial, _ = self.states()
        xi_n = torch.zeros(2, 7)

        def objective():
            state = initial
            total = torch.zeros(2)
            for step in range(6):
                action = actor(state[4])
                next_state = world.step(state, action, xi_n)
                total = total + .99 ** step * .5 * CONTROL_DT * learner.reward(state, next_state)
                state = next_state
            return total.mean()

        objective().backward()
        gradient = actor.weight.grad
        self.assert_nonzero(gradient)
        index = tuple(int(value) for value in torch.unravel_index(gradient.abs().argmax(), gradient.shape))
        expected = gradient[index].item()
        epsilon = .005
        with torch.no_grad():
            actor.weight[index] += epsilon
            plus = objective().item()
            actor.weight[index] -= 2 * epsilon
            minus = objective().item()
            actor.weight[index] += epsilon
        self.assertAlmostEqual((plus - minus) / (2 * epsilon), expected,
                               delta=max(3e-6, abs(expected) * .05))
        self.assertTrue(all(parameter.grad is None for parameter in world.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))
        self.assertEqual(learner.replay_size, 0)

    def test_production_policy_update_gets_style_only_actor_gradient_on_cpu(self):
        self.motion = expert_motion((12, 13))
        mdp = make_command_mdp()
        mdp.set_style_weight(.5)
        initial = recorded_state(self.motion, torch.tensor([9, 21]))
        actor = TerrainPolicy(mdp.obs_dim, mdp.goal_dim, hist=10, num_joints=3, action_limit=4.)
        world = TerrainWorldModel(mdp.field, 3, hidden=(16, 16))
        with torch.no_grad():
            world.net[-1].weight.normal_(std=.04)
        critic = Critic(mdp.critic_dim)
        with torch.no_grad():
            critic.net[-1].weight.zero_()
            critic.net[-1].bias.zero_()
        learner = self.learner(history_steps=10)
        style_norm = copy.deepcopy(learner.norm.state_dict())
        world_before = copy.deepcopy(world.state_dict())
        style_before = copy.deepcopy(learner.discriminator.state_dict())
        command = torch.zeros(2, SCENARIO_DIM)
        command[:, 4] = mdp.target_clearance_m
        xi = torch.zeros(2, 7)
        xi[:, 0] = .75
        initial_flat = flatten_state(initial)
        data = {"s": initial_flat, "xi": xi, "c": command, "a_prev": torch.zeros(2, 3)}
        history = {key: value[:, None].expand(-1, 10, *value.shape[1:]) for key, value in data.items()}
        ids = torch.stack((torch.arange(10), torch.arange(12, 22)))
        history["s"] = flatten_state(recorded_state(self.motion, ids))
        args = SimpleNamespace(device="cpu", compile=False, profile=False, horizon=3, hist=10, pi_batch=2,
                               amp_weight=.5, gamma=.99, lam=.95,
                               critic_epochs=1, critic_minibatches=1)
        namespace = {
            "torch": torch, "args": args, "actor": actor, "world": world,
            "critic": critic, "style": learner,
            "style_reward_fn": learner.reward_from_features,
            "style_frame_fn": learner.frame_features,
            "regularization_reward_fn": lambda previous, state, action, xi: action.sum(-1) * 0,
            "upper_body_constraint": None,
            "entry_style_reward_fn": None,
            "action_constraint": None,
            "mdp": mdp, "advance_command": advance_command,
            "normalize_xi": normalize_xi, "unflatten_state": unflatten_state,
            "sample_starts": lambda _: (data, history),
            "opt_pi": torch.optim.Adam(actor.parameters(), lr=1e-4),
            "opt_v": torch.optim.Adam(critic.parameters(), lr=1e-4),
            "critic_dim": mdp.critic_dim,
            "critic_loss_fn": lambda inputs, target, indices: (
                critic(inputs[indices]) - target[indices]).square().mean(),
            "reward_term": lambda state, *rest: (state[0][:, 0] * 0, state[0][:, 0] * 0),
        }
        load_training_functions(namespace, ("imag_step", "critic_input", "value_estimate", "policy_update"))
        with patch.object(MotionLib, "ref_state", side_effect=AssertionError("reference leak")), \
                patch.object(MotionLib, "goal", side_effect=AssertionError("goal playback")), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("GPU access")):
            result = namespace["policy_update"]()
        self.assertTrue(all(bool(torch.isfinite(value)) for value in result))
        self.assert_nonzero(actor.dec[-1].weight.grad)
        self.assert_nonzero(actor.enc[0].weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in world.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in learner.parameters()))
        self.assert_state_equal(world_before, world)
        self.assert_state_equal(style_before, learner.discriminator)
        self.assert_state_equal(style_norm, learner.norm)
        self.assertEqual(learner.replay_size, 0)

    def test_actor_observation_and_command_goal_use_actual_state_and_no_reference(self):
        mdp = make_command_mdp()
        S0, _ = self.states()
        c = torch.zeros(2, SCENARIO_DIM)
        c[:, 1:4], c[:, 4] = torch.tensor([.4, -.2, .3]), mdp.target_clearance_m
        xi, previous = torch.zeros(2, 7), torch.zeros(2, 3)
        with patch.object(MotionLib, "ref_state", side_effect=AssertionError("reference pose")), \
                patch.object(MotionLib, "goal", side_effect=AssertionError("reference goal")):
            obs, goal = mdp.obs(S0, previous, xi, c), mdp.goal(S0, c)
            torch.testing.assert_close(obs[:, :3], S0[4] - mdp.default_q)
            torch.testing.assert_close(obs[:, 6:9],
                                       (S0[1].transpose(-1, -2) @ S0[3][..., None]).squeeze(-1) * .25)
            torch.testing.assert_close(obs[:, 9:12], -S0[1][:, 2, :])
            torch.testing.assert_close(goal[:, :2], c[:, 1:3] / mdp.field.scale)
            torch.testing.assert_close(goal[:, 7:], torch.zeros(2, 5))
            self.motion.root_p.add_(1000.)
            self.motion.root_R.zero_()
            self.motion.q.add_(100.)
            torch.testing.assert_close(mdp.obs(S0, previous, xi, c), obs, atol=0, rtol=0)
            torch.testing.assert_close(mdp.goal(S0, c), goal, atol=0, rtol=0)
        self.assertFalse(hasattr(mdp, "motion"))

    def test_style_terminal_payout_replaces_continuation_and_has_goal_gradient(self):
        if not hasattr(CommandMDP, "set_style_weight"):
            self.skipTest("CommandMDP style-aware terminal payout has not landed")
        for profile in ("stairs", "mountain", "parkour"):
            for gamma in (.9, .99, .999):
                mdp = make_command_mdp(profile, gamma=gamma)
                mdp.set_style_weight(.5)
                maximum = COMMAND_MAX_STEP_REWARD + .5
                expected = SUCCESS_BONUS + gamma * maximum / (1 - gamma)
                self.assertAlmostEqual(mdp.success_reward, expected)
                goal_step_return = maximum + expected
                for delay in (1, 32, 500):
                    delayed = (maximum * (1 - gamma ** delay) / (1 - gamma)
                               + gamma ** delay * goal_step_return)
                    self.assertGreater(goal_step_return, delayed)
        mdp = make_command_mdp("stairs")
        mdp.set_style_weight(.5)
        c = torch.zeros(1, SCENARIO_DIM)
        c[:, 0], c[:, 4] = mdp.field.terrain.centers[0], mdp.target_clearance_m
        p = torch.tensor([[mdp.field.terrain.goal_x, c[0, 0], 1.2]], requires_grad=True)
        S = (p, torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3),
             mdp.default_q[None], torch.zeros(1, 3))
        terminal = mdp.soft_terminal_reward(S, c)
        terminal.sum().backward()
        self.assertGreater(p.grad[0, 0].item(), 0)
        for profile in ("velocity", "stealth"):
            mdp = make_command_mdp(profile)
            mdp.set_style_weight(3.)
            self.assertEqual(mdp.success_reward, 0)
            self.assertEqual(mdp.terminal_reward(torch.tensor([False]), torch.tensor([True])).item(),
                             -FALL_PENALTY)


if __name__ == "__main__":
    unittest.main()

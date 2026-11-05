"""CPU contracts for reference-free command locomotion and its Warp driver."""

from contextlib import nullcontext
import inspect
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.buffer import ReplayBuffer
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy, TerrainWorldModel
from trackerLab.kinematics.supermimic.rotations import so3_exp, yaw_mat
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp.supermimic import flatten_state
from trackerLab.tracker_env.mdp.supermimic_command import (
    COMMAND_FIELDS, COMMAND_VERSION, COMMAND_MAX_STEP_REWARD, CONTROL_DT, GOAL_DIM, JUMP_DURATION,
    SCENARIO_DIM, CommandMDP, advance_command, body_velocities, jump_pulse,
    target_clearance,
)
from trackerLab.tracker_env.mdp.supermimic_terrain import (
    FALL_PENALTY, MAX_STEP_REWARD, SUCCESS_BONUS, HeightField,
)
from trackerLab.tracker_env.supermimic.command import CommandSim
from trackerLab.tracker_env.supermimic.terrain import TerrainSim
from trackerTask.supermimic.registry import TASK_REGISTRY, get_task as registry_task
from trackerTask.supermimic.gait import FOOT_BODIES
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import (
    ALL_TASKS, COMMAND_TASKS, PROFILES, ROBOTS, STEALTH_TASKS, TASKS, generate, get_task,
)


def make_mdp(profile="velocity", n=4, clearance_ratio=1., gamma=.99):
    field = HeightField(generate(profile, .76, 42), .76, "cpu")
    return CommandMDP(field, torch.zeros(n), torch.full((n,), -10.),
                      torch.full((n,), 10.), gamma=gamma, profile=profile,
                      clearance_ratio=clearance_ratio)


def make_state(mdp, batch=1):
    p = torch.zeros(batch, 3)
    if mdp.terrain_navigation:
        p[:, 1] = mdp.field.terrain.centers[0]
    p[:, 2] = mdp.target_clearance_m + mdp.field.sample(p[:, :2])
    return (p, torch.eye(3).expand(batch, 3, 3).clone(), torch.zeros(batch, 3),
            torch.zeros(batch, 3), mdp.default_q.expand(batch, -1).clone(),
            torch.zeros(batch, mdp.num_joints))


def make_command(mdp, batch=1):
    c = torch.zeros(batch, SCENARIO_DIM)
    if mdp.terrain_navigation:
        c[:, 0] = mdp.field.terrain.centers[0]
    c[:, 4] = mdp.target_clearance_m
    return c


def make_reset_motion(sim, frames=1):
    p = torch.tensor([[12., -8., sim.target_clearance_m]]).expand(frames, -1).clone()
    R = (so3_exp(torch.tensor([[0., 0., math.pi/2]]))
         @ so3_exp(torch.tensor([[.15, -.1, 0.]]))).expand(frames, -1, -1).clone()
    q = torch.tensor([[.1, -.2, .15, -.1]]).expand(frames, -1).clone()
    feet = len(sim.spec.tracked_bodies)
    return SimpleNamespace(
        root_p=p, root_R=R, root_v=torch.tensor([[.1, .4, .02]]).expand(frames, -1).clone(),
        root_w=torch.tensor([[.1, -.05, .2]]).expand(frames, -1).clone(),
        q=q, qd=torch.tensor([[.1, .2, .3, .4]]).expand(frames, -1).clone(),
        body_p=torch.tensor([[[12., -8., .03]]]).expand(frames, feet, -1).clone(),
        body_R=torch.eye(3).expand(frames, feet, 3, 3).clone(),
        ref_state=Mock(side_effect=AssertionError("reference playback")),
    )


class CpuCommandSim(CommandSim):
    """Exercise the production reset/step methods with stationary stub physics."""

    def __init__(self, profile="velocity", batch=1, max_episode_steps=1000,
                 jump_height=0., clearance_ratio=1., action_limit=4.,
                 stealth_keyboard_mode=False):
        self.mdp = make_mdp(profile, clearance_ratio=clearance_ratio)
        self.task = get_task(f"locomotion/g1/{profile}")
        self.robot, self.device, self.num_envs = "g1", "cpu", batch
        self.height, self.scale = .76, 1.
        self.spec = SimpleNamespace(num_joints=self.mdp.num_joints,
                                    tracked_bodies=FOOT_BODIES[self.robot])
        self.contract = {}
        self.self_collisions = False
        self.stealth_keyboard_mode = bool(stealth_keyboard_mode)
        self.terrain = self.mdp.field.terrain
        self.centers = torch.tensor(self.terrain.centers, dtype=torch.float32)
        self.default_q = self.mdp.default_q
        self.action_scale = torch.full_like(self.default_q, .25)
        self.qlo, self.qhi = torch.full_like(self.default_q, -10.), torch.full_like(self.default_q, 10.)
        n = self.spec.num_joints
        self.qpos, self.qvel = torch.zeros(batch, 7+n), torch.zeros(batch, 6+n)
        self.ctrl = torch.zeros(batch, n)
        self._root_qpos, self._root_qvel = 0, 0
        self._qpos_ids, self._qvel_ids, self._actuator_ids = torch.arange(7, 7+n), torch.arange(6, 6+n), torch.arange(n)
        self.ep_step = torch.zeros(batch, dtype=torch.int32)
        self.ep_return, self.a_prev = torch.zeros(batch), torch.zeros(batch, n)
        self.xi = torch.zeros(batch, n+4)
        self.xi[:, 0] = .75
        self.c = torch.zeros(batch, 2)
        self.levels = torch.zeros(batch, dtype=torch.long)
        self.max_episode_steps = max_episode_steps
        self.curriculum = self.mdp.terrain_navigation
        self.physical_ray_validation = False
        self.target_clearance_m, self.jump_height = self.mdp.target_clearance_m, jump_height
        self.action_limit, self.command_resample_steps, self.jump_probability = action_limit, (100, 250), .15
        self.min_success_steps = math.ceil(10. / CONTROL_DT)
        self.physics_graph = object()
        self._stream = object()
        self.model, self.data = object(), object()
        self.wp = SimpleNamespace(from_torch=lambda value: value, capture_launch=Mock())
        self.mw = SimpleNamespace(reset_data=self._reset_data, forward=Mock())
        self._reset_state(torch.arange(batch))

    def _warp_context(self):
        return nullcontext()

    def _randomize(self, ids, initialize=False):
        pass

    def _reset_data(self, model, data, mask):
        self.qpos[mask] = 0
        self.qvel[mask] = 0


class CommandLocomotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_task_union_preserves_terrain_registry_count_and_ids(self):
        expected = [f"locomotion/{robot}/{profile}" for robot in ROBOTS for profile in PROFILES]
        self.assertEqual(list(TASKS), expected)
        self.assertEqual((len(TASKS), len(COMMAND_TASKS), len(STEALTH_TASKS), len(ALL_TASKS)),
                         (18, 6, 5, 29))
        self.assertEqual(sum(entry.runtime_task == "terrain" for entry in TASK_REGISTRY.values()), 18)
        self.assertEqual(sum(entry.runtime_task == "command" for entry in TASK_REGISTRY.values()), 11)
        for name, task in ALL_TASKS.items():
            self.assertIs(get_task(name), task)
            description = task.describe()
            self.assertEqual(description["objective"], "command-driven-locomotion")
            self.assertEqual(description["objective_version"], COMMAND_VERSION)
            self.assertFalse(description["runtime_reference_motion"])
            registry_description = registry_task(name).describe()
            self.assertEqual(registry_description["scenario_dim"], SCENARIO_DIM)
        for lookup in (get_task, registry_task):
            with self.assertRaises(ValueError):
                lookup("locomotion/go2/stealth")

    def test_command_heightfields_are_flat_finite_symmetric_and_scaled(self):
        for profile in ("velocity", "stealth"):
            full, half = generate(profile, .76), generate(profile, .38, 9)
            self.assertEqual(full.profile, profile)
            self.assertGreaterEqual(min(full.heights.shape), 2)
            self.assertTrue(np.isfinite(full.heights).all())
            self.assertTrue(np.isfinite(full.normalized).all())
            self.assertTrue(np.all(full.heights == 0))
            self.assertEqual(full.x_min, -full.x_max)
            self.assertEqual(full.y_min, -full.y_max)
            self.assertLess(full.x_min, -20)
            self.assertEqual(half.x_min, full.x_min * .5)
            self.assertEqual(len(full.centers), 6)
            self.assertGreater(full.size[2], 0)
        for invalid in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                generate("velocity", invalid)

    def test_every_profile_has_compatible_actor_and_goal_dimensions(self):
        for robot in ROBOTS:
            n = get_robot(robot).num_joints
            actor = None
            for profile in (*PROFILES, "velocity", "stealth"):
                mdp = make_mdp(profile, n)
                S, c = make_state(mdp, 2), make_command(mdp, 2)
                xi, previous = torch.zeros(2, n+4), torch.zeros(2, n)
                obs, goal = mdp.obs(S, previous, xi, c), mdp.goal(S, c)
                self.assertEqual(mdp.obs_dim, 3*n + 196)
                self.assertEqual(obs.shape, (2, mdp.obs_dim))
                self.assertEqual(goal.shape, (2, GOAL_DIM))
                self.assertEqual(mdp.scenario_dim, 7)
                self.assertEqual(mdp.obs_noise.shape, (mdp.obs_dim,))
                self.assertEqual(mdp.critic_features(S, xi, c).shape, (2, mdp.critic_dim))
                if actor is None:
                    actor = TerrainPolicy(mdp.obs_dim, mdp.goal_dim, hist=2, num_joints=n)
                self.assertEqual(actor(obs[:, None].expand(-1, 2, -1), goal).shape, (2, n))

    def test_no_motion_library_or_reference_state_is_consumed(self):
        with patch.object(MotionLib, "from_config", side_effect=AssertionError("motion load")), \
                patch.object(MotionLib, "ref_state", side_effect=AssertionError("reference state")):
            mdp = make_mdp()
            S, c = make_state(mdp), make_command(mdp)
            a, xi = torch.zeros(1, mdp.num_joints), torch.zeros(1, mdp.num_joints+4)
            for value in (mdp.obs(S, a, xi, c), mdp.goal(S, c), mdp.reward(S, a, a, c),
                          mdp.critic_features(S, xi, c)):
                self.assertTrue(torch.isfinite(value).all())
            sim = CpuCommandSim()
            sim.step(a)
            self.assertFalse(hasattr(sim, "motion"))
            self.assertFalse(hasattr(mdp, "motion"))

    def test_body_velocity_and_yaw_commands_use_full_body_frame(self):
        mdp = make_mdp()
        S = list(make_state(mdp, 4))
        axes = torch.tensor([[0., 0., 0.], [0., 0., math.pi/2],
                             [0., 0., math.pi], [.25, -.2, .7]])
        S[1] = so3_exp(axes)
        local_v = torch.tensor([[.6, -.3, .1]]).expand(4, -1)
        local_w = torch.tensor([[.1, -.1, .8]]).expand(4, -1)
        S[2] = (S[1] @ local_v.unsqueeze(-1)).squeeze(-1)
        S[3] = (S[1] @ local_w.unsqueeze(-1)).squeeze(-1)
        v, w = body_velocities(S)
        torch.testing.assert_close(v, local_v)
        torch.testing.assert_close(w, local_w)
        c = make_command(mdp, 4)
        c[:, 1:4] = torch.tensor([.6, -.3, .8])
        _, terms = mdp.reward(S, S[4], S[4], c, terms=True)
        torch.testing.assert_close(terms["body_velocity"], torch.full((4,), .04))
        torch.testing.assert_close(terms["yaw_rate"], torch.full((4,), .04))
        torch.testing.assert_close(mdp.command_metrics(S, c)["velocity_error_mps"], torch.zeros(4),
                                   atol=1e-6, rtol=0)

    def test_stop_reverse_sidestep_and_turn_have_no_forward_bonus(self):
        mdp = make_mdp()
        S, c = list(make_state(mdp, 6)), make_command(mdp, 6)
        c[:, 1:4] = torch.tensor([[0., 0., 0.], [.7, 0., 0.], [-.7, 0., 0.],
                                  [0., .7, 0.], [0., -.7, 0.], [0., 0., -.8]])
        S[2][:, :2], S[3][:, 2] = c[:, 1:3], c[:, 3]
        reward, terms = mdp.reward(S, S[4], S[4], c, terms=True)
        torch.testing.assert_close(reward, reward[:1].expand(6))
        for name in ("progress", "lane", "heading"):
            torch.testing.assert_close(terms[name], torch.zeros(6))
        stop = make_command(mdp, 6)
        stop_reward = mdp.reward(S, S[4], S[4], stop)
        self.assertGreater(stop_reward[0].item(), stop_reward[1:].max().item())

    def test_flat_goal_has_zero_navigation_and_observation_has_no_heading_target(self):
        mdp = make_mdp()
        S, c = make_state(mdp), make_command(mdp)
        a, xi = torch.zeros(1, 4), torch.zeros(1, 8)
        original_obs, original_goal = mdp.obs(S, a, xi, c), mdp.goal(S, c)
        changed = list(S)
        changed[0] = torch.tensor([[15., 7., .76]])
        changed[1] = so3_exp(torch.tensor([[0., 0., 1.4]]))
        altered = c.clone()
        altered[:, 0] = -1234.
        torch.testing.assert_close(mdp.obs(changed, a, xi, altered), original_obs)
        torch.testing.assert_close(mdp.goal(changed, altered), original_goal)
        torch.testing.assert_close(original_goal[:, 7:], torch.zeros(1, 5))
        self.assertFalse(mdp.termination(changed, altered).item())

    def test_terrain_navigation_rewards_and_goal_crossing_stay_enabled(self):
        mdp = make_mdp("stairs")
        S, c = list(make_state(mdp, 2)), make_command(mdp, 2)
        S[2][:, 0] = torch.tensor([1., -1.])
        _, terms = mdp.reward(S, S[4], S[4], c, terms=True)
        self.assertGreater(terms["progress"][0].item(), terms["progress"][1].item())
        self.assertTrue((mdp.goal(S, c)[:, -1] == 1).all())
        S[0][0, 1] += 2.
        self.assertTrue(mdp.termination(S, c)[0].item())

    def test_configured_stealth_clearance_is_physical_and_changes_height_target(self):
        mdp = make_mdp("stealth", clearance_ratio=.65)
        self.assertAlmostEqual(mdp.target_clearance_m, .76*.65)
        S, c = list(make_state(mdp, 2)), make_command(mdp, 2)
        S[0][1, 2] = .76
        reward = mdp.reward(S, S[4], S[4], c)
        self.assertGreater(reward[0].item(), reward[1].item())
        self.assertFalse(mdp.termination(S, c)[0].item())
        self.assertAlmostEqual(mdp.goal(S, c)[0, 3].item(), .76*.65, places=6)
        explicit = CommandMDP(mdp.field, mdp.default_q, mdp.soft_lo, mdp.soft_hi,
                               target_clearance_m=.43)
        self.assertEqual(explicit.target_clearance_m, .43)
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                make_mdp(clearance_ratio=value)

    def test_jump_pulse_uses_clock_and_stops_after_point_eight_seconds(self):
        mdp = make_mdp()
        c = make_command(mdp, 6)
        c[:, 5] = .2
        c[:, 6] = torch.tensor([-.1, 0., .2, .4, .8, 1.6])
        torch.testing.assert_close(jump_pulse(c), torch.tensor([0., 0., .1, .2, 0., 0.]))
        torch.testing.assert_close(target_clearance(c), c[:, 4] + jump_pulse(c))
        c[:, 5] = 0
        torch.testing.assert_close(jump_pulse(c), torch.zeros(6))

    def test_advance_command_is_pure_differentiable_and_never_resamples(self):
        c = make_command(make_mdp(), 3).requires_grad_()
        original = c.detach().clone()
        next_c = advance_command(c)
        torch.testing.assert_close(c, original)
        torch.testing.assert_close(next_c[:, :6], c[:, :6])
        torch.testing.assert_close(next_c[:, 6], torch.full((3,), CONTROL_DT))
        next_c.sum().backward()
        torch.testing.assert_close(c.grad, torch.ones_like(c))
        for _ in range(100):
            next_c = advance_command(next_c)
        torch.testing.assert_close(next_c[:, 6], torch.full((3,), JUMP_DURATION))
        torch.testing.assert_close(next_c[:, :6], original[:, :6])
        for dt in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                advance_command(c, dt)
        with self.assertRaises(ValueError):
            advance_command(torch.zeros(1, 2))
        self.assertEqual(len(COMMAND_FIELDS), SCENARIO_DIM)

    def test_advance_command_fullgraph_compile_matches_eager_values_and_gradients(self):
        compiled = torch.compile(advance_command, backend="aot_eager", fullgraph=True)
        for batch in ((3,), (2, 3)):
            for dt in (CONTROL_DT, .04):
                c = torch.zeros(*batch, SCENARIO_DIM)
                c[..., 1:4] = torch.tensor([-.5, .2, -.3])
                c[..., 4:6] = torch.tensor([.76, .2])
                c[..., 6] = torch.tensor([.1, .79, .8])
                eager_input, compiled_input = c.clone().requires_grad_(), c.clone().requires_grad_()
                expected = advance_command(eager_input, dt)
                actual = compiled(compiled_input, dt)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(compiled_input, c)
                torch.testing.assert_close(
                    torch.autograd.grad(actual.square().sum(), compiled_input)[0],
                    torch.autograd.grad(expected.square().sum(), eager_input)[0],
                )
        default_input = torch.zeros(2, SCENARIO_DIM)
        torch.testing.assert_close(compiled(default_input), advance_command(default_input))

    def test_jump_height_and_clock_have_reward_gradients(self):
        mdp = make_mdp()
        c = make_command(mdp)
        c[:, 5:7] = torch.tensor([.2, .2])
        c.requires_grad_()
        S = make_state(mdp)
        reward = mdp.reward(S, S[4], S[4], c)
        reward.sum().backward()
        self.assertTrue(torch.isfinite(c.grad).all())
        self.assertGreater(c.grad[:, 5:7].abs().min().item(), 0.)
        self.assertTrue(torch.isfinite(mdp.goal(S, c)).all())

    def test_soft_failures_ignore_flat_goal_and_lane_but_include_field_edges(self):
        mdp = make_mdp()
        S, c = list(make_state(mdp, 3)), make_command(mdp, 3)
        S[0][:, :2] = torch.tensor([[25., 12.], [0., 0.], [33., 0.]])
        S[0][1, 2] = .1
        success, fall = mdp.soft_outcome_probabilities(S, c)
        torch.testing.assert_close(success, torch.zeros(3))
        self.assertLess(fall[0].item(), .001)
        self.assertGreater(fall[1:].min().item(), .99)
        self.assertEqual(mdp.termination(S, c).tolist(), [False, True, True])
        S[2][0, 1] = float("nan")
        self.assertTrue(mdp.termination(S, c)[0].item())
        self.assertEqual(mdp.soft_outcome_probabilities(S, c)[1][0].item(), 1.)

    def test_imagined_world_model_rewards_and_critic_use_command_context(self):
        mdp = make_mdp("parkour")
        S, c = make_state(mdp, 2), make_command(mdp, 2)
        S = tuple(value.clone().requires_grad_() for value in S)
        a, previous = torch.full((2, 4), .1, requires_grad=True), torch.zeros(2, 4)
        world = TerrainWorldModel(mdp.field, 4, hidden=(32, 32))
        xi = torch.zeros(2, 8)
        next_state, next_c = world.step(S, a, xi), advance_command(c)
        reward, done = mdp.imagined_reward(next_state, a, previous, next_c)
        torch.testing.assert_close(
            reward, mdp.reward(next_state, a, previous, next_c) + mdp.soft_terminal_reward(next_state, next_c))
        torch.testing.assert_close(done, mdp.soft_termination(next_state, next_c))
        reward.sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(a.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(mdp.critic_features(next_state, xi, next_c)).all())

    def test_nonfinite_root_positions_reach_failure_masks_without_sampler_errors(self):
        mdp = make_mdp()
        for index in range(3):
            for value in (float("nan"), float("inf"), -float("inf")):
                S, c = make_state(mdp), make_command(mdp)
                S[0][0, index] = value
                self.assertTrue(mdp.termination(S, c).item())
                success, fall = mdp.soft_outcome_probabilities(S, c)
                self.assertEqual(success.item(), 0.)
                self.assertEqual(fall.item(), 1.)

    def test_parent_scenario_allocation_is_backward_compatible(self):
        self.assertEqual(inspect.signature(TerrainSim).parameters["scenario_dim"].default, 2)
        sim = CpuCommandSim(batch=2)
        self.assertEqual(sim.c.shape, (2, 7))
        self.assertEqual(sim._command_age.shape, (2,))
        sim.c[1, 1:4] = torch.tensor([.4, -.2, 1.])
        preserved = sim.c[1].clone()
        sim.reset(torch.tensor([0]), levels=torch.tensor([4]))
        torch.testing.assert_close(sim.c[1], preserved)
        self.assertEqual(sim.levels[0].item(), 4)
        terrain = CpuCommandSim("stairs")
        terrain.curriculum = False
        terrain.reset(levels=torch.tensor([4]))
        self.assertEqual(terrain.levels.item(), 4)
        self.assertEqual(terrain.c[0, 0].item(), terrain.centers[4].item())

    def test_real_and_imagined_jump_advance_match_without_teleporting_state(self):
        sim = CpuCommandSim()
        sim.c[:, 1:4], sim.c[:, 5:7] = 0, torch.tensor([.2, 0.])
        imagined = sim.c.clone()
        original_state = flatten_state(sim.state())
        replay = ReplayBuffer(40, 1, "cpu", num_joints=4, scenario_dim=SCENARIO_DIM)
        for _ in range(32):
            before = imagined.clone()
            imagined = advance_command(imagined)
            transition = sim.step(torch.zeros(1, 4))
            torch.testing.assert_close(transition["c"], before)
            torch.testing.assert_close(transition["c1"], imagined)
            torch.testing.assert_close(sim.c, imagined)
            torch.testing.assert_close(transition["s1"], original_state)
            S1 = sim.state()
            expected = sim.mdp.reward(S1, transition["a"], transition["a_prev"], imagined)
            torch.testing.assert_close(transition["r"], expected)
            replay.add(transition)
        self.assertEqual(replay.data["c"].shape, (40, 1, 7))

    def test_real_commands_cover_stop_reverse_sideways_turns_and_jump_opt_in(self):
        sim = CpuCommandSim(batch=2048)
        command = sim.c[:, 1:4]
        self.assertTrue((command == 0).all(-1).any())
        self.assertTrue(((command[:, 0] < 0) & (command[:, 1:] == 0).all(-1)).any())
        self.assertTrue(((command[:, 0] == 0) & (command[:, 1] != 0) & (command[:, 2] == 0)).any())
        self.assertTrue(((command[:, :2] == 0).all(-1) & (command[:, 2] != 0)).any())
        self.assertTrue(((sim._command_interval >= 100) & (sim._command_interval <= 250)).all())
        self.assertTrue((sim.c[:, 5] == 0).all())
        sim.jump_probability = 1.
        sim.sample_commands(jump_height=.15)
        torch.testing.assert_close(sim.c[:, 5], torch.full((2048,), .15))
        sim.sample_commands(jump_height=0.)
        self.assertTrue((sim.c[:, 5] == 0).all())

    def test_real_resampling_happens_after_transition_and_holds_for_100_actions(self):
        sim = CpuCommandSim(batch=32)
        sim._command_interval.fill_(100)
        original = sim.c.clone()
        for _ in range(99):
            transition = sim.step(torch.zeros(32, 4))
            self.assertFalse(transition["command_resampled"].any())
            torch.testing.assert_close(sim.c[:, :6], original[:, :6])
        transition = sim.step(torch.zeros(32, 4))
        self.assertTrue(transition["command_resampled"].all())
        torch.testing.assert_close(transition["c"][:, :6], original[:, :6])
        torch.testing.assert_close(transition["c1"][:, :6], original[:, :6])
        self.assertTrue((sim._command_age == 0).all())
        self.assertTrue((sim.c[:, 6] == 0).all())
        self.assertFalse(torch.equal(sim.c[:, 1:4], original[:, 1:4]))

    def test_actions_are_bounded_before_physical_targets_and_replay(self):
        sim = CpuCommandSim()
        transition = sim.step(torch.tensor([[20., -20., 4.5, -5.]]))
        torch.testing.assert_close(transition["a"], torch.tensor([[4., -4., 4., -4.]]))
        torch.testing.assert_close(transition["policy_action"], torch.tensor([[20., -20., 4.5, -5.]]))
        torch.testing.assert_close(sim.ctrl, transition["a"] * .25)
        torch.testing.assert_close(sim.a_prev, transition["a"])

    def test_flat_success_requires_ten_seconds_and_episode_tracking(self):
        for tracked, length, expected in ((1., 1, False), (1., 499, False),
                                           (1., 500, True), (.5, 500, False)):
            sim = CpuCommandSim(max_episode_steps=length)
            sim.c[:, 1:4] = 0
            sim.ep_step[:] = length - 1
            sim._episode_tracking[:] = (length - 1) * tracked
            transition = sim.step(torch.zeros(1, 4))
            self.assertEqual(transition["episode_success"].item(), expected)
            self.assertTrue(transition["done"].item())
            self.assertTrue(transition["truncated"].item())
            self.assertFalse(transition["terminated"].item())
            self.assertTrue(transition["episode_survived"].item())
            self.assertEqual(transition["reward_terms"]["terminal"].item(), 0.)
            self.assertEqual(sim.ep_step.item(), 0)

    def test_stationary_policy_cannot_pass_low_speed_moving_commands(self):
        sim = CpuCommandSim(batch=6, max_episode_steps=500)
        sim.set_command_ranges([[.15, .20], [0., 0.], [0., 0.]])
        sim.sample_commands()
        sim._command_interval.fill_(1000)
        for _ in range(500):
            transition = sim.step(torch.zeros(6, 4))
            self.assertFalse(transition["command_metrics"]["command_tracking"].any())
        self.assertTrue(transition["episode_survived"].all())
        self.assertFalse(transition["episode_success"].any())
        mdp = make_mdp()
        state, command = list(make_state(mdp, 3)), make_command(mdp, 3)
        command[:, 1:3] = torch.tensor([[.15, 0.], [-.15, 0.], [0., .15]])
        state[2][:, :2] = .8 * command[:, 1:3]
        self.assertTrue(mdp.command_metrics(state, command)["command_tracking"].all())

    def test_compiled_transition_outputs_match_physical_reward_metrics_and_failures(self):
        for profile in ("velocity", "stealth", "stairs"):
            mdp = make_mdp(profile)
            state, command = list(make_state(mdp, 3)), make_command(mdp, 3)
            command[:, 1] = .35
            state[2][:, 0] = torch.tensor([.35, 0., -.15])
            state[0][2, 2] = .05
            action = torch.full((3, mdp.num_joints), .1)
            previous = torch.zeros_like(action)
            expected = (mdp.reward(state, action, previous, command, terms=True),
                        mdp.command_metrics(state, command), mdp.termination(state, command))
            compiled = torch.compile(mdp.transition_outputs, backend="aot_eager", fullgraph=True)
            reward, terms, metrics, fallen = compiled(tuple(state), action, previous, command)
            torch.testing.assert_close((reward, terms), expected[0])
            torch.testing.assert_close(metrics, expected[1])
            torch.testing.assert_close(fallen, expected[2])
        sim = CpuCommandSim()
        sim.command_outputs_fn = Mock(wraps=sim.mdp.transition_outputs)
        sim.step(torch.zeros(1, 4))
        sim.command_outputs_fn.assert_called_once()

    def test_flat_field_goal_is_ignored_and_terrain_goal_terminates(self):
        for profile, reached in (("velocity", False), ("stealth", False), ("stairs", True)):
            sim = CpuCommandSim(profile)
            sim.qpos[:, 0] = sim.terrain.goal_x + .2
            xy = sim.qpos[:, :2]
            sim.qpos[:, 2] = sim.mdp.field.sample(xy) + .76
            transition = sim.step(torch.zeros(1, 4))
            self.assertEqual(transition["episode_success"].item(), reached)
            self.assertEqual(transition["done"].item(), reached)
            self.assertEqual(transition["terminated"].item(), reached)

    def test_fall_overrides_timeout_and_preserves_terminal_state(self):
        sim = CpuCommandSim(max_episode_steps=1)
        sim.qpos[:, 2] = .05
        terminal = flatten_state(sim.state())
        transition = sim.step(torch.zeros(1, 4))
        self.assertTrue(transition["episode_fall"].item())
        self.assertFalse(transition["episode_success"].item())
        self.assertFalse(transition["truncated"].item())
        torch.testing.assert_close(transition["s1"], terminal)
        self.assertGreater(sim.qpos[0, 2].item(), .7)
        self.assertEqual(transition["reward_terms"]["terminal"].item(), -FALL_PENALTY)

    def test_runtime_rejects_invalid_configuration_before_cuda_initialization(self):
        mdp, task = make_mdp(), get_task("locomotion/g1/velocity")
        for kwargs in ({"action_limit": 0}, {"jump_height": -1}, {"jump_probability": 2},
                       {"target_clearance_m": 0}, {"command_resample_steps": (99, 250)},
                       {"command_resample_steps": (1000, 1001)},
                       {"command_resample_steps": (1001, 1001)}, {"max_episode_steps": 0}):
            with self.assertRaises(ValueError):
                CommandSim(task, mdp, device="cpu", **kwargs)
        from trackerLab.tracker_env.supermimic.terrain import TerrainSim
        def stub_parent(self, *args, **kwargs):
            self.contract = {}
        with patch.object(TerrainSim, "__init__", new=stub_parent):
            sim = CommandSim(task, mdp, device="cpu", max_episode_steps=1000,
                             command_resample_steps=(1000, 1000))
        self.assertEqual(sim.command_resample_steps, (1000, 1000))

    def test_expert_command_ranges_use_physical_units_without_robot_rescaling(self):
        sim = CpuCommandSim(batch=256)
        ranges = torch.tensor([[.3, .55], [-.04, .04], [-.1, .1]])
        sim.set_command_ranges(ranges)
        sim.sample_commands()
        self.assertTrue(bool((sim.c[:, 1:4] >= ranges[:, 0]).all()))
        self.assertTrue(bool((sim.c[:, 1:4] <= ranges[:, 1]).all()))
        self.assertGreater(sim.c[:, 1].std().item(), .04)
        with self.assertRaises(ValueError):
            sim.set_command_ranges([[1., 0.], [0., 0.], [0., 0.]])

    def test_keyboard_stealth_samples_forward_pivot_and_curved_motion(self):
        from trackerLab.tracker_env.mdp.supermimic_command import STEALTH_KEYBOARD_MODES

        sim = CpuCommandSim("stealth", batch=4096, stealth_keyboard_mode=True)
        sim.sample_commands()
        valid = torch.tensor([row[1:4] for row in STEALTH_KEYBOARD_MODES])
        sampled = sim.c[:, 1:4]
        matches = (sampled[:, None, :] == valid[None]).all(-1)
        self.assertTrue(bool(matches.any(-1).all()))
        selected = matches.float().argmax(-1)
        for index, _ in enumerate(STEALTH_KEYBOARD_MODES):
            self.assertGreater(int((selected == index).sum()), 350)

    def test_style_weight_replaces_total_positive_continuation_at_terrain_goals(self):
        for gamma in (.9, .99, .999):
            for profile in PROFILES:
                mdp = make_mdp(profile, gamma=gamma)
                dimensions = (mdp.obs_dim, mdp.goal_dim, mdp.scenario_dim, mdp.critic_dim)
                for weight in (0., .5, 2.):
                    mdp.set_style_weight(weight)
                    maximum = COMMAND_MAX_STEP_REWARD + weight
                    expected = SUCCESS_BONUS + gamma * maximum / (1 - gamma)
                    self.assertAlmostEqual(mdp.success_reward, expected)
                    goal_step_return = maximum + expected
                    self.assertAlmostEqual(mdp.max_step_reward, maximum)
                    self.assertEqual(mdp.style_weight, weight)
                    self.assertEqual(dimensions, (mdp.obs_dim, mdp.goal_dim,
                                                  mdp.scenario_dim, mdp.critic_dim))
                    for delay in (1, 32, 500):
                        delayed = (maximum * (1 - gamma ** delay) / (1 - gamma)
                                   + gamma ** delay * goal_step_return)
                        self.assertGreater(goal_step_return, delayed)
                mdp.set_style_weight(.5)
                self.assertAlmostEqual(mdp.success_reward, SUCCESS_BONUS + gamma * .652 / (1 - gamma))
                before = mdp.success_reward
                for invalid in (-1., float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        mdp.set_style_weight(invalid)
                    self.assertEqual(mdp.success_reward, before)

    def test_flat_terminal_payout_stays_zero_with_amp_weight(self):
        for profile in ("velocity", "stealth"):
            mdp = make_mdp(profile)
            mdp.set_style_weight(2.)
            self.assertEqual(mdp.success_reward, 0.)
            self.assertEqual(mdp.terminal_reward(torch.tensor([True]), torch.tensor([False])).item(), 0.)
            self.assertEqual(mdp.terminal_reward(torch.tensor([False]), torch.tensor([True])).item(),
                             -FALL_PENALTY)

    def test_stealth_command_ranges_are_sixty_percent_with_identical_layout(self):
        velocity, stealth = CpuCommandSim(batch=1024), CpuCommandSim("stealth", batch=1024,
                                                                   clearance_ratio=.65)
        torch.manual_seed(123)
        velocity.sample_commands()
        torch.manual_seed(123)
        stealth.sample_commands()
        torch.testing.assert_close(stealth.c[:, 1:4], velocity.c[:, 1:4] * .6)
        self.assertTrue((stealth.c[:, 4] == stealth.target_clearance_m).all())
        self.assertEqual((velocity.mdp.obs_dim, velocity.mdp.goal_dim, velocity.mdp.scenario_dim),
                         (stealth.mdp.obs_dim, stealth.mdp.goal_dim, stealth.mdp.scenario_dim))
        source = TerrainPolicy(velocity.mdp.obs_dim, velocity.mdp.goal_dim, hist=2, num_joints=4)
        target = TerrainPolicy(stealth.mdp.obs_dim, stealth.mdp.goal_dim, hist=2, num_joints=4)
        target.load_state_dict(source.state_dict(), strict=True)

    def test_reference_resets_reanchor_tilt_clearance_velocities_and_joint_offsets(self):
        sim = CpuCommandSim("stairs", batch=3)
        motion = make_reset_motion(sim)
        sim.mdp.field.values.fill_(.27)
        sim.xi[:, 1:-3] = torch.tensor([[.01, -.02, .03, -.01]])
        source_states = tuple(getattr(motion, name).clone() for name in
                              ("root_p", "root_R", "root_v", "root_w", "q", "qd"))
        receipt = sim.set_reference_resets(motion, probability=1.)
        self.assertEqual(receipt["eligible_frames"], 1)
        self.assertTrue(receipt["foot_support_filter"])
        self.assertTrue(receipt["training_only"])
        untouched = sim.qpos[1].clone()
        ids, spawn = torch.tensor([0, 2]), {}
        apply = sim._apply_reference_resets

        def inspect_spawn(selected):
            spawn["state"] = tuple(value.clone() for value in sim.state())
            apply(selected)

        with patch.object(sim, "_apply_reference_resets", side_effect=inspect_spawn):
            sim.reset(ids, levels=torch.tensor([1, 4]))
        p, R, v, w, q, qd = sim.state()
        ref_p, ref_R, ref_v, ref_w, ref_q, ref_qd = source_states
        alignment = yaw_mat(spawn["state"][1][ids]) @ yaw_mat(ref_R).transpose(-1, -2)
        expected_R = alignment @ ref_R
        torch.testing.assert_close(p[ids, :2], spawn["state"][0][ids, :2])
        torch.testing.assert_close(p[ids, 2], torch.full((2,), .27) + ref_p[:, 2])
        torch.testing.assert_close(R[ids], expected_R)
        torch.testing.assert_close(R[ids, 2], ref_R[:, 2].expand(2, -1))
        torch.testing.assert_close(v[ids], (alignment @ ref_v[..., None]).squeeze(-1))
        torch.testing.assert_close(w[ids], (alignment @ ref_w[..., None]).squeeze(-1))
        torch.testing.assert_close(q[ids], (ref_q + sim.xi[ids, 1:-3]).clamp(sim.qlo, sim.qhi))
        torch.testing.assert_close(qd[ids], ref_qd.expand(2, -1))
        torch.testing.assert_close(sim.ctrl[ids], q[ids])
        torch.testing.assert_close(sim.a_prev[ids],
                                   (q[ids] - sim.default_q - sim.xi[ids, 1:-3]) / sim.action_scale)
        torch.testing.assert_close(sim.qpos[1], untouched)
        self.assertFalse(sim.mdp.termination(sim.state(), sim.c).any())
        for name, original in zip(("root_p", "root_R", "root_v", "root_w", "q", "qd"), source_states):
            torch.testing.assert_close(getattr(motion, name), original)
        motion.ref_state.assert_not_called()

    def test_reference_filter_rejects_fallen_airborne_invalid_and_extreme_frames(self):
        sim = CpuCommandSim(batch=64)
        motion = make_reset_motion(sim, frames=11)
        motion.root_R[1] = so3_exp(torch.tensor([1.3, 0., 0.]))
        motion.root_p[2, 2] = .1
        motion.root_p[3, 2] = 1.2
        motion.root_v[4, 2] = 1.
        motion.body_p[5, :, 2] = .5
        motion.root_v[6, 0] = float("nan")
        motion.q[7, 0] = 15.
        motion.qd[8, 1] = 30.
        motion.root_R[9] = torch.eye(3) * 2
        motion.root_R[10, 0, 0] = -motion.root_R[10, 0, 0]
        motion.root_R[10, 1, 0] = -motion.root_R[10, 1, 0]
        motion.root_R[10, 2, 0] = -motion.root_R[10, 2, 0]
        receipt = sim.set_reference_resets(motion, probability=1.)
        self.assertEqual(receipt["eligible_frames"], 1)
        sim.reset()
        torch.testing.assert_close(sim.qpos[:, sim._qpos_ids], motion.q[:1].expand(64, -1))
        self.assertFalse(sim.mdp.termination(sim.state(), sim.c).any())

    def test_reference_stealth_frames_match_low_physical_clearance(self):
        sim = CpuCommandSim("stealth", batch=8, clearance_ratio=.65)
        motion = make_reset_motion(sim, frames=2)
        motion.root_p[1, 2] = .76
        receipt = sim.set_reference_resets(motion, probability=1.)
        self.assertEqual(receipt["eligible_frames"], 1)
        sim.reset()
        torch.testing.assert_close(sim.state()[0][:, 2], torch.full((8,), sim.target_clearance_m))

    def test_reference_reset_samples_multiple_frames_with_matching_velocities(self):
        sim = CpuCommandSim(batch=128)
        motion = make_reset_motion(sim, frames=2)
        motion.q[1] *= 2
        motion.qd[1] *= 3
        motion.root_p[1, 2] = .72
        receipt = sim.set_reference_resets(motion, probability=1.)
        self.assertEqual(receipt["eligible_frames"], 2)
        sim.reset()
        sampled = (sim.qpos[:, sim._qpos_ids[0]] > .15).long()
        self.assertEqual(set(sampled.tolist()), {0, 1})
        torch.testing.assert_close(sim.qpos[:, sim._qpos_ids], motion.q[sampled])
        torch.testing.assert_close(sim.qvel[:, sim._qvel_ids], motion.qd[sampled])
        torch.testing.assert_close(sim.state()[0][:, 2], motion.root_p[sampled, 2])

    def test_reference_reset_keeps_joint_limits_and_rejects_unreachable_pd_targets(self):
        sim = CpuCommandSim(batch=8)
        sim.qlo[:] = -.2
        sim.xi[:, 1:-3] = -.01
        motion = make_reset_motion(sim)
        sim.set_reference_resets(motion, probability=1.)
        sim.reset()
        q = sim.qpos[:, sim._qpos_ids]
        self.assertTrue((q >= sim.qlo).all())
        self.assertTrue((q <= sim.qhi).all())
        torch.testing.assert_close(sim.ctrl, q)
        hold = (sim.default_q + sim.xi[:, 1:-3] + sim.a_prev * sim.action_scale).clamp(
            sim.qlo, sim.qhi)
        torch.testing.assert_close(hold, sim.ctrl)
        invalid = make_reset_motion(sim)
        invalid.q[:, 0] = 1.5
        with self.assertRaisesRegex(ValueError, "no safe"):
            sim.set_reference_resets(invalid)

    def test_reference_reset_uses_named_joint_and_actuator_indices(self):
        sim = CpuCommandSim()
        sim._qpos_ids = torch.tensor([9, 7, 10, 8])
        sim._qvel_ids = torch.tensor([9, 8, 6, 7])
        sim._actuator_ids = torch.tensor([2, 0, 3, 1])
        sim.xi[:, 1:-3] = torch.tensor([[.01, -.01, .02, -.02]])
        motion = make_reset_motion(sim)
        sim.set_reference_resets(motion, probability=1.)
        sim.reset()
        target = motion.q + sim.xi[:, 1:-3]
        torch.testing.assert_close(sim.qpos[:, sim._qpos_ids], target)
        torch.testing.assert_close(sim.qvel[:, sim._qvel_ids], motion.qd)
        torch.testing.assert_close(sim.ctrl[:, sim._actuator_ids], target)

    def test_reference_cache_is_detached_and_excluded_from_normal_steps_and_inference(self):
        sim = CpuCommandSim()
        self.assertFalse(any(name.startswith("_reference_reset") for name in vars(sim)))
        motion = make_reset_motion(sim)
        motion.q.requires_grad_(True)
        sim.set_reference_resets(motion, probability=1.)
        cached = sim._reference_reset_states[4].clone()
        with torch.no_grad():
            motion.q.add_(1.)
        torch.testing.assert_close(sim._reference_reset_states[4], cached)
        self.assertTrue(all(not value.requires_grad for value in sim._reference_reset_states))
        self.assertFalse(hasattr(sim, "motion"))
        sim.reset()
        with patch.object(sim, "_apply_reference_resets", side_effect=AssertionError("step reference")):
            sim.step(torch.zeros(1, 4))
        sim.set_reference_resets(None)
        self.assertFalse(any(name.startswith("_reference_reset") for name in vars(sim)))
        self.assertFalse(sim.contract["reference_resets"]["enabled"])
        sim.reset()
        torch.testing.assert_close(sim.qpos[:, sim._qpos_ids], sim.default_q.unsqueeze(0))
        sim.set_reference_resets(make_reset_motion(sim), probability=1.)
        sim.set_reference_resets(object(), probability=0.)
        self.assertFalse(any(name.startswith("_reference_reset") for name in vars(sim)))

    def test_reference_probability_applies_independently_per_reset_environment(self):
        sim = CpuCommandSim(batch=1024)
        sim.set_reference_resets(make_reset_motion(sim), probability=.7)
        torch.manual_seed(11)
        sim.reset()
        changed = sim.qpos[:, sim._qpos_ids[0]] != sim.default_q[0]
        self.assertGreater(changed.float().mean().item(), .6)
        self.assertLess(changed.float().mean().item(), .8)
        torch.testing.assert_close(sim.qvel[~changed], torch.zeros_like(sim.qvel[~changed]))

    def test_reference_reset_validates_inputs_and_preserves_prior_cache_on_error(self):
        sim = CpuCommandSim()
        motion = make_reset_motion(sim)
        for invalid in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sim.set_reference_resets(motion, probability=invalid)
        for invalid in (object(), make_reset_motion(sim, frames=0)):
            with self.assertRaises(ValueError):
                sim.set_reference_resets(invalid)
        sim.set_reference_resets(motion)
        original = sim._reference_reset_states
        invalid = make_reset_motion(sim)
        invalid.root_p[:, 2] = .01
        with self.assertRaisesRegex(ValueError, "no safe"):
            sim.set_reference_resets(invalid)
        self.assertIs(sim._reference_reset_states, original)
        invalid = make_reset_motion(sim)
        invalid.q = torch.zeros(1, 5)
        with self.assertRaisesRegex(ValueError, "joint dimensions"):
            sim.set_reference_resets(invalid)
        without_fk = make_reset_motion(sim)
        del without_fk.body_p
        del without_fk.body_R
        receipt = sim.set_reference_resets(without_fk)
        self.assertFalse(receipt["foot_support_filter"])

    def test_reference_go2_support_filter_uses_calf_to_foot_offset(self):
        sim = CpuCommandSim()
        sim.robot = "go2"
        sim.spec.tracked_bodies = FOOT_BODIES["go2"]
        motion = make_reset_motion(sim, frames=2)
        motion.body_p[0, :, 2] = .243
        motion.body_p[1, :, 2] = .6
        receipt = sim.set_reference_resets(motion)
        self.assertTrue(receipt["foot_support_filter"])
        self.assertEqual(receipt["eligible_frames"], 1)

    def test_provided_grounded_mask_replaces_pi_like_foot_origin_heuristic(self):
        for profile, ratio in (("velocity", 1.), ("stealth", .65)):
            sim = CpuCommandSim(profile, batch=8, clearance_ratio=ratio, action_limit=20.)
            sim.robot = "pi_plus"
            sim.spec.tracked_bodies = FOOT_BODIES["pi_plus"]
            sim.height, sim.scale = .351, .351/.76
            sim.target_clearance_m = sim.height * ratio
            motion = make_reset_motion(sim, frames=2)
            motion.body_p[0, :, 2], motion.body_p[1, :, 2] = .05, .11
            with self.assertRaisesRegex(ValueError, "no safe"):
                sim.set_reference_resets(motion, probability=1.)
            mask = np.array([False, True], dtype=bool)
            motion.q[1, 0] = .4
            receipt = sim.set_reference_resets(motion, probability=1., grounded_mask=mask)
            self.assertEqual(receipt["eligible_frames"], 1)
            self.assertEqual(receipt["support_filter"], "provided_physical_grounded_mask")
            self.assertTrue(receipt["grounded_mask_provided"])
            self.assertEqual(receipt["grounded_mask_true_frames"], 1)
            sim.reset()
            torch.testing.assert_close(sim.qpos[:, sim._qpos_ids], motion.q[1:2].expand(8, -1))
            self.assertFalse(hasattr(sim, "grounded_mask"))

    def test_grounded_mask_keeps_other_filters_and_no_safe_error(self):
        sim = CpuCommandSim(action_limit=20.)
        motion = make_reset_motion(sim, frames=7)
        motion.q[1, 0] = 5.1
        motion.q[2, 0] = 11.
        motion.root_R[3] = so3_exp(torch.tensor([1.3, 0., 0.]))
        motion.root_p[4, 2] = .1
        motion.root_v[5, 2] = 1.
        motion.qd[6, 1] = 30.
        receipt = sim.set_reference_resets(motion, grounded_mask=torch.ones(7, dtype=torch.bool))
        self.assertEqual(receipt["eligible_frames"], 1)
        original = sim._reference_reset_states
        with self.assertRaisesRegex(ValueError, "no safe"):
            sim.set_reference_resets(motion, grounded_mask=torch.zeros(7, dtype=torch.bool))
        self.assertIs(sim._reference_reset_states, original)

    def test_grounded_mask_requires_boolean_per_frame_and_skips_unused_fk(self):
        sim = CpuCommandSim()
        motion = make_reset_motion(sim, frames=2)
        for invalid in (torch.ones(2), np.array([0, 1]), torch.ones(2, 1, dtype=torch.bool),
                        torch.ones(1, dtype=torch.bool), True):
            with self.assertRaisesRegex(ValueError, "Grounded mask"):
                sim.set_reference_resets(motion, grounded_mask=invalid)
        motion.body_p = torch.full((2, 1), float("nan"))
        receipt = sim.set_reference_resets(motion, grounded_mask=torch.ones(2, dtype=torch.bool))
        self.assertEqual(receipt["eligible_frames"], 2)
        self.assertEqual(receipt["support_filter"], "provided_physical_grounded_mask")
        normal = sim.set_reference_resets(make_reset_motion(sim))
        self.assertEqual(normal["support_filter"], "foot_origin_height_heuristic")
        self.assertFalse(normal["grounded_mask_provided"])

    def test_reference_action_limit_twenty_accepts_thirteen_and_rejects_twenty_one(self):
        narrow, expanded = CpuCommandSim(), CpuCommandSim(action_limit=20.)
        motion = make_reset_motion(expanded, frames=2)
        motion.q[:, 0] = torch.tensor([13., 21.]) * expanded.action_scale[0]
        mask = torch.ones(2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "no safe"):
            narrow.set_reference_resets(motion, grounded_mask=mask)
        receipt = expanded.set_reference_resets(motion, probability=1., grounded_mask=mask)
        self.assertEqual(receipt["eligible_frames"], 1)
        expanded.reset()
        self.assertAlmostEqual(expanded.a_prev[0, 0].item(), 13.)
        torch.testing.assert_close(expanded.ctrl, motion.q[:1])


if __name__ == "__main__":
    unittest.main()

"""Entry commands and arm acceleration remain consistent across execution paths."""

import ast
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from train_terrain import noise_padded_actor_history

from test_command_locomotion import (
    CpuCommandSim, make_mdp, make_state, make_command, make_reset_motion,
)
import test_stealth_upper_body as upper_body_tests
from trackerLab.tracker_env.mdp.supermimic_command import entry_command, advance_command
from trackerLab.algorithms.supermimic.stealth_upper_body import UpperBodyConstraint
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy, TerrainWorldModel
from trackerLab.tracker_env.mdp.supermimic import normalize_xi


class StealthEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make_scheduled_constraint(self, **overrides):
        fixture = upper_body_tests.UpperBodyConstraintTests()
        names = tuple(fixture.reference_q)
        config = {
            "entry_times_s": [0., 1.9, 3.34, 3.9],
            "entry_reference_q": [
                dict(zip(names, pose)) for pose in (
                    (0., .1, -.1, 0.), (.2, .4, -.18, .25),
                    (.46, .56, -.28, .38), (.5, .6, -.3, .4))
            ],
            "residual_radius_rad": .02,
            "posture_tolerance_rad": .25,
            "max_joint_acceleration_rad_s2": 3.,
        }
        config.update(overrides)
        return fixture.make_constraint(**config)

    def test_quintic_entry_reaches_commands_and_has_zero_endpoint_slopes(self):
        c = torch.zeros(5, 7)
        c[:, 1:4] = torch.tensor([.25, 0., -.55])
        c[:, 4] = .7
        c[:, 6] = torch.tensor([0., .001, 1., 1.999, 2.])
        c.requires_grad_()
        effective = entry_command(c, 2., .76)
        torch.testing.assert_close(effective[0, 1:4], torch.zeros(3))
        self.assertAlmostEqual(float(effective[0, 4].detach()), .76, places=6)
        torch.testing.assert_close(effective[2, 1:4], c[2, 1:4] / 2)
        torch.testing.assert_close(effective[-1, 1:5], c[-1, 1:5])
        gradient = torch.autograd.grad(effective[:, 1].sum(), c)[0][:, 6]
        self.assertLess(gradient[0].abs(), 1e-6)
        self.assertLess(gradient[-1].abs(), 1e-6)
        self.assertGreater(gradient[2], 0.)

    def test_command_resampling_keeps_completed_entry_and_reset_restarts_it(self):
        sim = CpuCommandSim("stealth", batch=8, stealth_keyboard_mode=True)
        sim.mdp.entry_duration_s = 2.
        sim.c[:, 6] = 2.
        sim.sample_commands()
        torch.testing.assert_close(sim.c[:, 6], torch.full((8,), 2.))
        sim.reset()
        torch.testing.assert_close(sim.c[:, 6], torch.zeros(8))
        sim.step(torch.zeros(8, 4))
        torch.testing.assert_close(sim.c[:, 6], torch.full((8,), .02))
        self.assertLess(sim.mdp.effective_command(sim.c)[:, 1:4].abs().max(), .001)

    def test_legacy_commands_and_clock_are_unchanged(self):
        c = torch.randn(3, 7)
        self.assertIs(entry_command(c, 0., .76), c)
        c[:, 6] = .79
        torch.testing.assert_close(advance_command(c)[:, 6], torch.full((3,), .8))
        torch.testing.assert_close(advance_command(c, duration=2.)[:, 6],
                                   torch.full((3,), .81))

    def test_training_reference_starts_skip_arm_entry_but_standing_restarts_it(self):
        sim = CpuCommandSim("stealth", batch=8, stealth_keyboard_mode=True)
        sim.mdp.entry_duration_s = 4.84
        sim.set_reference_resets(make_reset_motion(sim), probability=1.)
        sim.reset()
        torch.testing.assert_close(sim.c[:, 6], torch.full((8,), 4.84))
        sim.sample_commands()
        torch.testing.assert_close(sim.c[:, 6], torch.full((8,), 4.84))
        sim.set_reference_resets(None)
        sim.reset()
        torch.testing.assert_close(sim.c[:, 6], torch.zeros(8))

    def test_padded_entry_history_repeats_the_same_noisy_first_observation(self):
        observations = torch.zeros(3, 10, 4)
        history_steps = torch.tensor([1, 3, 10])
        mdp = SimpleNamespace(add_noise=lambda value: value + torch.randn_like(value))
        torch.manual_seed(31)
        noisy = noise_padded_actor_history(mdp, observations, history_steps)
        torch.testing.assert_close(noisy[0], noisy[0, :1].expand(10, -1))
        torch.testing.assert_close(noisy[1, :8], noisy[1, 7:8].expand(8, -1))
        self.assertGreater((noisy[1, 8] - noisy[1, 7]).abs().max(), 0.)
        torch.manual_seed(31)
        independently_noised = mdp.add_noise(observations)
        torch.testing.assert_close(noisy[2], independently_noised[2])
        torch.testing.assert_close(noisy[1, 7:], independently_noised[1, 7:])

    def test_arm_velocity_and_acceleration_are_bounded_from_neutral(self):
        fixture = upper_body_tests.UpperBodyConstraintTests()
        constraint = fixture.make_constraint(max_joint_acceleration_rad_s2=3.)
        n = len(fixture.joint_names)
        older = torch.zeros(3, n)
        previous = older.clone()
        previous_velocity = torch.zeros_like(previous)
        for step in range(200):
            raw = torch.full_like(previous, 5. if step < 100 else -5.)
            applied = constraint(raw, previous, older)
            velocity = (applied - previous) * constraint.action_scale / .02
            acceleration = (velocity - previous_velocity) / .02
            self.assertLessEqual(velocity[:, constraint.upper_indices].abs().max().item(), 1.2001)
            self.assertLessEqual(acceleration[:, constraint.upper_indices].abs().max().item(), 3.001)
            torch.testing.assert_close(applied[:, :2], raw[:, :2])
            older, previous = previous, applied
            previous_velocity = velocity
        restored = UpperBodyConstraint.from_dict(constraint.to_dict())
        self.assertEqual(restored.max_joint_acceleration_rad_s2, 3.)
        scripted = torch.jit.script(restored)
        torch.testing.assert_close(scripted(raw, previous, older),
                                   restored(raw, previous, older))

    def test_waypoint_entry_preserves_synchronized_path_and_leg_actions(self):
        constraint = self.make_scheduled_constraint()
        times = torch.tensor([0., .95, 1.9, 2.62, 3.34, 3.62, 3.9])
        raw = torch.full((7, constraint.num_joints), 20.)
        previous, older = torch.full_like(raw, -3.), torch.full_like(raw, 4.)
        applied = constraint(raw, previous, older, entry_time=times)
        poses = constraint.entry_reference_q
        expected = torch.stack((
            poses[0], (poses[0] + poses[1]) / 2, poses[1],
            (poses[1] + poses[2]) / 2, poses[2], (poses[2] + poses[3]) / 2, poses[3]))
        upper = constraint.upper_indices
        actual = constraint.default_q[upper] + applied[:, upper] * constraint.action_scale[upper]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(constraint.reference_at(times), expected)
        torch.testing.assert_close(applied[:, :2], raw[:, :2], rtol=0, atol=0)
        # Histories and actor arm residuals cannot deform the entry path.
        changed = constraint(-raw, older, previous, entry_time=times)
        torch.testing.assert_close(changed[:, upper], applied[:, upper], rtol=0, atol=0)
        torch.testing.assert_close(constraint.reference_at(torch.tensor(-1.)), poses[0])
        torch.testing.assert_close(constraint.reference_at(torch.tensor(5.)), poses[-1])

    def test_entry_analytic_speed_acceleration_and_waypoint_continuity(self):
        constraint = self.make_scheduled_constraint().double()
        times, poses = constraint.entry_times_s, constraint.entry_reference_q
        duration, delta = times[1:] - times[:-1], poses[1:] - poses[:-1]
        phases = torch.tensor([0., (3. - math.sqrt(3.)) / 6., .5,
                               (3. + math.sqrt(3.)) / 6., 1.], dtype=torch.float64)
        query = (times[:-1, None] + duration[:, None] * phases).flatten().requires_grad_()
        reference = constraint.reference_at(query)
        rates, accelerations = [], []
        for joint in range(len(constraint.upper_joint_names)):
            rate = torch.autograd.grad(reference[:, joint].sum(), query,
                                       create_graph=True, retain_graph=True)[0]
            rates.append(rate)
            accelerations.append(torch.autograd.grad(rate.sum(), query, retain_graph=True)[0])
        rate = torch.stack(rates, -1).reshape(3, 5, -1)
        acceleration = torch.stack(accelerations, -1).reshape(3, 5, -1)
        torch.testing.assert_close(rate[:, 2], 1.875 * delta / duration[:, None])
        torch.testing.assert_close(acceleration[:, 1],
                                   (10. / math.sqrt(3.)) * delta / duration[:, None].square())
        torch.testing.assert_close(acceleration[:, 3], -acceleration[:, 1])
        self.assertLessEqual(rate.abs().max().item(), constraint.max_joint_rate_rad_s)
        self.assertLessEqual(acceleration.abs().max().item(), constraint.max_joint_acceleration_rad_s2)
        torch.testing.assert_close(rate[:, [0, -1]], torch.zeros_like(rate[:, [0, -1]]), atol=1e-12, rtol=0)
        torch.testing.assert_close(acceleration[:, [0, -1]],
                                   torch.zeros_like(acceleration[:, [0, -1]]), atol=1e-12, rtol=0)
        torch.testing.assert_close(constraint.reference_at(times), poses, rtol=0, atol=0)
        torch.testing.assert_close(constraint.reference_at(times - 1e-5),
                                   constraint.reference_at(times + 1e-5), atol=1e-12, rtol=0)

    def test_entry_and_static_handoff_bound_discrete_speed_and_acceleration(self):
        constraint = self.make_scheduled_constraint().double()
        upper = constraint.upper_indices
        older = torch.zeros(1, constraint.num_joints, dtype=torch.float64)
        previous = older.clone()
        velocity = older.clone()
        raw = torch.full_like(previous, 20.)
        for index in range(246):
            time = torch.tensor([index * constraint.control_dt], dtype=torch.float64)
            applied = constraint(raw, previous, older, entry_time=time)
            next_velocity = (applied - previous) * constraint.action_scale / constraint.control_dt
            acceleration = (next_velocity - velocity) / constraint.control_dt
            self.assertLessEqual(next_velocity[:, upper].abs().max().item(),
                                 constraint.max_joint_rate_rad_s + 1e-6)
            self.assertLessEqual(acceleration[:, upper].abs().max().item(),
                                 constraint.max_joint_acceleration_rad_s2 + 1e-6)
            if index == 195:
                q = constraint.default_q[upper] + applied[:, upper] * constraint.action_scale[upper]
                torch.testing.assert_close(q[0], constraint.reference_q, atol=1e-12, rtol=0)
            older, previous, velocity = previous, applied, next_velocity

        target_action = torch.zeros_like(previous)
        target_action[:, upper] = (
            constraint.reference_q - constraint.default_q[upper]) / constraint.action_scale[upper]
        config = constraint.to_dict()
        config.pop("entry_times_s")
        config.pop("entry_reference_q")
        static = UpperBodyConstraint.from_dict(config).double()
        time = torch.tensor([constraint.entry_duration_s + .02], dtype=torch.float64)
        torch.testing.assert_close(
            constraint(raw, target_action, target_action, entry_time=time),
            static(raw, target_action, target_action), rtol=0, atol=0)

    def test_entry_gradients_follow_clock_and_keep_actor_legs_trainable(self):
        constraint = self.make_scheduled_constraint().double()
        raw = torch.zeros(2, constraint.num_joints, dtype=torch.float64, requires_grad=True)
        previous = torch.ones_like(raw, requires_grad=True)
        older = torch.zeros_like(raw)
        clock = torch.tensor([.7, 2.5], dtype=torch.float64, requires_grad=True)
        result = constraint(raw, previous, older, entry_time=clock)
        raw_grad, previous_grad, time_grad = torch.autograd.grad(result.sum(), (raw, previous, clock))
        torch.testing.assert_close(raw_grad[:, constraint.upper_indices],
                                   torch.zeros_like(raw_grad[:, constraint.upper_indices]))
        torch.testing.assert_close(raw_grad[:, :2], torch.ones_like(raw_grad[:, :2]))
        torch.testing.assert_close(previous_grad, torch.zeros_like(previous_grad))
        self.assertTrue(bool(torch.isfinite(time_grad).all()))
        self.assertTrue(bool((time_grad.abs() > 0).all()))
        self.assertTrue(torch.autograd.gradcheck(
            constraint.reference_at, (clock.detach().requires_grad_(),)))

    def test_waypoint_reward_uses_scheduled_target_and_independent_tolerance(self):
        constraint = self.make_scheduled_constraint()
        times = torch.tensor([0., .7, 2.4, 3.9, 5.])
        q = constraint.default_q.expand(5, -1).clone()
        q[:, constraint.upper_indices] = constraint.reference_at(times)
        qd = torch.zeros_like(q)
        state = (None, None, None, None, q, qd)
        torch.testing.assert_close(constraint.posture_reward(state, entry_time=times), torch.zeros(5))
        q = q.clone()
        q[:, constraint.upper_indices] += .1
        q.requires_grad_()
        state = (None, None, None, None, q, qd)
        reward = constraint.posture_reward(state, entry_time=times)
        comparison = self.make_scheduled_constraint(residual_radius_rad=.25)
        torch.testing.assert_close(reward, comparison.posture_reward(state, entry_time=times), rtol=0, atol=0)
        reward.sum().backward()
        self.assertTrue(bool(torch.isfinite(q.grad).all()))
        self.assertGreater(q.grad[:, constraint.upper_indices].abs().sum().item(), 0.)
        with self.assertRaisesRegex(ValueError, "or blend"):
            constraint.posture_reward(state, blend=torch.ones(5), entry_time=times)

    def test_entry_config_roundtrip_and_script_trace_exports(self):
        constraint = self.make_scheduled_constraint().eval()
        config = constraint.to_dict()
        restored = UpperBodyConstraint.from_dict(config)
        self.assertEqual(config, restored.to_dict())
        self.assertAlmostEqual(restored.entry_duration_s, 3.9, places=6)
        self.assertEqual(config["posture_tolerance_rad"], .25)
        self.assertEqual(config["residual_radius_rad"], .02)
        self.assertEqual(config["entry_times_s"], [0., 1.9, 3.34, 3.9])
        self.assertEqual(set(config["entry_reference_q"][1]), set(constraint.upper_joint_names))
        scripted = torch.jit.script(constraint)
        example = (torch.randn(2, constraint.num_joints), torch.zeros(2, constraint.num_joints),
                   torch.zeros(2, constraint.num_joints), torch.tensor([.2, .8]))
        traced = torch.jit.trace(constraint, example)
        for module in (scripted, traced):
            archive = io.BytesIO()
            torch.jit.save(module, archive)
            archive.seek(0)
            loaded = torch.jit.load(archive)
            for times in (torch.tensor([0.]), torch.tensor([.7, 1.9, 2.7, 3.9, 4.2])):
                raw = torch.randn(times.numel(), constraint.num_joints)
                previous, older = torch.zeros_like(raw), torch.zeros_like(raw)
                torch.testing.assert_close(loaded(raw, previous, older, times),
                                           constraint(raw, previous, older, times))
                torch.testing.assert_close(loaded.reference_at(times), constraint.reference_at(times))

    def test_entry_duration_keeps_configured_seconds_across_roundtrip(self):
        constraint = self.make_scheduled_constraint(entry_times_s=[0., 2., 3.5, 4.82])
        self.assertEqual(constraint.entry_duration_s, 4.82)
        self.assertEqual(constraint.to_dict()["entry_times_s"], [0., 2., 3.5, 4.82])
        restored = UpperBodyConstraint.from_dict(constraint.to_dict())
        self.assertEqual(restored.entry_duration_s, 4.82)

    def test_export_wrapper_restores_raw_seconds_and_advances_target_time(self):
        from export_command_policy import ConstrainedCommandPolicy

        constraint = self.make_scheduled_constraint(
            entry_times_s=[0., 2., 3.5, 4.82], residual_radius_rad=0.).eval()
        n = constraint.num_joints
        actor = TerrainPolicy(3 * n + 196, 12, hist=2, num_joints=n).eval()
        wrapped = ConstrainedCommandPolicy(actor, constraint, n, entry_duration_s=4.82).eval()
        times = torch.tensor([0., .98, 1.98, 3.48, 4.8, 4.82])
        goal = torch.zeros(len(times), 12)
        goal[:, 6] = times / 4.82
        history = torch.zeros(len(times), 2, 3 * n + 196)
        upper = constraint.upper_indices
        arm_default, arm_scale = constraint.default_q[upper], constraint.action_scale[upper]
        for index, offset in ((0, -.02), (1, 0.)):
            previous = history[:, index, 2 * n + 6:3 * n + 6]
            previous[:, upper] = (constraint.reference_at(times + offset) - arm_default) / arm_scale
        with torch.no_grad():
            expected_q = constraint.reference_at(times + .02)
            eager = wrapped(history, goal)
            torch.testing.assert_close(arm_default + eager[:, upper] * arm_scale, expected_q)
            traced = torch.jit.trace(wrapped, (history[:1], goal[:1]))
            torch.testing.assert_close(traced(history, goal), eager)

    def test_zero_residual_has_no_actor_arm_influence_and_keeps_finite_other_gradients(self):
        constraint = self.make_scheduled_constraint(residual_radius_rad=0.)
        upper = constraint.upper_indices
        raw = torch.full((2, constraint.num_joints), 10., requires_grad=True)
        previous = torch.zeros_like(raw)
        previous[:, upper] = (
            constraint.reference_q - constraint.default_q[upper]) / constraint.action_scale[upper]
        previous.requires_grad_()
        older = previous.detach().clone()
        times = torch.tensor([constraint.entry_duration_s + .02, 10.])
        applied = constraint(raw, previous, older, entry_time=times)
        q = constraint.default_q[upper] + applied[:, upper] * constraint.action_scale[upper]
        torch.testing.assert_close(q, constraint.reference_q.expand(2, -1))
        raw_grad, previous_grad = torch.autograd.grad(applied.sum(), (raw, previous))
        torch.testing.assert_close(raw_grad[:, upper], torch.zeros_like(raw_grad[:, upper]), rtol=0, atol=0)
        torch.testing.assert_close(raw_grad[:, :2], torch.ones_like(raw_grad[:, :2]), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(previous_grad).all()))
        raw2 = raw.detach().clone()
        raw2[:, upper] = -20.
        torch.testing.assert_close(constraint(raw2, previous, older, entry_time=times),
                                   applied, rtol=0, atol=0)

        displaced = previous.detach().clone()
        displaced[:, upper] += .5
        displaced.requires_grad_()
        approaching = constraint(raw.detach(), displaced, displaced.detach(), entry_time=times)
        history_grad, = torch.autograd.grad(approaching[:, upper].sum(), displaced)
        self.assertTrue(bool(torch.isfinite(history_grad).all()))
        self.assertGreater(history_grad[:, upper].abs().sum().item(), 0.)
        state_q = constraint.default_q.expand(2, -1).clone().requires_grad_()
        state = (None, None, None, None, state_q, torch.zeros_like(state_q))
        reward = constraint.posture_reward(state, entry_time=times)
        gradient, = torch.autograd.grad(reward.sum(), state_q)
        self.assertTrue(bool(torch.isfinite(reward).all()))
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(gradient[:, upper].abs().sum().item(), 0.)

        restored = UpperBodyConstraint.from_dict(constraint.to_dict())
        self.assertEqual(restored.residual_radius_rad, 0.)
        self.assertEqual(restored.posture_tolerance_rad, .25)
        scripted = torch.jit.script(restored)
        traced = torch.jit.trace(restored, (raw.detach(), previous.detach(), older, times))
        for module in (scripted, traced):
            for clock in (times, torch.tensor([.3, 2.])):
                actual = module(raw.detach(), previous.detach(), older, clock)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                torch.testing.assert_close(actual, restored(raw.detach(), previous.detach(), older, clock))

    def test_absent_schedule_keeps_legacy_config_buffers_and_blend(self):
        constraint = upper_body_tests.UpperBodyConstraintTests().make_constraint()
        self.assertEqual(constraint.entry_duration_s, 0.)
        for name in ("entry_times_s", "entry_reference_q", "posture_tolerance_rad"):
            self.assertNotIn(name, constraint.to_dict())
        self.assertEqual(set(constraint.state_dict()),
                         {"upper_indices", "default_q", "action_scale", "reference_q"})
        times = torch.tensor([0., 3., 5.])
        torch.testing.assert_close(constraint.reference_at(times), constraint.reference_q.expand(3, -1))
        blend = torch.tensor([0., .3, 1.])
        q = constraint.default_q.expand(3, -1).clone()
        upper = constraint.upper_indices
        q[:, upper] += blend[:, None] * (constraint.reference_q - constraint.default_q[upper])
        state = (None, None, None, None, q, torch.zeros_like(q))
        torch.testing.assert_close(constraint.posture_reward(state, blend), torch.zeros(3))
        automatic = upper_body_tests.UpperBodyConstraintTests().make_constraint(residual_radius_rad=.02)
        self.assertEqual(automatic.posture_tolerance_rad, .02)
        self.assertNotIn("posture_tolerance_rad", automatic.to_dict())

    def test_entry_requires_clock_and_checks_batch_shape(self):
        constraint = self.make_scheduled_constraint()
        raw = torch.zeros(2, constraint.num_joints)
        state = (None, None, None, None, raw, raw)
        with self.assertRaisesRegex(ValueError, "entry_time is required"):
            constraint(raw, raw, raw)
        with self.assertRaisesRegex(ValueError, "entry_time is required"):
            constraint.posture_reward(state)
        with self.assertRaisesRegex(ValueError, "batch shape"):
            constraint(raw, raw, raw, entry_time=torch.zeros(2, 1))
        with self.assertRaisesRegex(ValueError, "batch shape"):
            constraint.posture_reward(state, entry_time=torch.zeros(2, 1))
        with self.assertRaisesRegex(ValueError, "dtype"):
            constraint.reference_at(torch.tensor([.1], dtype=torch.float64))
        torch.testing.assert_close(
            constraint(raw, raw, raw, entry_time=torch.tensor(.5)),
            constraint(raw, raw, raw, entry_time=torch.full((2,), .5)))

    def test_invalid_entry_schedules_and_tolerances_are_rejected(self):
        valid = self.make_scheduled_constraint().to_dict()
        poses = valid["entry_reference_q"]
        invalid = (
            {"entry_times_s": None}, {"entry_reference_q": None},
            {"entry_times_s": []}, {"entry_times_s": [0.]},
            {"entry_times_s": [.1, 1.9, 3.34, 3.9]},
            {"entry_times_s": [0., 1.9, 1.9, 3.9]},
            {"entry_times_s": [0., 1.9, 1., 3.9]},
            {"entry_times_s": [0., -1., 3.34, 3.9]},
            {"entry_times_s": [0., float("nan"), 3.34, 3.9]},
            {"entry_times_s": [0., 1.9, 3.34, float("inf")]},
            {"entry_reference_q": poses[:-1]}, {"entry_reference_q": [poses[1], *poses[1:]]},
            {"entry_reference_q": [*poses[:-1], poses[2]]},
            {"entry_reference_q": [poses[0], {"left_elbow_joint": .2}, *poses[2:]]},
            {"entry_reference_q": [poses[0], [0., .1, float("nan"), 0.], *poses[2:]]},
            {"max_joint_rate_rad_s": .01}, {"max_joint_acceleration_rad_s2": .01},
            {"action_limit": 1.}, {"residual_radius_rad": -.01},
            {"residual_radius_rad": 0., "posture_tolerance_rad": None},
            {"residual_radius_rad": True}, {"residual_radius_rad": float("inf")},
            *({"posture_tolerance_rad": value} for value in (0., -.1, True, float("nan"), float("inf"), ".25")),
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                UpperBodyConstraint.from_dict(dict(valid, **overrides))
        # A zero acceleration limit keeps its existing meaning: disabled.
        self.make_scheduled_constraint(max_joint_acceleration_rad_s2=0.)

    def test_production_imagination_uses_entry_clock_and_action_history(self):
        mdp = make_mdp("stealth")
        mdp.entry_duration_s = 2.
        n = mdp.num_joints
        actor = TerrainPolicy(mdp.obs_dim, mdp.goal_dim, hist=10, num_joints=n)
        world = TerrainWorldModel(mdp.field, n, hidden=(16, 16))
        state, c = make_state(mdp, 2), make_command(mdp, 2)
        c[:, 6] = 1.
        xi = torch.zeros(2, n+4)
        history = mdp.obs(state, torch.zeros(2, n), xi, c)[:, None].expand(-1, 10, -1)
        namespace = {
            "torch": torch, "actor": actor, "world": world, "mdp": mdp,
            "spec": SimpleNamespace(num_joints=n), "upper_body_constraint": None,
            "action_constraint": None,
        }
        tree = ast.parse((REPO / "scripts/rls/supermimic/train_terrain.py").read_text("utf-8"))
        nodes = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "imag_step"]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "imag_step", "exec"), namespace)
        *_, next_command = namespace["imag_step"](state, history, xi, normalize_xi(xi), c)
        torch.testing.assert_close(next_command[:, 6], torch.full((2,), 1.02))


if __name__ == "__main__":
    unittest.main()

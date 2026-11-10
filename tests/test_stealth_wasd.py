"""CPU contracts for opt-in WASD sampling and direction-symmetric tracking."""

import itertools
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from test_command_locomotion import CpuCommandSim, make_command, make_reset_motion, make_state
from trackerLab.kinematics.supermimic.rotations import so3_exp
from trackerLab.tracker_env.mdp.supermimic import unflatten_state
from trackerLab.tracker_env.mdp.supermimic_command import (
    COMMAND_MAX_STEP_REWARD, COMMAND_VERSION, CONTROL_DT, GOAL_DIM, SCENARIO_DIM,
    STEALTH_KEYBOARD_MODES, STEALTH_WASD_KEYS_V1, STEALTH_WASD_MODES,
    STEALTH_WASD_MODES_V1, STEALTH_WASD_MODES_VERSION, WASD_CONTROL_VERSION,
    WASD_STATIONARY_VELOCITY_TOLERANCE, WASD_VELOCITY_TOLERANCE, WASD_YAW_TOLERANCE,
    CommandMDP, stealth_keyboard_modes, wasd_keyboard_metadata, wasd_keys_to_twist,
)
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField
from trackerLab.tracker_env.supermimic.command import CommandSim
from trackerLab.tracker_env.supermimic.terrain import TerrainSim
from trackerTask.supermimic.terrain import generate, get_task


def wasd_mdp(*, control_mode="wasd", profile="stealth", entry_duration_s=0., scale=1.,
             wasd_reward_version=1):
    field = HeightField(generate(profile, .76 * scale, 42), .76 * scale, "cpu")
    return CommandMDP(field, torch.zeros(4), torch.full((4,), -10.), torch.full((4,), 10.),
                      profile=profile, control_mode=control_mode, entry_duration_s=entry_duration_s,
                      wasd_reward_version=wasd_reward_version)


def wasd_sim(batch=1, *, entry_duration_s=0., clearance_ratio=.92):
    sim = CpuCommandSim("stealth", batch=batch, clearance_ratio=clearance_ratio)
    sim.mdp = wasd_mdp(entry_duration_s=entry_duration_s, scale=1.)
    sim.mdp.target_clearance_m = sim.mdp.field.height * clearance_ratio
    sim.mdp.clearance_ratio = clearance_ratio
    sim.set_wasd_keyboard_mode()
    return sim


class StealthWasdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_broader_reward_has_starting_gradients_without_relaxing_success_or_maximum(self):
        old, current = wasd_mdp(), wasd_mdp(wasd_reward_version=2)
        commands = make_command(current, 2)
        commands[:, 1:4] = torch.tensor([[.25, 0., 0.], [0., 0., .55]])
        state = list(make_state(current, 2))
        state[2] = torch.zeros_like(state[2], requires_grad=True)
        state[3] = torch.zeros_like(state[3], requires_grad=True)
        action = torch.zeros(2, 4)
        old_reward = old.reward(state, action, action, commands)
        new_reward = current.reward(state, action, action, commands)
        old_grad = torch.autograd.grad(old_reward.sum(), state[2:4], retain_graph=True)
        new_grad = torch.autograd.grad(new_reward.sum(), state[2:4])
        self.assertGreater(new_grad[0][0, 0].item(), .04)
        self.assertGreater(new_grad[1][1, 2].item(), 20 * old_grad[1][1, 2].item())
        torch.testing.assert_close(current.command_metrics(state, commands),
                                   old.command_metrics(state, commands))
        for mdp in (old, current):
            exact = list(make_state(mdp, 2))
            exact[2][:, :2], exact[3][:, 2] = commands[:, 1:3], commands[:, 3]
            reward = mdp.reward(exact, action, action, commands)
            self.assertTrue(bool((reward <= COMMAND_MAX_STEP_REWARD).all()))
        for version in (0, 4, True, "2"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                wasd_mdp(wasd_reward_version=version)
        with self.assertRaises(ValueError):
            wasd_mdp(control_mode="legacy", wasd_reward_version=2)

    def test_versioned_named_modes_and_all_key_combinations(self):
        expected = (
            ("forward", .25, 0., 0., .20), ("backward", -.20, 0., 0., .15),
            ("turn_left", 0., 0., .55, .10), ("turn_right", 0., 0., -.55, .10),
            ("forward_left", .25, 0., .45, .10), ("forward_right", .25, 0., -.45, .10),
            ("backward_left", -.20, 0., .45, .075), ("backward_right", -.20, 0., -.45, .075),
            ("idle", 0., 0., 0., .10),
        )
        self.assertEqual(STEALTH_WASD_MODES_V1, expected)
        self.assertIs(STEALTH_WASD_MODES, STEALTH_WASD_MODES_V1)
        self.assertEqual(STEALTH_WASD_MODES_VERSION, 1)
        self.assertEqual(WASD_CONTROL_VERSION, 1)
        self.assertAlmostEqual(sum(mode.probability for mode in STEALTH_WASD_MODES), 1.)
        for mode, keys in zip(STEALTH_WASD_MODES, STEALTH_WASD_KEYS_V1):
            self.assertEqual(mode.name, mode[0])
            self.assertEqual(wasd_keys_to_twist(keys), (mode.vx, mode.vy, mode.yaw))
        for pressed in itertools.product((False, True), repeat=4):
            w, a, s, d = pressed
            keys = tuple(key for key, enabled in zip("wasd", pressed) if enabled)
            translation, turn = int(w) - int(s), int(a) - int(d)
            vx = .25 if translation > 0 else -.20 if translation < 0 else 0.
            yaw = turn * (.45 if translation else .55)
            self.assertEqual(wasd_keys_to_twist(keys), (vx, 0., yaw))
            self.assertEqual(wasd_keys_to_twist("".join(keys).upper()), (vx, 0., yaw))
        self.assertEqual(wasd_keys_to_twist(["W", "W", "a"]), (.25, 0., .45))
        self.assertEqual(wasd_keys_to_twist(), (0., 0., 0.))
        for keys in (None, 1, ["w", "x"], [True], ["KeyW"], ["wa"]):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                wasd_keys_to_twist(keys)

    def test_shared_metadata_and_legacy_table_are_preserved(self):
        self.assertEqual(STEALTH_KEYBOARD_MODES, (
            ("forward", .25, 0., 0., .40), ("turn_left", 0., 0., .55, .15),
            ("turn_right", 0., 0., -.55, .15), ("forward_left", .25, 0., .45, .15),
            ("forward_right", .25, 0., -.45, .15)))
        self.assertIs(stealth_keyboard_modes("keyboard"), STEALTH_KEYBOARD_MODES)
        self.assertIs(stealth_keyboard_modes("legacy"), STEALTH_KEYBOARD_MODES)
        self.assertIs(stealth_keyboard_modes("wasd"), STEALTH_WASD_MODES)
        metadata = wasd_keyboard_metadata()
        self.assertEqual(metadata["version"], 1)
        self.assertEqual(len(metadata["command_modes"]), 9)
        for row in metadata["command_modes"]:
            self.assertEqual(tuple(row["command"]), wasd_keys_to_twist(row["keys"]))
        metadata["command_modes"][0]["command"][0] = 99.
        self.assertEqual(wasd_keyboard_metadata()["command_modes"][0]["command"][0], .25)
        for value in (None, False, "fixed", "WASD", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stealth_keyboard_modes(value)

    def test_sampler_covers_nine_modes_with_declared_probabilities(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(114)
            sim = wasd_sim(batch=40000)
        commands = torch.tensor([row[1:4] for row in STEALTH_WASD_MODES])
        torch.testing.assert_close(sim.c[:, 1:4], commands[sim._stealth_mode_ids], rtol=0, atol=0)
        frequencies = torch.bincount(sim._stealth_mode_ids, minlength=9).float() / sim.num_envs
        torch.testing.assert_close(frequencies,
                                   torch.tensor([row.probability for row in STEALTH_WASD_MODES]),
                                   rtol=0, atol=.012)
        self.assertEqual(sim.contract["command_mode_set"], "wasd")
        self.assertEqual(sim.contract["wasd_keyboard"], wasd_keyboard_metadata())

    def test_fixed_mode_preserves_ids_through_reset_resample_and_reference_start(self):
        sim = wasd_sim(batch=12, entry_duration_s=4.)
        for index, mode in enumerate(STEALTH_WASD_MODES):
            with self.subTest(mode=mode.name):
                sim.set_fixed_keyboard_mode(mode.name)
                sim.c[:, 6] = 2.
                sim.sample_commands(torch.tensor([0, 4, 8]))
                torch.testing.assert_close(sim.c[:, 6], torch.full((12,), 2.))
                sim.reset()
                torch.testing.assert_close(sim.c[:, 6], torch.zeros(12))
                torch.testing.assert_close(sim._stealth_mode_ids, torch.full((12,), index))
                torch.testing.assert_close(
                    sim.c[:, 1:4], torch.tensor(mode[1:4]).expand(12, 3), rtol=0, atol=0)
                self.assertEqual(sim.contract["fixed_keyboard_mode"], mode.name)
                sim._command_interval.fill_(1)
                result = sim.step(torch.zeros(12, 4))
                torch.testing.assert_close(result["stealth_mode_id"], torch.full((12,), index))
                self.assertTrue(bool(result["command_resampled"].all()))
                torch.testing.assert_close(sim._stealth_mode_ids, result["stealth_mode_id"])
        sim.set_fixed_keyboard_mode("backward_right")
        sim.set_reference_resets(make_reset_motion(sim, frames=4), probability=1.)
        sim.reset()
        torch.testing.assert_close(sim.c[:, 1:4], torch.tensor([-.20, 0., -.45]).expand(12, 3))
        torch.testing.assert_close(sim._stealth_mode_ids, torch.full((12,), 7))
        torch.testing.assert_close(sim.c[:, 6], torch.full((12,), 4.))
        with self.assertRaisesRegex(ValueError, "Clear the fixed"):
            sim.set_stealth_keyboard_mode()
        self.assertEqual(sim.command_mode_set, "wasd")
        sim.set_fixed_keyboard_mode(None)
        self.assertNotIn("fixed_keyboard_mode", sim.contract)
        sim.set_stealth_keyboard_mode()
        self.assertEqual(len(sim._keyboard_modes()), 5)
        self.assertNotIn("wasd_keyboard", sim.contract)
        for invalid in ("idle", "backward", False, 5, "Forward"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sim.set_fixed_keyboard_mode(invalid)
        disabled = CpuCommandSim("stealth")
        with self.assertRaisesRegex(ValueError, "Enable a keyboard"):
            disabled.set_fixed_keyboard_mode("forward")

    def test_key_release_stays_stealth_and_preserves_entry_clock_and_hands(self):
        sim = wasd_sim(batch=1, entry_duration_s=4.)
        sim.set_fixed_keyboard_mode("forward")
        sim.c[:, 6] = 1.8
        target_height = sim.mdp.field.height * .92
        arm_target = sim.c[:, 1:4].clone()
        sim.set_fixed_keyboard_mode("idle")
        self.assertEqual(sim.contract["wasd_keyboard"]["idle_semantics"].startswith(
            "key release selects zero body translation"), True)
        torch.testing.assert_close(sim.c[:, 1:4], torch.zeros(1, 3))
        torch.testing.assert_close(sim.c[:, 4], torch.full((1,), sim.target_clearance_m))
        torch.testing.assert_close(sim.c[:, 6], torch.full((1,), 1.8))
        self.assertAlmostEqual(sim.target_clearance_m, target_height, places=6)
        torch.testing.assert_close(arm_target, torch.tensor([[.25, 0., 0.]]))
        sim.set_fixed_keyboard_mode("backward")
        torch.testing.assert_close(sim.c[:, 1:4], torch.tensor([[-.20, 0., 0.]]))
        torch.testing.assert_close(sim.c[:, 4], torch.full((1,), target_height))
        torch.testing.assert_close(sim.c[:, 6], torch.full((1,), 1.8))
        self.assertEqual(sim.contract["fixed_keyboard_mode"], "backward")
        self.assertNotIn("standing", sim.contract["fixed_keyboard_mode"])

    def test_partial_resampling_respects_fixed_mode_without_changing_other_worlds(self):
        sim = wasd_sim(batch=8)
        sim.set_fixed_keyboard_mode("backward")
        ids = torch.tensor([1, 4, 7])
        others = torch.tensor([0, 2, 3, 5, 6])
        sim.c[others, 1:4] = torch.tensor([.1, .2, .3])
        sim._stealth_mode_ids[others] = 88
        sim.sample_commands(ids)
        torch.testing.assert_close(sim.c[ids, 1:4], torch.tensor([-.2, 0., 0.]).expand(3, 3))
        torch.testing.assert_close(sim._stealth_mode_ids[ids], torch.ones(3, dtype=torch.long))
        torch.testing.assert_close(sim.c[others, 1:4], torch.tensor([.1, .2, .3]).expand(5, 3))
        torch.testing.assert_close(sim._stealth_mode_ids[others], torch.full((5,), 88))
        sim.sample_commands(torch.empty(0, dtype=torch.long))

    def test_evaluation_cycle_covers_all_modes_and_survives_resets_and_resampling(self):
        table = torch.tensor([row[1:4] for row in STEALTH_WASD_MODES])
        for batch in (90, 108):
            with self.subTest(batch=batch):
                sim = wasd_sim(batch=batch, entry_duration_s=4.84)
                expected_ids = torch.arange(batch).remainder(9)
                sim.c[:, 6] = 4.84
                with patch("torch.multinomial", side_effect=AssertionError("random mode sampling")):
                    sim.set_keyboard_evaluation_cycle()
                    torch.testing.assert_close(sim.c[:, 6], torch.full((batch,), 4.84))
                    sim.sample_commands()
                    torch.testing.assert_close(sim._stealth_mode_ids, expected_ids)
                    torch.testing.assert_close(sim.c[:, 1:4], table[expected_ids], rtol=0, atol=0)
                    torch.testing.assert_close(torch.bincount(sim._stealth_mode_ids, minlength=9),
                                               torch.full((9,), batch // 9))
                    torch.testing.assert_close(sim.c[:, 4], torch.full((batch,), .92 * .76))
                    sim.reset()
                    torch.testing.assert_close(sim.c[:, 6], torch.zeros(batch))
                    torch.testing.assert_close(sim._stealth_mode_ids, expected_ids)
                    torch.testing.assert_close(sim.c[:, 1:4], table[expected_ids], rtol=0, atol=0)
                    sim._command_interval.fill_(1)
                    transition = sim.step(torch.zeros(batch, 4))
                    self.assertTrue(bool(transition["command_resampled"].all()))
                    torch.testing.assert_close(transition["stealth_mode_id"], expected_ids)
                    torch.testing.assert_close(sim._stealth_mode_ids, expected_ids)
                    # The production step also performs timeout-triggered resets.
                    sim.max_episode_steps = 2
                    transition = sim.step(torch.zeros(batch, 4))
                    self.assertTrue(bool(transition["done"].all()))
                    torch.testing.assert_close(transition["stealth_mode_id"], expected_ids)
                    torch.testing.assert_close(sim._stealth_mode_ids, expected_ids)
                    torch.testing.assert_close(sim.c[:, 1:4], table[expected_ids], rtol=0, atol=0)
                self.assertEqual(sim.contract["keyboard_evaluation_layout"], {
                    "version": 1, "layout": "cycle",
                    "assignment": "environment_id % mode_count",
                    "mode_count": 9, "mode_names": [row.name for row in STEALTH_WASD_MODES],
                    "preserved_across_resets": True,
                })

    def test_evaluation_cycle_subsets_use_absolute_env_ids_and_preserve_other_worlds(self):
        sim = wasd_sim(batch=108, entry_duration_s=4.84)
        sim.set_keyboard_evaluation_cycle()
        sim.c[:, 6] = 4.84
        ids = torch.tensor([107, 0, 53, 8, 24])
        other_mask = torch.ones(sim.num_envs, dtype=torch.bool)
        other_mask[ids] = False
        original_commands = sim.c.clone()
        original_mode_ids = sim._stealth_mode_ids.clone()
        sim.c[ids, 1:4] = 99.
        sim._stealth_mode_ids[ids] = 99
        with patch("torch.multinomial", side_effect=AssertionError("random mode sampling")):
            sim.sample_commands(ids)
            torch.testing.assert_close(sim.c, original_commands, rtol=0, atol=0)
            torch.testing.assert_close(sim._stealth_mode_ids, original_mode_ids)
            sim.reset(ids)
            torch.testing.assert_close(sim.c[other_mask], original_commands[other_mask], rtol=0, atol=0)
            torch.testing.assert_close(sim._stealth_mode_ids, original_mode_ids)
            torch.testing.assert_close(sim.c[ids, 1:4], original_commands[ids, 1:4], rtol=0, atol=0)
            torch.testing.assert_close(sim.c[ids, 6], torch.zeros(len(ids)))
            sim.sample_commands(torch.empty(0, dtype=torch.long))

    def test_evaluation_cycle_disable_restores_random_mode_probabilities(self):
        sim = wasd_sim(batch=90, entry_duration_s=4.84)
        self.assertFalse(getattr(sim, "_keyboard_evaluation_cycle", False))
        self.assertNotIn("keyboard_evaluation_layout", sim.contract)
        sim.set_keyboard_evaluation_cycle(True)
        sim.c[:, 6] = 4.84
        # Force the RNG result to make re-enabling probability sampling observable.
        chosen = torch.full((90, 1), 8, dtype=torch.long)
        with patch("torch.multinomial", return_value=chosen) as sample:
            sim.set_keyboard_evaluation_cycle(False)
            sample.assert_called_once()
            probabilities, draws = sample.call_args.args
            torch.testing.assert_close(probabilities, torch.tensor(
                [row.probability for row in STEALTH_WASD_MODES]).expand(90, -1))
            self.assertEqual(draws, 1)
        self.assertFalse(sim._keyboard_evaluation_cycle)
        self.assertNotIn("keyboard_evaluation_layout", sim.contract)
        torch.testing.assert_close(sim._stealth_mode_ids, chosen[:, 0])
        torch.testing.assert_close(sim.c[:, 1:4], torch.zeros(90, 3))
        torch.testing.assert_close(sim.c[:, 6], torch.full((90,), 4.84))
        sim.set_fixed_keyboard_mode("backward")
        self.assertEqual(sim.contract["fixed_keyboard_mode"], "backward")

    def test_evaluation_cycle_rejects_fixed_mode_conflicts_and_invalid_flags_atomically(self):
        sim = wasd_sim(batch=9)
        sim.set_fixed_keyboard_mode("backward")
        before = sim.c.clone()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            sim.set_keyboard_evaluation_cycle(True)
        self.assertEqual(sim.contract["fixed_keyboard_mode"], "backward")
        self.assertNotIn("keyboard_evaluation_layout", sim.contract)
        torch.testing.assert_close(sim.c, before, rtol=0, atol=0)
        sim.set_fixed_keyboard_mode(None)
        sim.set_keyboard_evaluation_cycle(True)
        before = sim.c.clone()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            sim.set_fixed_keyboard_mode("idle")
        self.assertNotIn("fixed_keyboard_mode", sim.contract)
        self.assertTrue(sim._keyboard_evaluation_cycle)
        torch.testing.assert_close(sim.c, before, rtol=0, atol=0)
        for invalid in (None, 0, 1, "cycle", [], 1.):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "boolean"):
                sim.set_keyboard_evaluation_cycle(invalid)
            torch.testing.assert_close(sim.c, before, rtol=0, atol=0)
        disabled = CpuCommandSim("stealth")
        with self.assertRaisesRegex(ValueError, "Enable a keyboard"):
            disabled.set_keyboard_evaluation_cycle()
        disabled.set_keyboard_evaluation_cycle(False)
        self.assertNotIn("keyboard_evaluation_layout", disabled.contract)

    def test_evaluation_cycle_supports_legacy_table_and_updates_layout_on_table_switch(self):
        sim = CpuCommandSim("stealth", batch=90, stealth_keyboard_mode=True)
        with patch("torch.multinomial", side_effect=AssertionError("random mode sampling")):
            sim.set_keyboard_evaluation_cycle(True)
            torch.testing.assert_close(sim._stealth_mode_ids, torch.arange(90).remainder(5))
            layout = sim.contract["keyboard_evaluation_layout"]
            self.assertEqual(layout["mode_count"], 5)
            self.assertEqual(layout["mode_names"], [row[0] for row in STEALTH_KEYBOARD_MODES])
            sim.set_wasd_keyboard_mode()
            torch.testing.assert_close(sim._stealth_mode_ids, torch.arange(90).remainder(9))
            self.assertEqual(sim.contract["keyboard_evaluation_layout"]["mode_count"], 9)
            sim.set_stealth_keyboard_mode()
            torch.testing.assert_close(sim._stealth_mode_ids, torch.arange(90).remainder(5))
            self.assertEqual(sim.contract["keyboard_evaluation_layout"]["mode_count"], 5)

    def test_constructor_selects_wasd_before_parent_reset_and_validates_without_cuda(self):
        mdp, task = wasd_mdp(), get_task("locomotion/g1/stealth")

        def parent(sim, task, mdp, *args, **kwargs):
            sim.contract, sim.mdp = {}, mdp
            self.assertTrue(sim.stealth_keyboard_mode)
            self.assertEqual(len(sim._keyboard_modes()), 9)

        with patch.object(TerrainSim, "__init__", new=parent):
            sim = CommandSim(task, mdp, num_envs=2, device="cpu", command_mode_set="wasd")
        self.assertEqual(sim.contract["control_mode"], "wasd")
        self.assertEqual(sim.contract["control_mode_version"], 1)
        self.assertEqual(len(sim.contract["stealth_keyboard_modes"]), 9)
        self.assertEqual(sim.contract["wasd_keyboard"]["version"], 1)
        for invalid in ("fixed", None, False, 9):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "command_mode_set"):
                CommandSim(task, mdp, device="cpu", command_mode_set=invalid)
        with self.assertRaisesRegex(ValueError, "stealth"):
            CommandSim(get_task("locomotion/g1/velocity"), wasd_mdp(profile="velocity", control_mode="legacy"),
                       device="cpu", command_mode_set="wasd")
        with self.assertRaisesRegex(ValueError, "stealth"):
            wasd_mdp(profile="velocity")
        for invalid in ("keyboard", None, True, 1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "control_mode"):
                wasd_mdp(control_mode=invalid)

    def test_narrow_tracking_reward_reduces_under_speed_and_stationary_drift(self):
        wasd, legacy = wasd_mdp(), wasd_mdp(control_mode="legacy")
        state, command = list(make_state(wasd, 4)), make_command(wasd, 4)
        command[:, 1] = torch.tensor([.25, -.2, 0., 0.])
        command[:, 3] = torch.tensor([0., 0., .55, 0.])
        state[2][:, 0], state[3][:, 2] = torch.tensor([.133, -.1, .15, .15]), command[:, 3]
        action = torch.zeros(4, 4)
        _, new = wasd.reward(state, action, action, command, terms=True)
        _, old = legacy.reward(state, action, action, command, terms=True)
        self.assertAlmostEqual(new["body_velocity"][0].item(),
                               .04 * math.exp(-(.25 - .133)**2 / .15**2), places=7)
        self.assertAlmostEqual(new["body_velocity"][2].item(), .04 * math.exp(-.15**2 / .1**2), places=7)
        self.assertTrue(bool((new["body_velocity"] < old["body_velocity"]).all()))
        self.assertLess(new["dont_wait"][1].item(), -.004)
        self.assertEqual(old["dont_wait"][1].item(), 0.)
        self.assertTrue(bool(legacy.command_metrics(state, command)["command_tracking"][2:].all()))
        self.assertFalse(bool(wasd.command_metrics(state, command)["command_tracking"][2:].any()))
        state[2][:, 0] = command[:, 1]
        self.assertTrue(bool(wasd.command_metrics(state, command)["command_tracking"].all()))

    def test_correct_twists_win_all_modes_and_opposite_motion_loses(self):
        mdp = wasd_mdp()
        state, command = list(make_state(mdp, 9)), make_command(mdp, 9)
        command[:, 1:4] = torch.tensor([row[1:4] for row in STEALTH_WASD_MODES])
        action = torch.zeros(9, 4)
        state[2][:, :2], state[3][:, 2] = command[:, 1:3], command[:, 3]
        correct = mdp.reward(state, action, action, command)
        self.assertTrue(bool(mdp.command_metrics(state, command)["command_tracking"].all()))
        state[2] = -state[2]
        state[3] = -state[3]
        opposite = mdp.reward(state, action, action, command)
        self.assertTrue(bool((correct[:8] > opposite[:8]).all()))
        self.assertEqual(correct[-1].item(), opposite[-1].item())
        self.assertFalse(bool(mdp.command_metrics(state, command)["command_tracking"][:8].any()))

    def test_forward_backward_reward_and_finite_gradients_are_direction_symmetric(self):
        mdp = wasd_mdp()
        state, command = list(make_state(mdp, 4)), make_command(mdp, 4)
        command[:, 1] = torch.tensor([.2, -.2, .2, -.2])
        command[:, 3] = torch.tensor([.45, .45, -.45, -.45])
        state[2] = torch.zeros(4, 3, requires_grad=True)
        state[3] = torch.zeros(4, 3, requires_grad=True)
        action = torch.zeros(4, 4)
        reward, terms = mdp.reward(state, action, action, command, terms=True)
        torch.testing.assert_close(reward, reward[:1].expand(4), rtol=0, atol=0)
        self.assertTrue(bool((terms["dont_wait"] < -.009).all()))
        velocity_gradient, yaw_gradient = torch.autograd.grad(reward.sum(), (state[2], state[3]))
        self.assertTrue(bool(torch.isfinite(velocity_gradient).all()))
        self.assertTrue(bool(torch.isfinite(yaw_gradient).all()))
        torch.testing.assert_close(velocity_gradient[:, 0], velocity_gradient[0, 0] * command[:, 1].sign())
        self.assertTrue(bool((velocity_gradient[:, 0] * command[:, 1] > 0).all()))
        self.assertTrue(bool((yaw_gradient[:, 2] * command[:, 3] > 0).all()))
        state[2] = torch.zeros(4, 3)
        state[2][:, 0] = .16 * command[:, 1].sign()
        _, moving = mdp.reward(state, action, action, command, terms=True)
        self.assertTrue(bool((moving["dont_wait"] > terms["dont_wait"]).all()))

    def test_yaw_tolerance_and_body_frame_are_shared_by_reward_and_metrics(self):
        mdp = wasd_mdp()
        state, command = list(make_state(mdp, 3)), make_command(mdp, 3)
        state[1] = so3_exp(torch.tensor([[0., 0., 1.1], [0., 0., -1.4], [0., 0., .3]]))
        command[:, 1:4] = torch.tensor([[-.2, 0., .45]]).expand(3, -1)
        local_v = torch.tensor([[-.2, 0., 0.]]).expand(3, -1)
        local_w = torch.tensor([[0., 0., .45], [0., 0., .66], [0., 0., .64]])
        state[2] = (state[1] @ local_v[..., None]).squeeze(-1)
        state[3] = (state[1] @ local_w[..., None]).squeeze(-1)
        action = torch.zeros(3, 4)
        _, terms = mdp.reward(state, action, action, command, terms=True)
        torch.testing.assert_close(terms["body_velocity"], torch.full((3,), .04))
        torch.testing.assert_close(terms["yaw_rate"],
                                   .04 * torch.exp(-torch.tensor([0., .21, .19]).square() / .2**2))
        self.assertEqual(mdp.command_metrics(state, command)["command_tracking"].tolist(), [True, False, True])

    def test_entry_zero_command_remains_finite_and_reward_dimensions_do_not_change(self):
        mdp, old = wasd_mdp(entry_duration_s=4.), wasd_mdp(control_mode="legacy", entry_duration_s=4.)
        state, command = list(make_state(mdp, 3)), make_command(mdp, 3)
        command[:, 1:4], command[:, 6] = torch.tensor([[-.2, 0., .45]]), torch.tensor([0., 2., 4.])
        state[0][:, 2] = .76
        action, xi = torch.zeros(3, 4), torch.zeros(3, 8)
        reward, terms = mdp.reward(state, action, action, command, terms=True)
        self.assertTrue(bool(torch.isfinite(reward).all()))
        self.assertEqual(terms["dont_wait"][0].item(), 0.)
        self.assertLess(terms["dont_wait"][-1].item(), -.009)
        torch.testing.assert_close(mdp.obs(state, action, xi, command), old.obs(state, action, xi, command))
        torch.testing.assert_close(mdp.goal(state, command), old.goal(state, command))
        self.assertEqual((mdp.obs_dim, mdp.goal_dim, mdp.scenario_dim),
                         (old.obs_dim, GOAL_DIM, SCENARIO_DIM))
        self.assertEqual(COMMAND_VERSION, 4)

    def test_positive_reward_bound_and_style_continuation_are_unchanged(self):
        mdp = wasd_mdp()
        state, command = list(make_state(mdp, 9)), make_command(mdp, 9)
        command[:, 1:4] = torch.tensor([row[1:4] for row in STEALTH_WASD_MODES])
        state[2][:, :2], state[3][:, 2] = command[:, 1:3], command[:, 3]
        reward = mdp.reward(state, torch.zeros(9, 4), torch.zeros(9, 4), command)
        self.assertAlmostEqual(reward[-1].item(), .14, places=6)
        self.assertTrue(bool((reward <= COMMAND_MAX_STEP_REWARD).all()))
        for weight in (0., .25, 2.):
            mdp.set_style_weight(weight)
            self.assertAlmostEqual(mdp.max_step_reward, COMMAND_MAX_STEP_REWARD + weight)
            self.assertEqual(mdp.success_reward, 0.)
            self.assertTrue(bool((reward + weight <= mdp.max_step_reward).all()))
        self.assertEqual((WASD_VELOCITY_TOLERANCE, WASD_STATIONARY_VELOCITY_TOLERANCE,
                          WASD_YAW_TOLERANCE), (.15, .1, .2))

    def test_physical_step_and_compiled_imagination_share_wasd_rewards_and_gradients(self):
        sim = wasd_sim(batch=9)
        # Every selected mode is exercised by the unmodified production step.
        sim.c[:, 1:4] = torch.tensor([row[1:4] for row in STEALTH_WASD_MODES])
        sim._stealth_mode_ids = torch.arange(9)
        sim._command_interval.fill_(1000)
        transition = sim.step(torch.zeros(9, 4))
        state = unflatten_state(transition["s1"])
        expected = sim.mdp.reward(state, transition["a"], transition["a_prev"], transition["c1"])
        torch.testing.assert_close(transition["r"], expected, rtol=0, atol=0)
        torch.testing.assert_close(transition["stealth_mode_id"], torch.arange(9))
        torch.testing.assert_close(transition["command_metrics"],
                                   sim.mdp.command_metrics(state, transition["c1"]))
        compiled = torch.compile(sim.mdp.transition_outputs, backend="aot_eager", fullgraph=True)
        eager_state = (*state[:2], state[2].detach().clone().requires_grad_(),
                       state[3].detach().clone().requires_grad_(), *state[4:])
        compiled_state = (*state[:2], state[2].detach().clone().requires_grad_(),
                          state[3].detach().clone().requires_grad_(), *state[4:])
        eager = sim.mdp.transition_outputs(eager_state, transition["a"], transition["a_prev"], transition["c1"])
        actual = compiled(compiled_state, transition["a"], transition["a_prev"], transition["c1"])
        torch.testing.assert_close(actual, eager)
        eager_gradient = torch.autograd.grad(eager[0].sum(), eager_state[2:4])
        actual_gradient = torch.autograd.grad(actual[0].sum(), compiled_state[2:4])
        torch.testing.assert_close(actual_gradient, eager_gradient)
        for gradient in actual_gradient:
            self.assertTrue(bool(torch.isfinite(gradient).all()))


if __name__ == "__main__":
    unittest.main()

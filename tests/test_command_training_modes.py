"""CPU contracts for bounded training-only keyboard sampling."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from test_command_locomotion import CpuCommandSim
from trackerLab.tracker_env.mdp.supermimic_command import stealth_keyboard_modes


def make_sim(mode_set="wasd", batch=90):
    sim = CpuCommandSim("stealth", batch=batch, stealth_keyboard_mode=True)
    if mode_set == "wasd":
        sim.set_wasd_keyboard_mode()
    else:
        sim.set_stealth_keyboard_mode()
    return sim


class CommandTrainingModesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_subset_uses_canonical_order_probabilities_and_mode_ids(self):
        for mode_set, names in (
                ("keyboard", ["forward_right", "forward", "turn_left"]),
                ("wasd", ["idle", "backward_left", "forward"])):
            with self.subTest(mode_set=mode_set), torch.random.fork_rng(devices=[]):
                torch.manual_seed(125)
                sim = make_sim(mode_set, batch=16000)
                modes = stealth_keyboard_modes(mode_set)
                original_table = deepcopy(sim.contract["stealth_keyboard_modes"])
                selected = [i for i, row in enumerate(modes) if row[0] in names]
                total = sum(modes[i][4] for i in selected)
                expected = torch.tensor([row[4] / total if i in selected else 0.
                                         for i, row in enumerate(modes)])
                with patch("torch.multinomial", wraps=torch.multinomial) as sample:
                    sim.set_training_command_modes(names)
                probabilities, draws = sample.call_args.args
                torch.testing.assert_close(probabilities, expected.expand(sim.num_envs, -1))
                self.assertEqual(draws, 1)
                self.assertEqual(sim.contract["training_command_modes"], {
                    "training_only": True, "mode_names": [modes[i][0] for i in selected],
                    "probabilities": [modes[i][4] / total for i in selected],
                })
                self.assertEqual(set(sim._stealth_mode_ids.tolist()), set(selected))
                frequencies = torch.bincount(sim._stealth_mode_ids, minlength=len(modes)) / sim.num_envs
                torch.testing.assert_close(frequencies, expected, rtol=0, atol=.02)
                table = torch.tensor([row[1:4] for row in modes])
                torch.testing.assert_close(sim.c[:, 1:4], table[sim._stealth_mode_ids], rtol=0, atol=0)
                self.assertIs(sim._keyboard_modes(), modes)
                self.assertEqual(sim.contract["stealth_keyboard_modes"], original_table)
                names.clear()
                sim.reset()
                self.assertEqual(set(sim._stealth_mode_ids.tolist()), set(selected))
                sim._command_interval.fill_(1)
                transition = sim.step(torch.zeros(sim.num_envs, sim.spec.num_joints))
                self.assertTrue(bool(transition["command_resampled"].all()))
                self.assertEqual(set(transition["stealth_mode_id"].tolist()), set(selected))
                self.assertEqual(set(sim._stealth_mode_ids.tolist()), set(selected))

    def test_default_and_cleared_subset_preserve_original_sampler_and_rng(self):
        for mode_set in ("keyboard", "wasd"):
            with self.subTest(mode_set=mode_set), torch.random.fork_rng(devices=[]):
                sim = make_sim(mode_set, batch=512)
                probabilities = torch.tensor([row[4] for row in sim._keyboard_modes()])
                torch.manual_seed(73)
                torch.rand(sim.num_envs, 4)
                expected_ids = torch.multinomial(probabilities.expand(sim.num_envs, -1), 1).squeeze(-1)
                lo, hi = sim.command_resample_steps
                expected_intervals = torch.randint(lo, hi + 1, (sim.num_envs,))
                expected_rng = torch.get_rng_state()
                for clear in (False, True):
                    if clear:
                        sim.set_training_command_modes(["forward"])
                    torch.manual_seed(73)
                    with patch("torch.multinomial", wraps=torch.multinomial) as sample:
                        if clear:
                            sim.set_training_command_modes(None)
                        else:
                            sim.sample_commands()
                    sample.assert_called_once()
                    torch.testing.assert_close(sample.call_args.args[0],
                                               probabilities.expand(sim.num_envs, -1), rtol=0, atol=0)
                    torch.testing.assert_close(sim._stealth_mode_ids, expected_ids)
                    torch.testing.assert_close(sim._command_interval, expected_intervals)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng)
                    self.assertEqual(set(sim._stealth_mode_ids.tolist()), set(range(len(probabilities))))
                    self.assertNotIn("training_command_modes", sim.contract)

    def test_invalid_subsets_are_rejected_atomically(self):
        for mode_set in ("keyboard", "wasd"):
            sim = make_sim(mode_set)
            sim.set_training_command_modes(["turn_left", "forward"])
            before, contract = sim.c.clone(), deepcopy(sim.contract)
            selected, rng = sim._training_command_modes, torch.get_rng_state()
            invalid = [[], (), ["unknown"], ["forward", "unknown"], ["forward", "forward"],
                       ["Forward"], "forward", False, 7, [False], [["forward"]], {"forward": .9}]
            if mode_set == "keyboard":
                invalid.append(["idle"])
            for names in invalid:
                with self.subTest(mode_set=mode_set, names=names):
                    with patch.object(sim, "sample_commands") as sample, self.assertRaises(ValueError):
                        sim.set_training_command_modes(names)
                    sample.assert_not_called()
                    self.assertEqual(sim._training_command_modes, selected)
                    self.assertEqual(sim.contract, contract)
                    torch.testing.assert_close(sim.c, before, rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_subset_requires_keyboard_sampling(self):
        for profile in ("velocity", "stealth"):
            sim = CpuCommandSim(profile)
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "Enable a keyboard"):
                sim.set_training_command_modes(["forward"])
            self.assertNotIn("training_command_modes", sim.contract)
            sim.set_training_command_modes(None)
            self.assertNotIn("training_command_modes", sim.contract)

    def test_fixed_evaluation_can_select_every_canonical_mode(self):
        for mode_set in ("keyboard", "wasd"):
            sim = make_sim(mode_set)
            sim.set_training_command_modes(["forward"])
            receipt = deepcopy(sim.contract["training_command_modes"])
            with patch("torch.multinomial", side_effect=AssertionError("random mode sampling")):
                for index, row in enumerate(sim._keyboard_modes()):
                    with self.subTest(mode_set=mode_set, name=row[0]):
                        sim.set_fixed_keyboard_mode(row[0])
                        sim.set_training_command_modes(["forward"])
                        sim.reset()
                        sim.sample_commands(torch.tensor([89, 0, 43]))
                        torch.testing.assert_close(sim._stealth_mode_ids, torch.full((90,), index))
                        torch.testing.assert_close(sim.c[:, 1:4], torch.tensor(row[1:4]).expand(90, 3))
                        self.assertEqual(sim.contract["training_command_modes"], receipt)
            sim.set_fixed_keyboard_mode(None)
            torch.testing.assert_close(sim._stealth_mode_ids, torch.zeros(90, dtype=torch.long))

    def test_evaluation_cycle_uses_full_table_then_restores_saved_subset(self):
        for mode_set in ("keyboard", "wasd"):
            with self.subTest(mode_set=mode_set):
                sim = make_sim(mode_set)
                sim.set_training_command_modes(["forward"])
                modes = sim._keyboard_modes()
                expected_ids = torch.arange(90).remainder(len(modes))
                table = torch.tensor([row[1:4] for row in modes])
                with patch("torch.multinomial", side_effect=AssertionError("random mode sampling")):
                    sim.set_keyboard_evaluation_cycle()
                    sim.set_training_command_modes(["turn_left"])
                    sim.reset()
                    sim.sample_commands(torch.tensor([89, 0, 43, 17]))
                    torch.testing.assert_close(sim._stealth_mode_ids, expected_ids)
                    torch.testing.assert_close(sim.c[:, 1:4], table[expected_ids], rtol=0, atol=0)
                    sim._command_interval.fill_(1)
                    result = sim.step(torch.zeros(90, sim.spec.num_joints))
                    self.assertTrue(bool(result["command_resampled"].all()))
                    torch.testing.assert_close(result["stealth_mode_id"], expected_ids)
                    torch.testing.assert_close(sim._stealth_mode_ids, expected_ids)
                self.assertEqual(sim.contract["keyboard_evaluation_layout"]["mode_names"],
                                 [row[0] for row in modes])
                sim.set_keyboard_evaluation_cycle(False)
                left = [row[0] for row in modes].index("turn_left")
                torch.testing.assert_close(sim._stealth_mode_ids, torch.full((90,), left))
                self.assertEqual(sim.contract["training_command_modes"]["mode_names"], ["turn_left"])

    def test_incompatible_table_switch_requires_explicit_subset_clear(self):
        for cycle in (False, True):
            sim = make_sim()
            sim.set_training_command_modes(["idle", "forward"])
            sim.set_keyboard_evaluation_cycle(cycle)
            before, contract, rng = sim.c.clone(), deepcopy(sim.contract), torch.get_rng_state()
            with self.subTest(cycle=cycle), self.assertRaisesRegex(ValueError, "Clear the training"):
                sim.set_stealth_keyboard_mode()
            self.assertEqual(sim.command_mode_set, "wasd")
            self.assertEqual(sim._training_command_modes, ("forward", "idle"))
            self.assertEqual(sim.contract, contract)
            torch.testing.assert_close(sim.c, before, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng)
            sim.set_training_command_modes(None)
            sim.set_stealth_keyboard_mode()
            self.assertEqual(sim.command_mode_set, "keyboard")
            self.assertNotIn("training_command_modes", sim.contract)

    def test_compatible_table_switch_refreshes_probabilities_and_canonical_ids(self):
        for cycle in (False, True):
            sim = make_sim("keyboard")
            sim.set_training_command_modes(("turn_left", "forward"))
            sim.set_keyboard_evaluation_cycle(cycle)
            for mode_set in ("wasd", "keyboard"):
                with self.subTest(mode_set=mode_set, cycle=cycle):
                    if mode_set == "wasd":
                        sim.set_wasd_keyboard_mode()
                    else:
                        sim.set_stealth_keyboard_mode()
                    modes = sim._keyboard_modes()
                    selected = [i for i, row in enumerate(modes) if row[0] in ("forward", "turn_left")]
                    total = sum(modes[i][4] for i in selected)
                    self.assertEqual(sim.contract["training_command_modes"], {
                        "training_only": True, "mode_names": [modes[i][0] for i in selected],
                        "probabilities": [modes[i][4] / total for i in selected],
                    })
                    if cycle:
                        torch.testing.assert_close(sim._stealth_mode_ids, torch.arange(90).remainder(len(modes)))
                        self.assertEqual(sim.contract["keyboard_evaluation_layout"]["mode_count"], len(modes))
                    else:
                        self.assertTrue(set(sim._stealth_mode_ids.tolist()) <= set(selected))

    def test_selection_and_partial_resampling_preserve_entry_clock_height_and_physics(self):
        sim = make_sim(batch=8)
        sim.mdp.entry_duration_s = 4.84
        sim.c[:, 6] = torch.linspace(.1, 4.84, 8)
        sim.c[:, 0] = torch.arange(8)
        context = sim.c[:, [0, 4, 5, 6]].clone()
        physical = {name: getattr(sim, name).clone() for name in
                    ("qpos", "qvel", "ctrl", "xi", "a_prev", "ep_step", "ep_return")}
        for names in (["idle"], ["backward", "forward"], None):
            sim.set_training_command_modes(names)
            torch.testing.assert_close(sim.c[:, [0, 4, 5, 6]], context, rtol=0, atol=0)
            for name, value in physical.items():
                torch.testing.assert_close(getattr(sim, name), value, rtol=0, atol=0)
            sim._command_age.fill_(73)
            before, mode_ids = sim.c.clone(), sim._stealth_mode_ids.clone()
            intervals = sim._command_interval.clone()
            ids, others = torch.tensor([7, 0, 4]), torch.tensor([1, 2, 3, 5, 6])
            sim.sample_commands(ids)
            sim.sample_commands(torch.empty(0, dtype=torch.long))
            torch.testing.assert_close(sim.c[others], before[others], rtol=0, atol=0)
            torch.testing.assert_close(sim._stealth_mode_ids[others], mode_ids[others])
            torch.testing.assert_close(sim._command_age[others], torch.full((5,), 73))
            torch.testing.assert_close(sim._command_interval[others], intervals[others])
            torch.testing.assert_close(sim.c[:, [0, 4, 5, 6]], context, rtol=0, atol=0)
            self.assertEqual(sim.target_clearance_m, sim.mdp.target_clearance_m)


if __name__ == "__main__":
    unittest.main()

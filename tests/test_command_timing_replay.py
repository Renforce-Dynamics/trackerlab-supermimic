"""CPU contracts for pre-action command timing and replay start selection."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from trackerLab.algorithms.supermimic.buffer import ReplayBuffer
from trackerLab.tracker_env.mdp.supermimic_command import SCENARIO_DIM
from test_command_locomotion import CpuCommandSim


class CommandTimingReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_timing_counts_the_action_immediately_before_resampling(self):
        sim = CpuCommandSim(batch=2)
        sim.command_resample_steps = (123, 123)
        sim._command_interval[:] = torch.tensor([100, 170])
        sim._command_age[:] = torch.tensor([98, 160])
        command = sim.c.clone()
        action = torch.zeros_like(sim.a_prev)

        before = sim.step(action)
        self.assertEqual(before["command_time_left_steps"].tolist(), [2, 10])
        self.assertFalse(before["command_resampled"].any())
        transition = sim.step(action)
        self.assertEqual(transition["command_time_left_steps"].tolist(), [1, 9])
        self.assertEqual(transition["command_resampled"].tolist(), [True, False])
        self.assertFalse(transition["done"].any())
        torch.testing.assert_close(transition["c"][:, :6], command[:, :6])
        torch.testing.assert_close(transition["c1"][:, :6], command[:, :6])
        self.assertEqual(sim._command_age.tolist(), [0, 162])
        self.assertEqual(sim._command_interval.tolist(), [123, 170])

        following = sim.step(action)
        self.assertEqual(following["command_time_left_steps"].tolist(), [123, 8])
        self.assertEqual(before["command_time_left_steps"].tolist(), [2, 10])
        self.assertEqual(transition["command_time_left_steps"].tolist(), [1, 9])

    def test_terminal_receipts_keep_pre_step_timing_through_resets(self):
        sim = CpuCommandSim(batch=4, max_episode_steps=20)
        sim.command_resample_steps = (147, 147)
        sim._command_interval[:] = torch.tensor([100, 130, 200, 150])
        sim._command_age[:] = torch.tensor([99, 123, 180, 149])
        sim.ep_step[:] = torch.tensor([19, 5, 4, 7])
        sim.qpos[1, sim._root_qpos + 2] = .05
        command = sim.c.clone()
        action = torch.zeros_like(sim.a_prev)

        transition = sim.step(action)
        self.assertEqual(transition["command_time_left_steps"].tolist(), [1, 7, 20, 1])
        self.assertEqual(transition["done"].tolist(), [True, True, False, False])
        self.assertEqual(transition["truncated"].tolist(), [True, False, False, False])
        self.assertEqual(transition["terminated"].tolist(), [False, True, False, False])
        self.assertEqual(transition["command_resampled"].tolist(), [False, False, False, True])
        torch.testing.assert_close(transition["c"], command)
        self.assertEqual(sim._command_age.tolist(), [0, 0, 181, 0])
        self.assertEqual(sim.ep_step.tolist(), [0, 0, 5, 8])

        following = sim.step(action)
        self.assertEqual(following["command_time_left_steps"].tolist(), [147, 147, 19, 147])
        sim.reset()
        self.assertEqual(transition["command_time_left_steps"].tolist(), [1, 7, 20, 1])
        self.assertEqual(following["command_time_left_steps"].tolist(), [147, 147, 19, 147])

    def test_replay_wrap_keeps_timing_aligned_with_selected_start(self):
        sim = CpuCommandSim(batch=2, max_episode_steps=20)
        sim.command_resample_steps = (100, 100)
        sim._command_interval[:] = torch.tensor([100, 150])
        sim._command_age[:] = torch.tensor([92, 140])
        sim.ep_step[1] = 12
        replay = ReplayBuffer(5, 2, "cpu", num_joints=sim.spec.num_joints,
                              scenario_dim=SCENARIO_DIM, command_timing=True)
        self.assertEqual(replay.fields["command_time_left_steps"], ((), torch.int32))
        self.assertEqual(replay.data["command_time_left_steps"].shape, (5, 2))
        self.assertEqual(replay.data["command_time_left_steps"].dtype, torch.int32)
        records = []
        for index in range(11):
            action = torch.full_like(sim.a_prev, index / 10)
            transition = sim.step(action)
            records.append({key: transition[key].clone() for key in replay.fields})
            replay.add(transition)
        self.assertEqual((replay.ptr, replay.size), (1, 5))
        expected = {
            key: torch.stack([record[key] for record in records[-replay.T:]]).to(
                dtype=replay.data[key].dtype)
            for key in replay.fields
        }
        self.assertEqual(expected["command_time_left_steps"].tolist(),
                         [[2, 4], [1, 3], [100, 100], [99, 99], [98, 98]])
        for index in range(6, 11):
            torch.testing.assert_close(replay.data["command_time_left_steps"][index % replay.T],
                                       records[index]["command_time_left_steps"].int())

        ages = torch.arange(replay.size).repeat_interleave(sim.num_envs)
        envs = torch.arange(sim.num_envs).repeat(replay.size)
        with patch("trackerLab.algorithms.supermimic.buffer.torch.randint",
                   side_effect=[ages, envs]):
            start, history = replay.sample_starts(len(ages), hist=3)
        self.assertEqual(start["command_time_left_steps"].shape, (10,))
        self.assertEqual(start["command_time_left_steps"].dtype, torch.int32)
        self.assertNotIn("command_time_left_steps", history)
        for key in set(start) - {"history_steps"}:
            torch.testing.assert_close(start[key], expected[key][ages, envs])
        for key in history:
            torch.testing.assert_close(history[key][:, -1], start[key])
        # Environment 1 resets at retained age 2; its history repeats that state.
        reset = (ages == 2) & (envs == 1)
        self.assertEqual(start["history_steps"][reset].tolist(), [1])
        torch.testing.assert_close(history["c"][reset],
                                   expected["c"][2, 1].expand(1, 3, -1))

        with patch("trackerLab.algorithms.supermimic.buffer.torch.randint",
                   side_effect=[torch.tensor([0, 0]), torch.tensor([0, 1])]):
            windows = replay.sample_windows(2, 3)
        self.assertNotIn("command_time_left_steps", windows)
        for key in set(windows) - {"mask"}:
            torch.testing.assert_close(windows[key], expected[key][:3].transpose(0, 1))
        torch.testing.assert_close(windows["mask"],
                                   torch.tensor([[1., 1., 1.], [1., 1., 0.]]))

    def test_default_and_disabled_timing_preserve_old_buffer_fields(self):
        fields = {"s", "s1", "a", "policy_action", "a_prev", "xi", "r", "f",
                  "motion_id", "ep_step", "terminated", "truncated", "done"}
        for scenario_dim in (0, 2, SCENARIO_DIM):
            for options in ({}, {"command_timing": False}):
                with self.subTest(scenario_dim=scenario_dim, options=options):
                    replay = ReplayBuffer(3, 2, "cpu", num_joints=4,
                                          scenario_dim=scenario_dim, **options)
                    scenario = {"c"} if scenario_dim else set()
                    self.assertEqual(set(replay.fields), fields | scenario)
                    self.assertEqual(set(replay.data), fields | scenario)
                    transition = {key: torch.zeros_like(value[0])
                                  for key, value in replay.data.items()}
                    replay.add(transition)
                    transition["command_time_left_steps"] = torch.tensor([13, 17])
                    replay.add(transition)
                    start, history = replay.sample_starts(4, hist=2)
                    self.assertEqual(set(history), {"s", "a_prev", "xi", "f"} | scenario)
                    self.assertEqual(set(start), set(history) | {
                        "ep_step", "motion_id", "history_steps"})
                    self.assertEqual(set(replay.sample_windows(4, 2)), {
                        "s", "s1", "a", "policy_action", "xi", "done",
                        "terminated", "truncated", "mask"} | scenario)

    def test_enabled_timing_uses_existing_missing_field_validation(self):
        sim = CpuCommandSim()
        replay = ReplayBuffer(2, 1, "cpu", num_joints=sim.spec.num_joints,
                              scenario_dim=SCENARIO_DIM, command_timing=True)
        replay.add(sim.step(torch.zeros_like(sim.a_prev)))
        saved = {key: value.clone() for key, value in replay.data.items()}
        transition = sim.step(torch.zeros_like(sim.a_prev))
        timing = transition.pop("command_time_left_steps")
        with self.assertRaisesRegex(KeyError, "command_time_left_steps"):
            replay.add(transition)
        self.assertEqual((replay.ptr, replay.size), (1, 1))
        for key in saved:
            torch.testing.assert_close(replay.data[key], saved[key])
        transition["command_time_left_steps"] = timing
        replay.add(transition)
        torch.testing.assert_close(replay.data["command_time_left_steps"][1], timing.int())

    def test_command_timing_constructor_rejects_invalid_options(self):
        for invalid in (None, 0, 1, "", "true", [], {}, torch.tensor(True)):
            with self.subTest(command_timing=invalid):
                with self.assertRaisesRegex(ValueError, "command_timing must be a boolean"):
                    ReplayBuffer(2, 1, "cpu", scenario_dim=SCENARIO_DIM,
                                 command_timing=invalid)
        for invalid in (0, -1, None, True, 1.5, "7"):
            with self.subTest(scenario_dim=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer scenario_dim"):
                    ReplayBuffer(2, 1, "cpu", scenario_dim=invalid, command_timing=True)


if __name__ == "__main__":
    unittest.main()

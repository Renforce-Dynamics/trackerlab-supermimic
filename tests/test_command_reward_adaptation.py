"""Reward-only continuation remains explicit and preserves all saved state."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from train_terrain import parse_args, validate_command_curriculum_options, validate_command_sampling_resume
from test_command_locomotion import make_command, make_state
from test_stealth_wasd import wasd_mdp


def args(*flags):
    return parse_args(["--task", "locomotion/g1/stealth", "--run-dir", "unused-reward-test",
                       "--resume", "original.pt", "--stealth-control-mode", "wasd",
                       "--command-reward-mode", "wasd", "--wasd-reward-version", "2", *flags])


class CommandRewardAdaptationTests(unittest.TestCase):
    def test_legacy_to_v2_requires_both_explicit_admissions(self):
        saved = {"arguments": {"stealth_control_mode": "keyboard"},
                 "iteration": 12000, "actor": {"test": torch.tensor([1., 2.])}}
        original = deepcopy(saved)
        for flags in ((), ("--allow-command-sampling-change",), ("--allow-command-reward-change",)):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_command_sampling_resume(saved, args(*flags))
        receipt = validate_command_sampling_resume(saved, args(
            "--allow-command-sampling-change", "--allow-command-reward-change"))
        self.assertIsNone(receipt["reward_mode_preserved"])
        self.assertTrue(receipt["command_sampling_changed"])
        self.assertEqual(receipt["reward_adaptation"]["source_mode"], "legacy")
        self.assertEqual(receipt["reward_adaptation"]["target_mode"], "wasd")
        self.assertEqual(receipt["reward_adaptation"]["target_version"], 2)
        self.assertFalse(receipt["reward_adaptation"]["model_and_optimizer_resets"])
        self.assertIn("saved critic retained", receipt["reward_adaptation"]["critic_initialization"])
        self.assertEqual(saved["arguments"], original["arguments"])
        self.assertEqual(saved["iteration"], original["iteration"])
        torch.testing.assert_close(saved["actor"], original["actor"], rtol=0, atol=0)

    def test_already_wasd_legacy_changes_only_reward(self):
        saved = {"arguments": {"stealth_control_mode": "wasd", "command_reward_mode": "legacy"}}
        receipt = validate_command_sampling_resume(saved, args("--allow-command-reward-change"))
        self.assertFalse(receipt["command_sampling_changed"])
        self.assertIn("reward_adaptation", receipt)

    def test_v2_continuation_requires_no_adaptation_flag(self):
        saved = {"arguments": {"stealth_control_mode": "wasd", "command_reward_mode": "wasd",
                               "wasd_reward_version": 2}}
        receipt = validate_command_sampling_resume(saved, args())
        self.assertEqual(receipt["reward_mode_preserved"], "wasd")
        self.assertNotIn("reward_adaptation", receipt)
        self.assertFalse(receipt["command_sampling_changed"])

    def test_other_reward_transitions_stay_rejected(self):
        for saved in ({"stealth_control_mode": "wasd", "wasd_reward_version": 1},
                      {"stealth_control_mode": "fixed", "command_reward_mode": "legacy"}):
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                validate_command_sampling_resume({"arguments": saved}, args(
                    "--allow-command-sampling-change", "--allow-command-reward-change"))
        requested = args("--wasd-reward-version", "1")
        with self.assertRaisesRegex(ValueError, "reward") :
            validate_command_sampling_resume({"arguments": {"stealth_control_mode": "wasd",
                "wasd_reward_version": 2}}, requested)

    def test_adaptation_requires_full_resume_and_v2_target(self):
        for mutation in ({"resume": None}, {"warm_start": Path("other.pt")},
                         {"wasd_reward_version": 1}, {"command_reward_mode": "legacy"}):
            requested = args("--allow-command-reward-change")
            for key, value in mutation.items():
                setattr(requested, key, value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_command_curriculum_options(requested)

    def test_selected_existing_v2_strengthens_reverse_and_idle_without_obs_change(self):
        old, new = wasd_mdp(control_mode="legacy"), wasd_mdp(wasd_reward_version=2)
        command = make_command(new, 4)
        command[:, 1:4] = torch.tensor([[.2, 0., 0.], [-.2, 0., 0.], [0., 0., 0.], [0., 0., .55]])
        state = list(make_state(new, 4))
        state[2] = torch.zeros(4, 3, requires_grad=True)
        state[3] = torch.zeros(4, 3, requires_grad=True)
        action = torch.zeros(4, 4)
        _, old_terms = old.reward(state, action, action, command, terms=True)
        reward, terms = new.reward(state, action, action, command, terms=True)
        self.assertEqual(old_terms["dont_wait"][1].item(), 0.)
        torch.testing.assert_close(terms["dont_wait"][0], terms["dont_wait"][1], rtol=0, atol=0)
        self.assertLess(terms["dont_wait"][1].item(), -.009)
        grad_v, grad_w = torch.autograd.grad(reward.sum(), (state[2], state[3]))
        self.assertGreater(grad_v[0, 0].item(), 0.)
        self.assertLess(grad_v[1, 0].item(), 0.)
        self.assertGreater(grad_w[3, 2].item(), 0.)
        torch.testing.assert_close(grad_v[0, 0], -grad_v[1, 0])
        state[2] = torch.zeros(4, 3)
        state[2][2, 0] = .1
        _, drifting_old = old.reward(state, action, action, command, terms=True)
        _, drifting_new = new.reward(state, action, action, command, terms=True)
        self.assertLess(drifting_new["body_velocity"][2], drifting_old["body_velocity"][2])
        torch.testing.assert_close(old.goal(state, command), new.goal(state, command))
        torch.testing.assert_close(old.obs(state, action, torch.zeros(4, 8), command),
                                   new.obs(state, action, torch.zeros(4, 8), command))


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for sampler curriculum validation and legacy reward continuity."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from train_terrain import (
    parse_args, validate_command_curriculum_options, validate_command_sampling_resume,
)
from test_command_locomotion import make_command, make_mdp, make_state
from trackerLab.kinematics.supermimic.rotations import so3_exp
from trackerLab.tracker_env.mdp.supermimic_command import (
    CONTROL_DT, JUMP_DURATION, STEALTH_WASD_MODES, CommandMDP,
    WASD_STATIONARY_VELOCITY_TOLERANCE, WASD_VELOCITY_TOLERANCE, WASD_YAW_TOLERANCE,
)


def arguments(*flags):
    return parse_args(["--task", "locomotion/g1/stealth",
                       "--run-dir", "unused-command-curriculum-test", *flags])


class CommandSamplingResumeTests(unittest.TestCase):
    def test_cli_defaults_and_explicit_curriculum_flags(self):
        defaults = arguments()
        self.assertEqual(defaults.stealth_control_mode, "fixed")
        self.assertIsNone(defaults.command_reward_mode)
        self.assertIsNone(defaults.command_training_modes)
        self.assertFalse(defaults.allow_command_sampling_change)
        self.assertIsNone(defaults.resume)
        self.assertIsNone(validate_command_curriculum_options(defaults))
        requested = arguments(
            "--resume", "original.pt", "--stealth-control-mode", "wasd",
            "--command-reward-mode", "legacy", "--command-training-modes", "forward", "idle",
            "--allow-command-sampling-change")
        self.assertEqual(requested.resume, Path("original.pt"))
        self.assertEqual(requested.command_reward_mode, "legacy")
        self.assertEqual(requested.command_training_modes, ["forward", "idle"])
        self.assertTrue(requested.allow_command_sampling_change)
        self.assertIsNone(validate_command_curriculum_options(requested))

    def test_keyboard_implicit_legacy_to_wasd_subset_requires_explicit_permission(self):
        flags = ("--resume", "original.pt", "--stealth-control-mode", "wasd",
                 "--command-reward-mode", "legacy", "--command-training-modes", "forward", "idle")
        for explicit_none in (False, True):
            saved = {"stealth_control_mode": "keyboard"}
            if explicit_none:
                saved.update(command_reward_mode=None, command_training_modes=None)
            checkpoint = {"arguments": saved, "iteration": 12000}
            before = deepcopy(checkpoint)
            with self.subTest(explicit_none=explicit_none):
                with self.assertRaisesRegex(ValueError, "allow-command-sampling-change"):
                    validate_command_sampling_resume(checkpoint, arguments(*flags))
                receipt = validate_command_sampling_resume(
                    checkpoint, arguments(*flags, "--allow-command-sampling-change"))
                self.assertEqual(receipt, {
                    "source_control_mode": "keyboard", "target_control_mode": "wasd",
                    "source_training_modes": None, "target_training_modes": ["forward", "idle"],
                    "reward_mode_preserved": "legacy", "command_sampling_changed": True,
                })
                self.assertEqual(checkpoint, before)

    def test_same_source_reports_unchanged_without_permission_flag(self):
        cases = (
            ({}, (), "fixed", "legacy"),
            ({"stealth_control_mode": "keyboard"},
             ("--stealth-control-mode", "keyboard", "--command-reward-mode", "legacy"),
             "keyboard", "legacy"),
            ({"stealth_control_mode": "wasd"}, ("--stealth-control-mode", "wasd"), "wasd", "wasd"),
            ({"stealth_control_mode": "wasd", "command_reward_mode": None},
             ("--stealth-control-mode", "wasd", "--command-reward-mode", "wasd"), "wasd", "wasd"),
            ({"stealth_control_mode": "wasd", "command_reward_mode": "legacy",
              "command_training_modes": ["forward", "idle"]},
             ("--stealth-control-mode", "wasd", "--command-reward-mode", "legacy",
              "--command-training-modes", "forward", "idle"), "wasd", "legacy"),
        )
        for saved, flags, mode, reward in cases:
            with self.subTest(saved=saved):
                checkpoint = {"arguments": saved}
                before = deepcopy(checkpoint)
                receipt = validate_command_sampling_resume(
                    checkpoint, arguments("--resume", "original.pt", *flags))
                self.assertEqual(receipt, {
                    "source_control_mode": mode, "target_control_mode": mode,
                    "source_training_modes": saved.get("command_training_modes"),
                    "target_training_modes": saved.get("command_training_modes"),
                    "reward_mode_preserved": reward, "command_sampling_changed": False,
                })
                self.assertEqual(checkpoint, before)

    def test_wasd_subset_change_and_clear_require_permission(self):
        saved = {"stealth_control_mode": "wasd", "command_reward_mode": "legacy",
                 "command_training_modes": ["forward"]}
        for subset in (["backward", "idle"], None):
            flags = ["--resume", "original.pt", "--stealth-control-mode", "wasd",
                     "--command-reward-mode", "legacy"]
            if subset is not None:
                flags.extend(["--command-training-modes", *subset])
            with self.subTest(subset=subset):
                with self.assertRaisesRegex(ValueError, "allow-command-sampling-change"):
                    validate_command_sampling_resume({"arguments": saved}, arguments(*flags))
                receipt = validate_command_sampling_resume(
                    {"arguments": saved}, arguments(*flags, "--allow-command-sampling-change"))
                self.assertIs(receipt["command_sampling_changed"], True)
                self.assertEqual(receipt["source_training_modes"], ["forward"])
                self.assertEqual(receipt["target_training_modes"], subset)
                self.assertEqual(receipt["reward_mode_preserved"], "legacy")

    def test_reward_change_is_rejected_even_with_permission(self):
        for saved, requested in (
            ({"stealth_control_mode": "keyboard"}, None),
            ({"stealth_control_mode": "wasd"}, "legacy"),
            ({"stealth_control_mode": "wasd", "command_reward_mode": "legacy"}, "wasd"),
        ):
            flags = ["--resume", "original.pt", "--stealth-control-mode", "wasd",
                     "--allow-command-sampling-change"]
            if requested is not None:
                flags.extend(["--command-reward-mode", requested])
            with self.subTest(saved=saved, requested=requested), self.assertRaisesRegex(ValueError, "reward"):
                validate_command_sampling_resume({"arguments": saved}, arguments(*flags))

    def test_unsupported_sampler_transitions_are_rejected_even_with_permission(self):
        for old, new in (("fixed", "keyboard"), ("fixed", "wasd"), ("wasd", "keyboard"),
                         ("wasd", "fixed"), ("keyboard", "fixed")):
            saved = {"stealth_control_mode": old, "command_reward_mode": "legacy"}
            requested = arguments(
                "--resume", "original.pt", "--stealth-control-mode", new,
                "--command-reward-mode", "legacy", "--allow-command-sampling-change")
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                validate_command_sampling_resume({"arguments": saved}, requested)


class CommandCurriculumOptionsTests(unittest.TestCase):
    def test_supported_reward_and_subset_options_return_none(self):
        for flags in (
            (), ("--stealth-control-mode", "keyboard"), ("--stealth-control-mode", "wasd"),
            ("--stealth-control-mode", "keyboard", "--command-reward-mode", "legacy"),
            ("--stealth-control-mode", "wasd", "--command-reward-mode", "legacy"),
            ("--stealth-control-mode", "wasd", "--wasd-reward-version", "2"),
            ("--stealth-control-mode", "wasd", "--command-reward-mode", "wasd",
             "--wasd-reward-version", "2", "--command-training-modes", "backward", "idle"),
            ("--stealth-control-mode", "wasd", "--command-training-modes",
             *(row.name for row in STEALTH_WASD_MODES)),
        ):
            with self.subTest(flags=flags):
                self.assertIsNone(validate_command_curriculum_options(arguments(*flags)))

    def test_malformed_unknown_duplicate_and_empty_subsets_raise_value_error(self):
        saved = {"arguments": {"stealth_control_mode": "wasd", "command_reward_mode": "legacy"}}
        for subset in ([], (), "forward", False, 1, {"forward": .5}, {"forward"},
                       ["unknown"], ["Forward"], ["forward", "standing"], ["forward", "forward"],
                       [None], [1], [["forward"]]):
            requested = arguments(
                "--resume", "original.pt", "--stealth-control-mode", "wasd",
                "--command-reward-mode", "legacy", "--allow-command-sampling-change")
            # Config-loaded values can bypass argparse's nargs/choice checks.
            requested.command_training_modes = subset
            with self.subTest(subset=subset):
                with self.assertRaises(ValueError):
                    validate_command_curriculum_options(requested)
                with self.assertRaises(ValueError):
                    validate_command_sampling_resume(saved, requested)

    def test_subset_requires_wasd_sampler(self):
        for mode in ("fixed", "keyboard"):
            requested = arguments("--stealth-control-mode", mode, "--command-training-modes", "forward")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_command_curriculum_options(requested)

    def test_unsupported_reward_modes_and_combinations_raise_value_error(self):
        for reward in ("", "keyboard", False, 2, []):
            requested = arguments("--stealth-control-mode", "wasd")
            requested.command_reward_mode = reward
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                validate_command_curriculum_options(requested)
        for mode in ("fixed", "keyboard"):
            requested = arguments("--stealth-control-mode", mode, "--command-reward-mode", "wasd")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_command_curriculum_options(requested)
        for mode, reward in (("fixed", None), ("keyboard", None), ("fixed", "legacy"),
                             ("keyboard", "legacy"), ("wasd", "legacy")):
            flags = ["--stealth-control-mode", mode, "--wasd-reward-version", "2"]
            if reward is not None:
                flags.extend(["--command-reward-mode", reward])
            with self.subTest(mode=mode, reward=reward), self.assertRaises(ValueError):
                validate_command_curriculum_options(arguments(*flags))

    def test_permission_flag_requires_full_resume_even_with_warm_start(self):
        for flags in ((), ("--warm-start", "original.pt")):
            requested = arguments("--stealth-control-mode", "wasd",
                                  "--allow-command-sampling-change", *flags)
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "resume"):
                validate_command_curriculum_options(requested)


class CommandSamplerRewardContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)

    def make_mdps(self, duration):
        legacy = make_mdp("stealth", clearance_ratio=.92)
        legacy.entry_duration_s = duration
        options = dict(profile="stealth", clearance_ratio=.92, entry_duration_s=duration,
                       control_mode="wasd")
        sampler = CommandMDP(legacy.field, legacy.default_q, torch.full_like(legacy.default_q, -10.),
                             torch.full_like(legacy.default_q, 10.), reward_control_mode="legacy", **options)
        wasd = CommandMDP(legacy.field, legacy.default_q, torch.full_like(legacy.default_q, -10.),
                          torch.full_like(legacy.default_q, 10.), **options)
        return legacy, sampler, wasd

    def test_all_nine_modes_preserve_legacy_rewards_gradients_obs_and_entry_clock_exactly(self):
        table = torch.tensor([row[1:4] for row in STEALTH_WASD_MODES])
        clocks = torch.tensor([0., .7, 2.42, 4.84, 5.2]).repeat_interleave(len(table))
        batch = len(clocks)
        for duration in (0., 4.84):
            with self.subTest(entry_duration_s=duration):
                legacy, sampler, _ = self.make_mdps(duration)
                command = make_command(legacy, batch)
                command[:, 1:4], command[:, 6] = table.repeat(5, 1), clocks
                original_command = command.clone()
                command.requires_grad_()
                state = list(make_state(legacy, batch))
                state[0][:, 0] = torch.linspace(-1., 1., batch)
                state[0][:, 2] += .03
                state[1][:] = so3_exp(torch.tensor([[.12, -.09, .4]]))
                for index, scale in ((2, .4), (3, .6), (4, .3), (5, .8)):
                    state[index][:] = torch.linspace(-scale, scale, state[index].numel()).reshape_as(state[index])
                state = tuple(value.requires_grad_() for value in state)
                action = torch.linspace(-.2, .5, batch * legacy.num_joints).reshape(batch, -1).requires_grad_()
                previous = (-.3 * action.detach() + .06).requires_grad_()
                expected = legacy.reward(state, action, previous, command, terms=True)
                actual = sampler.reward(state, action, previous, command, terms=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                inputs = (*state, action, previous, command)
                expected_grad = torch.autograd.grad(expected[0].sum(), inputs)
                actual_grad = torch.autograd.grad(actual[0].sum(), inputs)
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
                for gradient in actual_grad:
                    self.assertTrue(bool(torch.isfinite(gradient).all()))
                for gradient in actual_grad[2:8]:
                    self.assertGreater(float(gradient.abs().sum()), 0.)
                if duration:
                    self.assertGreater(float(actual_grad[-1][:, 6].abs().sum()), 0.)
                xi = torch.linspace(-.1, .1, batch * (legacy.num_joints + 4)).reshape(batch, -1)
                torch.testing.assert_close(sampler.obs(state, previous, xi, command),
                                           legacy.obs(state, previous, xi, command), rtol=0, atol=0)
                torch.testing.assert_close(sampler.goal(state, command), legacy.goal(state, command), rtol=0, atol=0)
                torch.testing.assert_close(sampler.effective_command(command),
                                           legacy.effective_command(command), rtol=0, atol=0)
                advanced = sampler.advance_command(command)
                torch.testing.assert_close(advanced, legacy.advance_command(command), rtol=0, atol=0)
                torch.testing.assert_close(advanced[:, :6], original_command[:, :6], rtol=0, atol=0)
                torch.testing.assert_close(advanced[:, 6], (clocks + CONTROL_DT).clamp(0, max(JUMP_DURATION, duration)),
                                           rtol=0, atol=0)
                torch.testing.assert_close(command, original_command, rtol=0, atol=0)

    def test_legacy_reward_sampler_retains_wasd_metric_thresholds_for_all_nine_modes(self):
        table = torch.tensor([row[1:4] for row in STEALTH_WASD_MODES])
        for duration, clock in ((0., 0.), (4.84, 0.), (4.84, 2.42), (4.84, 4.84)):
            legacy, sampler, wasd = self.make_mdps(duration)
            command = make_command(legacy, len(table))
            command[:, 1:4], command[:, 6] = table, clock
            effective = sampler.effective_command(command)
            moving = effective[:, 1:3].norm(dim=-1) > .1 * sampler.field.scale
            velocity_tolerance = sampler.field.scale * torch.where(
                moving, WASD_VELOCITY_TOLERANCE, WASD_STATIONARY_VELOCITY_TOLERANCE)
            for axis in ("velocity", "yaw"):
                for outside in (False, True):
                    with self.subTest(duration=duration, clock=clock, axis=axis, outside=outside):
                        state = make_state(legacy, len(table))
                        state[0][:, 2] = effective[:, 4]
                        state[2][:, :2], state[3][:, 2] = effective[:, 1:3], effective[:, 3]
                        margin = .001 if outside else -.001
                        if axis == "velocity":
                            state[2][:, 1] += velocity_tolerance + margin
                        else:
                            state[3][:, 2] += WASD_YAW_TOLERANCE + margin
                        metrics = sampler.command_metrics(state, command)
                        torch.testing.assert_close(metrics, wasd.command_metrics(state, command), rtol=0, atol=0)
                        self.assertEqual(metrics["command_tracking"].tolist(), [not outside] * len(table))
                        self.assertTrue(bool(legacy.command_metrics(state, command)["command_tracking"].all()))


if __name__ == "__main__":
    unittest.main()

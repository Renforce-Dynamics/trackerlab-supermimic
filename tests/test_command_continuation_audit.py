"""CPU checks for command continuation checkpoint audits."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from verify_command_continuation import verify


SOURCE_ITERATION = 12000
OPTIMIZER_STEPS = {
    "opt_pi": (12000, 11990, 11995),
    "opt_wm": (36000, 35991),
    "opt_v": (96000, 95992, 95996),
}
OPTIMIZER_MULTIPLIERS = {"opt_pi": 1, "opt_wm": 3, "opt_v": 8}
AMP_OPTIMIZER_STEPS = (24000, 23990)


def optimizer(steps):
    return {
        "state": {index: {"step": torch.tensor(float(step)),
                          "exp_avg": torch.tensor([.1, -.2]),
                          "exp_avg_sq": torch.tensor([.01, .04])}
                  for index, step in enumerate(steps)},
        "param_groups": [{"params": list(range(len(steps))), "lr": 2e-4}],
    }


def checkpoints(updates=5):
    before = {
        "iteration": SOURCE_ITERATION,
        "arguments": {"wm_batches": 3, "critic_epochs": 2, "critic_minibatches": 4, "amp_updates": 2},
        "actor": {
            "obs_norm.mean": torch.tensor([.1, -.2]),
            "obs_norm.var": torch.tensor([.5, .7]),
            "obs_norm.count": torch.tensor(2048.),
            "goal_norm.mean": torch.tensor([.3, -.4]),
            "goal_norm.var": torch.tensor([.6, .8]),
            "goal_norm.count": torch.tensor(4096.),
            "dec.0.weight": torch.tensor([[.1, .2], [-.3, .4]]),
        },
        "stealth_upper_body_constraint": {
            "reference_q": {"left_elbow_joint": .5, "right_elbow_joint": -.5},
            "residual": .25,
        },
        "upper_body_rate_limit": None,
        "amp_style": {"expert_hash": "ab" * 32, "updates": 24000,
                      "optimizer": optimizer(AMP_OPTIMIZER_STEPS)},
        **{name: optimizer(steps) for name, steps in OPTIMIZER_STEPS.items()},
    }
    after = deepcopy(before)
    after["iteration"] += updates
    after["actor"]["dec.0.weight"].add_(.01)
    for name, multiplier in OPTIMIZER_MULTIPLIERS.items():
        for state in after[name]["state"].values():
            state["step"].add_(updates * multiplier)
    after["amp_style"]["updates"] += updates * 2
    for state in after["amp_style"]["optimizer"]["state"].values():
        state["step"].add_(updates * 2)
    after["resume_receipt"] = {"source_iteration": SOURCE_ITERATION,
                               "command_sampling_changed": True}
    return before, after


def checkpoint_optimizers(checkpoint):
    return {**{name: checkpoint[name] for name in OPTIMIZER_STEPS},
            "amp_optimizer": checkpoint["amp_style"]["optimizer"]}


class CommandContinuationAuditTests(unittest.TestCase):
    def test_reports_exact_optimizer_increments_and_preserved_state(self):
        for updates in (1, 5, 17):
            with self.subTest(updates=updates):
                before, after = checkpoints(updates)
                original = deepcopy((before, after))
                report = verify(before, after, SOURCE_ITERATION)
                expected_steps = {
                    name: {"source_min": min(steps), "source_max": max(steps),
                           "current_min": min(steps) + updates * OPTIMIZER_MULTIPLIERS[name],
                           "current_max": max(steps) + updates * OPTIMIZER_MULTIPLIERS[name],
                           "expected_increment": updates * OPTIMIZER_MULTIPLIERS[name]}
                    for name, steps in OPTIMIZER_STEPS.items()
                }
                expected_steps["amp_optimizer"] = {
                    "source_min": min(AMP_OPTIMIZER_STEPS), "source_max": max(AMP_OPTIMIZER_STEPS),
                    "current_min": min(AMP_OPTIMIZER_STEPS) + updates * 2,
                    "current_max": max(AMP_OPTIMIZER_STEPS) + updates * 2,
                    "expected_increment": updates * 2,
                }
                self.assertEqual(report, {
                    "source_iteration": SOURCE_ITERATION, "iteration": SOURCE_ITERATION + updates,
                    "additional_updates": updates, "actor_normalization_preserved": True,
                    "arm_controller_preserved": True, "expert_identity_preserved": True,
                    "optimizer_steps": expected_steps, "amp_update_increment": updates * 2,
                    "resume_receipt": after["resume_receipt"],
                })
                for checkpoint, expected in zip((before, after), original):
                    for name in ("actor", *OPTIMIZER_STEPS):
                        torch.testing.assert_close(checkpoint[name], expected[name], rtol=0, atol=0)
                    torch.testing.assert_close(checkpoint["amp_style"]["optimizer"],
                                               expected["amp_style"]["optimizer"], rtol=0, atol=0)
                    for name in ("iteration", "arguments", "stealth_upper_body_constraint",
                                 "upper_body_rate_limit", "resume_receipt"):
                        self.assertEqual(checkpoint.get(name), expected.get(name))
                    for name in ("expert_hash", "updates"):
                        self.assertEqual(checkpoint["amp_style"][name], expected["amp_style"][name])

    def test_optimizer_dictionary_order_does_not_affect_continuity(self):
        before, after = checkpoints()
        for checkpoint in (before, after):
            for value in checkpoint_optimizers(checkpoint).values():
                group = value["param_groups"][0]
                value["param_groups"] = [{**group, "params": group["params"][:1]},
                                         {**group, "params": group["params"][1:]}]
        expected = verify(before, after, SOURCE_ITERATION)
        for value in checkpoint_optimizers(after).values():
            value["state"] = dict(reversed(list(value["state"].items())))
        self.assertEqual(verify(before, after, SOURCE_ITERATION), expected)

    def test_resume_receipt_is_optional(self):
        before, after = checkpoints()
        del after["resume_receipt"]
        self.assertIsNone(verify(before, after, SOURCE_ITERATION)["resume_receipt"])

    def test_unexpected_source_iteration_is_rejected(self):
        before, after = checkpoints()
        for source in (SOURCE_ITERATION - 1, SOURCE_ITERATION + 1, after["iteration"]):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "Unexpected source"):
                verify(before, after, source)

    def test_zero_and_negative_iteration_deltas_are_rejected(self):
        for updates in (0, -1, -100):
            before, after = checkpoints(updates)
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, "completed continuation updates"):
                verify(before, after, SOURCE_ITERATION)

    def test_each_actor_normalization_value_must_be_unchanged(self):
        for prefix in ("obs_norm", "goal_norm"):
            for field in ("mean", "var", "count"):
                before, after = checkpoints()
                key = f"{prefix}.{field}"
                after["actor"][key].add_(1.)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Actor normalization changed"):
                    verify(before, after, SOURCE_ITERATION)

    def test_missing_source_normalization_is_rejected(self):
        before, after = checkpoints()
        for checkpoint in (before, after):
            checkpoint["actor"] = {"dec.0.weight": checkpoint["actor"]["dec.0.weight"]}
        with self.assertRaisesRegex(ValueError, "Actor normalization changed"):
            verify(before, after, SOURCE_ITERATION)

    def test_changed_or_removed_arm_constraint_is_rejected(self):
        for change in ("target", "residual", "removed"):
            before, after = checkpoints()
            if change == "target":
                after["stealth_upper_body_constraint"]["reference_q"]["left_elbow_joint"] += .1
            elif change == "residual":
                after["stealth_upper_body_constraint"]["residual"] += .1
            else:
                after["stealth_upper_body_constraint"] = None
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Arm controller changed"):
                verify(before, after, SOURCE_ITERATION)

    def test_changed_expert_identity_is_rejected(self):
        for identity in ("cd" * 32, "", None):
            before, after = checkpoints()
            after["amp_style"]["expert_hash"] = identity
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "expert identity changed"):
                verify(before, after, SOURCE_ITERATION)

    def test_unchanged_rate_limiter_is_preserved(self):
        before, after = checkpoints()
        limiter = {"response_version": 2, "max_rate": 1.5}
        for checkpoint in (before, after):
            checkpoint["stealth_upper_body_constraint"] = None
            checkpoint["upper_body_rate_limit"] = deepcopy(limiter)
        self.assertIs(verify(before, after, SOURCE_ITERATION)["arm_controller_preserved"], True)

    def test_added_removed_or_changed_rate_limiter_is_rejected(self):
        limiter = {"response_version": 2, "max_rate": 1.5}
        for old, new in ((None, limiter), (limiter, None), (limiter, {**limiter, "max_rate": 2.})):
            before, after = checkpoints()
            before["stealth_upper_body_constraint"] = after["stealth_upper_body_constraint"] = None
            before["upper_body_rate_limit"], after["upper_body_rate_limit"] = old, new
            with self.subTest(old=old, new=new), self.assertRaisesRegex(ValueError, "Arm controller changed"):
                verify(before, after, SOURCE_ITERATION)

    def test_mismatched_optimizer_cardinality_is_rejected(self):
        for name in (*OPTIMIZER_STEPS, "amp_optimizer"):
            for change in ("removed", "added", "missing_step", "empty"):
                before, after = checkpoints()
                state = checkpoint_optimizers(after)[name]["state"]
                if change == "removed":
                    del state[0]
                elif change == "added":
                    state[len(state)] = deepcopy(state[0])
                elif change == "missing_step":
                    del state[0]["step"]
                else:
                    state.clear()
                with self.subTest(optimizer=name, change=change), self.assertRaisesRegex(
                        ValueError, f"Optimizer entries differ: {name}"):
                    verify(before, after, SOURCE_ITERATION)

    def test_missing_source_optimizer_counters_are_rejected(self):
        for name in (*OPTIMIZER_STEPS, "amp_optimizer"):
            for empty in (False, True):
                before, after = checkpoints()
                state = checkpoint_optimizers(before)[name]["state"]
                if empty:
                    state.clear()
                else:
                    for value in state.values():
                        del value["step"]
                with self.subTest(optimizer=name, empty=empty), self.assertRaisesRegex(
                        ValueError, f"Optimizer entries differ: {name}"):
                    verify(before, after, SOURCE_ITERATION)

    def test_each_optimizer_counter_must_have_the_expected_increment(self):
        for name, steps in {**OPTIMIZER_STEPS, "amp_optimizer": AMP_OPTIMIZER_STEPS}.items():
            for index in range(len(steps)):
                for error in (-1, 1, None):
                    before, after = checkpoints()
                    step = checkpoint_optimizers(after)[name]["state"][index]["step"]
                    if error is None:
                        step.zero_()
                    else:
                        step.add_(error)
                    with self.subTest(optimizer=name, index=index, error=error), self.assertRaisesRegex(
                            ValueError, f"Optimizer counter continuity differs: {name}"):
                        verify(before, after, SOURCE_ITERATION)

    def test_swapping_per_id_optimizer_steps_is_rejected(self):
        for name in (*OPTIMIZER_STEPS, "amp_optimizer"):
            before, after = checkpoints()
            state = checkpoint_optimizers(after)[name]["state"]
            state[0]["step"], state[1]["step"] = state[1]["step"], state[0]["step"]
            with self.subTest(optimizer=name), self.assertRaisesRegex(
                    ValueError, f"Optimizer counter continuity differs: {name}"):
                verify(before, after, SOURCE_ITERATION)

    def test_changed_optimizer_state_ids_are_rejected(self):
        for name in (*OPTIMIZER_STEPS, "amp_optimizer"):
            before, after = checkpoints()
            state = checkpoint_optimizers(after)[name]["state"]
            new_id = max(state) + 1
            state[new_id] = state.pop(0)
            self.assertEqual(len(state), len(checkpoint_optimizers(before)[name]["state"]))
            with self.subTest(optimizer=name), self.assertRaisesRegex(
                    ValueError, f"Optimizer entries differ: {name}"):
                verify(before, after, SOURCE_ITERATION)

    def test_changed_parameter_group_identity_and_order_are_rejected(self):
        for name in (*OPTIMIZER_STEPS, "amp_optimizer"):
            for change in ("parameter_id", "parameter_order", "group_split", "group_order"):
                before, after = checkpoints()
                old, new = checkpoint_optimizers(before)[name], checkpoint_optimizers(after)[name]
                group = new["param_groups"][0]
                if change == "parameter_id":
                    group["params"][0] = len(new["state"])
                elif change == "parameter_order":
                    group["params"].reverse()
                else:
                    groups = [{**group, "params": group["params"][:1]},
                              {**group, "params": group["params"][1:]}]
                    new["param_groups"] = groups
                    if change == "group_order":
                        old["param_groups"] = deepcopy(groups)
                        new["param_groups"].reverse()
                with self.subTest(optimizer=name, change=change), self.assertRaisesRegex(
                        ValueError, f"Optimizer entries differ: {name}"):
                    verify(before, after, SOURCE_ITERATION)

    def test_reset_amp_optimizer_is_rejected_even_with_correct_amp_update_count(self):
        before, after = checkpoints(updates=5)
        for state in after["amp_style"]["optimizer"]["state"].values():
            state["step"].zero_()
        self.assertEqual(after["amp_style"]["updates"] - before["amp_style"]["updates"], 10)
        with self.assertRaisesRegex(ValueError, "Optimizer counter continuity differs: amp_optimizer"):
            verify(before, after, SOURCE_ITERATION)

    def test_wrong_amp_update_counts_are_rejected(self):
        for increment in (-1, 0, 9, 11):
            before, after = checkpoints(updates=5)
            after["amp_style"]["updates"] = before["amp_style"]["updates"] + increment
            with self.subTest(increment=increment), self.assertRaisesRegex(ValueError, "AMP update continuity differs"):
                verify(before, after, SOURCE_ITERATION)


if __name__ == "__main__":
    unittest.main()

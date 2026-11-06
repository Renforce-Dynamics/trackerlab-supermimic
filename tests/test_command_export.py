"""Bounded CPU export checks for a source-independent command actor."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

from export_command_policy import export
from trackerLab.algorithms.supermimic import networks
from trackerLab.algorithms.supermimic.amp_style import AMPStyle
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerLab.algorithms.supermimic.stealth_upper_body import UpperBodyConstraint
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp.supermimic_command import COMMAND_VERSION, GOAL_FIELDS
from trackerTask.supermimic.robots import get_robot


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CommandExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        cls.addClassCleanup(setattr, networks, "AMP_DTYPE", networks.AMP_DTYPE)
        torch.set_num_threads(2)
        networks.AMP_DTYPE = None
        temporary = tempfile.TemporaryDirectory(prefix="command-export-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.source, cls.output = cls.root / "source.pt", cls.root / "portable"
        cls.spec = get_robot("go2")
        cls.hist, cls.joints = 2, 12
        cls.obs_dim, cls.goal_dim = 3 * cls.joints + 196, len(GOAL_FIELDS)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(614)
            cls.actor = TerrainPolicy(cls.obs_dim, cls.goal_dim, hist=cls.hist,
                                      num_joints=cls.joints, action_limit=4.).eval()
            cls.actor.obs_norm.update(.2 + .6 * torch.randn(16, cls.hist, cls.obs_dim))
            cls.actor.goal_norm.update(.1 + .4 * torch.randn(16, cls.goal_dim))
            with torch.no_grad():
                cls.actor.dec[-1].bias.copy_(torch.linspace(-1., 1., cls.joints))
        # Deliberately omit all motion configuration, paths and expert arrays.
        # Training-only sentinels must survive in source and vanish from exports.
        cls.checkpoint = {
            "training_stage": "command-style", "objective_version": COMMAND_VERSION,
            "task": "locomotion/go2/velocity", "robot": "go2", "iteration": 17,
            "arguments": {"hist": cls.hist},
            "actor": cls.actor.state_dict(),
            "world": {"training_world_sentinel": torch.tensor([12345.])},
            "amp_style": {"discriminator": {"training_discriminator_sentinel": torch.tensor([23456.])}},
            "critic": {"training_critic_sentinel": torch.tensor([34567.])},
            "opt_pi": {"state": {}, "param_groups": []},
        }
        torch.save(cls.checkpoint, cls.source)
        cls.source_digest = file_digest(cls.source)
        with patch.object(MotionLib, "from_config", side_effect=AssertionError("expert source load")), \
                patch.object(MotionLib, "__init__", side_effect=AssertionError("motion construction")), \
                patch.object(AMPStyle, "__init__", side_effect=AssertionError("discriminator construction")), \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("GPU work")):
            cls.metadata = export(cls.source, cls.output)
        cls.scripted = torch.jit.load(str(cls.output / "policy.ts"), map_location="cpu").eval()

    def test_cpu_trace_matches_eager_for_batch_one_and_seven(self):
        generator = torch.Generator().manual_seed(822)
        self.assertEqual(self.spec.num_joints, self.joints)
        with torch.no_grad():
            for batch in (1, 7):
                with self.subTest(batch=batch):
                    observations = torch.randn(batch, self.hist, self.obs_dim, generator=generator)
                    goal = torch.randn(batch, self.goal_dim, generator=generator)
                    expected = self.actor(observations, goal)
                    actual = self.scripted(observations, goal)
                    self.assertEqual(actual.shape, (batch, self.joints))
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertTrue(bool(torch.isfinite(actual).all()))
                    self.assertTrue(bool((actual.abs() <= 4.).all()))
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                    # Dynamic batching must preserve each row's independent action.
                    individual = torch.cat([
                        self.scripted(observations[index:index+1], goal[index:index+1])
                        for index in range(batch)
                    ])
                    torch.testing.assert_close(actual, individual, atol=1e-5, rtol=1e-5)

    def test_metadata_declares_actor_only_runtime_and_input_contract(self):
        metadata = json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata, json.loads(json.dumps(self.metadata)))
        self.assertEqual(metadata["task"], self.checkpoint["task"])
        self.assertEqual(metadata["robot"], "go2")
        self.assertEqual(metadata["iteration"], 17)
        self.assertEqual(metadata["inputs"], {
            "observation_history": ["batch", 2, 232], "command_goal": ["batch", self.goal_dim],
        })
        self.assertEqual(metadata["output"], ["batch", 12])
        self.assertEqual(metadata["goal_fields"], list(GOAL_FIELDS))
        self.assertEqual(sum(width for _, width in metadata["observation_fields"]), self.obs_dim)
        self.assertEqual(metadata["joint_names"], list(self.spec.joint_names))
        self.assertEqual(metadata["default_joint_positions"], list(self.spec.default_q))
        self.assertEqual(metadata["action_scale"], list(self.spec.action_scale))
        self.assertEqual(metadata["height_scan_shape"], [17, 11])
        self.assertEqual(metadata["control_hz"], 50)
        for key in ("runtime_reference_motion", "runtime_discriminator", "runtime_world_model"):
            self.assertIs(metadata[key], False)
        self.assertIs(metadata["normalization_included"], True)
        self.assertEqual(len(metadata["validation_max_error"]), 2)
        self.assertTrue(all(0 <= error <= 1e-5 for error in metadata["validation_max_error"]))
        self.assertNotIn("amp_motion_config", metadata)
        self.assertNotIn("expert_source", metadata)
        self.assertEqual(metadata["source_checkpoint_sha256"], self.source_digest)

    def test_export_contains_only_actor_weights_and_preserves_normalization(self):
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"actor.pt", "policy.ts", "metadata.json"})
        saved = torch.load(self.output / "actor.pt", map_location="cpu", weights_only=True)
        self.assertEqual(set(saved), {
            "actor", "robot", "hist", "obs_dim", "goal_dim", "action_limit", "objective_version",
        })
        self.assertEqual((saved["robot"], saved["hist"], saved["obs_dim"], saved["goal_dim"]),
                         ("go2", 2, 232, self.goal_dim))
        self.assertEqual(saved["action_limit"], 4.)
        self.assertEqual(saved["objective_version"], COMMAND_VERSION)
        expected, traced = self.actor.state_dict(), self.scripted.state_dict()
        self.assertEqual(set(saved["actor"]), set(expected))
        self.assertEqual(set(traced), set(expected))
        for name, tensor in expected.items():
            with self.subTest(tensor=name):
                self.assertEqual(saved["actor"][name].device.type, "cpu")
                torch.testing.assert_close(saved["actor"][name], tensor, atol=0, rtol=0)
                torch.testing.assert_close(traced[name], tensor, atol=0, rtol=0)
        self.assertGreater(saved["actor"]["obs_norm.count"].item(), 0.)
        self.assertGreater(saved["actor"]["goal_norm.count"].item(), 0.)
        for name, _ in self.scripted.named_modules():
            self.assertFalse(any(part in name.lower() for part in
                                 ("world", "critic", "discriminator", "motion", "replay")))

    def test_export_preserves_the_complete_source_checkpoint(self):
        self.assertTrue(self.source.is_file())
        self.assertEqual(file_digest(self.source), self.source_digest)
        source = torch.load(self.source, map_location="cpu", weights_only=True)
        self.assertEqual(source["arguments"], {"hist": 2})
        self.assertNotIn("amp_motion_config", source["arguments"])
        torch.testing.assert_close(source["world"]["training_world_sentinel"], torch.tensor([12345.]))
        torch.testing.assert_close(source["amp_style"]["discriminator"]["training_discriminator_sentinel"],
                                   torch.tensor([23456.]))
        self.assertEqual(source["opt_pi"], self.checkpoint["opt_pi"])
        for name, tensor in self.actor.state_dict().items():
            torch.testing.assert_close(source["actor"][name], tensor, atol=0, rtol=0)

    def test_wrong_objective_is_rejected_before_artifacts_are_written(self):
        for index, fields in enumerate((
            {"training_stage": "tracking", "objective_version": COMMAND_VERSION},
            {"training_stage": "command-style", "objective_version": COMMAND_VERSION - 1},
        )):
            with self.subTest(fields=fields):
                source, output = self.root / f"invalid_{index}.pt", self.root / f"invalid_{index}"
                torch.save(fields, source)
                digest = file_digest(source)
                with self.assertRaisesRegex(ValueError, "command/style"):
                    export(source, output)
                self.assertFalse(output.exists())
                self.assertEqual(file_digest(source), digest)

    def test_nonfinite_actor_is_rejected_before_artifacts_are_written(self):
        actor = dict(self.actor.state_dict())
        actor["goal_norm.mean"] = actor["goal_norm.mean"].clone()
        actor["goal_norm.mean"][0] = float("nan")
        source, output = self.root / "nonfinite.pt", self.root / "nonfinite"
        torch.save({**self.checkpoint, "actor": actor}, source)
        digest = file_digest(source)
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            export(source, output)
        self.assertFalse(output.exists())
        self.assertEqual(file_digest(source), digest)
        self.assertTrue(bool(torch.isfinite(self.actor.goal_norm.mean).all()))

    def test_g1_export_includes_same_stateless_upper_body_limits(self):
        spec = get_robot("g1")
        hist, obs_dim = 2, 3 * spec.num_joints + 196
        actor = TerrainPolicy(
            obs_dim, self.goal_dim, hist=hist, num_joints=spec.num_joints,
            action_limit=20.,
        ).eval()
        reference = {
            name: -.8 if "shoulder_pitch" in name else float(spec.default_q[index])
            for index, name in enumerate(spec.joint_names)
            if any(part in name for part in ("shoulder", "elbow", "wrist"))
        }
        constraint = UpperBodyConstraint(
            spec.joint_names, spec.default_q, spec.action_scale, reference,
        )
        source, output = self.root / "g1-upper.pt", self.root / "g1-upper"
        torch.save({
            "training_stage": "command-style", "objective_version": COMMAND_VERSION,
            "task": "locomotion/g1/stealth", "robot": "g1", "iteration": 3,
            "arguments": {"hist": hist, "action_limit": 20.,
                          "physical_domain_randomization": True},
            "actor": actor.state_dict(),
            "stealth_upper_body_constraint": constraint.to_dict(),
        }, source)
        export(source, output)
        exported = torch.jit.load(str(output / "policy.ts")).eval()
        observations = torch.zeros(5, hist, obs_dim)
        goals = torch.randn(5, self.goal_dim)
        n = spec.num_joints
        with torch.no_grad():
            for _ in range(100):
                previous = observations[:, -1, 2 * n + 6:3 * n + 6].clone()
                raw = actor(observations, goals)
                expected = constraint(raw, previous)
                actual = exported(observations, goals)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                delta = (actual - previous) * torch.tensor(spec.action_scale)
                self.assertLessEqual(delta[:, constraint.upper_indices].abs().max().item(),
                                     .024 + 1e-5)
                torch.testing.assert_close(actual[:, :15], raw[:, :15], atol=1e-5, rtol=1e-5)
                observations[:, -1, 2 * n + 6:3 * n + 6] = actual
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["stealth_upper_body_constraint"], constraint.to_dict())
        self.assertTrue(metadata["physical_domain_randomization"])


if __name__ == "__main__":
    unittest.main()

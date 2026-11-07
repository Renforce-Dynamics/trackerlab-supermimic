"""CPU checks for training-only command subsets and independent reward metadata."""

import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()

import evaluate_terrain
from command_envelope import keyboard_stealth_envelope
from export_command_policy import command_training_metadata, export
from trackerLab.algorithms.supermimic import networks
from trackerLab.algorithms.supermimic.amp_style import AMPStyle
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp import supermimic_command as command_mdp
from trackerLab.tracker_env.supermimic import command as command_sim
from trackerTask.supermimic.robots import get_robot
from test_command_export import file_digest
from test_command_locomotion import CpuCommandSim, make_command, make_state


class CommandCurriculumEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        cls.source = REPO / "scripts/rls/supermimic/evaluate_terrain.py"
        tree = ast.parse(cls.source.read_text(encoding="utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")

        def assigns(node, name):
            return (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets))

        # Execute the production setup on CPU, stopping before actor/AMP construction.
        start = next(i for i, node in enumerate(main.body) if assigns(node, "command_mode"))
        stop = next(i for i, node in enumerate(main.body) if assigns(node, "actor"))
        cls.setup_code = compile(
            ast.Module(body=main.body[start:stop], type_ignores=[]), str(cls.source), "exec")
        report = next(node for node in main.body if assigns(node, "result"))
        cls.report_code = compile(
            ast.Module(body=[report], type_ignores=[]), str(cls.source), "exec")
        loop = next(node for node in main.body if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "step_index")
        cls.sequence_code = compile(
            ast.Module(body=[loop.body[0]], type_ignores=[]), str(cls.source), "exec")

    def run_setup(self, arguments, *, held=None, layout="random", sequence=None):
        arguments = {"hist": 2, "max_episode_steps": 1000, "seed": 42, **arguments}
        checkpoint = {
            "arguments": arguments, "training_stage": "command-style",
            "task": "locomotion/g1/stealth", "iteration": 0,
        }
        mode = arguments.get("stealth_control_mode", "fixed")
        if mode in ("keyboard", "wasd"):
            checkpoint["command_envelope"] = keyboard_stealth_envelope(mode)
        args = SimpleNamespace(
            command_mode=held, mode_layout=layout, command_sequence=sequence,
            command_stage_seconds=3., steps=1050 if sequence else 1, num_envs=9,
            seed=731, checkpoint=Path("cpu-fixture.pt"))
        spec = SimpleNamespace(default_q=[0.] * 4, joint_names=["j0", "j1", "j2", "j3"],
                               num_joints=4)
        model = SimpleNamespace(
            joint=lambda name: SimpleNamespace(id=spec.joint_names.index(name)),
            jnt_range=np.array([[-10., 10.]] * 4))

        def simulation(task, mdp, num_envs, device, seed, solver_iterations,
                       max_episode_steps, **options):
            self.assertEqual(device, "cpu")
            self.assertNotIn("command_training_modes", options)
            self.assertNotIn("training_command_modes", options)
            sim = CpuCommandSim(
                task.profile, batch=num_envs, max_episode_steps=max_episode_steps,
                clearance_ratio=arguments.get("clearance_ratio", 1.),
                stealth_keyboard_mode=options["stealth_keyboard_mode"])
            sim.mdp = mdp
            sim.command_mode_set = options.get("command_mode_set", "keyboard")
            sim.command_resample_steps = options["command_resample_steps"]
            return sim

        namespace = {
            **vars(evaluate_terrain), "ck": checkpoint, "args": args, "device": "cpu",
            "parser": argparse.ArgumentParser(),
            "robot_parameters": lambda robot: (spec, .76, []),
            "build_model": lambda *args, **kwargs: (model,),
        }
        with patch.object(command_mdp, "CommandMDP", wraps=command_mdp.CommandMDP) as constructor, \
                patch.object(command_sim, "CommandSim", side_effect=simulation), \
                patch.object(CpuCommandSim, "set_training_command_modes", create=True,
                             side_effect=AssertionError("evaluation applied training subset")):
            exec(self.setup_code, namespace)
        self.assertEqual(constructor.call_count, 1)
        namespace["mdp_options"] = constructor.call_args.kwargs
        return namespace

    def test_missing_and_none_reward_modes_keep_historical_defaults(self):
        for control in (None, "fixed", "keyboard", "wasd"):
            for explicit_none in (False, True):
                with self.subTest(control=control, explicit_none=explicit_none):
                    arguments = {} if control is None else {"stealth_control_mode": control}
                    if explicit_none:
                        arguments["command_reward_mode"] = None
                    expected = "wasd" if control == "wasd" else "legacy"
                    self.assertEqual(command_training_metadata({"arguments": arguments}), {
                        "command_reward_mode": expected, "training_command_modes": None,
                    })
                    setup = self.run_setup(arguments)
                    self.assertEqual(setup["mdp_options"]["reward_control_mode"], expected)
                    self.assertEqual(setup["mdp"].reward_control_mode, expected)
                    self.assertEqual(setup["mdp"].control_mode, expected)

    def test_explicit_reward_and_exact_training_argument_are_independent(self):
        modes = ["forward", "forward_left"]
        arguments = {
            "stealth_control_mode": "wasd", "command_reward_mode": "legacy",
            "command_training_modes": modes, "training_command_modes": ["idle"],
        }
        metadata = command_training_metadata({"arguments": arguments})
        self.assertEqual(metadata, {
            "command_reward_mode": "legacy", "training_command_modes": modes,
        })
        metadata["training_command_modes"].append("backward")
        self.assertEqual(modes, ["forward", "forward_left"])
        self.assertIsNone(command_training_metadata({
            "arguments": {"training_command_modes": ["idle"]},
        })["training_command_modes"])
        base = {"stealth_control_mode": "wasd"}
        self.assertEqual(evaluate_terrain.command_runtime_options({"arguments": arguments}),
                         evaluate_terrain.command_runtime_options({"arguments": base}))
        for invalid in ("", "keyboard", False):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "command_reward_mode"):
                command_training_metadata({"arguments": {"command_reward_mode": invalid}})

    def test_wasd_sampler_can_preserve_legacy_reward_and_observations(self):
        legacy = self.run_setup({"stealth_control_mode": "keyboard"})["mdp"]
        setup = self.run_setup({
            "stealth_control_mode": "wasd", "command_reward_mode": "legacy",
            "command_training_modes": ["forward"],
        })
        mdp = setup["mdp"]
        self.assertEqual(setup["mdp_options"]["control_mode"], "wasd")
        self.assertEqual(setup["mdp_options"]["reward_control_mode"], "legacy")
        self.assertEqual(mdp.control_mode, "wasd")
        self.assertEqual(mdp.reward_control_mode, "legacy")
        state, command = make_state(legacy, 9), make_command(legacy, 9)
        command[:, 1:4] = torch.tensor([row[1:4] for row in command_mdp.stealth_keyboard_modes("wasd")])
        action = torch.zeros(9, 4)
        xi = torch.zeros(9, 8)
        torch.testing.assert_close(mdp.reward(state, action, action, command),
                                   legacy.reward(state, action, action, command), atol=0, rtol=0)
        torch.testing.assert_close(mdp.obs(state, action, xi, command),
                                   legacy.obs(state, action, xi, command), atol=0, rtol=0)
        torch.testing.assert_close(mdp.goal(state, command), legacy.goal(state, command), atol=0, rtol=0)

    def test_training_subset_cannot_restrict_held_evaluation(self):
        canonical = command_mdp.stealth_keyboard_modes("wasd")
        for name, vx, vy, yaw, _ in canonical:
            with self.subTest(mode=name):
                setup = self.run_setup({
                    "stealth_control_mode": "wasd", "command_reward_mode": "legacy",
                    "command_training_modes": ["forward"],
                }, held=name)
                env = setup["env"]
                self.assertEqual(setup["mode_names"], [row[0] for row in canonical])
                self.assertEqual(env._keyboard_modes(), canonical)
                torch.testing.assert_close(env.c[:, 1:4], torch.tensor([vx, vy, yaw]).expand(9, -1))
                env.reset()
                torch.testing.assert_close(env.c[:, 1:4], torch.tensor([vx, vy, yaw]).expand(9, -1))

    def test_training_subset_cannot_restrict_cycle_evaluation(self):
        setup = self.run_setup({
            "stealth_control_mode": "wasd", "command_reward_mode": "legacy",
            "command_training_modes": ["forward"],
        }, layout="cycle")
        env = setup["env"]
        canonical = command_mdp.stealth_keyboard_modes("wasd")
        expected = torch.tensor([row[1:4] for row in canonical])
        for _ in range(3):
            torch.testing.assert_close(env.c[:, 1:4], expected)
            torch.testing.assert_close(env._stealth_mode_ids, torch.arange(9))
            env.reset()
        self.assertEqual(env.contract["keyboard_evaluation_layout"]["mode_count"], 9)

    def test_training_subset_cannot_restrict_release_evaluation(self):
        setup = self.run_setup({
            "stealth_control_mode": "wasd", "command_reward_mode": "legacy",
            "command_training_modes": ["forward"], "entry_duration_s": 4.84,
        }, sequence="wasd-release-regression")
        env = setup["env"]
        plan = setup["sequence_plan"]
        self.assertEqual([stage["mode"] for stage in plan["stages"]],
                         [row[0] for row in evaluate_terrain.WASD_RELEASE_REGRESSION])
        self.assertEqual(plan["total_steps"], 1050)
        self.assertEqual(env.max_episode_steps, 1051)
        setup["sequence_stage_index"] = 0
        env.c[:, 6] = 2.4
        clock = env.c[:, 6].clone()
        for stage in plan["stages"]:
            setup["step_index"] = stage["start_step"]
            exec(self.sequence_code, setup)
            self.assertEqual(env._fixed_keyboard_mode_name, stage["mode"])
            torch.testing.assert_close(env.c[:, 6], clock)
            expected = next(row[1:4] for row in env._keyboard_modes() if row[0] == stage["mode"])
            torch.testing.assert_close(env.c[:, 1:4], torch.tensor(expected).expand(9, -1))

    def test_evaluation_report_records_effective_reward_and_training_subset(self):
        for arguments, reward, subset in (
            ({}, "legacy", None),
            ({"stealth_control_mode": "wasd"}, "wasd", None),
            ({"stealth_control_mode": "wasd", "command_reward_mode": "legacy",
              "command_training_modes": ["forward"]}, "legacy", ["forward"]),
        ):
            with self.subTest(arguments=arguments):
                setup = self.run_setup(arguments)
                setup.update(rate_limiter_config=None, lanes=[])
                exec(self.report_code, setup)
                report = json.loads(json.dumps(setup["result"]))
                self.assertEqual(report["command_reward_mode"], reward)
                self.assertEqual(report["training_command_modes"], subset)
                self.assertEqual(report["lanes"], [])


class CommandCurriculumExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        cls.addClassCleanup(setattr, networks, "AMP_DTYPE", networks.AMP_DTYPE)
        torch.set_num_threads(2)
        networks.AMP_DTYPE = None
        temporary = tempfile.TemporaryDirectory(prefix="command-curriculum-export-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.spec = get_robot("g1")
        cls.hist, cls.obs_dim = 2, 3 * cls.spec.num_joints + 196
        cls.goal_dim = len(command_mdp.GOAL_FIELDS)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(914)
            cls.actor = TerrainPolicy(
                cls.obs_dim, cls.goal_dim, hist=cls.hist, num_joints=cls.spec.num_joints,
                action_limit=4.).eval()
            cls.actor.obs_norm.update(torch.randn(16, cls.hist, cls.obs_dim))
            cls.actor.goal_norm.update(torch.randn(16, cls.goal_dim))
        cls.exports = {}
        cases = {
            "legacy": {"stealth_control_mode": "keyboard"},
            "old-wasd": {"stealth_control_mode": "wasd"},
            "staged": {"stealth_control_mode": "wasd", "command_reward_mode": "legacy",
                       "command_training_modes": ["forward", "forward_left"]},
        }
        for name, settings in cases.items():
            source, output = cls.root / f"{name}.pt", cls.root / name
            checkpoint = {
                "training_stage": "command-style", "objective_version": command_mdp.COMMAND_VERSION,
                "task": "locomotion/g1/stealth", "robot": "g1", "iteration": 0,
                "arguments": {"hist": cls.hist, **settings}, "actor": cls.actor.state_dict(),
                "command_envelope": keyboard_stealth_envelope(settings["stealth_control_mode"]),
            }
            torch.save(checkpoint, source)
            digest = file_digest(source)
            with patch.object(MotionLib, "__init__", side_effect=AssertionError("motion load")), \
                    patch.object(AMPStyle, "__init__", side_effect=AssertionError("AMP load")), \
                    patch.object(torch.cuda, "synchronize", side_effect=AssertionError("GPU work")):
                metadata = export(source, output)
            cls.exports[name] = (source, output, metadata, digest, settings)

    def test_export_metadata_and_actor_payload_roundtrip_training_settings(self):
        for name, (source, output, metadata, digest, settings) in self.exports.items():
            with self.subTest(case=name):
                decoded = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
                saved = torch.load(output / "actor.pt", map_location="cpu", weights_only=True)
                expected = {
                    "command_reward_mode": "wasd" if name == "old-wasd" else "legacy",
                    "training_command_modes": settings.get("command_training_modes"),
                }
                self.assertEqual(decoded, json.loads(json.dumps(metadata)))
                for document in (decoded, saved):
                    self.assertEqual({key: document[key] for key in expected}, expected)
                self.assertEqual(file_digest(source), digest)
                self.assertEqual(decoded["source_checkpoint_sha256"], digest)
                checkpoint = torch.load(source, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["arguments"], {"hist": self.hist, **settings})
                self.assertEqual(set(saved["actor"]), set(self.actor.state_dict()))
                for key, value in self.actor.state_dict().items():
                    torch.testing.assert_close(saved["actor"][key], value, atol=0, rtol=0)
                if settings["stealth_control_mode"] == "wasd":
                    self.assertEqual(saved["stealth_control_mode"], "wasd")
                    self.assertEqual(decoded["keyboard_control"], command_mdp.wasd_keyboard_metadata())
                    self.assertEqual(len(decoded["trained_command_envelope"]["command_modes"]), 9)

    def test_training_settings_leave_actor_inputs_and_outputs_unchanged(self):
        generator = torch.Generator().manual_seed(422)
        observations = torch.randn(7, self.hist, self.obs_dim, generator=generator)
        goals = torch.randn(7, self.goal_dim, generator=generator)
        with torch.no_grad():
            expected = self.actor(observations, goals)
            for name, (_, output, metadata, _, _) in self.exports.items():
                with self.subTest(case=name):
                    self.assertEqual(metadata["inputs"], {
                        "observation_history": ["batch", self.hist, self.obs_dim],
                        "command_goal": ["batch", self.goal_dim],
                    })
                    self.assertEqual(sum(width for _, width in metadata["observation_fields"]), self.obs_dim)
                    for flag in ("runtime_reference_motion", "runtime_discriminator", "runtime_world_model"):
                        self.assertIs(metadata[flag], False)
                    portable = torch.jit.load(str(output / "policy.ts"), map_location="cpu").eval()
                    self.assertEqual(set(portable.state_dict()), set(self.actor.state_dict()))
                    for batch in (1, 7):
                        torch.testing.assert_close(
                            portable(observations[:batch], goals[:batch]), expected[:batch],
                            atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

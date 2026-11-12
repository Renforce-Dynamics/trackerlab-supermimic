"""Absolute root clearance survives command training, evaluation and CPU export."""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts/rls/supermimic"
sys.path.insert(0, str(SCRIPTS))
from bootstrap import configure_paths
configure_paths()

import evaluate_terrain
import train_terrain
from command_envelope import keyboard_stealth_envelope
from export_command_policy import command_clearance_options, command_training_metadata, export
from trackerLab.algorithms.supermimic import networks
from trackerLab.algorithms.supermimic.terrain_networks import TerrainPolicy
from trackerLab.tracker_env.mdp.supermimic_command import COMMAND_VERSION, CommandMDP
from trackerLab.tracker_env.mdp.supermimic_terrain import HeightField
from trackerLab.tracker_env.supermimic.command import CommandSim
from trackerLab.tracker_env.supermimic.terrain import TerrainSim
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import generate, get_task
from test_command_locomotion import make_state


HEIGHT = 0.6651475726945968
BASE = ["--task", "locomotion/g1/stealth", "--run-dir", "unused-clearance"]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(networks, "AMP_DTYPE", None)
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU work is outside these clearance tests")
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    yield
    torch.set_num_threads(threads)


def arguments(*options):
    return train_terrain.parse_args([
        *BASE, "--stealth-control-mode", "wasd", "--clearance-ratio", ".92",
        "--entry-duration-s", ".6", "--device", "cpu", "--num-envs", "9", *options])


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def _main_body(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return next(node.body for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")


def _execute(body, module, namespace):
    code = compile(ast.Module(body=body, type_ignores=[]), module.__file__, "exec")
    exec(code, namespace)
    return namespace


def _cpu_parent(self, task, mdp, num_envs, device, seed, solver_iterations,
                max_episode_steps, physical_ray_validation, **options):
    """Replace GPU allocation only; use the real command constructor and sampler."""
    assert device == "cpu"
    self.task, self.mdp, self.num_envs, self.device = task, mdp, num_envs, device
    self.height, self.scale = mdp.field.height, mdp.field.scale
    self.contract = {}
    self.c = torch.zeros(num_envs, 7)
    self._command_age = torch.zeros(num_envs, dtype=torch.long)
    self._command_interval = torch.zeros_like(self._command_age)
    self.sample_commands()


def training_setup(args):
    spec, task = get_robot("g1"), get_task(args.task)
    field = HeightField(generate(task.profile, task.height, args.seed), task.height, "cpu")
    namespace = {
        **vars(train_terrain), "args": args, "task": task, "field": field,
        "default_q": torch.tensor(spec.default_q),
        "qlo": torch.full((29,), -10.), "qhi": torch.full((29,), 10.),
        "CommandMDP": CommandMDP, "CommandSim": CommandSim,
    }
    body = _main_body(train_terrain)
    start = next(i for i, node in enumerate(body) if _assigns(node, "mdp"))
    end = next(i for i, node in enumerate(body) if _assigns(node, "actor"))
    with patch.object(TerrainSim, "__init__", new=_cpu_parent):
        return _execute(body[start:end], train_terrain, namespace)


def evaluation_setup(saved):
    spec = get_robot("g1")
    checkpoint = {
        "task": "locomotion/g1/stealth", "training_stage": "command-style",
        "arguments": {"hist": 2, "max_episode_steps": 1000, "seed": 42,
                      "stealth_control_mode": "wasd", **saved},
        "command_envelope": keyboard_stealth_envelope("wasd"),
    }
    model = SimpleNamespace(
        joint=lambda name: SimpleNamespace(id=spec.joint_names.index(name)),
        jnt_range=np.array([[-10., 10.]] * 29))
    namespace = {
        **vars(evaluate_terrain), "ck": checkpoint, "device": "cpu",
        "args": SimpleNamespace(command_mode=None, mode_layout="cycle", command_sequence=None,
                                command_stage_seconds=3., steps=1000, seed=101, num_envs=9),
        "parser": argparse.ArgumentParser(),
        "robot_parameters": lambda robot: (spec, .76, []),
        "build_model": lambda *args, **kwargs: (model,),
    }
    body = _main_body(evaluate_terrain)
    start = next(i for i, node in enumerate(body) if _assigns(node, "command_mode"))
    end = next(i for i, node in enumerate(body) if _assigns(node, "levels"))
    with patch.object(TerrainSim, "__init__", new=_cpu_parent):
        return _execute(body[start:end], evaluate_terrain, namespace)


def test_actual_parser_accepts_fullbody_preset_and_explicit_override():
    fullbody = SCRIPTS / "configs/command_stealth_full_body.json"
    parsed = train_terrain.parse_args([*BASE, "--config", str(fullbody)])
    assert parsed.target_clearance_m == HEIGHT
    assert parsed.iterations == 5500 and parsed.wm_only_iters == 500
    assert parsed.stealth_control_mode == "wasd" and parsed.amp_weight == .1
    overridden = train_terrain.parse_args([
        *BASE, "--config", str(fullbody), "--target-clearance-m", ".61"])
    assert overridden.target_clearance_m == .61
    assert train_terrain.parse_args(BASE).target_clearance_m is None
    old = fullbody.with_name("command_stealth_beyondamp.json")
    assert train_terrain.parse_args([*BASE, "--config", str(old)]).target_clearance_m is None


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_cli_height_fails_during_parsing(value):
    with pytest.raises(SystemExit):
        arguments(f"--target-clearance-m={value}")


@pytest.mark.parametrize("value", [True, False, 0, -.1, float("nan"), float("inf"), "0.665"])
def test_invalid_saved_or_preset_height_rejected(tmp_path, value):
    config = tmp_path / "bad-height.json"
    config.write_text(json.dumps({"target_clearance_m": value}), encoding="utf-8")
    with pytest.raises(SystemExit):
        train_terrain.parse_args([*BASE, "--config", str(config)])
    saved = {"arguments": {"target_clearance_m": value}}
    for restore in (command_training_metadata, evaluate_terrain.command_runtime_options):
        with pytest.raises(ValueError, match="target_clearance_m"):
            restore(saved)
    with pytest.raises(ValueError, match="target_clearance_m"):
        train_terrain.validate_resume_clearance(saved, arguments())


def test_train_and_evaluator_use_absolute_height_for_every_wasd_command():
    args = arguments("--target-clearance-m", str(HEIGHT))
    training = training_setup(args)
    evaluation = evaluation_setup(vars(args))
    for setup in (training, evaluation):
        mdp, sim = setup["mdp"], setup["env"]
        assert mdp.target_clearance_m == sim.target_clearance_m == HEIGHT
        assert sim.contract["target_clearance_m"] == HEIGHT
        assert HEIGHT != .92 * .76
        sim.set_keyboard_evaluation_cycle()
        sim.sample_commands()
        assert sim._stealth_mode_ids.tolist() == list(range(9))
        state = make_state(mdp, 9)
        zeros = torch.zeros(9, 29)
        observation = mdp.obs(state, zeros, torch.zeros(9, 33), sim.c)
        assert observation.shape == (9, 283)
        for elapsed, blend in ((0., 0.), (.3, .5), (.6, 1.), (10., 1.)):
            sim.c[:, 6] = elapsed
            sim.sample_commands(torch.tensor([0, 4, 8]))
            torch.testing.assert_close(sim.c[:, 4], torch.full((9,), HEIGHT), atol=0, rtol=0)
            torch.testing.assert_close(sim.c[:, 6], torch.full((9,), elapsed), atol=0, rtol=0)
            goal = mdp.goal(state, sim.c)
            assert goal.shape == (9, 12)
            torch.testing.assert_close(goal[:, 3],
                torch.full((9,), (.76 + blend * (HEIGHT-.76)) / mdp.field.scale))
    assert evaluation["training_metadata"]["target_clearance_m"] == HEIGHT
    assert evaluate_terrain.command_runtime_options({"arguments": vars(args)})[
        "target_clearance_m"] == HEIGHT


def test_missing_and_none_preserve_sampler_rng_observations_goals_and_rewards_exactly():
    results = []
    saved = {"clearance_ratio": .92, "entry_duration_s": .6}
    for settings in (saved, saved | {"target_clearance_m": None}):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(910)
            setup = evaluation_setup(settings)
            mdp, sim = setup["mdp"], setup["env"]
            sim.c[:, 6] = .6
            state, action, xi = make_state(mdp, 9), torch.zeros(9, 29), torch.zeros(9, 33)
            results.append([
                tensor.detach().numpy().tobytes() for tensor in (
                    sim.c, torch.get_rng_state(), mdp.obs(state, action, xi, sim.c),
                    mdp.goal(state, sim.c), mdp.reward(state, action, action, sim.c))])
            assert mdp.target_clearance_m == .92 * .76
            assert "target_clearance_m" not in setup["training_metadata"]
            assert "target_clearance_m" not in evaluate_terrain.command_runtime_options(
                {"arguments": settings})
    assert results[0] == results[1]
    assert command_clearance_options({}) == command_clearance_options(
        {"target_clearance_m": None}) == {}


@pytest.mark.parametrize("saved,current,valid", [
    ({}, None, True), ({"target_clearance_m": None}, None, True),
    ({"target_clearance_m": HEIGHT}, HEIGHT, True),
    ({}, HEIGHT, False), ({"target_clearance_m": HEIGHT}, None, False),
    ({"target_clearance_m": HEIGHT}, .92 * .76, False),
])
def test_actual_resume_prefix_requires_identical_optional_height(saved, current, valid):
    args = arguments(*([] if current is None else ["--target-clearance-m", str(current)]))
    ck = {"task": args.task, "training_stage": "command-style",
          "objective_version": COMMAND_VERSION, "arguments": saved}
    before = deepcopy(ck)
    body = _main_body(train_terrain)
    resume = next(node for node in body if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Attribute) and node.test.attr == "resume")
    end = next(i for i, node in enumerate(resume.body) if _assigns(node, "sampling_receipt"))
    namespace = {**vars(train_terrain), "resume_checkpoint": ck, "args": args,
                 "COMMAND_VERSION": COMMAND_VERSION}
    if valid:
        _execute(resume.body[:end], train_terrain, namespace)
    else:
        with pytest.raises(ValueError, match="target_clearance_m"):
            _execute(resume.body[:end], train_terrain, namespace)
    assert ck == before


def test_actual_checkpoint_assignment_keeps_absolute_height_and_physical_contract(tmp_path):
    args = arguments("--target-clearance-m", str(HEIGHT), "--amp-motion-config", "pool/train.json")
    setup = training_setup(args)
    body = _main_body(train_terrain)
    node = next(node for statement in body for node in ast.walk(statement)
                if _assigns(node, "checkpoint"))
    namespace = {
        "args": args, "iteration": 749, "COMMAND_VERSION": COMMAND_VERSION,
        "initialization": None, "resume_receipt": None, "task": setup["task"],
        "command_envelope": keyboard_stealth_envelope("wasd"),
        "quality": SimpleNamespace(sole_height=torch.zeros(2)),
        "style": SimpleNamespace(checkpoint=lambda: {}),
        "entry_style_config": None, "aim_style_config": None, "waist_config": None,
        "waist_reference_receipt": None, "upper_body_config": None, "rate_limiter_config": None,
        "upper_body_reference_receipt": None, "env": setup["env"], "run": tmp_path,
        **{name: SimpleNamespace(state_dict=lambda: {}) for name in (
            "actor", "world", "critic", "opt_wm", "opt_pi", "opt_v")},
    }
    _execute([node], train_terrain, namespace)
    saved = namespace["checkpoint"]
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["arguments"]["target_clearance_m"] == HEIGHT
    assert restored["simulator_contract"]["target_clearance_m"] == HEIGHT
    train_terrain.validate_resume_clearance(restored, args)


@pytest.mark.parametrize("settings", [{}, {"target_clearance_m": None},
                                    {"target_clearance_m": HEIGHT}])
def test_cpu_export_preserves_height_and_leaves_actor_contract_unchanged(tmp_path, settings):
    spec = get_robot("g1")
    actor = TerrainPolicy(283, 12, hist=2, num_joints=29, action_limit=20.).eval()
    checkpoint = {
        "task": "locomotion/g1/stealth", "robot": "g1", "iteration": 5500,
        "training_stage": "command-style", "objective_version": COMMAND_VERSION,
        "arguments": {"hist": 2, "action_limit": 20., "clearance_ratio": .92,
                      "entry_duration_s": .6, "stealth_control_mode": "wasd", **settings},
        "actor": actor.state_dict(), "command_envelope": keyboard_stealth_envelope("wasd"),
    }
    source, output = tmp_path / "source.pt", tmp_path / "\u6a21\u578b\u5bfc\u51fa"
    torch.save(checkpoint, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = export(source, output)
    payload = torch.load(output / "actor.pt", map_location="cpu", weights_only=True)
    decoded = json.loads((output / "metadata.json").read_text())
    assert decoded == json.loads(json.dumps(metadata))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    explicit = settings.get("target_clearance_m") is not None
    for document in (metadata, payload):
        if explicit:
            assert document["target_clearance_m"] == HEIGHT
        else:
            assert "target_clearance_m" not in document
    assert metadata["inputs"] == {"observation_history": ["batch", 2, 283],
                                 "command_goal": ["batch", 12]}
    assert metadata["output"] == ["batch", 29]
    assert list(metadata["joint_names"]) == list(spec.joint_names)
    assert metadata["action_scale"] == spec.action_scale
    assert metadata["actuators"] is None
    assert len(metadata["keyboard_control"]["command_modes"]) == 9
    assert (payload["obs_dim"], payload["goal_dim"]) == (283, 12)
    mdp = CommandMDP(
        HeightField(generate("stealth", .76, 42), .76, "cpu"),
        torch.tensor(spec.default_q), torch.full((29,), -10.), torch.full((29,), 10.),
        profile="stealth", clearance_ratio=.92, **command_clearance_options(payload))
    state = make_state(mdp, 2)
    commands = torch.zeros(2, 7)
    commands[:, 4] = mdp.target_clearance_m
    goal = mdp.goal(state, commands)
    torch.testing.assert_close(goal[:, 3],
        torch.full((2,), HEIGHT if explicit else .92 * .76), atol=0, rtol=0)
    with (output / "policy.ts").open("rb") as stream:
        portable = torch.jit.load(stream, map_location="cpu").eval()
    history = torch.zeros(2, 2, 283)
    with torch.no_grad():
        torch.testing.assert_close(portable(history, goal), actor(history, goal),
                                   atol=1e-5, rtol=1e-5)


@pytest.fixture
def declared_fullbody_pool(tmp_path, monkeypatch):
    import prepare_style_campaign
    pool = tmp_path / "declared-pool"
    pool.mkdir()
    config = pool / "train.json"
    config.write_text(json.dumps({"root": "."}), encoding="utf-8")
    auxiliary = pool / "full-body-style.json"
    payload = json.dumps({"target_clearance_m": HEIGHT}).encode()
    auxiliary.write_bytes(payload)
    manifest = {
        "schema": "style-campaign-pool-v1", "style": "stealth", "recipe_sha256": "a" * 64,
        "auxiliary_files": {"full_body_style": {"file": auxiliary.name,
            "sha256": hashlib.sha256(payload).hexdigest()}},
    }
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Motion/kinematic auditing belongs to the existing and parent-owned verifiers.
    verified = []
    def verify(root, robot, style):
        assert (root, robot, style) == (pool, "g1", "stealth")
        verified.append(root)
        return manifest
    monkeypatch.setattr(prepare_style_campaign, "verify_pool", verify)
    return pool, manifest, verified


def test_declared_fullbody_pool_requires_exact_explicit_height(declared_fullbody_pool):
    pool, manifest, verified = declared_fullbody_pool
    result = train_terrain.validate_style_references(
        pool / "train.json", "g1", target_clearance_m=HEIGHT)
    assert verified == [pool]
    assert result == {
        "kind": "immutable-style-pool", "style": "stealth",
        "manifest": str(pool / "manifest.json"), "recipe_sha256": manifest["recipe_sha256"]}


@pytest.mark.parametrize("height", [None, .92 * .76, float(np.nextafter(HEIGHT, 1.))])
def test_declared_pool_height_omission_or_mismatch_stops_before_cuda(
        declared_fullbody_pool, monkeypatch, height):
    pool, _, verified = declared_fullbody_pool
    options = [] if height is None else ["--target-clearance-m", str(height)]
    monkeypatch.setattr(sys, "argv", [
        "train_terrain.py", *BASE, "--amp-motion-config", str(pool / "train.json"), *options])
    with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA boundary")) as cuda:
        with pytest.raises(ValueError, match="explicit target_clearance_m"):
            train_terrain.main()
    cuda.assert_not_called()
    assert verified == []


def test_matching_declared_pool_passes_the_real_trainer_pre_cuda_gate(
        declared_fullbody_pool, monkeypatch):
    pool, _, verified = declared_fullbody_pool
    monkeypatch.setattr(sys, "argv", [
        "train_terrain.py", *BASE, "--amp-motion-config", str(pool / "train.json"),
        "--target-clearance-m", str(HEIGHT)])
    with patch.object(torch.cuda, "is_available", return_value=False) as cuda:
        with pytest.raises(RuntimeError, match="Terrain training requires CUDA"):
            train_terrain.main()
    cuda.assert_called_once_with()
    assert verified == [pool]


@pytest.mark.parametrize("tamper", ["document", "manifest-hash"])
def test_fullbody_auxiliary_hash_must_match(declared_fullbody_pool, tamper):
    pool, manifest, verified = declared_fullbody_pool
    if tamper == "document":
        (pool / "full-body-style.json").write_text(
            json.dumps({"target_clearance_m": .92 * .76}), encoding="utf-8")
    else:
        manifest["auxiliary_files"]["full_body_style"]["sha256"] = "b" * 64
        (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="auxiliary hash"):
        train_terrain.validate_style_references(pool / "train.json", "g1", target_clearance_m=HEIGHT)
    assert verified == []


@pytest.mark.parametrize("record", [None, {}, {"file": "../full-body-style.json", "sha256": "a" * 64}])
def test_malformed_fullbody_declaration_fails_closed(declared_fullbody_pool, record):
    pool, manifest, verified = declared_fullbody_pool
    manifest["auxiliary_files"]["full_body_style"] = record
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="must declare full-body-style.json"):
        train_terrain.validate_style_references(pool / "train.json", "g1", target_clearance_m=HEIGHT)
    assert verified == []


@pytest.mark.parametrize("document", [{}, {"target_clearance_m": None}, {"target_clearance_m": 0}, []])
def test_hashed_fullbody_document_requires_a_valid_height(declared_fullbody_pool, document):
    pool, manifest, verified = declared_fullbody_pool
    payload = json.dumps(document).encode()
    (pool / "full-body-style.json").write_bytes(payload)
    manifest["auxiliary_files"]["full_body_style"]["sha256"] = hashlib.sha256(payload).hexdigest()
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        train_terrain.validate_style_references(pool / "train.json", "g1", target_clearance_m=HEIGHT)
    assert verified == []


@pytest.mark.parametrize("auxiliary", [None, {}, {"aim_style": {}}])
def test_legacy_pool_without_fullbody_declaration_keeps_existing_validation(
        declared_fullbody_pool, auxiliary):
    pool, manifest, verified = declared_fullbody_pool
    manifest["auxiliary_files"] = auxiliary
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # An undeclared file has no effect on a legacy pool.
    (pool / "full-body-style.json").write_bytes(b"ignored legacy file")
    legacy = train_terrain.validate_style_references(pool / "train.json", "g1")
    explicit = train_terrain.validate_style_references(
        pool / "train.json", "g1", target_clearance_m=HEIGHT)
    assert legacy == explicit
    assert verified == [pool, pool]

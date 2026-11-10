"""Shared aim data and workflow must preserve command locomotion contracts."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from prepare_stealth_shared_aim import replace_upper
from train_terrain import parse_args, validate_upper_body_options, load_shared_aim_reference
from trackerTask.supermimic.robots import get_robot
from trackerTask.supermimic.terrain import get_task
from test_command_locomotion import CpuCommandSim


def test_shared_pose_changes_only_named_upper_fields():
    spec = get_robot("g1")
    values = np.arange(4*36).reshape(4, 36)/100.
    stream = io.StringIO()
    csv.writer(stream).writerows(values)
    names = [name for name in spec.joint_names if any(
        p in name.split("_") for p in ("waist", "shoulder", "elbow", "wrist"))]
    pose = {name: (i-8)*.02 for i, name in enumerate(names)}
    changed = replace_upper(stream.getvalue().encode(), pose, list(spec.joint_names), 4)
    result = np.loadtxt(io.BytesIO(changed), delimiter=",")
    np.testing.assert_array_equal(result[:, :19], values[:, :19])
    for name, value in pose.items():
        np.testing.assert_array_equal(result[:, 7+spec.joint_names.index(name)], [value]*4)
    with pytest.raises(ValueError):
        replace_upper(stream.getvalue().encode(), pose, list(spec.joint_names), 5)


def test_new_preset_keeps_controls_and_changes_only_style_strength():
    base = ["--task", "locomotion/g1/stealth", "--run-dir", "unused", "--warm-start", "source.pt"]
    folder = REPO/"scripts/rls/supermimic/configs"
    old = parse_args([*base, "--config", str(folder/"command_stealth_velocity_led.json")])
    new = parse_args([*base, "--config", str(folder/"command_stealth_shared_aim.json"),
                      "--aim-style-reference", "reviewed.json"])
    changed = {key for key in vars(old) if getattr(old,key) != getattr(new,key)}
    assert changed == {"config", "amp_weight", "aim_style_reference"}
    assert new.amp_weight == .1 and new.entry_duration_s == .6
    assert not new.warm_start_reset_arms and new.warm_start_require_control_match
    validate_upper_body_options(new, get_task(new.task))
    new.stealth_upper_body_constraint = True
    with pytest.raises(ValueError):
        validate_upper_body_options(new, get_task(new.task))


def test_physical_aim_cost_uses_completed_transition_and_is_included_once():
    sim = CpuCommandSim(max_episode_steps=1)
    observed = []
    def aim(state, c):
        observed.append(c.clone())
        return torch.full_like(c[:, 0], -.007)
    sim.aim_style_reward_fn = aim
    transition = sim.step(torch.zeros(1, 4))
    torch.testing.assert_close(observed[0], transition["c1"])
    torch.testing.assert_close(transition["reward_terms"]["aim_style"], torch.tensor([-.007]))
    terms = transition["reward_terms"]
    torch.testing.assert_close(transition["r"], sum(terms.values()))


def test_reward_target_is_bound_to_the_amp_pool_recipe(tmp_path):
    document = {"source": {"canonical_sha256": "a"*64}, "weight": .01}
    payload = json.dumps(document).encode()
    target = tmp_path / "aim-style-reference.json"
    target.write_bytes(payload)
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"root": "."}))
    manifest = {
        "auxiliary_files": {"aim_style": {"file": target.name,
            "sha256": hashlib.sha256(payload).hexdigest()}},
        "recipe": {"guidance": document, "source": document["source"]},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_shared_aim_reference(target, config) == document
    elsewhere = tmp_path / "other.json"
    elsewhere.write_bytes(payload)
    with pytest.raises(ValueError, match="declared"):
        load_shared_aim_reference(elsewhere, config)
    target.write_text(json.dumps(dict(document, weight=.02)))
    with pytest.raises(ValueError, match="hash"):
        load_shared_aim_reference(target, config)
    target.write_bytes(payload)
    manifest["recipe"]["source"] = {"canonical_sha256": "b"*64}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source"):
        load_shared_aim_reference(target, config)

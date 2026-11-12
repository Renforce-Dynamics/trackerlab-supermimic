"""Trainer entry-point checks for the bounded G1 sole-speed AMP recipe."""

from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/rls/supermimic"))
import train_terrain


def arguments(task="locomotion/g1/stealth"):
    return train_terrain.parse_args([
        "--task", task, "--run-dir", "unused-sole-speed-run",
        "--config", str(REPO / "scripts/rls/supermimic/configs/command_stealth_full_body_sole_speed.json"),
        "--describe",
    ])


def test_sole_speed_cli_preserves_actor_and_world_warm_start_contract():
    args = arguments()
    assert args.style_feature_mode == "postural_end_effectors_vx_sole_speed"
    assert args.iterations - args.wm_only_iters == 5000
    assert args.freeze_actor_normalization and args.warm_start_require_control_match
    assert not args.warm_start_reset_arms and not args.warm_start_reset_world
    assert not args.allow_style_feature_change
    old = vars(args) | {"style_feature_mode": "postural_end_effectors_vx"}
    assert train_terrain.can_transfer_world_model(old, vars(args))


def test_sole_speed_rejects_other_robot_before_describe_or_simulation(monkeypatch, capsys):
    parsed = arguments("locomotion/a3/stealth")
    monkeypatch.setattr(train_terrain, "parse_args", lambda: parsed)
    with pytest.raises(SystemExit, match="require G1 geometry"):
        train_terrain.main()
    assert capsys.readouterr().out == ""

"""Force traces must not mix first-episode dynamics with post-reset buffers."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"))
from bootstrap import configure_paths
configure_paths()
from evaluate_terrain import (
    FirstEpisodeRecorder, control_trace_frame, control_trace_metadata, command_runtime_options,
)
from export_command_policy import command_actuator_metadata
from test_command_locomotion import CpuCommandSim
from trackerLab.tracker_env.supermimic.command import (
    CommandSim, CONTROL_TRACE_FIELDS, CONTROL_TRACE_PHASES,
)


def test_named_trace_is_frozen_before_env_and_input_buffers_change():
    n, batch = 3, 4
    force = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    env = SimpleNamespace(
        device="cpu", num_envs=batch,
        default_q=torch.tensor([.1, .2, .3]), action_scale=torch.tensor([.5, 1., 2.]),
        qlo=torch.full((3,), -1.), qhi=torch.ones(3),
        _actuator_ids=torch.tensor([2, 0, 1]), _qvel_ids=torch.tensor([4, 2, 3]),
        wp=SimpleNamespace(to_torch=lambda value: value),
        data=SimpleNamespace(**{name: force.clone() + offset for name, offset in
             (("actuator_force", 0.), ("qfrc_actuator", 100.),
              ("qfrc_constraint", 200.), ("qfrc_bias", 300.))}))
    position = torch.zeros(batch, 3)
    state = (position, torch.eye(3).repeat(batch, 1, 1), position, position,
             torch.zeros(batch, n), torch.full((batch, n), .7))
    action = torch.tensor([[0., 0., 0.], [1., 2., 3.], [.1, .2, .3], [.4, .5, .6]])
    target = torch.tensor([[.1, .2, .3], [.65, 1., 1.], [.2, .4, .6], [.3, .5, .8]])
    record_ids = torch.tensor([3, 1, 2])
    CommandSim.set_control_trace_ids(env, record_ids)
    record_ids.fill_(0)
    trace = CommandSim._capture_control_trace(env, action, target, state)
    transition = {"control_trace": trace, "done": torch.tensor([False, False, True, False])}
    for values in vars(env.data).values():
        values.fill_(999.)
    action.fill_(-99.)
    target.fill_(-99.)
    state[5].fill_(-99.)
    env._control_trace_ids.fill_(0)
    result = control_trace_frame(transition, [3, 1, 2], [False, True, True])
    np.testing.assert_array_equal(result["control_trace_valid"], [False, True, False])
    np.testing.assert_array_equal(result["applied_actions"][1], [1., 2., 3.])
    np.testing.assert_allclose(result["joint_targets"][1], [.65, 1., 1.])
    np.testing.assert_allclose(result["joint_velocities"][1], [.7, .7, .7])
    np.testing.assert_array_equal(result["actuator_force"][1], [7., 5., 6.])
    np.testing.assert_array_equal(result["qfrc_actuator"][1], [109., 107., 108.])
    np.testing.assert_array_equal(result["qfrc_constraint"][1], [209., 207., 208.])
    np.testing.assert_array_equal(result["qfrc_bias"][1], [309., 307., 308.])
    for name in CONTROL_TRACE_FIELDS:
        assert np.isnan(result[name][[0, 2]]).all()
        assert torch.isfinite(trace[name]).all()
    np.testing.assert_array_equal(trace["recorded_env_ids"], [3, 1, 2])
    assert trace["phases"] == CONTROL_TRACE_PHASES


@pytest.mark.parametrize("trace_enabled", [False, True])
@pytest.mark.parametrize("terminated_id", [None, 0, 1])
def test_step_captures_selected_worlds_before_any_batch_reset_forward(trace_enabled, terminated_id):
    sim = CpuCommandSim(batch=4, max_episode_steps=10)
    sim._actuator_ids = torch.tensor([2, 0, 3, 1])
    sim._qvel_ids = torch.tensor([9, 6, 8, 7])
    fields = ("actuator_force", "qfrc_actuator", "qfrc_constraint", "qfrc_bias")
    sim.data = SimpleNamespace(**{name: torch.full((4, 10), -500.) for name in fields})
    sim.wp.to_torch = lambda values: values
    force = torch.arange(40, dtype=torch.float32).reshape(4, 10)
    post_velocity = torch.arange(16, dtype=torch.float32).reshape(4, 4) * .01
    def complete_physics(*args, **kwargs):
        for index, name in enumerate(fields):
            getattr(sim.data, name).copy_(force + index * 100.)
        sim.qvel[:, sim._qvel_ids] = post_velocity
    def reset_forward(*args, **kwargs):
        # Even an unrecorded world's reset may recompute the whole batch.
        for name in fields:
            getattr(sim.data, name).fill_(777.)
    sim.wp.capture_launch = Mock(side_effect=complete_physics)
    sim.mw.forward = Mock(side_effect=reset_forward)
    if terminated_id is not None:
        sim.ep_step[terminated_id] = sim.max_episode_steps - 1
    if trace_enabled:
        sim.set_control_trace_ids([3, 1])
    action = torch.linspace(-1., 1., 16).reshape(4, 4)
    transition = sim.step(action)
    assert sim.wp.capture_launch.call_count == 1
    assert sim.mw.forward.call_count == int(terminated_id is not None)
    assert ("control_trace" in transition) == trace_enabled
    if not trace_enabled:
        return
    trace = transition["control_trace"]
    assert trace["actuator_force"].shape == (2, 4)
    torch.testing.assert_close(trace["recorded_env_ids"], torch.tensor([3, 1]))
    torch.testing.assert_close(trace["applied_actions"], action[[3, 1]])
    torch.testing.assert_close(trace["joint_targets"], action[[3, 1]] * .25)
    torch.testing.assert_close(trace["joint_velocities"], post_velocity[[3, 1]])
    for index, name in enumerate(fields):
        columns = sim._actuator_ids if name == "actuator_force" else sim._qvel_ids
        torch.testing.assert_close(trace[name], (force + index * 100.)[[3, 1]][:, columns])
    if terminated_id is not None:
        assert (sim.data.actuator_force == 777.).all()
    result = control_trace_frame(transition, [3, 1], [True, True])
    np.testing.assert_array_equal(result["control_trace_valid"], [True, terminated_id != 1])
    for name in CONTROL_TRACE_FIELDS:
        assert np.isfinite(result[name][0]).all()
        if terminated_id == 1:
            assert np.isnan(result[name][1]).all()
        else:
            assert np.isfinite(result[name][1]).all()


def test_control_trace_ids_validation_and_disabling():
    sim = CpuCommandSim(batch=4)
    assert "control_trace" not in sim.step(torch.zeros(4, 4))
    sim.set_control_trace_ids([3, 1])
    sim.set_control_trace_ids(None)
    assert "control_trace" not in sim.step(torch.zeros(4, 4))
    for ids in ([], [True], [1.], [1, 1], [-1], [4], [[1]], 1):
        with pytest.raises(ValueError, match="Control trace IDs"):
            sim.set_control_trace_ids(ids)


def test_evaluator_requires_frozen_snapshot_matching_ids_and_phase():
    with pytest.raises(ValueError, match="frozen pre-reset"):
        control_trace_frame({}, [0], [True])
    trace = {"recorded_env_ids": torch.tensor([1, 0]), "phases": dict(CONTROL_TRACE_PHASES)}
    transition = {"control_trace": trace, "done": torch.zeros(2, dtype=torch.bool)}
    with pytest.raises(ValueError, match="recorder order"):
        control_trace_frame(transition, [0, 1], [True, True])
    with pytest.raises(ValueError, match="activity"):
        control_trace_frame(transition, [1, 0], [True])
    trace["phases"]["forces"] = "post_reset_forward"
    with pytest.raises(ValueError, match="phase metadata"):
        control_trace_frame(transition, [1, 0], [True, True])


def test_trace_metadata_preserves_ids_and_separates_force_state_phases():
    metadata = control_trace_metadata([3, 1], ["waist_roll_joint", "left_knee_joint"])
    roundtrip = json.loads(json.dumps(metadata))
    assert roundtrip["recorded_env_ids"] == [3, 1]
    assert roundtrip["joint_names"] == ["waist_roll_joint", "left_knee_joint"]
    assert roundtrip["phases"]["same_time_pd_reconstruction"] is False
    assert roundtrip["phases"]["forces"].startswith("last_physics_substep")
    assert roundtrip["phases"]["state"] == "poststep_after_final_integration"
    assert "NaN" in roundtrip["invalid_rows"]
    roundtrip["phases"]["forces"] = "changed"
    assert CONTROL_TRACE_PHASES["forces"].startswith("last_physics_substep")


def test_npz_preserves_trace_phase_and_initial_invalid_rows(tmp_path):
    metadata = control_trace_metadata([3, 1], ["waist_roll_joint", "left_knee_joint"])
    recorder = FirstEpisodeRecorder(np.zeros((2, 9), dtype=np.float32),
                                    np.zeros((2, 187), dtype=np.float32), 0, np.array([7, 8]))
    trace = {name: np.full((1, 2, 2), np.nan, dtype=np.float32) for name in CONTROL_TRACE_FIELDS}
    trace["control_trace_valid"] = np.zeros((1, 2), dtype=bool)
    output = tmp_path / "trajectory.npz"
    recorder.save(output, recorded_env_ids=np.array([3, 1], dtype=np.int64),
                  control_trace_metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
                  **trace)
    with np.load(output, allow_pickle=False) as saved:
        restored = json.loads(str(saved["control_trace_metadata_json"].item()))
        assert restored == metadata
        np.testing.assert_array_equal(saved["recorded_env_ids"], restored["recorded_env_ids"])
        assert saved["control_trace_valid"].shape == saved["qpos"].shape[:2]
        assert not saved["control_trace_valid"].any()
        for name in CONTROL_TRACE_FIELDS:
            assert np.isnan(saved[name]).all()


def test_evaluation_restores_waist_gains_without_changing_historical_defaults():
    assert "waist_pd_scale" not in command_runtime_options({"arguments": {}})
    assert command_runtime_options({"arguments": {"waist_pd_scale": 3.}})["waist_pd_scale"] == 3.


def test_export_requires_named_saved_motor_contract_when_physics_changed():
    spec = SimpleNamespace(joint_names=("waist_roll_joint",), num_joints=1)
    assert command_actuator_metadata({"arguments": {}}, spec) is None
    checkpoint = {"arguments": {"waist_pd_scale": 3.}}
    with pytest.raises(ValueError, match="Saved simulator"):
        command_actuator_metadata(checkpoint, spec)
    checkpoint["simulator_contract"] = {
        "waist_pd_scale": 3., "joint_names": ["waist_roll_joint"],
        "motors_effort_velocity_kp_kd_armature": [[50., 37., 85.5, 3.14, .0072]]}
    result = command_actuator_metadata(checkpoint, spec)
    assert result["nominal_motors"][0][2] == 85.5
    checkpoint["simulator_contract"]["joint_names"] = ["left_ankle_roll_joint"]
    with pytest.raises(ValueError, match="named joints"):
        command_actuator_metadata(checkpoint, spec)

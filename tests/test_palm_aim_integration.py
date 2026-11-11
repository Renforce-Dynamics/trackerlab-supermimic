"""Optional palm geometry must share the exact aim source and entry schedule."""

import copy

import numpy as np
import pytest
import torch

from test_stealth_aim_style import (
    AimKinematics, AimStyleGuidance, command, guidance_config, reference_q, state_with)
from test_stealth_palm_separation import _config as palm_config
from trackerLab.algorithms.supermimic.stealth_palm_separation import PalmSeparationGuidance


def combined_config():
    config = guidance_config()
    palm = palm_config()
    p = np.asarray(config["reference_wrist_positions_torso"])
    R = np.asarray(config["reference_wrist_rotations_torso"])
    normal = p[1]-p[0]
    normal /= np.linalg.norm(normal)
    hulls = [np.asarray(v) for v in palm["hull_vertices_wrist"]]
    projections = [(p[i]+hulls[i] @ R[i].T) @ normal for i in range(2)]
    gap = float(projections[1].min()-projections[0].max())
    assert gap > 0
    palm.update(source=copy.deepcopy(config["source"]),
        reference_wrist_positions_torso=p.tolist(),
        reference_wrist_rotations_torso=R.tolist(),
        separator_normal_torso=normal.tolist(),
        reference_gap_m=gap, reference_geom_distance_m=gap, margin_m=gap+.01)
    config["palm_separation"] = palm
    return config


def test_geometry_cost_is_added_once_with_the_same_entry_blend():
    kin, config = AimKinematics(), combined_config()
    combined = AimStyleGuidance(kin, config)
    plain_config = copy.deepcopy(config)
    del plain_config["palm_separation"]
    plain = AimStyleGuidance(kin, plain_config)
    state = state_with(reference_q(), root_pitch=.14, root_yaw=.7)
    palm = PalmSeparationGuidance(config["palm_separation"])
    p = torch.tensor(config["reference_wrist_positions_torso"], dtype=torch.float64)
    R = torch.tensor(config["reference_wrist_rotations_torso"], dtype=torch.float64)
    expected = palm.reward(p, R)
    assert expected < 0
    for elapsed, blend in ((0., 0.), (.3, .5), (.6, 1.), (4., 1.)):
        difference = combined.reward(state, command(elapsed))-plain.reward(state, command(elapsed))
        torch.testing.assert_close(difference, expected*blend, atol=1e-12, rtol=0.)
    assert combined.to_dict() == config
    assert "palm_separation" not in plain.to_dict()
    assert "palm_plane_gap_m" in combined.metrics(state)
    assert "palm_plane_gap_m" not in plain.metrics(state)


def test_palm_gradients_reach_learned_arms_without_any_action_override():
    kin, config = AimKinematics(), combined_config()
    combined = AimStyleGuidance(kin, config)
    plain_config = copy.deepcopy(config)
    del plain_config["palm_separation"]
    plain = AimStyleGuidance(kin, plain_config)
    q = reference_q().requires_grad_()
    state = state_with(q, root_pitch=.14, root_yaw=.7)
    extra = combined.reward(state, command(1.))-plain.reward(state, command(1.))
    gradient, = torch.autograd.grad(extra, q)
    assert torch.isfinite(gradient).all()
    assert gradient[15:].norm() > 1e-5
    torch.testing.assert_close(gradient[:12], torch.zeros_like(gradient[:12]), atol=1e-12, rtol=0.)
    torch.testing.assert_close(q.detach(), reference_q(), rtol=0., atol=0.)


@pytest.mark.parametrize("change", ["source", "pose"])
def test_different_palm_source_or_reference_frames_are_rejected(change):
    config = combined_config()
    if change == "source":
        config["palm_separation"]["source"]["asset_sha256"] = "f"*64
    else:
        p = np.asarray(config["palm_separation"]["reference_wrist_positions_torso"])
        config["palm_separation"]["reference_wrist_positions_torso"] = (p+.001).tolist()
    with pytest.raises(ValueError, match="selected aiming source"):
        AimStyleGuidance(AimKinematics(), config)

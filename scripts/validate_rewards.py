"""Check Table A.2 rewards, derivatives and vibed parity without starting Isaac Sim."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from trackerlab_supermimic.kinematics import G1Kinematics
from trackerlab_supermimic.mdp import MAX_STEP_REWARD, MDP, REWARD_TERMS, RewardCfg
from trackerlab_supermimic.motion import MotionLib
from trackerlab_supermimic.robot import default_joint_pos_list
from trackerlab_supermimic.rotations import so3_exp


def table_cases():
    n = 10
    p = torch.zeros(n, 3)
    R = torch.eye(3).expand(n, 3, 3).clone()
    zero3, zero29 = torch.zeros(n, 3), torch.zeros(n, 29)
    motion = SimpleNamespace(
        root_p=p.clone(), root_R=R.clone(),
        body_p=p[:, None].expand(n, 14, 3).clone(),
        body_R=R[:, None].expand(n, 14, 3, 3).clone(),
        body_v=torch.zeros(n, 14, 3), body_w=torch.zeros(n, 14, 3),
    )
    kin = SimpleNamespace(
        q_lo=-torch.ones(29), q_hi=torch.ones(29),
        forward=lambda p, R, v, w, q, qd: (
            p[:, None].expand(n, 14, 3), R[:, None].expand(n, 14, 3, 3),
            v[:, None].expand(n, 14, 3), w[:, None].expand(n, 14, 3),
        ),
    )
    mdp = MDP(kin, motion, torch.zeros(29), obs_mode="relative_ori99")
    v, w, q, a = zero3.clone(), zero3.clone(), zero29.clone(), zero29.clone()
    p[1, 2] = 0.3
    R[2] = so3_exp(torch.tensor([0.4, 0.0, 0.0]))
    v[3, 0], w[4, 2], p[5, 0] = 1.0, 3.14, 0.3
    a[6, 0], q[7, 0], q[8, 0] = 1.0, 1.1, -1.1
    R[9] = so3_exp(torch.tensor([0.0, 0.0, 0.4]))
    state = (p, R, v, w, q, zero29.clone())
    reward, terms = mdp.reward(state, torch.arange(n), a, zero29, terms=True)
    base = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.0, 0.0])
    expected = base.repeat(n, 1)
    expected[1, [0, 4]] *= math.exp(-1)
    expected[2, [1, 5]] *= math.exp(-1)
    expected[3, 2] *= math.exp(-1)
    expected[4, 3] *= math.exp(-1)
    expected[5, 4] *= math.exp(-1)
    expected[6, 6] = -0.1 * 0.02
    expected[7:9, 7] = -2 * 0.2 * 0.02
    expected[9, 5] *= math.exp(-1)
    actual = torch.stack([terms[key] for key in REWARD_TERMS], -1)
    # The shared acos clamp produces a sub-microradian reward bias at exact identity.
    torch.testing.assert_close(actual, expected, atol=4e-7, rtol=1e-5)
    torch.testing.assert_close(actual.sum(-1), reward, atol=2e-8, rtol=1e-6)
    assert abs(reward[0].item() - MAX_STEP_REWARD) < 1e-6
    assert bool((reward <= MAX_STEP_REWARD).all())
    p[0, 2], p[1, 2] = 0.249, 0.251
    R[2] = so3_exp(torch.tensor([math.acos(0.21), 0.0, 0.0]))
    R[3] = so3_exp(torch.tensor([math.acos(0.19), 0.0, 0.0]))
    done = mdp.termination(state, torch.arange(n))
    assert done[:4].tolist() == [False, True, False, True]
    soft = mdp.soft_termination(state, torch.arange(n))
    assert bool(torch.isfinite(soft).all()) and bool(((soft >= 0) & (soft <= 1)).all())
    return {
        "cases": n, "maximum_component_error": float((actual - expected).abs().max()),
        "component_sum_error": float((actual.sum(-1) - reward).abs().max()),
        "termination_boundaries": "passed", "perfect_reward": float(reward[0]),
    }


def real_data_parity(upstream_path: Path, motion_dir: Path):
    name = "trackerlab_supermimic.validation_upstream"
    spec = importlib.util.spec_from_file_location(name, upstream_path)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[name] = upstream
    spec.loader.exec_module(upstream)
    kin = G1Kinematics("cpu")
    motion = MotionLib([str(motion_dir / "walk1_subject1.csv")], kin, "cpu")
    default = torch.tensor(default_joint_pos_list())
    ours = MDP(kin, motion, default, obs_mode="relative_ori99")
    public = upstream.MDP(kin, motion, default)
    frames = torch.linspace(20, motion.num_frames - 40, 32).long()
    state = list(motion.ref_state(frames))
    state[0] += 0.1 * torch.randn_like(state[0])
    state[1] = so3_exp(0.3 * torch.randn(32, 3)) @ state[1]
    state[2] += 0.3 * torch.randn_like(state[2])
    state[3] += 0.4 * torch.randn_like(state[3])
    state[4] += 0.08 * torch.randn_like(state[4])
    state = tuple(t.detach().requires_grad_(True) for t in state)
    a = (0.2 * torch.randn(32, 29)).requires_grad_(True)
    prev = 0.2 * torch.randn(32, 29)
    xi = torch.zeros(32, 33)
    xi[:, 0] = 1.0
    actual, terms = ours.reward(state, frames, a, prev, terms=True)
    expected = public.reward(state, frames, a, prev)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    ga = torch.autograd.grad(actual.sum(), (*state, a), retain_graph=True)
    ge = torch.autograd.grad(expected.sum(), (*state, a), retain_graph=True)
    grad_error = 0.0
    for left, right in zip(ga, ge):
        assert bool(torch.isfinite(left).all())
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)
        grad_error = max(grad_error, float((left - right).abs().max()))
    torch.testing.assert_close(sum(terms[key] for key in REWARD_TERMS), actual, atol=1e-7, rtol=1e-6)
    parity = {}
    for method, values in (
        ("obs", (state, prev, xi, frames)),
        ("termination", (state, frames)),
        ("soft_termination", (state, frames)),
        ("critic_features", (state, frames, xi)),
    ):
        left, right = getattr(ours, method)(*values), getattr(public, method)(*values)
        torch.testing.assert_close(left, right)
        parity[method] = float((left.float() - right.float()).abs().max().detach())
    yaw = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    yawed = ((yaw @ state[0].unsqueeze(-1)).squeeze(-1), yaw @ state[1],
             (yaw @ state[2].unsqueeze(-1)).squeeze(-1),
             (yaw @ state[3].unsqueeze(-1)).squeeze(-1), *state[4:])
    delta = (ours.obs(state, prev, xi, frames) - ours.obs(yawed, prev, xi, frames)).abs().max()
    assert delta > 0.5
    return {
        "reward_max_error": float((actual - expected).abs().max().detach()),
        "gradient_max_error": grad_error, "other_heads_max_error": parity,
        "yaw90_observation_change": float(delta.detach()), "finite_gradients": True,
        "upstream_sha256": hashlib.sha256(upstream_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream_mdp", type=Path, help="Optional pinned vibed mdp.py for value/gradient parity")
    parser.add_argument("--motion_dir", type=Path, default=REPO / "data/lafan1_g1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(170)
    report = {
        "table": "SuperMimic Appendix A, Table A.2 (8 motion-tracking terms); dt=0.02",
        "reward_config": asdict(RewardCfg()),
        "analytic_cases": table_cases(),
        "vibed_parity": (
            real_data_parity(args.upstream_mdp, args.motion_dir)
            if args.upstream_mdp else {"status": "not_run", "reason": "--upstream_mdp not provided"}
        ),
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

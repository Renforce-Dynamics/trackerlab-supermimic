"""Compare this layout against the previous standalone release using real motions."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from bootstrap import configure_paths
configure_paths()

import torch

from trackerLab.algorithms.supermimic.networks import Policy, WorldModel
from trackerLab.kinematics.supermimic.g1 import G1Kinematics
from trackerLab.managers.supermimic import SuperMimicMotionManager
from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
from trackerLab.tracker_env.mdp.supermimic import MDP, MAX_STEP_REWARD
from trackerTask.supermimic.robot import default_joint_pos_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="v0.1 standalone checkout")
    parser.add_argument("--motion_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.baseline.resolve() / "src"))
    old_motion = importlib.import_module("trackerlab_supermimic.motion")
    old_mdp = importlib.import_module("trackerlab_supermimic.mdp")
    old_networks = importlib.import_module("trackerlab_supermimic.networks")
    kin = G1Kinematics("cpu")
    current = MotionLib.from_dir(str(args.motion_dir), kin, "cpu")
    previous = old_motion.MotionLib.from_dir(str(args.motion_dir), kin, "cpu")
    fields = ("root_p", "root_R", "root_v", "root_w", "q", "qd", "body_p", "body_R",
              "body_v", "body_w", "clip_start", "clip_end", "clip_len", "clip_id",
              "bin_start", "bin_clip", "frame_bin", "root_v_local", "root_w_local", "root_gz")
    for field in fields:
        torch.testing.assert_close(getattr(current, field), getattr(previous, field), rtol=0, atol=0)
    frames = torch.linspace(0, current.num_frames - 1, 128).long()
    torch.testing.assert_close(current.goal(frames, 10), previous.goal(frames, 10), rtol=0, atol=0)
    defaults = torch.tensor(default_joint_pos_list())
    mdp = MDP(kin, current, defaults, obs_mode="relative_ori99")
    prior_mdp = old_mdp.MDP(kin, previous, defaults, obs_mode="relative_ori99")
    torch.manual_seed(52)
    state = tuple(t.detach().clone().requires_grad_(True) for t in current.ref_state(frames))
    action = torch.randn(128, 29).requires_grad_(True)
    prev_action = torch.randn(128, 29)
    xi = torch.randn(128, 33) * .01
    reward, terms = mdp.reward(state, frames, action, prev_action, terms=True)
    old_reward, old_terms = prior_mdp.reward(state, frames, action, prev_action, terms=True)
    for field in terms:
        torch.testing.assert_close(terms[field], old_terms[field], rtol=0, atol=0)
    actual_grads = torch.autograd.grad(reward.sum(), (*state, action), retain_graph=True)
    old_grads = torch.autograd.grad(old_reward.sum(), (*state, action))
    for actual, expected in zip(actual_grads, old_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    observation = mdp.obs(state, prev_action, xi, frames)
    torch.testing.assert_close(observation, prior_mdp.obs(state, prev_action, xi, frames), rtol=0, atol=0)
    torch.testing.assert_close(mdp.termination(state, frames), prior_mdp.termination(state, frames))

    # Execute the actual pre-refactor sampling methods without launching a simulator.
    source = (args.baseline / "src/trackerlab_supermimic/env.py").read_text()
    env_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "TrackingSim")
    methods = [node for node in env_class.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_sample_frames", "_update_bin_returns")]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "<baseline sampling>", "exec"), scope)
    manager = SuperMimicMotionManager(current, 128, "cpu", .99, MAX_STEP_REWARD)
    manager.bin_returns.copy_(torch.linspace(0, manager.max_bin_return, current.num_bins))
    old = SimpleNamespace(motion=previous, device="cpu", gamma=.99,
                          bin_returns=manager.bin_returns.clone(), max_bin_return=manager.max_bin_return,
                          bin_past_weights=manager.bin_past_weights.clone())
    ids = torch.arange(128)
    torch.manual_seed(738)
    expected_ids, expected_frames = scope["_sample_frames"](old, ids)
    torch.manual_seed(738)
    actual_ids, actual_frames = manager.sample_frames(ids)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_frames, expected_frames, rtol=0, atol=0)
    manager.reset(ids, actual_frames)
    old.reset_bin = manager.reset_bin.clone()
    old.ep_step = torch.randint(1, 501, (128,))
    old.ep_return = torch.rand(128)
    old.ep_discounted_return = torch.rand(128)
    terminated = torch.rand(128) > .5
    scope["_update_bin_returns"](old, ids, terminated)
    manager.update_returns(ids, terminated, old.ep_step, old.ep_return, old.ep_discounted_return)
    torch.testing.assert_close(manager.bin_returns, old.bin_returns, rtol=0, atol=0)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = Policy(99, 730, hist=10)
    prior_policy = old_networks.Policy(99, 730, hist=10)
    policy.load_state_dict(checkpoint["pi"])
    prior_policy.load_state_dict(checkpoint["pi"])
    wm, old_wm = WorldModel(), old_networks.WorldModel()
    wm.load_state_dict(checkpoint["wm"])
    old_wm.load_state_dict(checkpoint["wm"])
    history = observation.detach().unsqueeze(1).repeat(1, 10, 1)
    goal = current.goal(frames, 10)
    with torch.no_grad():
        torch.testing.assert_close(policy(history, goal), prior_policy(history, goal), rtol=0, atol=0)
        predicted = wm.step(state, action.detach(), xi)
        old_predicted = old_wm.step(state, action.detach(), xi)
        for actual, expected in zip(predicted, old_predicted):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(wm.targets(state, predicted), old_wm.targets(state, old_predicted),
                                   rtol=0, atol=0)
        torch.testing.assert_close(wm.state_error(predicted, state), old_wm.state_error(old_predicted, state),
                                   rtol=0, atol=0)
    result = {
        "status": "passed", "motions": len(current.names), "reference_frames": current.num_frames,
        "reference_fields_equal": list(fields), "checkpoint_iteration": checkpoint["iteration"],
        "goal_observation_reward_gradient_policy_world_model_max_error": 0.0,
        "reset_sampling_and_return_updates": "bitwise_equal",
        "world_model_targets_and_state_error": "bitwise_equal",
        "scope": "CPU numerical parity; physical rollout is validated separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

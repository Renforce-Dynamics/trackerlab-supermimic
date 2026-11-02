"""Export a normalized actor as TorchScript and check CPU inference equivalence."""

import argparse
from copy import deepcopy
from pathlib import Path

from common import digest, load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Export output already exists")
    import torch
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization

    checkpoint = args.checkpoint.resolve()
    config = load(checkpoint.parent / "run_config.json")
    contract = load(checkpoint.parent / "simulator_contract.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    observations = contract["training"]["observation_dim"]
    actions = len(contract["joint_names"])
    policy_cfg = deepcopy(config["ppo"]["policy"])
    policy_cfg.pop("class_name")
    model = ActorCritic(observations, observations, actions, **policy_cfg)
    model.load_state_dict(payload["model_state_dict"])
    normalizer = EmpiricalNormalization(shape=[observations], until=1e8)
    normalizer.load_state_dict(payload["obs_norm_state_dict"])
    policy = torch.nn.Sequential(normalizer, model.actor).eval()
    sample = torch.randn(16, observations)
    with torch.inference_mode():
        scripted = torch.jit.trace(policy, sample)
        torch.testing.assert_close(scripted(sample), policy(sample), rtol=1e-5, atol=1e-6)
    args.output.mkdir(parents=True)
    target = args.output / "policy.ts"
    scripted.save(str(target))
    save(args.output / "contract.json", contract)
    save(args.output / "manifest.json", {
        "checkpoint_sha256": digest(checkpoint), "policy_sha256": digest(target),
        "task": config["arguments"]["task"], "completed_updates": payload["iter"] + 1,
        "input": "Raw observation in the recorded simulator contract; normalization included.",
        "output": "Raw actor action; runtime clamps to [-3,3] then uses recorded PD action scale.",
        "equivalence": "CPU eager versus TorchScript, 16 random observations, rtol=1e-5 atol=1e-6.",
        "hardware_validation": False,
    })


if __name__ == "__main__":
    main()

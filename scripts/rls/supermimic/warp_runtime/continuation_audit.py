"""Small, startup/boundary-only receipts for a simulator continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value):
    """Hash values independently of device while retaining shape and dtype."""
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(
                ["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for entry in item:
                visit(entry)
        else:
            digest.update(json.dumps(
                [type(item).__name__, item], sort_keys=True, allow_nan=False).encode())

    visit(value)
    return digest.hexdigest()


def model_receipt(model, source_state=None):
    names = [name for name, _ in model.named_parameters()]
    current = model.state_dict()
    result = {
        "trainable_sha256": value_sha256({name: current[name] for name in names}),
        "state_sha256": value_sha256(current),
        "trainable_elements": sum(parameter.numel() for parameter in model.parameters()),
        "state_entries": len(current),
    }
    if source_state is not None:
        result["source_trainable_sha256"] = value_sha256(
            {name: source_state[name] for name in names})
        result["source_state_sha256"] = value_sha256(source_state)
        result["trainable_equal_to_source"] = (
            result["trainable_sha256"] == result["source_trainable_sha256"])
        result["state_equal_to_source"] = result["state_sha256"] == result["source_state_sha256"]
        if not result["trainable_equal_to_source"] or not result["state_equal_to_source"]:
            raise ValueError("A restored model differs from the pinned source checkpoint")
    return result


def optimizer_receipt(optimizer):
    import torch

    steps = []
    devices = {}
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        step = state["step"]
        device = str(step.device) if isinstance(step, torch.Tensor) else "python"
        devices[device] = devices.get(device, 0) + 1
        steps.append(float(step.detach().cpu()) if isinstance(step, torch.Tensor) else float(step))
    unique = sorted(set(steps))
    return {
        "state_sha256": value_sha256(optimizer.state_dict()),
        "parameter_state_sha256": value_sha256(optimizer.state_dict()["state"]),
        "step_devices": devices,
        "step_values": unique,
        "parameter_states": len(optimizer.state),
        "uniform_step_count": unique[0] if len(unique) == 1 else None,
    }


def write_receipt(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def adaptation_boundary(source_models, models, source_optimizers, optimizers, counters):
    current_models = {name: model_receipt(model) for name, model in models.items()}
    current_optimizers = {
        name: optimizer_receipt(optimizer) for name, optimizer in optimizers.items()
    }
    checks = {
        "actor_trainable_unchanged": current_models["pi"]["trainable_sha256"]
        == source_models["pi"]["trainable_sha256"],
        "critic_trainable_unchanged": current_models["critic"]["trainable_sha256"]
        == source_models["critic"]["trainable_sha256"],
        "actor_optimizer_moments_and_steps_unchanged": current_optimizers["pi"]["parameter_state_sha256"]
        == source_optimizers["pi"]["parameter_state_sha256"],
        "critic_optimizer_moments_and_steps_unchanged": current_optimizers["critic"]["parameter_state_sha256"]
        == source_optimizers["critic"]["parameter_state_sha256"],
        "zero_actor_update_attempts": counters["policy_update_attempts_this_run"] == 0,
        "zero_actor_updates": counters["policy_optimizer_steps_this_run"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError("World-model-only adaptation changed actor/critic training state")
    return {
        "checks": checks,
        "models": current_models,
        "optimizers": current_optimizers,
        "counters": dict(counters),
        "normalizers": "State buffers may update during data collection; trainable parameters are checked separately.",
        "optimizer_learning_rates": "Cumulative scheduled learning rates advance during adaptation; moments and step counters remain fixed for actor and critic.",
    }

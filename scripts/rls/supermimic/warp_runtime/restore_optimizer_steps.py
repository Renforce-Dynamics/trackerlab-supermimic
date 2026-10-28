"""Candidate resume repair; no production application performed.

PyTorch eager, non-capturable, non-fused Adam expects scalar step counters on
CPU. torch.load(map_location="cuda") can move serialized CPU counters to CUDA;
optimizer.load_state_dict preserves those locations for these parameter groups.
"""


def restore_eager_adam_step_counters(optimizer):
    import torch
    changed = []
    for group_index, group in enumerate(optimizer.param_groups):
        if group.get("capturable", False) or group.get("fused", False):
            continue
        for parameter_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            step = state.get("step")
            if isinstance(step, torch.Tensor) and step.device.type != "cpu":
                if step.numel() != 1 or step.requires_grad:
                    raise ValueError("Expected a scalar, nongradient Adam step counter")
                # Parameter values and moment tensors remain on their original devices.
                state["step"] = step.detach().cpu()
                changed.append((group_index, parameter_index))
    return changed

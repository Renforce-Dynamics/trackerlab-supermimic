"""Process-local defaults for the accepted BF16 world-model graph path."""

import os


FIXED_ENVIRONMENT = {
    "NVIDIA_TF32_OVERRIDE": "0",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
    "SM_GRAPH_SCOPE": "wm",
    "SM_FROZEN_NORM_GUARD": "0",
    "SM_COMPILE_CRITIC_LOSS": "0",
    "SM_GRAPH_VALIDATE": "1",
}
THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def configure_environment(environment=None):
    """Apply numerical settings before PyTorch imports; reject conflicts.

    Thread defaults can be overridden for the local host. The fixed numerical
    settings identify this tested training path and require a separate entrypoint
    for experiments with another graph or precision policy.
    """
    environment = os.environ if environment is None else environment
    for name, expected in FIXED_ENVIRONMENT.items():
        actual = environment.get(name, expected)
        if actual != expected:
            raise ValueError(f"{name} must be {expected!r} for this Warp runtime; got {actual!r}")
    for name, value in {**FIXED_ENVIRONMENT, **THREAD_DEFAULTS}.items():
        environment.setdefault(name, value)
    return {name: environment[name] for name in (*FIXED_ENVIRONMENT, *THREAD_DEFAULTS)}

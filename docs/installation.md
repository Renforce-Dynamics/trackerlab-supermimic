# Installation

## Tested Runtime

| Component | Version |
| --- | --- |
| Python | 3.11 |
| IsaacLab | v2.3.2, `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Isaac Sim | 5.1.0.0 |
| PyTorch | 2.7.0+cu128 |
| MuJoCo | 3.3.7, kinematics and offline rendering |
| GPU | RTX PRO 6000 Blackwell Server Edition |
| NVIDIA driver in the recorded run | 580.82.09 |

Use NVIDIA's versioned instructions for
[IsaacLab 2.3](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)
and [Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html).
The source commit in `resources.lock.json` identifies the exact code used.
IsaacLab's Python package reports `isaaclab==0.54.2` at that commit;
the repository version is 2.3.2.

## Python Environment

The following shows the pip-based arrangement used for headless training.
System-level NVIDIA driver, Vulkan/EGL libraries and simulator requirements
must already be available.

```bash
conda create -n trackerlab-supermimic python=3.11 -y
conda activate trackerlab-supermimic
python -m pip install setuptools==80.9.0 wheel==0.45.1
python -m pip install flatdict==4.0.1 --no-build-isolation
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
git -C ../IsaacLab checkout 37ddf626871758333d6ed89cf64ad702aef127d0
python -m pip install -e ../IsaacLab/source/isaaclab
python -m pip install ipython==8.30.0 psutil==5.9.8
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

These commands are run from the cloned `trackerlab-supermimic` directory.
The task imports the core `isaaclab` package; it needs no separate `robotlib`
installation and no complete TrackerLab checkout.

Isaac Sim prompts for its EULA on first use. After reading and agreeing to its
terms, its documented noninteractive option is:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

For root-owned containers, add `--kit_args=--allow-root` to `train.py`,
`evaluate.py` and `smoke_test.py`. The supervisor detects a root process
and supplies that Kit argument.

## Graphics And Compilation

Headless PhysX still requires Isaac Sim's system graphics stack.
On the validated Ubuntu host, these packages were installed:
`libxt6`, `libglu1-mesa`, `libegl1`, `libvulkan1`, `libvulkan-dev`,
and `vulkan-tools`. `vulkaninfo --summary` should enumerate the NVIDIA GPU.
The video renderer uses EGL and the NVIDIA driver.

`torch.compile` needs a working compiler toolchain and writable caches.
First-iteration compilation can take several minutes. Cache paths and CPU
thread counts can be selected through ordinary environment variables:

```bash
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/trackerlab-supermimic/inductor"
export TRITON_CACHE_DIR="$HOME/.cache/trackerlab-supermimic/triton"
```

Keep caches on a filesystem with free space. The smoke preset disables
compilation to check functionality quickly. The long-run presets enable it.

## Known Dependency Issue

The tested upstream stack contains a metadata conflict: Isaac Sim selects
`fastapi==0.115.7`, whose Starlette requirement is below 0.46; the pinned
IsaacLab source requests `starlette==0.49.1`. `pip check` reports this conflict.
The headless simulation, training, evaluation and export paths were exercised.
Web/livestream functionality has not been validated. This repository does not
claim a clean global `pip check` for that upstream combination.

See `docs/validation.md` for the exact scope of release testing.

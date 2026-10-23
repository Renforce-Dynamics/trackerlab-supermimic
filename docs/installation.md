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
The source commit in `scripts/rls/supermimic/resources.lock.json` identifies the exact code used.
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

## Existing TrackerLab

Use the host's Python environment and editable `trackerLab`/`trackerTask`
installations. From this SuperMimic checkout:

```bash
python scripts/install_trackerlab.py --trackerlab /path/to/trackerLab
python scripts/install_trackerlab.py --trackerlab /path/to/trackerLab --apply
cd /path/to/trackerLab
python -m pip install -r /path/to/trackerlab-supermimic/requirements.txt
python scripts/rls/supermimic/prepare_resources.py
python scripts/rls/supermimic/train.py \
  --config scripts/rls/supermimic/configs/smoke.json --headless \
  --motion_config data/configs/supermimic_lafan1.json --log_dir logs/supermimic_smoke
```

The installed source modules are byte-identical to the standalone modules.
The CLI selects that checkout's two source roots, so a separately installed
package cannot silently replace them. Use the root `pip install -e .` only for
the standalone environment; installing another distribution owning the same
top-level packages in the host environment creates overlapping installations.
The host still needs its existing task dependencies for its other tasks.

The installer copies only the SuperMimic module paths and
`scripts/rls/supermimic/`. It keeps root package `__init__.py` and `setup.py`
intact. Existing parent initializers for algorithms, kinematics, managers,
metrics, motion_buffer and tracker_env are retained as
`_supermimic_original_init.py`; a lazy initializer loads their original exports
on demand. This avoids loading unrelated tasks while launching SuperMimic.
Original import side effects are delayed until those exports are used.

Before writing, the installer checks every destination and saves overwritten
files in `.supermimic-backups/<timestamp>/`, mirroring the host paths. The
`.supermimic-install.json` record contains installed hashes. Local modifications
to extension files are reported as conflicts; `--replace_extensions` permits
replacement after review and retains backups. Modified parent wrappers or
preserved initializers require manual review. Repeated identical installations
make no changes. A failed file write restores the previous file contents.

To undo an overlay, review the installation record and backup directory:
restore overwritten files from the corresponding backup and remove only
new extension files created by that installation. Keep backups across upgrades.
The installer never changes datasets, logs, active processes or Git remotes.

Validation uses TrackerLab base `1e5ccc062b445712a0aa7308cfb99edd7296cc88`.
Custom forks can have other dependency or import requirements; the installer
does not claim compatibility with every generic TrackerLab task.

## Simulator Startup

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

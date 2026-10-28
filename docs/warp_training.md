# Optional G1 training with MuJoCo-Warp

`scripts/rls/supermimic/train_warp.py` provides a separate accelerated training
path for **G1, random initialization, BF16 MLPs, relative_ori99 observations
and ten goal frames**. The existing PhysX and multi-robot workflows keep their
entrypoints and environment requirements.

The runtime is imported from the teleoperation repository's 2026-10-28
scratch-training revision. Its source history, hashes and integration changes
are recorded in
[`SOURCE_PROVENANCE.json`](../scripts/rls/supermimic/warp_runtime/SOURCE_PROVENANCE.json).
The import includes source code and validation routines; datasets, weights,
remote controllers and private deployment records are excluded.

## Acceleration components

- MuJoCo-Warp runs the batched simulator on CUDA.
- A compact root83 cache is loaded onto the GPU once. Body references are
  computed in chunks and retained on the GPU, removing repeated CPU reference
  transfers during collection and reward computation.
- Fixed-shape collection functions use compilation. The world-model prediction
  window uses `torch.compile` with CUDA Graphs.
- Native BF16 autocast applies to MLP matrix products. Geometry, SO(3),
  integration, rewards, parameters, running statistics and Adam state use FP32.
- The shared `RunningNorm.update` keeps FP32 buffer addresses during active
  updates. This avoids graph recordings caused by replaced variance/count
  buffers while retaining the original arithmetic and update schedule.

The runtime uses the repository's repaired normalizer and motion-goal methods.
It verifies these implementations without replacing them with a second copy.
The genuine final frame retains its reference velocity; synthetic future
frames hold its pose with zero velocity.
[Endpoint behavior and compatibility](reference_tail_fix.md).

## Separate environment

Use Linux, Python 3.11 and a CUDA GPU supporting BF16. The source runtime used
PyTorch `2.7.0+cu128`, MuJoCo `3.16.0`, MuJoCo-Warp `3.16.0`, and Warp `1.18.0`.
These versions are recorded observations, rather than a claim about the latest
available packages.

```bash
python -m venv .venv-warp
source .venv-warp/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-warp.txt
```

Run from this checkout. The entrypoint imports this checkout's `source/`
packages, so an editable installation is unnecessary. The base project metadata
still describes the PhysX environment with MuJoCo `3.3.7`; keep that environment
separate from this Warp environment.

Prepare the G1 MJCF and URDF with the repository resource preparation tools.
The default asset directory is `data/assetslib/unitree/unitree_g1`.
`SUPERMIMIC_ASSETLIB_ROOT` or `SUPERMIMIC_ASSET_DIR` can select an existing
asset library. The 29-joint model and joint order must match the motion cache.

## Motion-cache contract

Supply a motion JSON and a prebuilt **root83** cache. This integration includes
the integrity-checking loader; the original cache conversion pipeline is
external. A fresh clone needs that prepared cache before training. Other mmap
cache formats cannot be substituted.

The motion JSON specifies:

- `files`: ordered source CSV paths with unique filename stems.
- `joint_names`: exactly the 29 G1 names in `RobotSpec.joint_names` order.
- `source_fps`: `50`; `quaternion_order`: `"xyzw"`.

The cache contains `manifest.json` and a little-endian FP32 binary array.
The manifest's schema is `supermimic-root-cache-v1`. It records ordered `names`,
`clip_lengths`, `num_frames`, and `source_config_sha256`. Each clip has at least
two frames. `groups.root` records the binary `file`, byte count, SHA256, width
`83`, and these ordered fields:

| Field | Column offset | Shape |
| --- | ---: | --- |
| `root_p` | 0 | `[3]` |
| `root_R` | 3 | `[3, 3]` |
| `root_v` | 12 | `[3]` |
| `root_w` | 15 | `[3]` |
| `q` | 18 | `[29]` |
| `qd` | 47 | `[29]` |
| `root_v_local` | 76 | `[3]` |
| `root_w_local` | 79 | `[3]` |
| `root_gz` | 82 | `[]` |

The binary size is `num_frames * 83 * 4`. The loader validates clip order,
configuration and binary hashes, dimensions and paths before copying data to
the GPU. The binary path must remain inside the cache directory. The prepared
values must follow the original 50 Hz rotation and finite-difference velocity
conventions. Existing caches need frame-count revalidation after the shared
50 Hz loader repair.

The historical source experiment selected 129,491 filtered clips and
47,344,725 frames. The preset uses the supplied selection; it does not apply
dataset exclusions or choose a train/evaluation split. Generalization requires
an independently defined evaluation collection.

## Configuration and launch

Inspect arguments without importing PyTorch, physics or motion data:

```bash
python scripts/rls/supermimic/train_warp.py \
  --config scripts/rls/supermimic/configs/g1_warp_bf16_8096.json \
  --motion_config data/configs/selected_motions.json \
  --motion_root_cache data/root83-cache \
  --log_dir logs/g1_warp_bf16_run01 --dry-run
```

Remove `--dry-run` to train. Each invocation needs a fresh log directory.
`--max_minutes` limits run time; creation of the path supplied to `--stop_file`
requests a saved checkpoint and a normal exit. The portable entrypoint currently
rejects checkpoint resumption and `--motion_storage mmap`.

| Setting | Preset |
| --- | ---: |
| Environments / target outer iterations | 8,096 / 320,000 |
| Replay length / updates per vectorized collection step | 1,000 / 0.4 |
| Observation width / history / goal frames | 99 / 10 / 10 |
| WM updates / batch / prediction window | 10 / 2,048 / 8 |
| Policy batch / imagined horizon | 4,096 / 32 |
| Critic epochs / minibatches | 5 / 4 |
| Simulator warmup / WM-only iterations | 64 / 500 |
| Learning rate / discount / lambda | 0.0002 / 0.99 / 0.95 |
| Control step / physics substeps | 20 ms / four 5 ms steps |
| Solver iterations / line-search iterations | 50 / 50 |
| Checkpoint / log interval | 2,000 / 50 |

A complete outer iteration performs 10 WM, one policy and 20 critic optimizer
updates. Initial WM-only iterations have a different workload. The preset keeps
action-noise scale `0.1`, friction `[0.3, 1.2]`, joint offset `0.01`, and torso
COM offsets `[0.025, 0.05, 0.05]`. The shared MDP and imported simulator retain
their reward, noise, randomization and sampling implementations.

The entrypoint checks these numerical settings before framework imports:

```text
NVIDIA_TF32_OVERRIDE=0
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
SM_GRAPH_SCOPE=wm
SM_FROZEN_NORM_GUARD=0
SM_COMPILE_CRITIC_LOSS=0
SM_GRAPH_VALIDATE=1
```

Conflicting values fail early. Thread defaults are OMP=8, MKL=2, OpenBLAS=1,
and NumExpr=1; explicit host overrides are supported. Critic loss stays eager.
Source and runtime numerical checks run before training results are accepted:
normalizer values/gradients/storage, actual WM-batch comparisons, resident
reference bodies, and finite/SO(3) actor-rollout checks.

Allocate the GPU and host-memory budget before launching. The historical
8,096-environment run had approximately 68 GiB of PyTorch device allocations,
with additional simulator, compilation and cache requirements. The script
does not manage other jobs, reserve a shared GPU, or automatically restart.

## Timing and metric interpretation

The source teleoperation revision measured **2.47115 complete outer iterations
per second** over 1,000 updates, iteration labels 541-1540, on one RTX PRO 6000.
The 404.67-second interval contained 10,000 WM, 1,000 policy and 20,000 critic
updates. Initial compilation, WM-only initialization and checkpoint writing
outside the measurement interval are excluded.

That measurement belongs to the source experiment. This repository integration
has CPU regression coverage; its fresh CUDA checks and a GPU timing run must
be performed on the deployment machine. It makes no new throughput-factor,
convergence, whole-training RNG equality, heldout or hardware-quality claim.

`train_log.csv` records training metrics. `wm_err8` is dimensionless normalized
squared state error. `term_frac` summarizes completed training episodes.
`pi_loss` is negative imagined return. These fields cannot be used as
whole-dataset failure counts or direct replacements for joint error in radians.
Paper comparisons require matched datasets, simulator conditions and metrics.

## CPU regression checks

```bash
python -m unittest discover -s tests -p 'test_warp_runtime_*.py' -v
python -m unittest discover -s tests -v
```

The tests cover command-line configuration, dependency/provenance closure,
normalizer formula/storage compatibility, goal boundaries and motion-cache
validation. They do not execute CUDA simulation or training. Runtime validation
results from a new GPU machine remain necessary for that machine's run.

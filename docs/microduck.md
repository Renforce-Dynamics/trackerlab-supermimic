# MicroDuck Native Locomotion

This optional task trains MicroDuck from random initialization using the
official **mjlab / MuJoCo Warp / RSL-RL PPO** recipe and BAM XL330 actuator.
It is separate from the IsaacLab SuperMimic imitation learner. It learns from
simulated velocity and pose commands, without a motion-capture dataset.
Its rewards and training iteration definition should not be compared directly
with the normalized imitation returns in the demo gallery.

## Pinned Sources

- Robot and task: `pollen-robotics/microduck_rl`,
  revision `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`.
- BAM: `Rhoban/bam`,
  revision `62bd8ce12154340be97e06f7f41a0ca8f116d967`, from the upstream lockfile.
- Code is Apache-2.0; robot models are CC-BY-NC-SA-4.0. Vendor files are
  downloaded separately and are not redistributed under this repository's MIT license.

## Task Layout

The task description is
`source/trackerTask/trackerTask/native/locomotion/robots/microduck/task.json`.
Its run preset is in `agents/ppo.json` beside it. The native launcher is
`scripts/rls/microduck/train.py`; the SuperMimic `tasks.py` registry remains
dedicated to the existing IsaacLab tasks.

The preset selects 4,096 environments, 10,000 PPO updates, seed 42, and
TensorBoard logging. The official actor and critic architecture, observation
normalization, 24 rollout steps, five PPO epochs, four minibatches, BAM model,
domain randomization and reward curricula remain unchanged. The launcher
records effective settings and source hashes in each run directory.

## Installation And Training

Use a dedicated Linux Python 3.12 environment. Keep the IsaacLab environment
separate:

```bash
git clone https://github.com/pollen-robotics/microduck_rl /path/to/microduck_rl
cd /path/to/microduck_rl
git checkout b02fdea4902c2b93b7d34a83a6f05da4ececb7d3
uv sync --frozen

cd /path/to/trackerlab-supermimic
/path/to/microduck_rl/.venv/bin/python scripts/rls/microduck/train.py \
  --upstream /path/to/microduck_rl --run-dir logs/microduck_smoke --smoke

/path/to/microduck_rl/.venv/bin/python scripts/rls/microduck/train.py \
  --upstream /path/to/microduck_rl --run-dir logs/microduck_4096_10k
```

Alternatively, `scripts/rls/microduck/supervise.py` accepts the same
`--upstream` and `--run-dir`, owns the training process, writes
`job_status.json`, and performs a 16-environment, 20-second-per-environment
command-tracking evaluation at completion. The final milestone contains
ONNX, a checkpoint, metrics, an environment-zero video and artifact hashes.
The evaluation retains official play randomization and push events, uses
seed 123, and includes automatic resets in its errors and termination counts.
Use `tmux` or an existing job scheduler for this long-running command.
Visual review remains a separate delivery check.

The smoke run uses 64 environments and five updates; its weights are separate
from fresh training. Existing run directories are protected from overwrite.
`--describe` displays the task and preset without importing a simulator.

Checkpoints use RSL-RL's native zero-based indices. Use the recorded iteration
and update count when reporting results. TensorBoard events contain task
rewards, episode length and PPO diagnostics. Video review and command-tracking
evaluation are required before making claims about the resulting gait.

## Export

Use the upstream exporter so observation normalization is included:

```bash
cd /path/to/microduck_rl
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /absolute/path/to/model_CHECKPOINT.pt \
  --num-envs 1 --onnx-file /absolute/path/to/policy.onnx
```

The exported policy is a native MicroDuck policy. It does not load into the
SuperMimic motion-goal inference API.

# SuperMimic Terrain Tasks

The registry adds 18 downstream objectives: stairs, mountain and parkour for
G1, Go2, A3, T1, Bumi and Pi+. Each task uses the repository's SuperMimic
world-model fitting, differentiable imagined policy optimization and GAE
critic. The terrain path uses MuJoCo-Warp for physics and 187 vertical height
rays for measured actor observations. The imagined actor, dynamics adapter and
critic receive heightfield samples at the predicted root state.

The height rays describe idealized geometry. This runtime does not include an
RGB/depth image encoder, sensor noise from a camera, or occlusion and material
rendering. The terrain tasks are velocity-commanded downstream objectives;
they do not require demonstration clips. Existing motion-tracking and Go2
downstream tasks remain registered separately.

The actor encodes each 187-sample scan into a learned 32-dimensional feature
before the SuperMimic observation-history decoder. The dynamics model receives
terrain samples at every predicted state, and the critic sees terrain-relative
clearance. New episodes start on the two easiest lanes and advance through
six difficulty levels using completed-episode success.

## List And Inspect

```bash
python scripts/rls/supermimic/tasks.py list
python scripts/rls/supermimic/tasks.py show --task locomotion/g1/stairs
python scripts/rls/supermimic/tasks.py smoke --task locomotion/g1/stairs
```

## Train

Train a single task to 10,000 outer iterations:

```bash
python scripts/rls/supermimic/train_terrain.py \
  --task locomotion/g1/stairs \
  --run-dir logs/supermimic-terrain/g1/stairs \
  --iterations 10000
```

Run the full 18-task queue on one GPU:

```bash
python scripts/rls/supermimic/supervise_terrain.py \
  --run-dir /path/to/isolated-workspace/runs/terrain-10000 \
  --num-envs 4096 --iterations 10000
```

The queue writes `campaign.json`, one run directory per task, training logs and
optimizer-complete `last.pt` checkpoints. Jobs run serially and can be resumed
from their latest checkpoint by restarting the supervisor. Create `STOP` in the
campaign directory to request a checkpointed pause. The next run resumes work
from the saved iteration. Stairs, mountain and parkour tasks can initialize
from a completed checkpoint for the same robot; each keeps its own 10,000
iteration counter and output directory. Each completed run receives a
1,500-step physics evaluation using a different terrain seed and all six
difficulty lanes. Per-lane success, falls, timeouts, return and progress
are written to `evaluation.json`.

The learner defaults use 4,096 environments, 10 world-model minibatches of
2,048 windows, 32 imagined steps, one actor update and 20 critic minibatches
per outer iteration after the 500-iteration world-model warmup. Training logs
record complete-iteration speed and optimizer losses. The `actor_critic_active`
column excludes the initial actor-compilation warmup from throughput analysis.
`training_speed_summary.json` records 1,000 complete iterations after this
warmup, together with the actual WM/actor/critic update workload.

## Throughput

Compare the median complete-iteration rate after warmup with the historical
Tele workload described in [warp_training.md](warp_training.md). Include
environment count, world-model batches, actor updates, critic optimizer steps,
compile time, simulator backend and measured window in any comparison. The
historical 2.47115 iterations/second measurement is a reference target, not a
result for these terrain tasks. Record each robot/profile rate from its own
training log before claiming parity.

## Validate

Run focused CPU contracts with `python -m pytest tests/test_supermimic_terrain.py`.
Before a full queue, validate every robot MJCF, CPU/GPU height-ray agreement,
partial resets, finite physics state and a short training update on the target
CUDA server. Check the resulting `run_config.json` and `run_summary.json` for
the exact simulator and optimizer settings.

```bash
python scripts/rls/supermimic/validate_terrain.py \
  --run-dir runs/terrain-validation
```

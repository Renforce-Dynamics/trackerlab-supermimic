# SuperMimic Terrain Tasks

This terrain campaign is retained for audit and historical review. New
locomotion runs use the gait-first, physical-quality-gated plan in
[SuperMimic Gait Foundation](supermimic_gait_foundation.md); the existing
terrain-specific low-success queue is retired.

The registry adds 18 downstream objectives: stairs, mountain and parkour for
G1, Go2, A3, T1, Bumi and Pi+. Each task uses the repository's SuperMimic
world-model fitting, differentiable imagined policy optimization and GAE
critic. The terrain path uses MuJoCo-Warp for physics and 187 terrain-height
samples for actor observations. The imagined actor, dynamics adapter and
critic receive heightfield samples at the predicted root state.

The height scans describe idealized geometry. MuJoCo-Warp physical rays are
used for validation against these scans. This runtime does not include an
RGB/depth image encoder, sensor noise from a camera, or occlusion and material
rendering. The terrain tasks are velocity-commanded downstream objectives;
they do not require demonstration clips. Existing motion-tracking and Go2
downstream tasks remain registered separately.

The actor encodes each 187-sample scan into a learned 32-dimensional feature
before the SuperMimic observation-history decoder. The dynamics model and
critic use a 54-point front-and-side subset of that same heightfield at each
predicted state; the critic also sees terrain-relative clearance. New episodes
start on the two easiest lanes and advance through six difficulty levels using
completed-episode success. A physical fall moves an environment down one level,
including falls after forward progress; timeouts leave the level unchanged.

Objective version 2 uses a `-0.5` fall penalty. At success, it replaces the
positive continuation lost when the episode terminates with
`gamma * 0.05 / (1 - gamma)`, plus a `+0.5` success bonus (total `5.45` at
`gamma=0.99`). Here `0.05` bounds the positive running reward. This makes
reaching the goal preferable to delaying termination to collect more running
reward. Imagined updates use soft success/fall probabilities and the same
terminal magnitudes as physical collection. Unit tests cover this preference
and the differentiable objective.

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
  --num-envs 8096 --iterations 10000
```

The queue writes `campaign.json`, one run directory per task, training logs and
optimizer-complete `last.pt` checkpoints. It also retains an
`iteration_XXXXXX.pt` checkpoint every 2,000 iterations. Jobs run serially and can be resumed
from their latest checkpoint by restarting the supervisor. Create `STOP` in the
campaign directory to request a checkpointed pause. The next run resumes work
from the saved iteration. Objective-version or discount changes require
`--warm-start` in a new directory instead of `--resume`. Warm starts load the
actor and world model, and reset the critic, optimizers and iteration counter.

Each completed run evaluates its retained checkpoints for 1,500 physics
steps on all six difficulty lanes using seed 43. The highest mean per-lane
success rate selects `best.pt`; lower fall rate breaks ties. The selected
checkpoint receives separate tests on seeds 44 and 45. `evaluation.json`
contains its selection-seed result, `evaluation_seed_44.json` and
`evaluation_seed_45.json` contain the test results, and
`validation_summary.json` combines them. Selection data and test data have
different roles and must be reported separately.

The queue warm-starts later tasks only from a same-robot checkpoint with at
least 50% mean per-lane success on **every** evaluated seed; it retains the
highest-scoring source rather than blindly forwarding the last task. An
explicit `--initial-checkpoints path/to/map.json` can supply initial policies
for the first task of each robot, with task IDs mapped to checkpoint paths.
Each new task still receives its own 10,000 iterations.

To repair low-scoring jobs and finish pending jobs while retaining a baseline:

```bash
python scripts/rls/supermimic/retrain_terrain.py \
  --baseline /path/to/previous-campaign \
  --run-dir /path/to/repair-campaign \
  --iterations 10000 --num-envs 8096
```

This preserves completed baseline tasks scoring at least 50% mean per-lane
success on the available baseline evaluation. The baseline gate is provisional
when only one seed exists. The saved `retrain_plan.json` records retained tasks,
new jobs and explicit initialization sources. Add `--plan` to inspect it without
launching. Each repaired job is tested on the three-seed protocol above.

To queue another 10,000 iterations for jobs whose completed validation remains
below 90% on at least one validation seed:

```bash
python scripts/rls/supermimic/queue_low_success_retrain.py \
  --source-campaign /path/to/terrain-10000/campaign.json \
  --run-dir /path/to/terrain-low-success-followup \
  --iterations 10000 --num-envs 8096 --threshold 0.90
```

The command waits for a running source campaign to finish, selects completed
jobs by `min_seed_macro_success`, orders them from lowest to highest, and
launches a separate sequential SuperMimic queue. Every selected task starts
from its own evaluated `best.pt`; the source campaign and its checkpoints are
preserved. Use `--plan` to inspect the follow-up without launching it.

The learner defaults use 8,096 environments, 10 world-model minibatches of
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

## Recorded Demos

Generate an MP4 and a local video gallery for every completed campaign job:

```bash
python scripts/rls/supermimic/render_terrain.py \
  --campaign /path/to/terrain-10000/campaign.json \
  --output /path/to/terrain-demos --seed 44
```

The evaluator records the first episode of environments 0 through 5, one for
each difficulty. It saves the physical pre-reset terminal poses and keeps
finished attempts frozen. The video shows difficulty 6/6 at 1x speed, retains
falls and timeouts, and labels the one-second terminal hold. A contact sheet
shows all six difficulty lanes at a fixed snapshot time. The inset visualizes
the 187 noise-free heightfield samples; the policy still uses observation noise.
MP4 decode, frame count, finite poses, and nonblank/moving pixels are checked.
Each `demo.json` records checkpoint and trajectory hashes, the exact checkpoint
iteration, held-out terrain seed and all first-attempt outcomes.

`index.html` opens directly in a browser. Checkpoint iterations may be below
10,000 because validation selected an earlier milestone. The displayed test
success rate comes from the separate campaign evaluation, rather than one
demo attempt. A retained baseline with only one seed remains labelled as such.
Repeat the command to add tasks completed since the previous recording; use
`--tasks locomotion/g1/stairs` to limit recording to specific completed tasks.
Add `--portable --overview` to create a standalone HTML containing every
video and a continuous MP4 compilation. `--gallery-only --output /path/to/terrain-demos`
rebuilds these viewing artifacts from the saved manifest without simulation.

## Validate

Run focused CPU contracts with `python -m pytest tests/test_supermimic_terrain.py`.
Before a full queue, validate every robot MJCF, CPU/GPU height-ray agreement,
partial resets, finite physics state and a short training update on the target
CUDA server. Independent checks compare the scan samples with MuJoCo-Warp
physical rays; training uses direct queries of the same heightfield, avoiding
repeated per-world ray traversal. Check the resulting `run_config.json` and `run_summary.json` for
the exact simulator and optimizer settings.

```bash
python scripts/rls/supermimic/validate_terrain.py \
  --run-dir runs/terrain-validation
```

# SuperMimic Gait Foundation

The former eighteen-task terrain policy set is retired from new runs. G1,
Go2, A3, T1, Bumi and Pi+ first train a flat-ground walking foundation with
SuperMimic's full-body motion-tracking, world-model imagination and GAE critic.
Terrain and parkour training remains gated until all six gait foundations pass
the physical policy-quality checks.

## Reference Preparation

The G1 foundation uses BONES-SEED neutral-walking recordings. Three distinct
recordings train the policy and a fourth recording stays held out. It converts
units and coordinate conventions, preserves frame timing, and aligns each
recording to a flat floor using the robot's physical foot colliders.

Go2 uses pinned, retargeted Kine2Go dog-walking recordings. Data handling
removes idle, bowing and excessive-tilt intervals, rescales the reference pace
to 75%, deduplicates identical motion files, aligns feet to the physical floor,
and keeps a separate walk recording for held-out testing.

A3, T1, Bumi and Pi+ use Mink inverse-kinematics retargeting from the audited
G1 walk set. Per-robot files undergo a fresh physical-foot-ground, stance,
swing-height, root-tilt and joint-limit audit. IK error and source/target model
hashes are recorded. The reference audit checks input quality and geometry;
policy training and held-out simulation provide separate acceptance evidence.

Prepare G1 and Go2 references:

```bash
python scripts/rls/supermimic/prepare_gait_references.py \
  --robot g1 --bones-source /path/to/BONES-SEED/extracted/g1/csv \
  --output data/gait-foundation-v5/g1

python scripts/rls/supermimic/prepare_kine2go.py \
  --source_dir data/raw/kine2go-gait-foundation \
  --output data/raw-go2-gait --motion_config data/raw-go2-gait.json \
  --clips ai4_dog_walk_00 ai4_dog_walk_01 ai4_dog_walk_02 \
          ai4_dog_walk_03 ai4_dog_trot_00 ai4_dog_trot_01

python scripts/rls/supermimic/prepare_gait_references.py \
  --robot go2 --source-config data/raw-go2-gait.json \
  --holdout ai4_dog_walk_02 \
  --output data/gait-foundation-v5/go2
```

Prepare the four humanoid references from G1:

```bash
for robot in a3 t1 bumi pi_plus; do
  python scripts/rls/supermimic/retarget_gait_references.py \
    --robot "$robot" --donor data/gait-foundation-v5/g1 \
    --output "data/gait-foundation-v5/$robot"
done
```

Each output includes `train.json`, `test.json` and `gait_audit.json`.
Provenance records name the upstream recordings and their hashes. Training and
test recordings must be disjoint.

## Training Queue

The queue runs one robot at a time on the shared GPU:

```bash
python scripts/rls/supermimic/supervise_gait.py \
  --references data/gait-foundation-v5/references.json \
  --run-dir logs/gait-foundation-v2/runs \
  --iterations 10000 --max-rounds 3 --num-envs 8096
```

`references.json` maps each robot to its audited reference directory. Each
round adds 10,000 iterations, with full actor, world-model, critic and Adam
state retained. A failed quality gate resumes the selected candidate with its
saved optimizer state for another round. Terrain tasks do not start
from this queue.

## Motion-Quality Gate

The selected gait checkpoint must pass on its training recordings and two
held-out seeds, across every held-out clip:

| Measure | Acceptance |
| --- | ---: |
| Full-clip survival | at least 95% |
| Mean tracked-body position error | at most 0.08 robot-height scales |
| Joint-position RMSE | at most 0.18 rad |
| Root orientation error | at most 0.15 rad |
| Tracked-foot position RMSE | at most 0.06 robot-height scales |
| Stance-foot horizontal speed | at most 0.12 robot-height scales/s |

All six per-robot reports must pass before the supervisor automatically queues
all eighteen terrain tasks in the separate `advanced-terrain-v1` directory.
The first terrain task for each robot initializes from its qualified gait
checkpoint. The
terrain initializer transfers the gait world model and matching proprioceptive
decoder weights. It creates a new terrain-goal encoder, height-scan encoder,
terrain adapter, value critic and optimizer states. Legacy terrain checkpoints
cannot be used to start or resume new training.

## Evaluation And Demo

An individual gait foundation can be checked against its held-out clips:

```bash
python scripts/rls/supermimic/evaluate_gait.py \
  --checkpoint logs/gait-foundation-v2/runs/g1/round_01/attempt_01/best.pt \
  --motion-config data/gait-foundation-v5/g1/test.json \
  --output logs/gait-foundation-v2/g1-heldout.json \
  --seed 44 --trajectory-output logs/gait-foundation-v2/g1-heldout.npz
```

The evaluator records full-clip outcomes, measured foot contacts and slip,
per-body and joint errors, seed, source configuration and checkpoint hash.
Recorded MuJoCo-Warp states can use `render_demo.py` for a policy/reference
comparison.

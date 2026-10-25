# Go2 Jump-Only Curriculum

The jump experiment is an isolated profile of the Go2 velocity/dodge task. It
starts from the rapid policy checkpoint, loads its actor and learned dynamics,
and initializes a new critic and fresh optimizers for the jump-specific reward.
The rapid run and its saved artifacts remain unchanged.

## Initial Course

At difficulty zero, each obstacle is a low bar approaching from the front.
Root speed is centered near 0.3 m/s, the bar is about 0.08 m high and travels
at 0.5-0.8 m/s. The lead time is long and encounter periods are 2.5-3.0 s.
Difficulty advances in 0.05 increments when a 100-update window contains at
least 250 completed encounters and at least 65% succeed. Height, speed,
command variation and frequency then approach the ranges in the task config.

## Reward And Measurements

The existing root-height term is reduced from 0.015 to 0.005; 0.010 is assigned
to a smooth foot/calf top-clearance reward. Their sum remains 0.015, so the
positive per-step reward maximum stays 0.1 and the `gamma=0.99` value bound
stays 10. Physical contact remains a penalty and an evaluation failure.

At each step, the clearance signal uses the Go2 calf frames and the configured
foot offset. It rewards points above the bar top during the crossing window.
The absolute root-height target is the bar top plus nominal root height and
0.08 m clearance: 0.475 m for a bar centered at 0.08 m. The timing gate blends
from nominal height to this absolute target; nominal height is counted once.
Evaluation separately records minimum signed foot/calf clearance, sample count,
contact force, crossing, landing episode survival and failure reason. A jump
attempt succeeds only under the existing natural-end, crossing, no-contact and
survival rules; a positive clearance sample alone does not count as success.

## Run

Use the Isaac Sim environment described in `installation.md`, including its
EULA environment setting for unattended runs.

```bash
python scripts/rls/supermimic/supervise.py \
  --config scripts/rls/supermimic/configs/go2_jump_dodge_8096.json \
  --warm_start_velocity logs/go2_rapid_8096_10k/last.pt \
  --motion_config data/configs/go2_kine2go.json \
  --run_dir logs/go2_jump_8096_6k \
  --num_envs 8096 --iterations 6000 --ckpt_every 1000 \
  --milestones 1000 3000 6000
```

This is a learned-reward obstacle task, not motion imitation of a jump clip.
The Kine2Go trot remains the cyclic reset/reference prior. The jump maneuver is
learned from explicit task rewards plus PhysX contact and crossing evaluation.

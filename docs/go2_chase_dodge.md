# Go2 Chase and Dodge

This downstream task follows a moving target while ducking under high bars and
crossing low bars. The task behavior was inspired by the user-supplied
Xiaohongshu video. The author's training code, reward formula, observation
specification and training time were unavailable during implementation. The
task configuration in this repository is our own extension.

## Transfer and Data

The initializer is the completed 10,000-iteration Go2 gait checkpoint trained
with 8,096 environments on Kine2Go `ai4_dog_trot_00`. Kine2Go supplies a
Go2-retargeted AI4Animation motion. In this task, that clip supplies reset poses,
phase-conditioned gait goals and a small style reward. Simulator transitions
supply the downstream learning data.

The actor retains its original motion encoder and action decoder. A new task
head adds an initially zero latent residual conditioned on 18 task features.
Given the same observation and motion goal, the initial output exactly matches
the gait actor. The last six observation dimensions now express orientation
relative to target velocity. A decaying teacher-action penalty supports the
first 2,500 task updates.

The learned dynamics retains its gait weights and gains an initially zero
residual conditioned on bar position, relative velocity and bar mode.
The new critic learns the downstream reward from scratch. Optimizer state is
also initialized for this task. Task iterations are counted separately from the
10,000 gait iterations.

## TrackerLab Layout

```text
data/configs/go2_kine2go.json
  -> motion_buffer/motion_lib/supermimic.py
  -> trackerTask/supermimic/go2_chase_dodge_cfg.py
  -> tracker_env/supermimic/chase_dodge.py
  -> algorithms/supermimic/chase_buffer.py
  -> algorithms/supermimic/chase_runner.py
  -> evaluate_chase.py -> render_chase.py -> milestones/*/ready.json
```

The same installer includes this task in an existing TrackerLab installation.
G1 and Go2 imitation use their original runner.

## Objective

The imagined-gradient actor update and GAE critic update follow the existing
SuperMimic implementation. Real and imagined rollouts use the same task reward:

| Positive term | Maximum per control step |
| --- | ---: |
| Target velocity | 0.035 |
| Target distance | 0.015 |
| Upright orientation | 0.010 |
| Desired height | 0.020 |
| Gait style | 0.010 |
| Survival | 0.010 |

Action changes, soft joint limits and geometric bar intersections contribute
nonpositive penalties. Thus `r <= 0.1`, and for `gamma=0.99` the bootstrap cap is
`0.1 / (1 - 0.99) = 10`. A reward change requires checking this bound again.

Bars are kinematic PhysX cuboids with collisions enabled. Physical bar forces
are measured every simulation substep. The differentiable reward uses a smooth
geometric collision proxy. These are separate measurements; the evaluation's
collision rate uses physical forces.

## Curriculum and Evaluation

The initial task is straight target following around 0.83 m/s. After iteration
300, difficulty grows over 5,000 iterations. Speeds range toward 0.65-1.5 m/s;
target lateral motion and approaching bar velocity increase with difficulty.
High bars lower from 0.65 m to 0.34 m, and low bars rise from 0.025 m to 0.12 m.
These values are initial local training choices, not estimates of the video's
training configuration.

Every milestone evaluates full difficulty with 16 fixed trials per mode.
Episodes last 8 seconds. A flat trial succeeds if it avoids falling; a bar trial
must pass the bar with no physical contact above 1 N and no fall. Reports also
include target-distance error and velocity RMSE, since survival alone does not
establish successful target following.

The videos replay recorded PhysX states using MuJoCo graphics. The first trial
of each mode is selected before evaluating outcomes. Generalization to
unseen obstacle families, camera observations and hardware is unmeasured.

## Run

```bash
python scripts/rls/supermimic/train.py \
  --config scripts/rls/supermimic/configs/go2_chase_dodge_8096.json \
  --motion_config data/configs/go2_kine2go.json \
  --warm_start /path/to/go2_gait/last.pt \
  --log_dir logs/go2_chase_dodge --headless
```

```bash
python scripts/rls/supermimic/supervise.py \
  --config scripts/rls/supermimic/configs/go2_chase_dodge_8096.json \
  --motion_config data/configs/go2_kine2go.json \
  --warm_start /path/to/go2_gait/last.pt \
  --run_dir logs/go2_chase_dodge_managed \
  --milestones 1000 3000 6000 10000
```

Task checkpoints can be evaluated with `evaluate_chase.py --checkpoint ...
--output ... --headless`, rendered with `render_chase.py --directory ...`, and
resumed with the original training configuration plus `--resume ...` in a new
log directory.

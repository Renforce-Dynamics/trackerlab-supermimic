# G1 WASD Adaptation Results

Recorded November 7, 2026, Asia/Shanghai.
Task design and keyboard semantics: [G1 WASD stealth](g1_stealth_wasd.md).

Releasing movement keys requests zero body translation and yaw while retaining
the stealth clearance and completed entry clock. Standing is a separate
policy selected by the host state machine.

## Completed Baselines

The following cycle evaluations use seed 101, 90 worlds, 1500 steps,
physical randomization, observation noise, and grounded standing starts.
Modes remain fixed per environment across resets. Full success requires
survival through timeout and tracking for at least 80% of episode actions.

| Experiment | Iterations | Episodes | Falls | Full Successes | Forward Body vx | Idle Tracking |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Natural v2 | 1000 | 90 | 0 | 1 | -0.0246 m/s | 53.4% |
| Natural v3 | 1000 | 90 | 0 | 10 | 0.0004 m/s | 97.3% |
| Natural v3 | 4000 | 91 | 1 | 10 | -0.0048 m/s | 87.1% |

All full successes in these evaluations belong to idle. The positive
forward target is 0.25 m/s. These are adaptation baselines, not usable
WASD locomotion policies. Longer v3 training did not resolve its stationary
solution.

The v3 1000-iteration release sequence exercised W, idle, S, WA, idle, SD,
idle for six seconds each. All 18 simulated worlds survived the 42-second
observation horizon. No episode timeout occurred within this extended horizon,
so its completed-episode success total is zero and must not be interpreted
as a failed survival rate. The six recorded first attempts retained a crouched
height, but movement tracking was poor and the arms remained low.

Local CPU pose replays found no self-penetration exceeding 1 mm in those six
sequence attempts or the nine cycle first attempts. Geometric sole slip
after three seconds was approximately 0.0085--0.0219 m/s in the sequence.
These are selected-by-environment first attempts, not outcome-filtered videos;
they do not certify all simulated states.

## Corrections And New Experiment

The earlier constrained actor's raw arm rows produced large targets when the
static arm override was removed. The initial migration reset those 14 output
rows and retained leg/waist outputs. A later audit found that its hard
acceleration clamp also zeroed arm gradients above a 0.001201-radian target
error from stationary history. Independent CPU autograd reproduced zero
gradient over a 32-step imagined rollout with a persistent 0.25-radian target.

The new limiter uses bounded smooth responses with target rate/acceleration
limits. Versioned checkpoint loading preserves old limiter behavior for old
models. WASD reward version 2 broadens movement and yaw reward widths while
retaining the prior success tolerances and maximum reward.

The new run is:

```text
/root/gpufree-data/runs/g1-stealth-wasd-smooth-v4-20261107
```

It starts from natural v3 `iteration_004000.pt`, retaining learned arm rows.
World model, critic and their optimizers restart; AMP restores because the
expert pool is unchanged. The actor learning rate is 2e-5, reference-reset
probability is 0.5, and sole-slip weight is 0.04. Its iteration count starts
at zero. The controller schedules 1000, 4000, 10000 and 20000 iterations,
with cycle/release evaluations and videos at each milestone. At publication,
the initial 500 world-model warmup iterations had completed; trained quality
of the new mapping remains pending.

## Throughput And Verification

Natural v3 measured **1.503 complete updates/s** over 1000 consecutive
actor-enabled updates on an RTX 4090 with 1024 simulator worlds. This is
below the requested 2.5 updates/s. The historical tele result used an RTX
PRO 6000 with a different workload.

Implementation `92e89cc` passed 540 local CPU tests with one optional skip,
and GitHub Actions run `42282998390`. Motion files and checkpoints remain in
ignored data/run directories; code changes were published on `main`.

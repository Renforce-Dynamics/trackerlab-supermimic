# G1 Stealth Command Curriculum

## Purpose

Continue the physically reviewed entry-v4 policy while expanding command coverage.
The algorithm remains SuperMimic with adversarial AMP. Inference consumes robot
observations and a body-frame velocity/yaw command; it needs no motion clip,
discriminator or world model. This experiment changes command sampling only.

The entry-extension iteration 12000 checkpoint retained forward speed at
0.206588 m/s against the original 10000 reference at 0.208927 m/s, with 0/24
falls in the matched seed-101, 1000-step evaluation. This is forward retention
evidence, not a full command or style acceptance result. The existing five-mode
extension is retired after its current stage and review; its outputs stay intact.
Its final iteration-15000 forward review fell in 4/26 completed episodes
(0.194540 m/s mean forward speed), failing the 10% fall guard. The curriculum
therefore uses iteration 12000, SHA256
`ba73ac067fb3a8e3e0f36889d368db1fc8fca4145d5cb364fab46d2888744b8d`.

## Commands

| Keys | vx (m/s) | vy (m/s) | yaw (rad/s) |
| --- | ---: | ---: | ---: |
| W | 0.25 | 0 | 0 |
| S | -0.20 | 0 | 0 |
| A / D | 0 | 0 | +0.55 / -0.55 |
| WA / WD | 0.25 | 0 | +0.45 / -0.45 |
| SA / SD | -0.20 | 0 | +0.45 / -0.45 |
| released | 0 | 0 | 0 |

Opposing translation/turn keys cancel on their respective axes. Releasing keys
retains crouched posture, arm targets and the current entry clock. Standing
belongs to an external state machine selecting another policy.

## Bounded Queue

`scripts/rls/supermimic/run_g1_command_curriculum.sh` resumes iteration 12000:

| Cumulative iteration | Additional updates in stage | Training modes |
| ---: | ---: | --- |
| 13000 | 1000 | W, A, D, WA, WD, released |
| 14000 | 1000 | previous modes plus S |
| 15000 | 1000 | all nine, adding SA and SD |
| 17000 | 2000 | all nine |
| 19000 | 2000 | all nine |
| 21000 | 2000 | all nine |
| 23000 | 2000 | all nine |
| 25000 | 2000 | all nine |

This is 13000 additional actor updates, including 11000 with all nine modes.
Existing canonical WASD probabilities are renormalized over each subset. No
new mode weights are introduced. Compared with the original five-mode sampler,
the relative sampling proportions also change to this existing WASD table.
Each stage retains the earlier directions. Each resume refills physical replay
through the existing collection path; checkpoint restoration does not restore
simulator state or its random-number stream.

The runner acquires the existing GPU lock for its whole lifetime and requires
a clean main checkout. `OUTPUT/STOP` requests a stage-boundary stop;
`OUTPUT/training/STOP` also stops the trainer at its checkpoint boundary.
It preserves every stage's run config, summary and immutable checkpoint.

```bash
bash scripts/rls/supermimic/run_g1_command_curriculum.sh \
  PYTHON OUTPUT SOURCE_12000 ORIGINAL_POOL ORIGINAL_FORWARD_REPORT \
  FIXED_PHYSICAL_BANK BANK_PROTOCOL
```

## Preserved Settings

- Full `--resume`: actor, world model, critic, discriminator and optimizers.
- Original single 106-frame Aiming2 expert and its calibrated arm reference.
- Original `command_stealth_entry.json` physical domain randomization, reset
  mixture, action limits, sole geometry and regularization coefficients.
- Original arm controller with zero learned arm residual and 4.82 s waypoints.
- Concurrent 4.84 s entry height/velocity ramp; key changes do not restart it.
- Original legacy reward, selected explicitly by `--command-reward-mode legacy`.
  WASD metrics keep the existing stricter thresholds regardless of reward.
- No actor normalizer count reset, arm reset, world-only re-warmup, learning-rate
  change or new replay command filter.

Changing only the sampler/subset requires `--allow-command-sampling-change`.
Reward changes and other incompatible full-resume settings still fail.
`verify_command_continuation.py` audits exact optimizer counter increments,
unchanged actor normalization, arm controller and expert identity at milestones.

## Evaluation

Every milestone records seed 101 held-forward (24 worlds, 1000 steps), all-nine
fixed command lanes (90 worlds, 1500 steps), and a 42-second release/command
sequence (18 worlds, 2100 steps). The latter keeps each command for six seconds
and records first attempts without outcome selection. Every direction receives
a video. Final iteration 25000 adds seed 103. Evaluation ignores the training
subset, so missing competence remains visible from the first review.

Continuation requires the existing matched forward-speed/fall gate and at most
10% completed-episode falls for each currently trained mode. Inactive modes are
reported, not gated yet. These are regression guards, not deployability gates;
they do not certify direction accuracy, transitions or natural motion. Each mode
reports actual vx/yaw, tracking fraction, sole slip and falls. Exported policies
remain experimental until physical and visual review.

The same recorded physical bank is used for 1/8/32-step world-model validation
with a common reference normalizer. Training WM loss alone is insufficient for
judging physical gait retention; discriminator scores alone cannot certify style.

## Limits

The single forward expert has no demonstrated natural reverse/idle gait. Those
commands can conflict with the motion prior, and the original `dont_wait` term
only encourages positive forward commands. More updates may improve coverage
but do not guarantee the requested appearance. This sampling-only experiment
keeps those factors explicit rather than changing data, rewards and controllers
together. Entry duration and fixed-arm expression cannot improve from additional
actor updates; controller retiming remains a separate physical experiment.

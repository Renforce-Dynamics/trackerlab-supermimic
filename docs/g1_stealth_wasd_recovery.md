# G1 WASD Recovery

The natural-arm migration lost the old policy's forward gait. Matched physical
counterfactuals on November 7, 2026 showed that changing its observation
normalization disrupted the old actor even before actor optimizer updates.
Removing its fixed arm trajectory also increased falls. Recovery starts from
the preserved entry-v4 iteration-10000 policy, not the stationary natural-v3
or smooth-v4 policies.

## Preserved Behavior

`command_stealth_wasd_recovery.json` uses SuperMimic and AMP, 1024 worlds,
physical domain randomization, and the existing nine WASD command modes.
Key release retains stealth clearance and the entry clock. Standing remains
an external state-machine policy.

The recovery preset freezes actor observation and goal normalization during
collection, including world-model-only warmup. Resume must keep this setting.
It preserves the actor output rows and requires exact equality of the saved
arm controller configuration. The source world model transfers when the
existing physical compatibility checks pass. Actor and critic optimizers
start fresh for the changed objective; actor learning rate is 2e-5.

Arm targets retain the accepted entry-v4 waypoint trajectory and fixed final
pose. This recovery stage does not learn new arm motion. Gradual arm changes
require separate physical review after commanded locomotion is recovered.

## Expert Pool

The ignored recovery derivative is:

```text
data/style-research-20261107/wasd-recovery-fixed-arms-v1/
  pools/g1/stealth-ad92cd79faafc566377d
```

It retains the natural WASD pool's original forward, backward and crouched-idle
chronology, one third of AMP history windows per role. Only the 14 named arm
CSV columns are replaced with the accepted entry-v4 final pose. Root, leg
and waist samples are preserved; the Aiming1 holdout is byte-identical.
The pool contains the exact original `upper-body-reference.json`, including
the runtime entry trajectory.

All 303 training poses passed the local geometry screen with zero counted
self-penetrating or negative audited hand/body-distance samples. Maximum
absolute normalized joint target is 8.454, below the existing action limit
of 20. These are source-pose checks, not learned-policy acceptance.

## Bounded Training Queue

The runner evaluates and renders the initial checkpoint under a held forward
command, then schedules 1000, 2000, 5000 and 10000 adaptation iterations.
The first 500 iterations update only the world model.

```sh
bash scripts/rls/supermimic/run_g1_wasd_recovery.sh \
  "$PYTHON" "$RUN" "$POOL" "$INITIAL" "$POOL/upper-body-reference.json"
```

Every milestone receives a matched 24-world, 1000-action forward evaluation.
`check_wasd_recovery.py` checks finite, compatible reports and requires:

- Mean body forward velocity at least 0.15 m/s and at least 75% of baseline.
- Mean body forward velocity no greater than 0.40 m/s.
- Fall fraction at most 10%.
- A valid baseline that satisfies its own speed/fall requirements.

A failed gate leaves its report, checkpoint and forward video, marks the
controller phase and prevents the next training stage. A passing gate is
followed by all-nine-mode evaluation, first-attempt videos, the 42-second
command/release sequence and export. The gate checks forward-gait preservation
only; complete WASD success and visual quality still require their own review.

The runner holds the existing GPU lock across training and review, honors
run/training STOP markers, and writes code/input hashes and terminal receipts.
The original checkpoints and completed experiments remain preserved.

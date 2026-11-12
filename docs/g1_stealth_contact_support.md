# G1 Contact-Support Reward Trial

`fullbody-contact` is a bounded trial on `locomotion/g1/stealth`. Its preset,
`command_stealth_full_body_contact.json`, differs from
`command_stealth_full_body.json` only in `stealth_gait_version=3`.
The existing corrected full-body pool and its data bytes remain unchanged.
Gait weight 0.02, swing height 0.04 m, AMP weight 0.1, all nine WASD modes,
0.6 s entry, domain randomization, measured clearance target and learned
29-action controller retain their existing settings.

## Evidence And Scope

The reviewed v2 checkpoint at iteration 10,500 achieved 90/90 numerical
successes and still failed visual review. The November 13, 2026 v3 trial
has now completed its full budget and recordings. It retains visual failures;
the measured results and the subsequent observability finding are below.

The saved first-attempt iteration-5,500 material was examined by the CPU
`support_gradient_audit.py` under
`../artifacts/g1-stealth-fullbody-review-20261112/`. That audit used AST
replacement of the support assignment alone. The proposed expression is:

```python
support = (.006 / hypot(relu(clearance / scale - .004), .006)) ** 2
```

All other v2 expressions and weights remain identical. The support estimate
has a plateau through 4 mm normalized clearance and a positive airborne
tail. It remains a geometric estimate, with no measured contact-force claim.
The following synthetic coordinate probes use `scale=1`.

| Coordinate probe | v2 dReward/dFootHeight | Candidate dReward/dFootHeight |
| --- | ---: | ---: |
| Both feet on the ground, each sliding at 0.1 m/s | -0.00816 /m | +0.09869 /m |
| One stationary planted foot and one swing foot, planted-foot derivative | -2.59 /m | 0 /m |
| Both feet airborne, landing derivative | Negative | Negative |

In the recorded left-pivot coordinate probes, the left/right foot
lowering-incentive fractions changed from 39.7% / 22.1% to 0% / 0%.
These are reward-coordinate probes holding other inputs fixed. They provide
neither feasible joint-action tests nor trained-policy acceptance, and do
not establish a solution to all skating. Full body motion, pivots, transitions
and contact behavior still require actual policy recordings.

The preserved 10,500 WM was also checked on 36 recorded starts, four per
command, using frozen CPU FP32 weights. All moving samples received finite,
nonzero leg-action gradients from the v3-minus-v2 reward change; idle gradients
were zero. Recorded-action sole-height prediction MAE was 0.196 mm at one
step, 0.818 mm at eight steps and 2.05 mm at 32 steps. The 32-step p95 error
was 8.40 mm. This verifies local action sensitivity and sampled prediction,
with root rates reconstructed from poses; it provides no physical intervention,
GPU parity or full-horizon gradient-fidelity claim.

## Initialization And Budget

Use a **standard warm start from the reviewed v2 iteration-10,500 model**:

```sh
bash scripts/rls/supermimic/run_g1_stealth_full_body_contact.sh \
  PYTHON NEW_RUN SOURCE_10500 CORRECTED_POOL SOURCE_COMMIT
```

`SOURCE_COMMIT` must identify clean `main` containing the recipe and the
v3 reward implementation and validators. The wrapper selects
`fullbody-contact` even when the caller has set another recipe environment
variable. `CORRECTED_POOL` is the same full-body pool used for the reviewed
10,500 model; rebuilding or editing it is outside this trial.

The ordinary warm-start path preserves actor weights, compatible world-model
weights and frozen actor normalization. It resets the critic and the actor,
world-model and critic optimizers. AMP state, including its optimizer, is
restored only when the saved expert identity matches the current expert.
The changed reward starts a new run and a new local iteration count:

- 500 world-model-only iterations.
- 5,000 actor updates.
- Final iteration 5,500, with the sole review at 5,500.

Passing the v2 source as a full `RESUME` is rejected. The new recipe also
rejects `--extra-actor-updates` and nonzero
`SUPERMIMIC_STEALTH_EXTRA_UPDATES`. The existing `fullbody` extension to
10,500 and the budgets of all other recipes retain their behavior.

## Interrupted Trial

An interrupted v3 run can continue within its original 5,500-iteration
budget, using a fresh controller directory:

```sh
bash scripts/rls/supermimic/run_g1_stealth_full_body_contact.sh \
  PYTHON RECOVERY_RUN SOURCE_10500 CORRECTED_POOL SOURCE_COMMIT CONTACT_V3_CHECKPOINT
```

The saved iteration must be a completed iteration from 1 through 5,499,
and its saved target must be 5,500. The checkpoint must match this v3 preset,
pool source identity, entry guidance, AMP identity and learned controller.
Actor, world model, critic, all three optimizers and AMP state are required.
This path uses the trainer's actual `--resume`; completed updates and the
remaining world-model warmup count carry forward. Replay buffers and physical
worlds follow the existing trainer resume startup.

Optional `SUPERMIMIC_STEALTH_PHASE_PROFILE_UPDATES=1000` retains the existing
timing observation. Its default is 0. The timing window must fit inside the
remaining budget, including the ordinary speed-observation interval; an
oversized window is rejected without increasing the target iteration.

## Controller Safeguards And Review

The shared runner requires `main`, a matching source commit and clean source.
It takes exclusive controller, repository GPU and shared scheduler leases,
then waits for the GPU owner to finish. Input receipts bind the source
checkpoint, optional resume checkpoint, preset, manifest, train index and
both source-bound auxiliary files: `aim-style-reference.json` and
`full-body-style.json`. Receipt checks after the GPU wait and after CPU
preflight detect changed inputs. CPU preflight rehashes the pool and
recomputes FK validation before training. Owner `STOP` markers remain active
during waiting, training and review; interrupted training retains its saved
checkpoint.

At iteration 5,500 the runner records six held-forward attempts, all nine
held-command lanes and the 42 s WASD release regression. It produces eleven
front/side recordings: one forward, nine held-command and one release
recording. Failed attempts remain in the evidence. The terminal status is
`completed_5500_awaiting_visual_review`; numerical success alone leaves
visual acceptance pending.

An unreviewed result remains excluded from policy export and other-robot
training. CPU checks verify the recipe contracts; target-server execution and
actual policy recordings provide separate evidence.

## Completed Trial: November 13, 2026

Training source `1d267aff134daf416c95bf3136ad95742c6d9b9f` completed
500 WM-only iterations and 5,000 actor updates. The final checkpoint label is
5,500, SHA256
`e8b26027ff02382c511e5ed9f942120589731443009afd7966fbea5f44e76b62`.
Actual saved optimizer counters are actor 5,000, WM 55,000, critic 100,000 and
AMP 30,000; AMP started at 20,000. The controller exited 0 after all recordings.
Actor normalization remained identical to the source checkpoint.

Held forward passed 24/24 episodes. All nine held commands passed 90/90 with
zero falls. Evaluation protocols, recorded commands, physical contexts and
initial poses match the retained v2 comparison. Eleven videos were
hash-verified and fully decoded, totaling 6,050 frames. Dense side-frame
inspection covered forward entry/stepping, left pivot and backward-left;
decode coverage alone makes no visual-acceptance claim.

First-attempt whole-sole lifts over 2..18 seconds use the same definition:
the lowest sole pole is above 2 cm for an observed span of at least 80 ms.

| Command | v2 at 10,500: Left/Right | v3 at 5,500: Left/Right |
| --- | ---: | ---: |
| W |25/26|21/21|
| S |8/12|11/13|
| A |0/0|0/0|
| D |18/10|28/27|
| W+A |26/27|21/21|
| W+D |27/27|23/23|
| S+A |0/6|0/15|
| S+D |0/25|1/25|
| Idle |0/0|0/0|

Forward geometric support slip decreased from 0.02668 to 0.02026 m/s.
Left-pivot slip remains 0.10692 m/s, with no sustained whole-sole lifts.
Backward-left retains a dragging left foot, and backward-right remains
asymmetric. In the matched held-forward first attempt, mean torso lean
increased from 13.76 to 15.02 degrees; the selected reference is 24.45 degrees.
Idle retains approximately 0.283 m fore/aft foot spacing.

The 42 second release sequence was recorded with 18 worlds. Its episode counters
are zero, so it supplies recorded transition evidence rather than an 18/18
completed-episode success rate.

**Visual acceptance remains false.** Preserve this partial improvement and
its source model. Export and other-robot training remain unqualified.
The bounded evidence is in
`../artifacts/g1-stealth-fullbody-review-20261112/contact-v3-005500/review-evidence.json`
(SHA256 `862b47be0f9b13583f81a2ec1fc17bafad9fdc0e5b78f1a0dc73082ea3c8423d`).
The ordinary 1,000-update speed window, iterations 541..1540, measured
1.211489 updates/s; shorter windows near 1.26/s are a different statistic.
Historical Tele speed parity remains unmet.

## AMP Observability Finding

A local CPU probe bound the unmodified production feature and history methods
to the saved 117-value `postural_end_effectors_vx` layout. With the exact
selected-pose CSV, zero root translation/joint velocity and rigid yaw rates
0 or +/-0.55 rad/s, all 1,170 values over ten frames were byte-identical. Actual
collision-sole material-point horizontal speeds changed from 0 to
0.0840..0.3021 m/s. Independent MuJoCo/analytic velocity agreement was within
5.58e-8 m/s.

This proves ambiguity in the feature map for that kinematic construction.
The selected pose includes sole heights from 0.003 to 0.1454 m; the probe establishes
neither contact feasibility nor the cause of the learned policy's failures.
Global-heading invariance was checked separately. A two-term posture probe
also found that upright and torso-style rewards compete, while their combined
gradient at the old forward samples still favors more forward tilt. It does
not support attributing the posture failure to the upright term alone.

A subsequent bounded candidate may append two per-foot RMS horizontal
sole-speed features derived from existing WM state and FK. Actor inputs,
reference data and task rewards would retain their current contracts.
The candidate requires its own discriminator training, gradient/geometry
checks and actual G1 visual review. The opt-in
[sole-speed trial implementation](g1_stealth_sole_speed.md) is now available;
training and visual qualification remain pending.

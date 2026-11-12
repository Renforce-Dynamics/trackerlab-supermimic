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
successes and still failed visual review. Version 3 has never been trained
or deployed at the time this recipe was prepared. Its effect on policy
behavior awaits the bounded trial and owner visual review.

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

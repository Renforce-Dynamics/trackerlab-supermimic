# G1 Full-Body Stealth Reference Repair

The owner rejected the completed25,500-iteration policy because its forward
motion looked upright instead of matching the chosen stealth posture.
This recipe corrects reference preparation for the existing
`locomotion/g1/stealth` task. It adds no task ID, PPO stage or runtime joint hold.

## Confirmed Reference Error

The previous preparation copied three waist and fourteen arm coordinates from
the selected pose while retaining each donor's root orientation and legs.
The selected waist coordinates depend on a14.43-degree forward root lean.
Copying those coordinates onto another root does not preserve torso orientation.

| Evidence | Mean Torso Forward Lean |
| --- | ---: |
| Selected pose |24.45degrees |
| Original single-forward expert |24.36degrees |
| Original10kforward policy |20.93degrees |
| Previous transformed forward expert |9.12degrees |
| Rejected25.5kforward policy |7.99degrees |

Policy measurements use fixed first attempts over2..18seconds. Different
training/evaluation settings prevent treating the comparison as a single-factor
optimizer ablation. The transformation error itself was reproduced with FK:
the later forward donor changed from28.45to9.12degrees of torso lean.

Solving only the waist while freezing each old root is insufficient. The idle
clip requires32.39..34.21degrees of waist pitch, above its29.79-degree limit.

## Corrected Preparation

`prepare_stealth_full_body.py` uses the early106-frame Aiming2 subject3 forward
clip and the existing101-frame chronological Aiming2 subject5 backward clip.
Idle is a101-frame stationary hold derived from the selected pose, with the
owner-requested front/back stagger. Original moving-source intervals,
hashes and licenses remain in the immutable manifest. The Aiming1 holdout is
copied byte for byte and never enters training.

Mink1.4/DAQP fits the source foot frames, selected torso orientation and a
world-Z root-height target. The mean root-height target is0.6651475726945968m,
from the selected pose; each donor's original vertical excursions remain.
Root, waist and leg coordinates are IK variables. Fourteen arm coordinates are
fixed only while constructing these private research reference clips.
The trained actor still learns all29outputs with the existing smooth arm-rate
limiter and no fixed waist or arm action override. The idle source is explicitly
marked as a constructed hold rather than a chronological LAFAN recording.

Idle retains the selected front-foot order, with28cmfore/aft and18cmlateral
separation. Both soles are flattened and placed3mmabove the nominal floor.
The settled pose is repeated without idle motion. Its nominal COM projection
has2.96cmof margin inside the sole-pole support hull, with no measured
self-penetration. Randomized dynamic balance still requires policy evaluation.

The world-height task has its own world-frame Jacobian. Weighting a tilted
body-frame translation residual alone does not isolate world Z. Joint ranges,
motor velocity limits and bounded free-root velocities are included in IK.

Preparation bundles the exact donor bytes and selected-pose receipt inside the
ignored pool. `--verify-pool` rehashes them and recomputes physical FK checks;
a stored `passed` field alone cannot qualify the input. The runner performs
this CPU verification before training and binds the pool and auxiliary files
to its checksum receipt.

## Geometry Evidence

The corrected preflight has maximum forward foot-origin error5.04mm,
vertical sole-pole error1.96mm, root-height fit error2.28mm, and torso rotation
error8.25e-6radians. The lowest forward sole pole remains1.05mm above the floor.
Backward foot errors are approximately1.26micrometers or less; idle is compared
against its explicitly constructed planted-foot targets.
The original stricter1mmrigid-foot criterion still fails for forward and is
explicitly retained in the report.

Retarget acceptance permits at most6mmfoot-origin error,3mmvertical-pole error
and5mm/sincrease in geometric support-foot speed, with independent floor,
joint/velocity, torso/height and self-intersection checks. Measured forward
support-speed increase is1.15mm/s; backward does not increase it and the idle
hold has zero kinematic motion.
These are kinematic data checks, not force-contact or dynamic-balance results.
Reference support-speed estimates are not themselves zero.

Backward root height still spans about0.619..0.719m because its original
vertical motion is retained. Natural appearance, body bobbing, physical foot
slip and command transitions require the actual policy videos.

## Bounded Training

The fullbody recipe uses an explicit warm start from the retained25.5kpolicy.
It preserves actor and compatible world-model weights, frozen actor input
normalization, action scaling, physics, randomization, all nine WASD modes,
the0.6sentry, AMP weight0.1and existing gait/hand reward weights.
The new expert identity fits a fresh AMP model. Critic and optimizer state
follow the existing warm-start reset contract; this is not full optimizer resume.

`command_stealth_full_body.json` sets the absolute clearance target to the
selected0.6651475726945968m. Training, evaluation and export must retain that
same target. The policy continues to require commands and body observations,
with no runtime reference trajectory.

```sh
bash scripts/rls/supermimic/run_g1_stealth_full_body.sh \
  PYTHON NEW_RUN SOURCE_25500 CORRECTED_POOL SOURCE_COMMIT
```

The budget is500world-model-only iterations plus5,000actor updates, ending
at5,500. The sole quality review is at that final checkpoint. The controller
records all nine held commands, six held-forward attempts and the42srelease
sequence, preserving failed attempts. Extra training follows a recorded review;
export and other robots require actual G1 visual acceptance. A corrected reference render
does not establish a corrected trained policy.

## Five-Thousand-Update Continuation

After reviewing the first trial, an explicit extension preserves the complete
checkpoint state and adds 5,000 actor updates. It starts at iteration 5,500
and ends at 10,500, with a single final review. There is no second WM-only
warmup or optimizer reset. A later interrupted extension may resume within
the same absolute budget; its saved target must be 10,500.
Replay buffers and physical worlds use the existing resume startup; the
serialized model, optimizer and normalization states carry forward.

```sh
SUPERMIMIC_STEALTH_EXTRA_UPDATES=5000 \
SUPERMIMIC_STEALTH_PHASE_PROFILE_UPDATES=1000 \
bash scripts/rls/supermimic/run_g1_stealth_full_body.sh \
  PYTHON NEW_RUN SOURCE_25500 CORRECTED_POOL SOURCE_COMMIT CHECKPOINT_5500
```

The pool, entry source, learned action controller, preset and saved AMP state
must retain their identities. The source checkpoint and first review remain
preserved. This continuation keeps the reward/data configuration and all nine
commands; its new run directory distinguishes the additional budget.

Optional phase observation is requested separately and follows the ordinary
1,000-update timing interval. The runner accepts zero or 1,000 observed updates.
If fewer than 2,040 updates remain in an interrupted extension, choose zero
to retain its existing training budget. The controller records the chosen
timing count and refuses a window that would exceed the target.

# G1 Stealth Combined Adaptation

November10,2026. The owner requested separate evidence for stepping and entry,
then a combined continuation with learned arms/waist and a1.21sentry.
AMP remains the main motion prior; a small training-only entry cost is the
sole additional reward component. Physical quality is unproven for this recipe.

## Completed Recovery

The preceding run completed1000->6000, exactly5000additional actor updates.
Final90-world cycle evaluation had no falls. Forward bodyvx.20393m/s,
backward-.13958m/s, versus+.25/-.20commands. Its recorded first forward
attempt had5/3sustained whole-foot lifts over2..18s; backward had0/0.
Held-forward24-worldvx.20777, but only3/6recorded worlds retained bilateral
lifting. The old slow entry still leaned backward. These results remain
visually unaccepted and are preserved as the parent comparison.

The original single-clip10k policy and longer adaptation demonstrate that
forward steps can exist under the existing AMP task. No preserved trial has
yet demonstrated natural fixed-facing backward stepping. Replacing the prior
with three clips, releasing arms or imposing waist targets each occurred with
other adaptation changes; none is an isolated proof of the regression cause.

The earlier better forward entry used an explicit waist pose, changed waist
gains and different arm/timing settings. Frozen hard-waist grafts increased
falls and were rejected. Their improvement cannot be credited to1.21stiming
or freely learned waist alone.

## Frozen AMP Evidence

The final6000discriminator, with weights and normalization frozen, scores:

| History | Mean unweighted AMP reward |
| --- | ---: |
| Its original training expert | .99528 |
| Same expert with leg pose held and leg velocity zero | .92717 |
| Recorded forward | .22772 |
| Recorded backward | .01031 |
| Unseen natural backward expert | 0 |
| Unseen backward expert with source fixed-arm pose | .38441 |

These are classifier counterfactuals or fixed recorded histories, not physical
interventions. Out-of-training-distribution scores do not imply that the
natural expert is intrinsically poor. About89%of squared joint-reward-gradient
energy in the sampled forward histories lies in arm coordinates. The current
fixed-arm output mapping prevents direct actor adjustment of those joints;
leg/body dynamics can still affect physical arm state indirectly. Gradient
shares depend on units and are not causal attribution through the WM.

AMP already includes all tracked root-relative positions, including both ankle
and wrist origins, plus wrist orientations and q/qd over ten frames. Foot
lifting information is present. The postural layout omits root velocity;
translation of otherwise identical idle histories has identical features.
Thus feature presence, expert/policy separation, and useful action gradients
are distinct properties. The actual WM action Jacobian remains unaudited here.

## Selected Changes

Source is the completed recovery6000 checkpoint with learned waist and nominal
gains. Actor input normalization and leg/waist output rows remain preserved.
The formerly overridden14arm decoder rows are initialized through the existing
warm-start path. Compatible pretrained WM weights transfer. Changed prior
data/feature shape require a fresh discriminator and its optimizer; critic and
actor/WM optimizers follow the existing warm-start reset behavior.

Use the natural-arm audited forward/backward/idle pool
`data/style-research-20261107/wasd-natural-audit-v1/pools/g1/stealth-3da9aaa22c000d40df9f`.
The existing `postural_end_effectors_vx` layout adds measured signed root-local
forward velocity to the existing endpoint features. Actual source speeds
about+.58/-.50m/s differ from commands+.25/-.20; this remains a measured risk.
No source retiming, new gait phase or reference-motion policy input is added.

All29raw joint outputs are learned. Arms retain differentiable rate2.4rad/s
and acceleration12rad/s2limits; waist and legs have no pose override. Nominal
waist gains, physical randomization, existing task/slip rewards and allnine
WASD modes are retained. Key release still requests crouched idle.

`EntryStyleGuidance` adds a nonpositive cost bounded by.008 per step only
during entry. It uses actual torso-up direction in the pelvis yaw-heading
frame, preserving torso/root pitch and roll relative to gravity. The target
comes from the verified forward expert's actual waist-medoid frame72.
Its source CSV/manifest hashes, motion index, expert hash and target vector
are stored in the checkpoint. The target blends from upright over1.21s,
and cost fades to exactly zero at the completed entry clock.

The shape is `-.008*(1-b)*e2/(e2+.35^2)`, where b is the existing quintic entry
blend and e2 is the squared torso-up error against its normalized blended
target. Weight.008 and scale.35 are explicit trial settings, not coefficients
proved by the historical experiments. This is training feedback, not an
actuator trajectory. FK/WM state gradients remain; all waist actions stay free.
Real rewards and imagined rewards use the same post-step state/clock.
Evaluation restores identical saved metadata; deployment needs no extra input.

An additional geometric swing reward was rejected during static analysis:
an initial formula made dual flight more profitable than grounded translation
when combined with slip penalties. No foot-height/airtime reward from that
prototype is integrated. Endpoint AMP remains responsible for stepping style.

## Run And Review

`run_g1_stealth_combined.sh PYTHON RUN SOURCE NATURAL_POOL SOURCE_COMMIT`
uses `command_stealth_combined.json`:5500total iterations,500WM-only plus
5000actor-enabled. It records2500and5500milestones, each with24held-forward
worlds,90allninecommand worlds and18release-sequence worlds. Videos preserve
the first attempts, including failures, with front/side views.

The full actor budget continues through valid quality regressions. Runtime
errors, source/input changes and explicitSTOP stop with receipts. No policy
export or quality claim is automatic. Validate forward/backward whole-sole
lift, sliding, asymmetry, hopping, startup lean, arms and release visually.
Old controllers and presets retain identical behavior when the new cost is
disabled. No new branch is used.

Primary-source comparison inspected local hash-verified snapshots:
OpenLegged/supermimic_AMP@596dfe37 and InstinctMJ@03563ee.
Both represent velocity in the prior. InstinctMJ also models contact/airtime;
its contact head and reward were not imported. The new torso-entry cost is
an explicitly identified local experiment.

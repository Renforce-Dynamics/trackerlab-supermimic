# G1 AMP Sole-Speed Trial

The completed [contact-support trial](g1_stealth_contact_support.md) retained
left-pivot sliding despite passing command tracking. Its posture-oriented AMP
features could represent static idle and a held pose undergoing rigid yaw with
identical histories. The new opt-in feature mode
`postural_end_effectors_vx_sole_speed` adds two values to distinguish horizontal
sole motion. Physical and visual improvement remain to be evaluated.

## Feature Contract

The original 117 frame features retain their ordering and arithmetic. Each foot
adds RMS horizontal material-point speed divided by robot height. All four
collision spheres per G1 foot participate:

```text
arm[k] = R_foot * sphere_center[k] - sphere_radius[k] * world_z
v[k] = V_foot + W_foot cross arm[k]
feature[foot] = sqrt(mean_k(v[k].x^2 + v[k].y^2)) / robot_height
```

The feature uses full FK from `p, R, v, w, q, qd`. It measures the velocities of
the sampled material points; differentiating the changing world-down pole
locations would describe a different quantity. A shared helper preserves the
existing locomotion reward's geometry and arithmetic.

Default G1 dimensions are 119 per frame and 1,190 per ten-frame history.
Constant global heading changes preserve the features. Time-varying rigid yaw
becomes observable through horizontal sole speed. A scaled norm retains finite
values and gradients at tiny or large finite speeds; the zero-speed
subgradient is zero.

The mode currently requires G1's joint/tracked-foot layout and four collision
spheres per foot. Sphere centers, radii, ordering and the material-point
convention participate in expert identity and scoring/restore compatibility.
Existing feature modes and defaults retain their behavior.

Policy inference retains its existing command, observation and 29-action
interfaces. The extra features belong to AMP training and imagined style
rewards. Contact-force sensors and contact timers are outside this feature
contract.

## Data And Budget

`command_stealth_full_body_sole_speed.json` differs from the contact-v3 preset
only in `style_feature_mode`. It retains the same 308 reference frames,
all nine WASD modes, 0.6 s entry, selected height, AMP weight 0.1, gait version 3,
reward weights, arm rate limiter, domain randomization and self-collisions.

Use the reviewed contact-v3 checkpoint at label 5,500, SHA256
`e8b26027ff02382c511e5ed9f942120589731443009afd7966fbea5f44e76b62`:

```sh
bash scripts/rls/supermimic/run_g1_stealth_full_body_sole_speed.sh \
  PYTHON NEW_RUN SOURCE_CONTACT_5500 CORRECTED_POOL SOURCE_COMMIT
```

The ordinary warm start preserves actor weights, compatible WM weights and
frozen actor normalization. Critic and actor/WM/critic optimizer states start
fresh. The changed discriminator layout and expert identity require a fresh AMP
discriminator and normalization; its update counter starts at zero.

The fixed budget is 500 WM-only iterations and 5,000 actor updates, ending at
label 5,500. The sole scheduled quality review is at that final checkpoint.
An interrupted run can resume matching full state within this budget. The
runner rejects full resume from an old feature mode, altered references,
controller changes and extra-update requests.

The existing controller checks clean `main`, source/input receipts and GPU
leases. It revalidates the full-body pool and records held forward, all nine
held commands and the 42 second release sequence. G1 visual acceptance remains
required before export or other-robot training.

## Preflight Evidence

The complete local CPU regression suite passed 1,697 tests with four
environment-related skips. Core sole-speed contracts run with a self-contained
G1-layout MJCF fixture; installed G1 assets add a second set of real-model
checks. Compilation tests isolate their Dynamo cache from other feature modes.

The corrected pool still hashes to
`dee05f4dc4960635447ad369ed05ffa7f8403214381f4b00c959d2374db78b8e`.
Its motion bytes and independent FK checks passed locally. The new feature
prefix was byte-identical to the old mode on the actual reference and sampled
policy states.

Using the constructed staggered idle reference, rigid yaw at +/-0.55 rad/s
kept the old prefix identical while producing about 0.1211 and 0.0959 m/s
left/right sole RMS speeds. This is a kinematic observability check.

A separate unmodified-policy evaluation recorded measured pre/post-physics
states, including root velocities, from nine selected worlds. On 36 starts
covering all commands, frozen CPU WM probes yielded nonzero finite leg-action
gradients for every moving sample. Mean sole-speed prediction errors at
1/8/32 recorded-action steps were 0.0152/0.0119/0.0143 m/s; the 32-step p95
was 0.0541 m/s. Earlier pose-derived root-rate diagnostics had larger errors
and remain preserved.

These sampled CPU results establish local sensitivity and prediction evidence.
They do not certify full-horizon policy gradients or learned gait quality.
GPU compilation parity, actual training resource use and trained-policy videos
remain separate checks. Raw FP32 feature hashes are device-specific;
deployment must bind the target GPU's expert identity.

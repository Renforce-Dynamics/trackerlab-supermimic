# G1 Stealth Velocity: Visual Comparison

The owner requests natural G1 stealth controlled by fixed translation and yaw
commands. Visual review takes precedence over reward, survival, WM loss and
velocity error. The previous 1500-iteration waist pilot remains unaccepted.

## Command Contract

Retain the versioned WASD table: W=+0.25 m/s, S=-0.20 m/s; A/D request
+/-0.55 rad/s at rest and +/-0.45 rad/s during forward/backward movement.
Opposing keys cancel independently. Releasing keys selects zero translation
and yaw, retaining the crouched stance and completed skill entry. Standing is
selected by the host state machine using another policy.

Inference uses the existing proprioception/history and velocity/yaw/entry goals.
No expert trajectory, phase tracking, discriminator or WM is required at runtime.

## Controlled Comparison

Both new candidates start from the same saved waist-pilot iteration1500.
They retain the three-clip chronological LAFAN Aiming2 forward/backward/idle
pool, fixed final arms, persistent waist target, WASD-v2 rewards, physical
randomization and fixed actor normalization. Actor weights transfer.
Critic and optimization states follow the existing explicit warm-start recipe.
The WM still observes executed actions. Changed waist PD gains are a physical
dynamics change: this explicitly starts a fresh WM and fits it during warmup.
The saved motor contract is carried through evaluation and any later export.

The `entry` candidate changes the existing arm trajectory's time scale to0.5:
4.82 seconds becomes2.41, and the concurrent command/clearance ramp is2.42.
Every arm waypoint, path segment, interpolation and final pose is retained.
The target-rate bound becomes2.4rad/s and the target-acceleration bound12rad/s2,
exactly the derivatives of a half-duration replay. The actual path peaks are
2.286rad/s and10.976rad/s2. Simulator effort and velocity limits are unchanged.
This is an engineered entry test, not a new learned transition.
Holding waist targets also remains an explicit control choice.

Local loaded-pose force calculations found insufficient restoring authority in
the original waist PD: the roll target is about-1.17degrees, while a conditional
static equilibrium is near-29.07degrees. This occurs below the50Nm cap.
Both candidates therefore share an explicit waist-only gain change. Only the
three named waist Kp/Kd values change; arm/leg gains, normalized action units
and effort limits are preserved. Actual force/target traces are recorded in
the new trials. Static calculations do not establish dynamic balance.
The selected multiplier is4.0 for Kp and2.0 for Kd. Nominal loaded roll error
is1.535degrees, and the maximum over72 sampled poses from18 recorded worlds
is3.268degrees. Factors2/3 yield maxima8.854/4.754degrees in that same subset.
These are fixed-other-coordinate equilibria, not free-base simulation results.

The `velocity` candidate uses the same timing and additionally selects the
existing `postural_end_effectors_vx` AMP feature. This appends measured signed
root-local forward velocity, allowing the prior to distinguish translation
with motion from a similar posture held while sliding. Its discriminator and
normalization are fitted afresh; the original posture-only candidate restores
the matching discriminator. The comparison consequently includes discriminator
refitting, which must be reported.

Forward and backward expert median speeds are approximately+0.58/-0.50m/s,
greater than the command magnitudes. The extra feature may expose this mismatch
and harm tracking. It is a testable comparison, not an assumed repair.
No source timing, root translation or reward weight is edited in this trial.
Neither comparison can establish that more iterations alone will resolve
backward dragging or torso lean.

## Run And Review

Prepare an immutable retimed arm document:

```sh
python scripts/rls/supermimic/retime_stealth_entry.py \
  --reference ORIGINAL_ARM_REFERENCE \
  --time-scale 0.5 --output NEW_ARM_REFERENCE

bash scripts/rls/supermimic/run_g1_velocity_visual.sh \
  PYTHON RUN POOL SOURCE_1500 NEW_ARM_REFERENCE WAIST_REFERENCE entry

bash scripts/rls/supermimic/run_g1_velocity_visual.sh \
  PYTHON DIFFERENT_RUN POOL SOURCE_1500 NEW_ARM_REFERENCE WAIST_REFERENCE velocity
```

Each independent trial has1500iterations, including500WM-only and1000actor
updates. It acquires both the repository lock and the existing external
scheduler's `/root/gpufree-data/ultra_ipingpong/queue/.runner.flock`, then waits
for shared GPU jobs and pending work to clear. A live external scheduler
holds that lease, so this controller waits without altering or stopping it.
An explicit owner-authorized exclusive time window is needed while that
scheduler remains enabled. The runner honors its own and parent STOP markers
during lock waits and between review stages; active training saves and exits
when either marker appears. Every candidate records both seeds101/103, nine fixed
commands for20seconds and a42second command/release sequence. The videos show
first attempts, with no outcome filtering. Front and side views are re-rendered
from recorded physics. Early falls shorten the first-attempt video and must be
reported, never hidden by looping or selecting another attempt.

Review forward entry, torso yaw/roll, arm motion, sustained forward/reverse
steps, foot dragging, pivots/arcs and release into crouched idle. Compare both
candidates to the preserved predecessor. The controller then exits at
`awaiting_visual_review_no_automatic_extension`; it does not export policies.
Further5000/10000/20000update stages depend on observed visual improvement.
This review stage is not evidence that the requested task is already trained.

Local target geometry was checked at7218poses from the previous two-seed,
nine-mode recordings with arms/waist replaced by the new targets: zero
physical self-penetrations beyond1mm. This counterfactual has no physics steps
and does not certify actual tracking, hand-mesh clearance or visual quality.
Evidence and independent implementation/source reviews are retained under
`artifacts/g1-stealth-velocity-research-20261108/` in the owner workspace.

## Primary Sources

The bounded source review inspected pinned code, not repository promotional
claims. Root velocity is represented by all four inspected priors; none proves
natural G1 LAFAN reverse locomotion on its own. No foreign PPO trainer, source
motion, or model weights are imported.

- [OpenLegged SuperMimic AMP](https://github.com/OpenLegged/supermimic_AMP/tree/596dfe37e0f75769b635386cd589960cce7faaa2): body velocities in AMP and backward walking examples.
- [InstinctMJ](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748): ten-frame local velocity features and differentiable style in imagined rollouts.
- [DeepMimic AMP](https://github.com/xbpeng/DeepMimic/tree/1f915c52fcd4b95b5f5f15b759ae91bd81e9a801): stealth heading task, root velocity and endpoint features. Heading-only evaluation does not prove fixed-facing reverse steps.
- [ASE](https://github.com/nv-tlabs/ASE/tree/6f9b4f1f289603eee6a4f45d082bc6e6d83fecec): velocity-bearing priors, weighted backward demonstrations, and a separate latent-skill mechanism.

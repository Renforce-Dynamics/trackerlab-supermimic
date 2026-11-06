# G1 Stealth Entry And Geometric Style

This opt-in experiment addresses standing entry, hand geometry, and foot
sliding in the single-clip G1 command policy. It retains the SuperMimic world
model and differentiable actor updates with a real-simulation AMP discriminator.
The actor continues to use body observations, the height scan, and velocity/yaw
commands. No reference motion is read during policy execution.

## Method

`configs/command_stealth_entry.json` adds:

- Grounded standing resets, using the actual four collision spheres on each
  foot, with small joint perturbations. The pilot mixes these starts equally
  with collision/floor-screened training-reference states. Reference starts
  use the completed entry clock; evaluation always starts from standing and
  requires no reference library.
- A configurable quintic entry schedule for requested velocity, yaw rate, and
  clearance. Its clock resets when the skill starts and continues across
  keyboard command changes. The existing goal clock carries this value.
- Early episode states in imagination, with repeated initial observations for
  missing actor history. AMP is masked until genuine temporal history exists,
  and its reward fades in during entry.
- Root-relative tracked-body positions and wrist rotations in the AMP
  discriminator. Actor observations and the portable actor dimensions stay
  unchanged. Joint angles already encode these quantities through forward
  kinematics; explicit geometric features change the discriminator's learning
  problem and weighting.
- Bounds on the rate and acceleration of commanded arm targets. Both real and
  imagined execution use the two preceding executed actions. These bounds
  concern targets; measured physical joint velocities can exceed them.
- Optional synchronized arm waypoints embedded in the skill's output
  transform. Their clock shares the existing goal phase. This is a finite
  startup controller, with no external motion-file input at runtime. Independent
  joint rate limits cannot replace this timing without changing its geometric
  path. The steady pose neighborhood resumes after the final waypoint.
- Slip measured at the spherical collision sole, including angular velocity.
  Support weights come from geometry and are detached in actor optimization.
  They are an estimate, not measured contact forces.

Standing resets improve coverage of the intended initial state. They provide
no global-optimality guarantee and can make learning harder when the transition
is poorly explored. A steady forward clip supplies no observed standing-to-aim
transition. The entry schedule is an engineered prior that requires physical
evaluation. Generated or recorded transition examples would be a separate
data experiment.

## Hand Reference

The original LAFAN BVH contains no finger/thumb landmarks, and its Hand End
Sites are zero length. An optional hand-forward retarget objective can use a
documented rest-axis convention and a robot mesh-axis proxy. It cannot recover
an anatomical palm normal or finger articulation from that source.

`retarget_lafan_aiming.py --hand-orientation forward` exposes this one-axis
objective. Its default conversion remains unchanged. Better directional error
alone does not certify collision-free arms.

When using an external static hand correction, the AMP expert and the arm
constraint must use the same corrected geometry. Preserve the original clip,
root/waist/leg values, and held-out Aiming1 data. Create a new immutable pool
with correction provenance instead of overwriting the original pool. Static
clearance, the complete entry path, and the permitted residual neighborhood
must be reviewed separately.

## Collision Compatibility

`collision_mask_version=2` explicitly refreshes MuJoCo body collision masks
from all attached geometry masks after `mj_setConst`. Version 1 preserves the
original builder behavior for old checkpoints. The installed training runtime
(MuJoCo 3.16.0, checked November 6, 2026) already refreshes these masks inside
`mj_setConst`; the old training builder had coherent masks and detected the
penetrating reference poses. A separate offline audit that changed geometry
masks after model construction had missed contacts.

The new self-contained test checks actual sibling-body contacts, including a
deliberately stale broadphase, and requires no downloaded G1 resources.
Checkpoint evaluation restores its saved mask version. A version change
conservatively refits the world model and requires a new warm-start run.

## Review

The preceding 10,000-iteration domain/arm stage completed with no falls in
192 held-out episodes, but no episodes met the full command success criterion.
Those results do not establish successful steering or natural entry.

A pose-replay audit of the fixed first six seed-71 attempts measured roughly
0.17--0.20 m/s geometric sole slip during the first three seconds, compared with
0.06--0.09 m/s afterward. The audit uses centered pose differences and sphere
material-point velocities; it is separate from the legacy ankle slip metric.

The v4 hand candidate has a 4.82-second synchronized arm entry and uses zero
actor arm residual initially, leaving the domain joint offsets active. A
kinematic sweep of 8,712 perturbed configurations had no self-penetration.
The source hand-direction proxy matches closely, but the fixed open-hand mesh
and unobserved palm roll still limit its appearance.

Applying this new controller to the old actor without actor optimization failed:
all six fixed first attempts fell, and the 96-world evaluation recorded 650
falls. Reverting only the goal ramp or standing reset did not resolve this.
The original checkpoint, evaluated with its original options in the same
runtime, completed twelve 20-second episodes without falls. The new entry and
hand controller therefore require adaptation; the software smoke is not a
trained-quality result. The bounded pilot mixes reference and standing states
to cover both sustained motion and entry before considering a longer run.

For each candidate, retain fixed first attempts and review entry and sustained
motion separately. Report falls, forward/yaw tracking, sole slip, observed arm
motion, and self-penetration. Render the complete entry as well as forward and
turning commands. A discriminator score or passing software smoke test alone
does not demonstrate natural motion.

## Reproduction

The hand pose and procedural entry are local research artifacts under ignored
data directories. Prepare a new immutable pool from the audited Aiming2/Aiming1
source pool:

```sh
python scripts/rls/supermimic/prepare_stealth_entry.py \
  --source-pool SOURCE_POOL --reference STATIC_ARM_REFERENCE_JSON \
  --entry-profile ENTRY_PROFILE_JSON --output data/stealth-entry-NEW \
  --provenance "Describe the local static correction and its geometric audit"
```

Use the printed `train_config` and `reference` paths with `train_terrain.py`,
the `command_stealth_entry.json` preset, and a new run directory. Set the command
entry duration to at least the final arm waypoint time. Warm-start the actor
from the prior G1 checkpoint, allow the fresh world-model warmup, and evaluate
the bounded pilot before extending it.

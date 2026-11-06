# G1 Stealth: Aiming Arms And Physical Randomization

The follow-up [entry and geometric style experiment](g1_stealth_entry.md)
adds grounded resets, explicit endpoint AMP features, collision-reviewed arm
entry profiles, and physical-sole slip measurements. This page describes the
preceding stage and its unchanged preset.

The existing keyboard run finishes its scheduled 20,000 iterations before this
stage starts. `continue_g1_stealth.py` waits for the exact completion receipt,
the immutable final checkpoint, and the recorded source process to exit.
It then acquires the repository-wide `.supermimic-terrain-gpu0.lock` before
starting any validation, training, rendering or export process. Child processes
inherit both the controller and queue locks, so a controller restart cannot
overlap an orphaned child.

The next stage uses the same single LAFAN training clip and the same five
velocity/yaw commands. It reuses the learned actor, creates fresh value and
optimizer state, and refits dynamics for 500 world-model-only iterations when
physical randomization changes. The first 2,000 iterations are evaluated and
rendered, then the same stage resumes through iteration 10,000.

## Upper Body

- A medoid of the shoulder, elbow and wrist angles in the single training clip
  supplies one fixed aiming posture. Legs and waist retain the actor output.
- Arm targets are softly restricted to a 0.25-radian neighborhood of that pose.
- Each arm joint target changes by at most 1.2 radians/second, or 0.024 radians
  per 50 Hz action. Starting from the ordinary neutral reset, the controller
  gradually raises the arms.
- Applied arm actions stay within the saved policy action limit, matching
  simulator clipping and imagined dynamics.
- A small joint-pose and joint-speed penalty is evaluated on physical and
  imagined transitions.
- Collection applies the arm constraint after exploration noise. Evaluation
  and the portable policy apply the identical stateless transform. The previous
  executed action comes from the existing observation history.

This is a fixed skill constraint. Runtime policy inputs contain no reference
trajectory, reference frame, discriminator output, or motion library. The G1
asset has wrist joints, but has no controlled finger articulation or firearm.
The target-rate bound concerns commanded joint targets; actual link speeds and
self-collision require physical evaluation.

## Physical Randomization

The original friction range (0.3--1.2), joint offsets (plus/minus 0.01 radians),
and center-of-mass shifts (plus/minus 0.025 scaled meters) remain represented in
the privileged context. The robust profile adds:

| Property | Scale |
| --- | --- |
| Body mass and corresponding inertia | 0.90--1.10 |
| Actuator position stiffness | 0.85--1.15 |
| Actuator velocity damping | 0.85--1.15 |
| Actuator force limits | 0.90--1.10 |

These four factors are deterministic functions of four independently sampled
joint-offset context channels. They are correlated with those offsets; this is
an explicit compatibility tradeoff that keeps actor, dynamics, replay and
critic dimensions unchanged. The factors, those four offsets, and center-of-mass
shifts remain fixed for each simulator world. Friction and other joint offsets
resample on episode resets.

Mass, inertia, subtree mass, inverse weights, actuator gains and force ranges use
per-world MuJoCo-Warp arrays. `set_const` recomputes mass-dependent quantities
after initialization. Every assignment starts from saved nominal parameters.
The actor never receives the physical scale factors as privileged inputs.

## Invocation

Use the verified training environment and its Torch/CUDA library path:

```sh
python scripts/rls/supermimic/continue_g1_stealth.py \
  --source-run /root/gpufree-data/runs/g1-stealth-keyboard-v3-extend10000 \
  --source-iterations 20000 \
  --source-pid SOURCE_PID --source-start-ticks SOURCE_START_TICKS \
  --gpu-lock /root/gpufree-data/work/trackerlab-supermimic/logs/.supermimic-terrain-gpu0.lock \
  --output /root/gpufree-data/runs/g1-stealth-upperbody-dr-v1
```

`status.json` records waiting, training, evaluation, rendering, export and
failure phases. Source, probe and final reviews retain first attempts without
outcome selection. Final reviews use two new simulator seeds. The stage exports
the constrained portable actor after training; physical deployment and motion
quality still require review.

# G1 Stealth: Free Arms, 20,000 Iterations

Owner-authorized experiment on November 8, 2026. This supersedes the pending
entry/velocity comparison. The old comparison exited with STOP and zero new
training iterations. This document describes the requested experiment, not a
completed or visually accepted policy.

## Changes

The command, clearance and AMP entry fade is 1.21 seconds, half the preceding
2.42-second preset. The fixed arm pose, entry waypoints and dedicated arm-posture
reward are disabled. All fourteen arm outputs are learned. The existing
pose-independent differentiable target filter remains, with a maximum target
rate of 2.4 rad/s and acceleration of 12 rad/s2. It has no pose attractor.
Thus 1.21 seconds is the task fade, not a guarantee that learned arms arrive at
a particular pose by that instant.

The existing source-derived waist target still ramps over 1.02 seconds and
holds afterwards, with the prepared waist Kp x4 / Kd x2 motor contract.
Twelve leg and fourteen arm outputs are learned; the three waist outputs
remain controlled by this explicit target.

The natural-arm expert pool restores chronological retargeted arm motion:

```text
data/style-research-20261107/wasd-natural-audit-v1/
  pools/g1/stealth-3da9aaa22c000d40df9f
```

Forward/backward experts are Aiming2 subject5 frames [2830,2891) and
[3620,3681); idle is Aiming2 subject3 [560,621). Each has 101 output frames at
50 Hz. Root, legs and waist match the earlier pool; fourteen arm channels differ.
The new pool requires a freshly source-bound waist JSON even though its target
angles are unchanged. Expert data remain outside Git.

AMP retains `postural_end_effectors`, including root-relative tracked-body
positions and wrist orientations. This experiment does not also introduce the
velocity-feature comparison. Task/quality/style reward coefficients, all nine
WASD modes, physical domain randomization, collisions and crouched release are
preserved from the previous entry preset.

## Initialization And Budget

Use the preserved waist-pilot iteration1500 checkpoint:

```text
/root/gpufree-data/runs/g1-stealth-visual-waist-pilot-20261108/training/iteration_001500.pt
sha256: 896cf1d3da76eb529b241d8bc986e35d66f4e02e7e9d730f7e8c8b0de685e8f6
```

Transfer actor weights and freeze saved input normalization. Reset only the
fourteen formerly overridden arm decoder rows; retain leg outputs. Fit fresh
WM dynamics for changed action distribution and waist gains. Changed experts
refit AMP. Critic and optimizers follow the existing warm-start behavior.

The budget is 20,000 new trainer iterations: 500 WM-only iterations followed
by 19,500 actor-enabled iterations. Record physics at 2,000 / 5,000 / 10,000 /
15,000 / 20,000, resuming the full saved learning state between milestones.
Resumption retains actor, WM, critic, discriminator and optimizer states; the
500-iteration warmup and arm-row reset occur only at the initial warm start.

## Operation

```sh
python scripts/rls/supermimic/prepare_stealth_waist_entry.py \
  --pool NATURAL_POOL --output NEW_WAIST_REFERENCE
bash scripts/rls/supermimic/run_g1_stealth_free_arms.sh \
  PYTHON NEW_RUN NATURAL_POOL SOURCE_1500 NEW_WAIST_REFERENCE
```

The repository GPU lock covers cooperating local runners. Exclusive admission
is the default. An optional explicit shared-capacity mode waits for at least
17,000 MiB free and caps this trainer's PyTorch CUDA allocator at 60% of total
GPU memory. The previous same-size run reserved 13.455 GiB; the observed 24 GiB
GPU had 17,199 MiB free with the unrelated job running. Sharing can lower speed.
The allocator cap excludes MuJoCo/Warp and other non-PyTorch allocations, so
the runner also watches total free memory and requests its own checkpointed
stop after two consecutive readings below 1,536 MiB.

An external scheduler's lifetime singleton lock is not an automatic per-job
handoff. It is only acquired when explicitly supplied for exclusive admission;
the capacity-sharing mode makes no exclusivity claim. A waiting controller is
not an active trainer. This experiment does not stop or change unrelated jobs.

Each milestone records all nine commands plus release/change sequences and
renders first attempts with front/side views. Preserve early falls. Inspect
forward entry, torso lean, arm expression, reverse foot lift, sliding, pivots,
combined commands and crouched release. The full 20,000 budget is authorized;
videos do not automatically grant visual acceptance or policy export.

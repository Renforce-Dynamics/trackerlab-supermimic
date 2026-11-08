# G1 Backward Stealth Style Trial

Owner review on November 8, 2026 rejected the 32000-iteration policy's
backward foot dragging and entry backward lean. The numeric command/fall guards
did not establish natural gait. That checkpoint remains an experimental baseline.

## Findings

The completed `g1-stealth-wasd-v2-20k-20261108` run used one 106-frame
Aiming2 subject-3 forward expert, including during reverse commands. Its
`postural_end_effectors` discriminator omits commanded root velocity and
yaw but retains joint, posture and endpoint histories. A negative velocity
goal therefore supplies no demonstrated reverse swing pattern.
This is a missing-data hypothesis, not proof that it explains every failure.

A forward-only motion prior can permit a learned reverse gait when its feature
invariances, task objective and physical exploration allow sufficient variation.
The absence of reverse demonstrations does not establish impossibility. The
current failed foot-lift measurements diagnose these policies, not all
forward-only AMP training. Data changes and velocity features remain separate
hypotheses requiring physical comparison.

The initial seed-101 release attempt has root Euler pitch from -8.17 to
+1.78 degrees over entry seconds [0, 4.84], followed by a minimum of
-14.44 degrees before second 6. Pitch is measured in the robot's forward/up
frame, with negative pitch indicating backward lean. Root pitch alone does
not characterize torso posture or balance.

## Primary Open-Source Comparison

These sources were read directly at the revisions below. No foreign source,
motion dataset, pretrained controller or PPO implementation was imported.

| Project | Revision | Inspected Evidence |
| --- | --- | --- |
| NVIDIA ASE | `6f9b4f1f289603eee6a4f45d082bc6e6d83fecec` | `ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml` explicitly samples `RL_Avatar_WalkBackward01_Motion.npy`, `WalkBackward02` and `RunBackward`. These are distinct backward demonstrations in a weighted dataset. |
| DeepMimic AMP | `1f915c52fcd4b95b5f5f15b759ae91bd81e9a801` | `args/train_amp_heading_humanoid3d_stealthy_args.txt` selects `data/datasets/humanoid3d_clips_stealthy.txt`, containing three stealth-walk clips, a task velocity goal and local-root AMP observations. Heading control alone does not demonstrate a fixed-facing reverse gait. |
| IsaacGymEnvs | `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` | `isaacgymenvs/cfg/task/HumanoidAMP.yaml` explicitly selects a reference motion. Its generic example does not provide evidence of backward-walking competence. |
| InstinctMJ SuperMimic | `03563ee02665f327425106372264070315673748` | `src/instinct_mj/rl/supermimic/style.py` learns a task-agnostic discriminator from real policy and reference windows; its style reward remains differentiable through imagined states. |
| OpenLegged SuperMimic AMP | `596dfe37e0f75769b635386cd589960cce7faaa2` | `src/assets/motions/g1/amp/WalkandRun/` contains backward walk/jog demonstrations. `supermimic/mdp/amp.py` samples adjacent expert frames and includes body velocities without command conditioning. `src/tasks/amp_loco/amp_env_cfg.py` includes negative vx commands and a contact-conditioned slip penalty. |
| Humanoid AMP MJLab | `80064abf5ab97279fccc0c4a209a0585d459930c` | `src/assets/motions/g1_seed/` contains a backward walk. `src/tasks/amp/config/g1/env_cfgs.py` permits negative vx and retains contact slip/torso terms while removing explicit clearance/gait rewards. Its AMP observation uses joint positions/velocities, showing that root-velocity features are not universal. |

Reproduction source roots:

```text
https://github.com/nv-tlabs/ASE/tree/6f9b4f1f289603eee6a4f45d082bc6e6d83fecec
https://github.com/xbpeng/DeepMimic/tree/1f915c52fcd4b95b5f5f15b759ae91bd81e9a801
https://github.com/isaac-sim/IsaacGymEnvs/tree/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3
https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748
https://github.com/OpenLegged/supermimic_AMP/tree/596dfe37e0f75769b635386cd589960cce7faaa2
https://github.com/yhx1203/humanoid_amp_mjlab/tree/80064abf5ab97279fccc0c4a209a0585d459930c
```

ASE includes an additional skill-latent mechanism. Its dataset membership is
evidence for backward demonstrations, not a claim that simply copying its
sampler reproduces its results. Mirroring left/right does not create a backward
example. This trial uses real chronological backward motion rather than
time reversal.

OpenLegged's listed evaluation scenarios do not include negative vx, and
neither G1 source establishes successful LAFAN aiming-style backward gait.
Their configuration/data coverage is not a backward-specific measured result.
The inspected implementations provide no reason to require a new gait-phase
clock, foot-airtime reward or command-conditioned discriminator for this trial.

## Bounded Data Adaptation

Run:

```text
/root/gpufree-data/runs/g1-stealth-lafan-reverse-5k-20261108
```

Warm-start source:

```text
/root/gpufree-data/runs/g1-stealth-wasd-v2-20k-20261108/training/iteration_032000.pt
SHA256 c5a5c80756691cbb361579eba4c845ed2914502c644d34aa383902f561b66f7d
```

Reuse the existing immutable, reviewed fixed-arm WASD pool:

```text
data/style-research-20261107/wasd-recovery-fixed-arms-v1/
  pools/g1/stealth-ad92cd79faafc566377d
manifest SHA256 69fea77f0ca082c167190df9848aca9e49a47b6c2b50a131c55cf64ff1b2d7f0
```

Its three equal-length clips each supply one third of complete AMP windows:

| Role | Original Aiming2 Source | Frames | Median Body vx |
| --- | --- | --- | ---: |
| Forward | `aiming2_subject5.bvh` | [2830, 2891) | +0.580 m/s |
| Backward | `aiming2_subject5.bvh` | [3620, 3681) | -0.501 m/s |
| Crouched idle | `aiming2_subject3.bvh` | [560, 621) | +0.022 m/s |

Each converted clip has 101 frames at 50 Hz. All 14 arm channels use the
existing accepted fixed pose; the original legs, waist and root remain.
The Aiming1 held-out CSV is unchanged. Source attribution, hashes and original
retarget receipts are retained in the pool; derivatives remain ignored/local.
The source gait speeds exceed the requested 0.25/-0.20 m/s command speeds.
No time scaling or root-velocity editing was introduced.

The existing recovery preset preserves network structure, WASD-v2 reward,
AMP weight 0.25, physical randomization, 50% reference resets and the exact
4.84-second entry/arm controller. It inherits actor and world-model weights.
Actor input means/variances are preserved and frozen. A new discriminator is
required because expert identity changes. The existing warm-start path resets
critic and optimizer states; actor LR is 2e-5 and command holds are 100--1000
steps. These differences from full resume are explicit experimental confounds.

The runner schedules 1000, 3000 and 5000 adaptation iterations. The first 500
iterations train only the world model, giving 4500 additional actor updates
at the final milestone. Each milestone must retain the existing forward
speed/fall guard before later training; cycle/release evaluations and first
attempt videos remain required. There is no automatic extension beyond 5000
or automatic visual approval.

At iteration 1000 the held-forward check recorded mean vx 0.168 m/s versus
the matched source 0.210 m/s, with zero falls in 24 episodes. It passed the
75%-retention/minimum-speed guard but exposes a forward-speed regression.
Backward swing and entry quality remain unaccepted pending recorded-trajectory
review. The first 500 actor updates do not establish final quality.

## Matched First-Review Gait Audit

CPU FK uses the original G1 XML SHA256
`76ad0c272806844755f35973e165d9a227039948edf7a903655d7e7c2e77c55c`,
MuJoCo 3.17.0, saved cycle seed-101 first attempts and no physics steps.
Both backward observations span episode seconds 5--20. A sustained lift
requires every foot collision sphere's bottom to exceed 2 cm for at least
0.08 seconds.

| Recording | Left / Right Sustained Lifts | Left / Right Maximum Whole-Foot Height |
| --- | --- | --- |
| Source 32000, cycle lane 1 | 0 / 0 | 1.17 / 0.70 cm |
| Adaptation 1000, cycle lane 1 | 0 / 0 | 0.76 / 1.18 cm |
| Actual backward expert, full 2 seconds | 1 / 2 | 4.76 / 5.47 cm |

Both policy recordings keep both feet within the 2 cm low region for the
entire measured interval. The expert has actual lifting, so the observed
absence of policy swing cannot be explained by a non-lifting reference.

Matched cycle lane-0 entry [0, 4.84] root pitch worsened from
[-8.99, +1.78] to [-14.99, +3.87] degrees. Torso backward lean projected along
root heading peaked at 24.54 versus 27.78 degrees. These are diagnostic
first-attempt measurements, not aggregate safety or anatomical validation.
The 1000 adaptation therefore has no visual approval.

Reports live under the local artifact directory:

```text
artifacts/g1-stealth-wasd-20261107/lafan-reverse-amp-5k-20261108/offline-gait-audit/
```

`audit_stealth_trajectory.py` reproduces whole-foot clearance, sustained
lifting, geometric support-pole material speed and root/torso pitch from
recorded qpos. Its source-comparison option reads explicit expert CSVs and
does not resample, re-ground or run the policy.

## Prepared Velocity-Feature Comparison

The candidate feature mode `postural_end_effectors_vx` appends one signed
root-local forward velocity, scaled by robot height, to the existing posture
and endpoint feature vector. Old feature modes and checkpoint interpretation
remain unchanged. Actor, world-model and critic input dimensions remain
unchanged; this is a discriminator-only observable, not a runtime motion input.

Identical local posture histories translated with different root velocities
are indistinguishable under the old posture feature construction. The extra
scalar can associate backward translation with the demonstrated reverse
joint history. This is a testable hypothesis motivated by OpenLegged's
velocity-bearing AMP, not a guaranteed repair. It also exposes the difference
between expert speed (about -0.50 m/s median) and the -0.20 m/s task goal.
With a forward-only expert pool, adding signed velocity can instead penalize
reverse motion more strongly. The prepared mode is therefore intended for
comparison with the explicit backward pool, not as a generic cure for a
forward-only prior.

Changing feature width requires a newly initialized/refit discriminator,
normalization and discriminator optimizer. It is not an admissible silent
resume. The running three-clip recovery experiment continues with its
original `postural_end_effectors` mode; preparation of the new mode does not
change or extend that queue. Startup lean remains a separate unresolved issue.

## Review Requirements

Review fixed first attempts, without outcome selection, for forward, S,
SA/SD, pivots and release. Compare whole-foot collider clearance, sustained
lift episodes, bilateral ground-drag duration, material-point support slip
and root/torso backward pitch during entry. A normal speed or low fall count
cannot approve a dragging gait. A new expert pool cannot be assumed to repair
the inherited arm/command entry controller without physical evidence.

CPU forward-kinematic diagnostics add no physical simulation steps and grant
no real-robot or deployment acceptance. Keep the baseline and all unsuccessful
trial outputs for matched comparisons.

# SuperMimic Command Locomotion With AMP

## Current G1 Trial And Migration

The November12,2026 G1 work uses the
[full-body stealth reference repair](g1_stealth_full_body.md), including the
owner-requested staggered idle hold. The earlier25.5kpolicy failed visual style
review despite improved command tracking and foot lifting.

The November13,2026 [contact-support trial](g1_stealth_contact_support.md)
changes one G1 gait-support approximation after the v2 run retained skating
despite passing command tracking. It keeps the reviewed references and weights.

The29command/style task definitions remain available. The historical v9batch
map and complete gait-foundation-v5 inputs described below remain unavailable in
the migrated31293workspace and current local checkout. Restore and verify their
inputs before a multi-robot campaign. G1 physical/visual acceptance precedes
other-robot training; registry presence and old smoke checks do not certify
learned performance or current-server readiness.

The November 4, 2026 command/style campaign replaces the retired
trajectory-tracking foundation route for new locomotion jobs.

The actor reads body observations, command goals and a local height scan.
Reference clips provide discriminator training data and optional random reset
poses during training. They are absent from actor inference. No PPO optimizer
is used in this campaign.

## Sources

Primary implementations inspected for this extension:

| Source | Revision | Contribution |
| --- | --- | --- |
| [InstinctMJ SuperMimic parkour](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic/parkour) | `03563ee02665f327425106372264070315673748` | World model, differentiable policy rollouts and real-data-only adversarial style learning |
| [InstinctMJ StyleReward](https://github.com/project-instinct/InstinctMJ/blob/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic/style.py) | same | Least-squares discriminator and quadratic style reward |
| [InstinctLab parkour AMP](https://github.com/project-instinct/InstinctLab/tree/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057/source/instinctlab/instinctlab/tasks/parkour) | `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057` | Terrain locomotion objectives and motion-prior task structure |
| [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour/tree/d2ffe27ba59a3229fad22a9fc94c38010bb1f519) | `d2ffe27ba59a3229fad22a9fc94c38010bb1f519` | Perception and terrain-curriculum comparison |
| [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/tree/94084601bacdf9cc3764b5c73daaeccae6035fac) | `94084601bacdf9cc3764b5c73daaeccae6035fac` | Aiming locomotion, obstacles and motion semantics |

InstinctMJ is a third-party SuperMimic extension. This repository does not
claim that the SuperMimic paper authors released an official AMP implementation.
Our integration implements the method using the existing TrackerLab networks
and MuJoCo-Warp runtime; it does not vendor InstinctMJ or Extreme Parkour code.
The cited Project Instinct and Extreme Parkour projects declare CC BY-NC 4.0;
their individual notices remain relevant to any future code reuse.

## Learning

Each full iteration collects real simulator transitions, updates the world
model, fits an AMP discriminator, differentiates an imagined rollout to update
the actor, and fits a GAE critic. The discriminator sees only detached real
transitions and ten-frame expert windows. No training sample crosses a
reference clip boundary.

The style reward is
`max(0, 1 - 0.25 * (D(window_t) - 1)^2)`.
The expert target is +1 and the policy target is -1. The discriminator uses a
ReLU MLP, AdamW, combined policy/expert input-gradient penalty, and the
weight-decay regularizers used by the inspected InstinctMJ configuration.

During actor updates, world-model, critic and discriminator parameters are
frozen; state/action gradients still flow through their computations.
The task's style coefficient multiplies the reward once per 20 ms policy
step, matching InstinctMJ. Terrain terminal compensation includes this
bounded positive reward so stopping at the goal remains preferable to
delaying completion.

The default `proprioceptive` discriminator features are projected gravity,
joint offsets, joint velocities, and body linear/angular velocities, over ten
consecutive frames. Joint velocity is scaled by 0.05. A full feature option additionally includes body
positions and terrain-relative height. The default avoids a fixed flat-ground
height target in the parkour discriminator. It still requires diverse motions
and physical testing; walking data alone cannot establish jumping quality.

## Actor Contract

For J joints, each observation contains `3J + 196` values:

- joint position relative to default and randomized offset: J
- joint velocity times 0.1: J (actor; AMP alone uses 0.05)
- body angular velocity times 0.25: 3
- projected gravity: 3
- preceding executed action: J
- body linear velocity divided by robot scale: 3
- local terrain scan: 187 (17 by 11)

The actor receives ten observation frames and twelve command-goal values.
Goals contain body vx/vy/yaw rate, root clearance, optional jump height and
clock, plus terrain navigation features. Flat velocity and stealth goals have
zero navigation features. No reference pose, reference phase, or future motion
frames enter either input.

`export_command_policy.py` exports the actor with its normalization to
TorchScript and records the exact feature order in `metadata.json`.
The exported module needs no discriminator, world model, or motion CSV.

Height scans and body velocity are simulator measurements in this campaign.
Real deployment requires a local height map and velocity estimator. This is a
height-map perception experiment, not an RGB-D policy reproduction.

Optional [phase timing](command_training_timing.md) separates command/AMP
work outside the ordinary throughput window. Enable it at run creation with
`--phase-profile-updates`; existing presets leave it disabled.

## Tasks And Queue

- 6 flat velocity tasks: `locomotion/{robot}/velocity`
- 18 existing terrain IDs: `locomotion/{robot}/{stairs,mountain,parkour}`
- 5 humanoid stealth tasks: `locomotion/{g1,a3,t1,bumi,pi_plus}/stealth`

Go2 uses quadruped references. Human held-object posture is not assigned to Go2.
The six velocity jobs run first. Dependent terrain/stealth jobs inherit actor
and dynamics from a velocity checkpoint that passes physical command tracking
and survival checks. The discriminator is restored only when expert identity
matches; a different style pool initializes a new discriminator.

```bash
python scripts/rls/supermimic/supervise_style.py \
  --style-map data/style-references-v9-stealth/style-map.json \
  --run-dir runs/supermimic-command-style-v4-20261105 \
  --iterations 10000 --max-rounds 3 --num-envs 8096
```

Each job receives 10,000 initial iterations. A failed evaluation queues another
10,000 iterations with actor, dynamics, critic, discriminator and optimizer
state retained, up to the configured round limit. Evaluation covers two
held-out seeds, saves fixed first-attempt trajectories, and renders a demo.
No outcome filtering is used when choosing demo attempts.

The physical gate combines completed episodes, success/fall rates, velocity
and yaw-rate errors, foot geometry and a minimum style score. A passing numeric
gate enters `awaiting_visual_review`; it cannot release dependent terrain jobs
or exports. A reviewer records `visual_review.json` with a reviewer, notes, and
SHA-256 hashes of the exact checkpoint and recorded first-attempt trajectory.
Changing either artifact invalidates that review.

The old `supervise_gait.py` campaign is stopped. Its checkpoints do not count
as command-controlled training progress.

## Motion Data

Ordinary walking uses the audited six-robot recordings under
`data/gait-foundation-v5`, with bounded LAFAN walk-family crops added for
humanoids. Terrain style data are separately selected from
locomotion/jump/obstacle recordings; the exact source files, crop intervals,
hashes and split membership are recorded in generated manifests.

The current humanoid terrain pools include walking/running, jumping and
original Obstacles records. Go2 additionally uses FK-verified Kine2Go run,
turn, strafe and forward-jump clips; forward jump B and run 02 are held out.
Aiming data currently use four training clips (560 frames) from subjects
2, 3 and 5 of take 2 and one held-out take 1 clip (400 frames).
Stealth diversity and command generalization remain limited.

Default command actions are bounded at magnitude 20 before joint-limit
clamping, matching the source controller's action range. A magnitude-four
cap excluded valid Pi+/T1 and held-object arm poses under their vendor
action scales. Optional reset frames are checked with actual MuJoCo foot
collider distances, rather than assuming a foot link origin lies on the floor.

The command-style map is published under the ignored
`data/style-references-v9-stealth/` tree. Moving-reference envelopes and the AMP
history sampler respect clip boundaries. Velocity AMP requires moving frames
in at least eight of ten samples and a moving window mean. Parkour AMP keeps
all complete windows, including airborne and takeoff/landing frames.

The LAFAN stealth extension preserves the original Aiming1 holdout and adds
three slower, source-range-pinned Aiming2 windows to each humanoid pool. Each
window passed the retargeted low-profile screen; the source-direction labels
are provenance, and physical performance remains subject to policy evaluation.

For a single-reference G1 stealth pilot, `--stealth-command 0.25 0 0` fixes the
training body-frame command at 0.25 m/s forward, zero lateral speed, and zero
yaw rate. The command is recorded in the checkpoint and held constant through
resampling. Use a fresh run directory when changing this command.
The single-clip G1 pilot uses a 0.92 root-clearance ratio and a stronger
geometry-based support-foot slip penalty; these settings are scoped to that
pilot preset and do not change other robot profiles.

The G1 keyboard preset samples five explicit modes: forward (40%), left/right
pivot in place (15% each), and forward arcs to the left/right (15% each). Forward is
`(vx=.25, vy=0, yaw=0)` in SI units; pivots use zero translation and yaw
`+/-0.55 rad/s`; arcs preserve `vx=.25 m/s` with yaw `+/-0.45 rad/s`.
The exported policy receives body-frame `vx`, `vy`, and yaw-rate fields at
every control step, so a runtime can update them directly while keys are held.
The G1 pilot's postural AMP features retain gravity, root clearance, joint
pose, and joint velocity while allowing commanded root translation and yaw to
vary independently of the single forward expert clip.
The pilot enables robot self-contact geometry; reset clips with hand/body
penetration are screened before reference-based episode starts. A keyboard
pilot warm start may retain the matching actor and posture discriminator while
resetting world-model/critic state when collision or solver physics changes.
Foot-slide and narrow-stance weights are strengthened for this G1 pilot preset.
The keyboard collision preset starts from the robot's default standing state;
the LAFAN clip remains an AMP expert source and does not seed colliding poses.

The opt-in [G1 entry experiment](g1_stealth_entry.md) extends this baseline with
geometric AMP features and separate entry quality checks. Its corrected arm
targets require a new immutable derivative pool matching the AMP expert, while
the original train clip and Aiming1 holdout remain untouched.

LAFAN1's original archive contains:

```text
aiming1_subject1.bvh
aiming1_subject4.bvh
aiming2_subject2.bvh
aiming2_subject3.bvh
aiming2_subject5.bvh
```

The public 40-clip G1 retarget subset omits these Aiming files. Stealth therefore
uses original BVH-to-G1 retargeting, with upper-body posture retained. Only
body motion is represented; no carried weapon object or weapon operation is
modeled. Source takes stay indivisible across training/testing, including
different subjects recorded simultaneously.

LAFAN1 data declare CC BY-NC-ND 4.0. Original files, local research derivatives,
and generated references are kept under ignored `data/` directories and are
excluded from code pushes. The preparation scripts preserve source hashes
and leave originals unchanged.

## Speed And Qualification

`training_speed_summary.json` reports full iterations after world-model warmup
and compilation, including real simulation, world-model updates, AMP updates,
actor BPTT and critic minibatches. The historical teleoperation reference is
2.47115 iterations/s; measured results for the new objective must be reported
separately. Small smoke tests do not establish full-size throughput or policy
quality.

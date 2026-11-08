# G1 Stealth Training Recipe

Recipe inspected November 8, 2026. Source synchronization does not constitute
new training or visual acceptance. The completed waist-pilot iteration1500
remains rejected; the new entry/velocity comparison is pending physical runs.
See [comparison and operating instructions](g1_stealth_velocity_visual.md).

## Data

The active immutable local AMP pool is:

```text
data/style-research-20261107/wasd-recovery-fixed-arms-v1/
  pools/g1/stealth-ad92cd79faafc566377d
```

| Role | Original LAFAN1 source | Original frames [start,stop) | Output | Median body vx |
| --- | --- | --- | --- | ---: |
| Forward | aiming2_subject5.bvh | [2830,2891) | 101 frames / 50 Hz | +0.580 m/s |
| Backward | aiming2_subject5.bvh | [3620,3681) | 101 frames / 50 Hz | -0.501 m/s |
| Crouched idle | aiming2_subject3.bvh | [560,621) | 101 frames / 50 Hz | +0.022 m/s |

Each clip lasts2seconds between endpoints. The303 training frames contain
original chronological, retargeted root/leg/waist motion. Fourteen arm channels
are explicitly replaced by the previously geometry-reviewed static pose. This
is a local LAFAN derivative, not unmodified full-body mocap. Aiming1 remains
held out and byte-identical. Source licenses apply; clips/checkpoints stay in
ignored local data/run paths and are excluded from Git pushes.

Ten-frame histories span180ms. Each training clip supplies92 valid windows;
uniform complete-history sampling assigns one third to each role. There is no
time reversal, synthetic backward playback, source speed editing or gait-phase
reference input. The source speeds exceed requested+0.25/-0.20m/s; this mismatch
is an explicit risk for the new velocity-bearing discriminator.

## Learner

The algorithm is the existing SuperMimic model-based actor/critic with AMP.
There is no PPO stage in this recipe.

1. Run1024 MuJoCo-Warp worlds; four5ms physics steps per20ms action.
2. Collect real states and executed/clamped actions, including upper-body output
   transforms. Store them in physical replay; fit the WM on eight-step windows.
3. Fit a least-squares AMP discriminator on real policy histories and expert
   histories. Imagined states are excluded from discriminator training data.
4. Optimize the actor through32 imagined steps (0.64seconds), differentiating
   task/style returns through the frozen WM and discriminator. Fit the critic
   to GAE/lambda-return targets. Gamma=.99, lambda=.95.

The candidate preset uses ten WM minibatches of2048, actor batch4096, five
critic epochs with four minibatches, and two AMP updates of2048 per actor-enabled
iteration. Actor LR=2e-5; WM/critic LR=2e-4. Physical action/control transforms
also run in imagination. Domain context is privileged to training, not a direct
actor command.

Both comparison trials warm-start the same prior1500 actor and freeze its input
normalization. New waist gains change physical dynamics, so the old WM is not
restored. Each bounded trial fits a fresh WM for500 iterations, followed by1000
actor-enabled iterations. Entry-only restores matching AMP; velocity mode
changes feature width and refits AMP normalization/discriminator/optimizer.
Further iterations require actual visual evidence; no automatic policy export.

Training resets mix50% grounded standing and50% screened expert states.
Commands are sampled independently of expert resets, so initial command/velocity
agreement is not guaranteed. Evaluation starts from grounded standing and uses
no reference initialization. Episodes last at most20seconds during training.

## Rewards

The actor uses `task + locomotion_quality + arm_posture + 0.25 * gate * AMP`.
The gate requires genuine history and uses the existing quintic entry blend.
These are per-action values. Flat stealth disables navigation progress, lane
and global-heading terms. Command tracking is body-frame velocity plus yaw rate.

| Term | Existing formula/weight |
| --- | --- |
| Planar velocity | `0.04*exp(-||v_xy-c_xy||^2/sigma_v^2)`; sigma_v=.30 moving, .10 stationary |
| Yaw rate | `0.04*exp(-(w_z-c_w)^2/.40^2)` |
| Alive | +.06 per step |
| Base tilt | `-.06*(1-R_zz)` |
| Root clearance | `-.004*((height-target)/(.2*.76))^2` |
| Joint posture | `-.0006*mean((q-default_q)^2)` |
| Action change | `-.0008*mean((a-a_prev)^2)` |
| Joint velocity | `-.00002*mean(qd^2)` |
| Joint soft limit | `-.04*sum(relu(q-soft_hi)+relu(soft_lo-q))` |
| Directional waiting | Up to-.01 when a moving command has insufficient projected progress |
| Foot sliding | `-.04*support_weighted_material_sole_speed` |
| Supporting foot tilt | `-.008*mean(tilt_xy_squared*support)` |
| Feet too close | `-.08*(relu(.10-signed_lateral_spacing)/.10)^2` |
| Joint acceleration | `-2.5e-9*mean(qdd^2)` |
| Nominal motor effort proxy | `-3e-9*sum(torque^2)` |
| Joint overspeed | `-.02*mean(relu(abs(qd)-.9*velocity_limit)^2)` |
| Base roll/pitch rate | `-.001*(body_wx^2+body_wy^2)` |
| Arm target error | `-.004*mean(smooth_L1((q_arm-target)/.25))` |
| Arm speed | `-.001*mean((qd_arm/max_arm_rate)^2)` |
| Fall | -.5 terminal penalty; imagined termination uses a differentiable approximation |
| AMP | `.25*max(0,1-.25*(D-1)^2)` after entry/history gates |

There is no explicit step-phase, foot-airtime, fixed swing-height or per-frame
reference tracking reward. Stepping style comes from AMP. Geometric sole support
is detached for actor differentiation and is a contact proxy, not measured force.
The effort penalty remains a nominal proxy under randomized gains; its small
coefficient and that limitation are documented rather than changed in this trial.

The old AMP feature includes gravity, height, q/qd, root-local body positions
and hand-link rotations. The new candidate additionally includes measured
signed body-forward velocity. It distinguishes an idle-like pose sliding
backward from the same local pose at rest; improvement remains a hypothesis.
The feature contains actual velocity, never the desired command as a substitute.

## Runtime Inputs And Outputs

The actor takes observation history `[B,10,283]` and current goal `[B,12]`.

| Observation block per frame | Width |
| --- | ---: |
| Joint position minus default and calibrated offset | 29 |
| Joint velocity scaled by0.1 | 29 |
| Body angular velocity scaled by0.25 | 3 |
| Projected gravity | 3 |
| Previous executed normalized action | 29 |
| Body linear velocity / robot-height scale | 3 |
| Local height scan (17x11) | 187 |

The12 goal fields are vx, vy, yaw rate, clearance, jump height, jump pulse,
entry clock, lane error, remaining world-X distance, two heading fields, and
terrain-navigation flag. In flat stealth jump and navigation fields are zero;
clearance remains0.6992m after entry. The host maps keys to twist and maintains
the skill-entry clock across releases/changes. Ten-frame history spans180ms.
The training height scan uses analytic simulator terrain; real sensing/state
estimation and hardware execution are outside this visual trial.

The raw actor produces29 normalized actions. Fourteen arm targets follow
embedded entry waypoints then the held pose; three waist targets follow the
source-derived1.02second ramp then the held pose. Twelve leg outputs remain
learned. Thus this is learned lower-body locomotion with explicit upper-body
control, not a fully learned29-DOF transition. The final action gives targets:

```text
q_target = clip(default_q + joint_offset + action_scale * clip(action,-20,20), joint_limits)
torque = clip(Kp * (q_target - q) - Kd * qd, motor_effort_limits)
```

Inference requires neither the source motion nor WM/discriminator/critic.
It outputs joint-position targets through the controller, not Cartesian paths
or direct task-space velocity commands. The new candidate uses2.41second arm
entry /2.42second command-clearance entry, with waist Kp x4 and Kd x2. Leg/arm
motor gains and motor effort limits remain unchanged. Wrist meshes have no
controlled fingers, and LAFAN lacks measured finger/thumb landmarks.

## Controls, Randomization And Review

W=+0.25m/s; S=-0.20m/s. A/D=+/-0.55rad/s in place, +/-0.45rad/s with W/S.
All nine combinations including idle are sampled, with holds2..20seconds.
Opposing keys cancel per axis. Release keeps crouched posture and completed
entry time; a host state machine selects a separate standing policy.

Physical variation includes friction.3..1.2, joint offsets+/-0.01rad, COM shifts,
mass/inertia+/-10%, and Kp/Kd+/-15%. Four privileged offset channels encode
correlated mass/motor factors. The advertised effort factor is.9..1.1, but the
retained XML joint cap limits effective effort to at most nominal. This existing
limit is preserved and explicitly disclosed. Observation noise is enabled.

Review uses seeds101/103, fixed first attempts for every command, and a42second
release/change sequence. Front/side recordings include commands and retain early
falls. Natural forward entry, arm expression, torso alignment, reverse foot lift,
slip, pivots/arcs and crouched release all matter. WM loss, AMP score and zero
falls alone do not grant visual acceptance.

# G1 Velocity-Led Stealth Repair

The previous combined recipe learned a crouched stationary solution. A frozen
scorer comparison showed that this behavior earned more combined per-step
reward than the earlier moving policy. This opt-in recipe gives velocity
commands priority while retaining the same learned arms, waist and motion prior.

Use task `locomotion/g1/stealth` with
`scripts/rls/supermimic/configs/command_stealth_velocity_led.json`.
No additional task ID or runtime motion input is introduced.

## Objective

WASD reward version3 changes the translation maximum from0.04to0.12 and yaw
maximum from0.04to0.08. Translation widths are0.15m/s while moving and0.10m/s
while stationary; yaw width is0.35rad/s. Directional progress is capped at0.04
once requested speed is reached. Translation and turning each have a bounded
0.04waiting penalty. Success thresholds are unchanged.

AMP weight changes from0.25to0.05. Its existing expert data, endpoint features,
signed rootvx feature, optimizer and normalization are retained when their
hash matches. It remains a style prior; forward/reverse/turn/idle commands
remain the task objective. Motion references are absent at inference.

Optional `stealth_gait_weight=0.02` adds a nonpositive command-gated sole cost.
Whole-sole clearance controls smooth support preference. A supported foot and
a translating lifted opposite foot reduce cost. Static single-leg poses gain
no swing relief, and opposite point velocities from pure foot rotation cancel.
The both-air penalty includes the maximum sole speed when all support points
cross the existing2cm slip cutoff, so takeoff cannot simply remove slip cost.
This compensation remains active during entry; other gait shaping fades in
with the command. Idle has zero gait cost.

This state-only shaping does not prescribe left/right phase or guarantee
alternation. Repeated stepping, heel/toe behavior, sliding and foot fidgeting
must be reviewed in recorded physical rollouts.

The skill-entry command and torso-guidance fade use0.60s instead of1.21s.
The learned-arm limits remain2.4rad/s and12rad/s2. The existing small entry
torso cost remains0.008and vanishes after entry. All29joint outputs remain
learned; release selects zero twist while retaining crouched stealth.

## Initialization And Runtime

Use a fresh warm-start from the previous combined iteration10500checkpoint,
which still moved forward/backward. This source had overspeed and idle drift;
it is an initialization, not a visually accepted result.

- Transfer all actor outputs, compatible WM and saved normalization.
- Freeze actor normalization; retain nominal randomized physical contracts.
- Initialize a fresh critic and actor/WM/critic optimizers for the changed reward.
- Restore AMP only when its expert identity matches.
- Preserve500world-only steps and50000new actor updates.
- Subsequent stages use full resume and reject changed reward, gait and entry.

`run_g1_stealth_velocity_led.sh` requires main and uses the existing controller,
GPU and scheduler locks. It selects only the explicit repair preset.
Reviews at2500,5500,10500,20500,30500,40500,50500record held forward, allnine
WASD modes and release commands. Quality failures remain recorded, with
visual acceptance required before any export. STOP/runtime/input failures
retain checkpointed terminal receipts.

## Offline Evidence

Same recorded first worlds,5..18s,130windows per command; root velocities
reconstructed from pose differences. With frozen30500AMP:

| Requested behavior | Earlier moving policy | Later stationary policy |
| --- | --- | --- |
| Forward, old objective | 0.2156 | 0.3183 |
| Forward, repaired objective | 0.2009 | 0.1297 |
| Backward, old objective | 0.1799 | 0.3276 |
| Backward, repaired objective | 0.1549 | 0.1389 |
| Idle, repaired objective | 0.1205 drifting | 0.2918 stationary |

The desired ordering is restored on these examples. This is a reward audit,
not evidence that the new policy is natural or successful. Fixed moving WM
validation and first-attempt video inspection remain necessary.

Reference: the inspected Project Instinct/InstinctMJ SuperMimic parkour
revision03563ee02665f327425106372264070315673748 uses real-only AMP updates,
differentiable imagined rewards and command-gated foot support/air-time terms.
This repair uses state-only geometry to avoid introducing unobserved timer
state into the current actor and critic.

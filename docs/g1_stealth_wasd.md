# G1 WASD Stealth Locomotion

## Control Contract

The exported policy accepts the existing body-frame velocity and yaw-rate
goals. WASD maps to nine trained modes:

| Keys | Mode | Target `(vx, vy, yaw rate)` |
| --- | --- | --- |
| W | Forward | `(0.25, 0, 0)` |
| S | Backward | `(-0.20, 0, 0)` |
| A / D | Turn in place | `(0, 0, +/-0.55)` |
| W+A / W+D | Forward arc | `(0.25, 0, +/-0.45)` |
| S+A / S+D | Reverse arc | `(-0.20, 0, +/-0.45)` |
| No keys | Stealth idle | `(0, 0, 0)` |

Releasing movement keys changes only the velocity and yaw-rate goals. The
clearance goal remains crouched, and the entry clock continues to its completed
value. The stealth policy has no stand-up command. An external state machine
selects the standing policy when standing is requested.

The command envelope, exported metadata, trainer, evaluator, and runtime mode
table share the same versioned mapping. Opposing translation keys and opposing
turn keys cancel independently.

## AMP Experts

`data/style-research-20261107/wasd-natural-audit-v1/` is a local-only,
immutable G1 derivative. Its three equal-length Aiming2 clips retain original
chronology:

- Subject 5 `[2830, 2891)`: low-profile forward motion.
- Subject 5 `[3620, 3681)`: low-profile backward motion.
- Subject 3 `[560, 621)`: low-motion crouched idle.

The AMP pool samples one third of its complete history windows from each
training clip. The original Aiming1 test bytes are inherited unchanged from
the prior verified pool. The expert preparation report distinguishes source
kinematic screening from physical policy validation.

Training uses `postural_end_effectors` features, including root-local tracked
body positions and the G1 wrist-yaw link rotations. The training objective
learns source-driven arms from these clips; it does not apply the previous
static upper-body lock. LAFAN's Hand End Sites contain no measured fingers or
thumb, and its zero-length endpoints leave palm roll unobserved. Wrist
orientation remains a documented retarget convention and fixed robot-hand
geometry remains a visual limitation.

All 303 raw retargeted training frames passed the local geometry screen:
no physical self-penetration or negative audited hand/body mesh distance.
The largest normalized arm target is 14.794, within the existing action
limit of 20. This checks expert poses, not learned rollout trajectories.

## Training And Review

`configs/command_stealth_wasd_natural.json` sets a 20,000-iteration run with a
10,000-iteration G1 stealth actor warm start, 1,024 MuJoCo-Warp worlds,
physical domain randomization, grounded standing starts, a 4.84-second
velocity/clearance entry ramp, G1 sole-slip regularization, and command holds
up to 20 seconds. AMP experts are training-only; the exported policy does not
load motion files, the discriminator, or the world model at runtime.

The initial 1,000-iteration adaptation is reviewed before a longer continuation.
The actor and compatible world-model weights transfer; the critic, optimizers,
and AMP discriminator for the new expert pool start fresh.
Review each saved milestone using fixed-mode and nine-mode-cycle evaluation.
The command-switch trajectory exercises forward, release-to-idle, backward,
forward arcs, and release-to-idle without resetting the entry clock. Render
first attempts from every mode and inspect balance, self-contact, sole slip,
entry continuity, idle posture, and upper-body style before selecting a
deployment checkpoint.

The runner accepts an explicit list of milestones:

```sh
bash scripts/rls/supermimic/run_g1_wasd.sh \
  "$PYTHON" "$RUN" "$POOL" "$INITIAL_CHECKPOINT" 1000

# After reviewing the first adaptation:
bash scripts/rls/supermimic/run_g1_wasd.sh \
  "$PYTHON" "$RUN" "$POOL" "$INITIAL_CHECKPOINT" 4000 10000 20000
```

The target 2.5 iterations per second is measured after compilation with the
trainer's completed-update speed probe. The historical tele result used
different hardware and workload, so the measured rate and comparison need to
remain explicit in each run report.

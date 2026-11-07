# G1 Original-Objective Extension

The owner requested additional training of the existing gait and an explanation
of end-effector handling, command coverage, slow entry and stiff walking.
This extension changes the training budget and review schedule only.
It uses full resume from the original entry-v4 iteration-10000 checkpoint.
No new reward, controller, expert mixture or network configuration is added.

## What The Original Baseline Learned

- SuperMimic + adversarial motion prior, 1024 worlds at 50 Hz.
- One 106-frame Aiming2 derivative, ten-frame AMP windows.
- Body-local tracked-body positions, including feet and hands, and wrist-link
  orientation are part of the postural/end-effector discriminator input.
- These are style features, not a per-frame Cartesian reference-tracking reward.
- Arm targets follow a fixed five-waypoint entry and then a fixed final pose.
  Zero arm residual means the actor cannot learn a different arm pose.
- Five command modes: W, A, D, WA and WD. Reverse, reverse arcs and no-key
  crouched idle are not part of this baseline's command sampling.
- Standing is reserved for a separate host state machine in the final design.

The existing entry schedule has four quintic segments, each with zero velocity
and acceleration at its endpoints. Arm entry takes 4.82 seconds, and commanded
velocity/yaw/height blend over 4.84 seconds. More policy updates cannot shorten
that authored schedule. A faster entry requires an independently audited path
or timing change, checked for physical balance and collision margins.

## Additional Training

The runner keeps the original `command_stealth_entry.json`, original expert pool
and `upper-body-reference.json`. Full actor, world, critic, discriminator and
optimizer states transfer. The command-time filter remains disabled: the bounded
comparison did not demonstrate a better gait with it enabled.

```sh
bash scripts/rls/supermimic/run_g1_entry_extension.sh \
  "$PYTHON" "$OUTPUT" "$SOURCE_10000" "$ORIGINAL_POOL" \
  "$BASELINE_FORWARD_JSON" "$HELDOUT_BANK" "$BANK_PROTOCOL"
```

The milestones are cumulative 12000, 15000 and 20000, up to 10000 additional
policy updates. Every milestone runs the same held-forward physical review,
all five fixed command modes with first-attempt videos, and 1/8/32-step WM
validation on the same held-out bank. The existing forward-retention gate
prevents automatic continuation if speed/fall thresholds fail. The final
milestone adds a second seed. Gate success alone does not certify motion style,
turning in place, reverse commands or no-key crouch.

The original checkpoint and failed experiments remain preserved. The separate
entry-timing review does not alter this run. Full WASD and relaxed, natural arm
motion require later isolated changes supported by the resulting evaluations.

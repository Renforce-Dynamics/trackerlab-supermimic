# Evidence-Based G1 Continuation

The owner requires changes grounded in the existing working policy, its final
WASD contract, and a verified public SuperMimic+AMP implementation. This batch
tests one training-behavior change rather than a new multi-setting recipe.

## Retained Contract

The final target remains G1 stealth locomotion with forward/reverse, turning
in place, translation plus turning, and zero commanded twist on key release.
Key release keeps the stealth posture; standing is a separate host-state-machine
policy. Runtime receives observations and commands, without reference playback.

The controlled comparison uses the existing entry-v4 iteration-10000
checkpoint and its exact single Aiming2 expert pool, accepted arm trajectory,
five-mode forward/turning command sampler, physical randomization, reward
coefficients and network sizes. This scope verifies preservation of its prior
abilities before extending commands again. Reverse and idle acceptance remain
outstanding and are not claimed by this comparison.

Both cases use full `--resume`, restoring actor, world model, critic,
discriminator, normalizers and optimizer states. No warm-start resets or arm-row
resets are performed. Actor normalization is already frozen by the source's
count above its existing `2e8` limit. Mean/variance and counts are preserved.
Each case starts independently from iteration 10000 and adds 500 updates.

## One Changed Behavior

Primary source: `project-instinct/InstinctMJ`, commit
`03563ee02665f327425106372264070315673748`, function
`ParkourSuperMimicRunner.sample_valid_starts`, in
`src/instinct_mj/rl/supermimic/parkour/runner.py`.
Its configuration sets `command_margin_steps=2`.

The source selects imagined-rollout starts only when the current command
continues beyond the horizon plus two control steps. The local implementation
records the number of control steps remaining at the pre-action state, stores
it in opt-in replay metadata, and applies the equivalent strict inequality:

```text
command_time_left_steps > horizon + command_margin_steps
```

For the existing horizon of 32, a remaining duration of 34 steps is excluded;
35 is admitted. WM training still uses recorded actions across command changes
and does not apply this actor-start restriction.

The control omits `--command-margin-steps`; old checkpoints retain the disabled
behavior. The treatment explicitly passes `--command-margin-steps 2`.
Subsequent resumes inherit the saved margin when no override is supplied.
An explicit change is recorded in `resume_receipt`, alongside the restored
source checkpoint and learning-state status.

## Validation

Each case has the same 24-world held-forward physical evaluation and original
first-attempt video. `check_wasd_recovery.py` compares speed and falls to the
existing exact-source baseline.

`validate_command_world.py` evaluates the old and candidate WMs on the same
recorded held-out physical trajectories at 1, 8 and 32 steps. It records the
bank and checkpoint hashes, uses one reference-model normalization for both
models, and excludes windows crossing a physical reset. This diagnostic does
not alter training or introduce a new model-loss weight.

The preserved bank comes from the measured seed-107 old-policy trajectory
audit, 90 worlds with fixed commands. These data are diagnostic-only and are
not inserted into replay. A smoother online training scalar alone does not
qualify either case.

The following remain hypotheses for later isolated comparisons:

- Matching reference reset direction and speed to requested commands.
- Staged expansion to reverse and crouched idle while retaining forward gait.
- Foot-contact/failure prediction heads and failure-focused replay sampling.
- Discriminator warmup scheduling and source-specific motion coverage.

No reward retuning, contact-head implementation, new expert mixture, or network
expansion is part of this controlled comparison.

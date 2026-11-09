# G1 Stealth: Signed-Velocity Style Continuation

The owner accepted the improved entry but rejected the shuffling gait.
The completed20kpolicy is preserved as the comparison checkpoint. This bounded
experiment changes AMP observability while retaining entry and learned control.

## Scope

The existing `postural_end_effectors_vx` feature appends measured root-local
signed forward velocity. The previous posture-only feature gave identical
style inputs to idle and the same idle history translated at+0.25/-0.20m/s.
The frozen20kdiscriminator scored all three at0.99999. Reference forward/backward
motions have clear foot lift, while the completed policy has little sustained
whole-sole clearance. These facts motivate the ablation; better gait is unproven.

Keep the1.21s command/clearance/style fade,1.02s waist ramp, waist motor gains,
learned arms, arm rate/acceleration limits, actor input normalization, nine WASD
commands, crouched release, randomization, batch sizes and every reward weight.
The natural chronological expert clips and their timing remain unchanged.
Their typical+0.58/-0.50m/s speeds exceed the commanded+.25/-.20, an explicit
risk to evaluate rather than conceal by changing source motion.

## State Transfer

Use `--resume` with `--allow-style-feature-change` in a new run directory.
Only G1 stealth posture AMP to signed-vx AMP is accepted. Reconstructing the
original feature-prefix fingerprint verifies that clips, layout and all prior
features are identical. All actor, WM and critic weights and optimizer states
restore in full. Their action controllers and physical contracts retain the
ordinary strict resume checks.

The new discriminator, its normalizer and optimizer are initialized afresh.
The actor runs without updates while200AMP optimizer steps fit real-policy
histories against experts. WM/critic/actor optimizer steps are absent during
this bootstrap; WM normalization is held during its additional collection.
The existing actor normalization remains frozen throughout.
Ordinary same-feature resume restores AMP and skips this bootstrap.

Preset: `scripts/rls/supermimic/configs/command_stealth_velocity_style.json`.
Incremental budget:2000actor-enabled iterations, absolute20001..22000,
plus200explicit AMP-only bootstrap updates. This is a bounded visual test,
with no automatic extension or policy export.

```sh
python scripts/rls/supermimic/train_terrain.py \
  --task locomotion/g1/stealth \
  --config scripts/rls/supermimic/configs/command_stealth_velocity_style.json \
  --run-dir NEW_RUN/training --resume SOURCE_20000 \
  --allow-style-feature-change --style-bootstrap-updates 200 \
  --amp-motion-config NATURAL_POOL/train.json --waist-entry-reference WAIST_JSON \
  --iterations 22000 --buffer-steps 500 --keep-every 500 --ckpt-every 250
```

Physical reviews use the same seed101, nine held commands and42s release
sequence as20k. Measure actual sole clearance, support slip, command response,
entry posture, arm movement and falls. Retain the first attempts and any failures.
The badminton scheduler remains stopped under owner instruction.

## Loss Reporting

Historical CSV `wm_loss` retains its summed-eight-step definition. Stdout now
also reports `wm_per_step=wm_loss/wm_window`. Optimized losses and gradients
are unchanged; existing logs are not rewritten. The previous0.198summed loss
corresponds to0.0248per-step time reduction. Tele has different channel floors,
so this reduction alone does not establish matched-model accuracy.

## CI Repair

The observed GitHub failure at3b8e31e included missingpytest and renderer
dependencies, plus platform-sensitive synthetic feature fingerprints. The
configuration description command also loaded optional simulator dependencies.
The complete CPU workflow must be checked on GitHub after the repairs; local
test success alone is not a completed CI verification.

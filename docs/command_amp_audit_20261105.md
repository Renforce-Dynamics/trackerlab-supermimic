# Command AMP Audit, November 5, 2026

The owner rejected the command-v1 videos. The campaign was stopped at
G1 10,000, Go2 10,000 and A3 2,416 updates. Completed numeric gates did not
establish acceptable motion appearance. The owner authorized removal of old
training outputs and a fresh campaign after correction.

## Primary Source

Inspected Project Instinct's
[InstinctMJ](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic/parkour),
revision `03563ee02665f327425106372264070315673748`. Files reviewed:
`style.py`, `parkour/config.py`, `parkour/mdp.py`, `parkour/runner.py` and
`tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`.

This is a third-party SuperMimic extension. Its configured parkour AMP has
ten frames, a `[1024,512]` ReLU discriminator, two 2,048-sample discriminator
updates, recent real policy windows, combined policy/expert gradient penalty
and a 0.25 style coefficient added per policy step.

## Confirmed Differences In Retired Training

- The prior saw two states separated by 20 ms, versus 180 ms of context
  in a ten-frame, 50 Hz window.
- The prior was multiplied by 0.02 again. G1's old maximum style contribution
  was 0.01 per step, versus 0.25 in the inspected upstream configuration.
- The discriminator was `[256,128]` ELU, with one 1,024-sample update and
  expert-only input gradient penalty.
- Command success did not measure foot slip, foot spacing or acceleration.
- G1's five training clips totaled 4,092 frames. Their mean speeds were
  approximately 0.24--0.55 m/s, predominantly forward. Commands additionally
  demanded reverse, sideways and fast turns.
- Go2's three training gait clips totaled 1,690 frames and averaged
  approximately 0.56--0.72 m/s. Its scaled forward commands topped out near
  0.43 m/s. Recorded velocity and finite-difference pose derivatives agreed.

These observations identify implementation and data-distribution problems.
Their individual contribution to appearance still requires physical
experiments; no single change is claimed to prove a better gait.

## Corrections And Remaining Differences

Ten-frame experts remain within one clip. Real histories exclude resets and
incomplete histories; actor imagination requires complete replay histories.
Style is a per-step coefficient. The prior uses ReLU, AdamW, both-side
gradient penalty and explicit regularization. Squared-gradient penalty
avoids the undefined second derivative of the L2 norm at zero.

New checkpoints declare activation, hidden widths, temporal length and
optimizer identity; scorer construction rejects missing or conflicting
metadata. Old checkpoints are incompatible with objective 4 and AMP
checkpoint format 2.

The default foundation command envelope is the 10th--90th percentile of
expert body-frame velocities in SI units. Reference resets use compatible
commands. This limits the current foundation's demonstrated coverage;
reverse/sideways/stop capability is not inferred from forward-gait results.
Go2's command pool adds verified native run/turn/strafe clips, excluding jumps.
Terrain goals retain their navigation semantics.

Reference-free foot-slide, foot-tilt, close-feet, joint-acceleration, motor
and joint-speed penalties use the same function in physics and imagination.
Support is estimated from geometry, not measured contact forces. Physical
evaluation reports those measurements independently of discriminator outputs.

This remains an adaptation: actor history is ten frames, actor qd scale is
0.1 and AMP qd scale is 0.05. AMP has a shared running normalization and
frame-major window layout. The logit penalty uses its configured 0.04
coefficient; upstream's documented compatibility quirk applies 1.0 when
enabled. Existing robot models, height-map perception and world-model
architecture are retained. The upstream contact-timer predictor and depth
camera encoder are not reproduced here.

The 964-update diagnostic using the initial ten-frame implementation achieved
zero command successes and about 0.01 m maximum forward progress despite a
0.669 style score. Its forward command lower percentile was about 0.003 m/s.
This is evidence of a stationary local solution, not a qualified foundation.
The revised walk pools contain sustained forward motion. Velocity AMP requires
at least eight moving frames per ten-frame window plus a moving window mean.
Envelope windows stay within a single source clip. Terrain AMP does not apply
the forward-motion filter, preserving complete jump and landing histories.

The T1/Bumi anomaly filter matches the original filename, including source
records whose output names are prefixed. The identified upright failure clip
is excluded from both walk and parkour pools by original source hash and exact
crop identity. Moving clips are cropped into immutable, source-hashed windows
with separate recording-level train/test membership.

## Fixed-Command G1 Diagnostic

The G1 velocity checkpoint completed 2,500 iterations with one 1.0 m/s-class
expert-derived command held for each episode. Two held-out seeds both survived
all six difficulty lanes to timeout. Mean projected velocity along the command
was 0.465 and 0.462 m/s; velocity RMSE was 0.222 and 0.215 m/s, while yaw-rate
RMSE was 0.720 and 0.717 rad/s. Geometry-based support-foot slip was 0.460 and
0.456 scale-m/s. Both evaluations had zero task successes, so this checkpoint
does not qualify for downstream terrain or stealth work. The rendered attempts
show forward steps with stiff posture and visible foot sliding; they are
retained for comparison rather than labeled successful.

The complete update window 541--1540 measured 1.9268 iterations/s, below the
teleoperation reference 2.47115. This is not a matched-workload result:
SuperMimic includes two AMP updates per actor update, whereas the referenced
teleoperation window used 20,000 critic updates per 1,000 outer iterations.
The AMP combined gradient-penalty path currently repeats discriminator
forwards; reusing its combined batch forward is a measured-workload-preserving
optimization candidate, not yet applied or benchmarked.

All six velocity foundations and dependent terrain/stealth jobs retain their
10,000-update initial target. A numerical pass now waits for a hash-bound
visual review of the exact checkpoint and trajectory before qualification or
dependent terrain work. Fresh physical validation and videos are required
before claiming improved motion quality.

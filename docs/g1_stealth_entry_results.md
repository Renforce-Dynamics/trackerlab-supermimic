# G1 Stealth Entry Continuation Results

The owner approved retaining the current v4 hand candidate and continuing the
same experiment through 10,000 iterations. The learner is SuperMimic with AMP,
with no PPO stage. Training, evaluation, first-attempt videos and actor export
completed from implementation commit
`5f4e5f81075b542e233f50ac3714913818a12cbe`.
Final reviews completed on November 7, 2026, Shanghai time.

## Settings

| Setting | Value |
| --- | --- |
| Task | `locomotion/g1/stealth` |
| Training hardware | RTX 4090, 1,024 MuJoCo-Warp worlds |
| Training iterations | 10,000, including 500 world-model-only warmup |
| Actor / critic / AMP iterations | 9,500 each |
| Control | 50 Hz, 20-second episodes |
| Expert | One 106-frame derivative of `aiming2_subject3.bvh` |
| Held-out motion | Original Aiming1 clip, unchanged and excluded from training |
| AMP | 10-frame postural/endpoint features, weight 0.25 |
| Actor history | 10 frames; 283 features per frame, 12 command-goal fields |
| Forward / forward arcs | 0.25 m/s, yaw rate 0 or +/-0.45 rad/s |
| In-place turns | Zero requested translation, +/-0.55 rad/s yaw |
| Command mixture | Forward 40%; each of four turn/arc modes 15% |
| Training resets | 50% grounded standing, 50% screened reference states |
| Evaluation resets | Grounded standing, no reference initialization |
| Arm entry | Five synchronized targets over 4.82 seconds |
| Command entry | 4.84-second quintic velocity/yaw/height ramp |
| Arm residual | Zero; final arm targets fixed to the accepted v4 pose |

The legs and waist remain learned policy outputs. The embedded arm entry and
fixed final target are explicit control constraints; adding AMP geometry does
not give the actor freedom to alter them. Runtime has no motion-file,
discriminator or world-model input. A caller still supplies the existing
command/entry clock and body/terrain observations.

Physical randomization includes friction 0.3--1.2, joint offsets +/-0.01 rad,
scaled center-of-mass shifts, mass/inertia +/-10%, stiffness/damping +/-15%,
and force limits +/-10%. Mass and motor scales are deterministic functions
of four fixed privileged joint-offset channels; they are correlated with
those offsets. This preserves the existing observation/context contract.

## Physical Evaluation

Each row uses 96 environments for 1,500 control steps with observation noise
and physical randomization. Episodes are capped at 20 seconds; early failures
can produce additional episodes. Counts include both failures and completed
timeouts. All evaluations start from standing. The seed is held out from
training.

| Iterations | Seed | Episodes | Falls | Full successes | Forward speed (m/s) | Forward tracking | Sole slip (m/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 101 | 107 | 24 | 0 | 0.123 | 48.9% | 0.208 |
| 2,000 | 103 | 102 | 25 | 0 | 0.122 | 47.7% | 0.209 |
| 4,000 | 101 | 96 | 0 | 0 | 0.133 | 50.8% | 0.093 |
| 6,000 | 101 | 96 | 0 | 0 | 0.168 | 45.9% | 0.115 |
| 6,000 | 103 | 96 | 0 | 0 | 0.159 | 46.3% | 0.114 |
| 10,000 | 101 | 96 | 0 | 0 | 0.201 | 73.2% | 0.093 |
| 10,000 | 103 | 97 | 1 | 0 | 0.193 | 73.7% | 0.096 |

Forward speed is projected body velocity during the forward mode and includes
entry ramp and mode changes. Full success requires survival plus command
tracking on at least 80% of episode actions; timeout survival alone is
insufficient. The two final seeds recorded one fall across 193 completed
episodes, but no full command successes. Slip uses geometric sole support,
not force measurements.

The fixed first six attempts of each final seed were also analyzed after the
entire 4.84-second entry and at least 0.5 seconds of unchanged command:

| Mode | Seed 101 body vx (m/s) | Seed 103 body vx (m/s) | Requested vx (m/s) |
| --- | --- | --- | --- |
| Forward | 0.228 | 0.226 | 0.25 |
| Turn left | 0.133 | 0.129 | 0 |
| Turn right | 0.166 | 0.147 | 0 |
| Forward left | 0.240 | 0.237 | 0.25 |
| Forward right | 0.242 | 0.219 | 0.25 |

This subset supports improved forward speed and exposes translation drift
during requested in-place turns. It covers only twelve fixed first attempts
and cannot establish aggregate performance across all environments. Recorded
poses in those attempts had no self-penetration beyond the audit's 1 mm
threshold. Their sole/hand dynamics and visual style still require review.

## Throughput And Artifacts

The last complete 1,000-update window (iterations 8,041--9,040) took
682.532 seconds: **1.465 full updates/s**, below the requested 2.5 updates/s.
The tele historical 2.471 updates/s result used an RTX PRO 6000, 8,096 worlds,
and no AMP discriminator; identical outer iteration counts do not imply the
same workload.

Final checkpoint on the training host:

```text
/root/gpufree-data/runs/g1-stealth-entry-v4-pilot-20261106/training/iteration_010000.pt
SHA256 55e36fb80141d55fc388c25efd37e04ffcc89bfad9f250dde2e965a7b1c26998
```

The same run contains `review/iter004000`, `review/iter006000`,
`review/iter010000`, and `export-iter010000`. Videos use the fixed first
attempts of environment IDs 0 and 5, with no outcome filtering. Export
contains `policy.ts`, `actor.pt`, and input/constraint metadata. Eager versus
scripted CPU validation had zero maximum error for batches 1 and 7.

The immutable expert pool remains under ignored `data/`; motion data and
training checkpoints are excluded from code pushes. The owner's UMR preview
named "LAFAN1 test Aiming, subject 5, 6 seconds" comes from
`aiming2_subject5.bvh`, frames [60,240). This experiment uses subject 3,
frames [340,460), with the documented static arm correction. They are related
Aiming motions rather than a matched-frame retarget comparison.

No longer continuation was queued after the 10,000-iteration experiment.
Remaining limitations are in-place-turn drift, residual foot sliding, strict
command success, visual fidelity, and throughput.

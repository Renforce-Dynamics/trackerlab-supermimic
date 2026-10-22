# Recorded Results

Recorded on 2026-10-23, Asia/Shanghai. These measurements come from the
TrackerLab parent implementation before extraction into this focused package.
Release smoke checks are documented separately in `validation.md`.

| Run | Iterations | Environments | Full-clip return | Failed clips | Training minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| TrackerLab, 99D + reset noise | 2000 | 8096 | 0.54160159 | 12/40 | 23.7112 |
| Previous TrackerLab, 93D | 2000 | 32768 | 0.39415966 | 11/40 | 40.65 |
| vibed author-reported | 2000 | 4096 | 0.541 | 12/40 | 9 |

TrackerLab uses PhysX; vibed uses Newton/MuJoCo-Warp. The public checkpoint
was not rerun in TrackerLab. Its score is the author's repository record at
`a1508d42c9d1699a52e8233ab16d08c488441ab9`. Timing includes each implementation's
own training measurement and compilation convention; this is not a controlled
speed comparison.

The current 2000-iteration run collected 40,998,144 simulator transitions,
completed 1,500 policy updates after 500 world-model-only iterations,
and recorded zero skipped policy updates. Frame-weighted return was
0.56960557. The checkpoint SHA256 is
`d4dccb935dfbf705b6a7a49c078fd5fd353f71cab60c6c5c3f6a5067683c3c33`.

The current return is 37.4% above the earlier TrackerLab run; the failed-motion
count is one higher. Multiple configuration changes and one seed per run
prevent attributing that difference to a single change. All 40 clips are
training data, so these results do not measure held-out generalization.
The 100,000-iteration experiment was still running when this record was
prepared; no final score is claimed.

## Reward Contributions

Each entry uses the full-clip, equal-clip normalization.

| Term | Contribution |
| --- | ---: |
| Body position | 0.15218611 |
| Body orientation | 0.07817497 |
| Body linear velocity | 0.12838919 |
| Body angular velocity | 0.11108772 |
| Root position | 0.00383448 |
| Root orientation | 0.07672079 |
| Action smoothness | -0.00797816 |
| Joint limits | -0.00081354 |

## Failed Motions

`dance1_subject1`, `dance2_subject4`,
`fallAndGetUp1_subject1`, `fallAndGetUp1_subject4`, `fallAndGetUp1_subject5`,
`fallAndGetUp2_subject2`, `fallAndGetUp2_subject3`, `fallAndGetUp3_subject1`,
`fight1_subject3`, `jumps1_subject2`, `walk3_subject1`, `walk3_subject4`.

The machine-readable record in `results_002000.json` contains the complete
per-motion values, source checkpoint hash and normalized components.
Model weights and motion data are not part of this Git distribution.

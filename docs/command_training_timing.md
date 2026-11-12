# Command Training Timing

`train_terrain.py --phase-profile-updates N` enables an optional measurement
window of 1 through 1,000 complete command/AMP updates. The default is zero,
so existing runs retain their current timing path.

The observer follows the existing Tele CUDA-event timing pattern. It measures
collection, all world-model updates, AMP fitting, actor sampling/history,
imagination, task rewards, style reward, objective construction, actor
backward/optimizer work, critic work, and normal bookkeeping/checkpoint I/O.
It leaves model inputs, losses, random draws and optimizer schedules in the
existing trainer.

## Timing Windows

The ordinary `training_speed_summary.json` measurement comes first. Phase
observation begins on the following iteration, so its event-recording overhead
stays outside that ordinary timing window. With 500 WM-only iterations, a
40-iteration actor compilation allowance and the default 1,000-update speed
window, the ordinary interval is 541 through 1,540. An optional 1,000-update
phase interval is 1,541 through 2,540.
These ranges use the trainer's one-based completed-iteration labels. The phase
JSON explicitly declares zero-based indices, so the same phase interval is
stored as 1,540 through 2,539.

For a resumed run, both windows start after at least 40 new iterations. The
requested total iteration target must accommodate the complete phase window.
An owner stop can still produce a partial observation.

Enable this option when creating a future run, alongside its existing task,
reference, initialization and training options:

```text
--phase-profile-updates 1000
```

`command_phase_profile.json` records the observed interval, device/runtime,
trainer hash, workload, per-phase host and CUDA stream spans, total wall time,
memory peaks and actual optimizer step states at both boundaries.
The four optimizer records cover the world model, actor, critic and AMP.
Uniform deltas require the same parameter identities and equal nonnegative
per-parameter increments. Missing states and irregular increments remain
visible in the receipt. Use a fresh run directory when a phase receipt already
exists.

CUDA events are recorded around existing calls on the device's current stream.
The observer synchronizes at measurement boundaries. Existing trainer log
synchronization retains its normal cadence. Stream spans include idle gaps;
nested spans overlap and their sum cannot be interpreted as total kernel time.
The whole wall interval includes normal logs and checkpoint writes inside the
window. Report serialization follows that interval.

## Current Evidence

The November 12 G1 full-body reference run at source `40e9615` uses the
existing ordinary timer. Iterations 541 through 1,540 completed in
818.881426 seconds, or 1.221178 complete updates/s, on an RTX 4090 with
1,024 environments. The schedule includes ten WM minibatches, one actor step,
twenty critic steps and two AMP steps per full iteration.

Historical Tele's 2.47115 updates/s used an RTX PRO 6000 and a different
environment count and workload. A phase receipt and matched hardware
measurement remain required to attribute the speed difference. This optional
observer has no measured GPU timing result yet, and speed observations provide
no motion-quality acceptance.

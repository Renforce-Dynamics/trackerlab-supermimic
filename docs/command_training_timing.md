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
measurement remain required to attribute the speed difference. Speed
observations provide no motion-quality acceptance.

The subsequent v2 full-body extension at `adb8f827` completed a 1,000-update
phase interval, zero-based indices 6540..7539, on the same RTX 4090. Its
instrumented speed was 1.258066 updates/s; the preceding ordinary window measured
1.262735 updates/s. Actual optimizer deltas were WM 10,000, actor 1,000,
critic 20,000 and AMP 2,000.

The largest mean current-stream span was actor backward/optimizer 417.10 ms,
followed by WM 120.57 ms, collection 73.50 ms and imagination 66.63 ms. These spans
include stream gaps. They are not exclusive kernel timings. An arithmetic
scenario subtracting the entire WM span while keeping all other observed
durations fixed gives about 1.48 updates/s; this is a prioritization calculation,
not a measured optimization.

The completed November 13 contact-v3 trial at `1d267aff` measured
1.211489 updates/s over its ordinary 541..1540 window. Recent 25-update windows
near 1.26/s must not replace that complete-window statistic.

Local Tele source `8dea656b` enforces `SM_GRAPH_SCOPE=wm` at its accepted
entrypoint. WM fitting requests `reduce-overhead`; imagination requests default
compilation. Command training currently requests `reduce-overhead` for both WM
and imagination. Both use BF16 MLP autocast with TF32 disabled. These source
settings alone establish neither actual backward graph replay nor a speedup.

After G1 visual review, a narrow benchmark candidate can disable CUDA graphs
only for command imagination while preserving compilation, workload, precision,
data and optimizer schedules. Full-horizon value/gradient checks, explicit RNG
accounting and equal complete-update timing windows are required before
adoption. No such performance candidate has been implemented or benchmarked.

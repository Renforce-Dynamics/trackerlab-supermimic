# Reference-tail and running-normalization repair

This revision ports three shared tracker repairs from the teleoperation
experiments into the canonical SuperMimic source. It retains the existing
PhysX environment and robot/task definitions.

## Recorded endpoints and synthetic goal frames

`MotionLib.goal(f, K)` builds future goals for requested frames
`f + 1, ..., f + K`. The stored frame index still clamps to the current clip's
inclusive last frame, `clip_end`.

The velocity rule uses the **requested index before clamping**:

| Requested frame | Position, orientation and joint position | Linear, angular and joint velocity |
| --- | --- | --- |
| Within the clip, including its recorded final frame | Recorded reference | Recorded reference |
| Beyond the recorded final frame | Held final reference pose | Zero |

For example, when the final recorded frame is 99 and the current frame is 98,
the first goal uses frame 99 with its reference velocities. Goals requesting
100 and later hold frame 99's pose with zero velocity. At current frame 99,
every future goal is a synthetic held-pose extension.

Previously the clamped frame reused its nonzero velocity for every synthetic
future request. Those goal features described a fixed pose together with
continuing motion. The repair removes this mismatch without overwriting the
actual reference samples or discarding the final recorded velocity.

The mask broadcasts across the joint dimension. Goal width remains
`K * (15 + 2 * num_joints)` for G1, Go2 and the other registered robots.
`future_frames`, `ref_state`, episode termination, reset sampling, rewards and
model dimensions retain their existing definitions. The goal horizon is
independent of the eight-step world-model training window.

## Preserve already-50Hz source samples

When the source rate equals the 50Hz control rate, the library now keeps the
source samples and frame count directly. The previous duration/floor
resampling could discard the final frame for lengths such as 30 or 48 because
of floating-point division.

Positions, joint positions and quaternions retain their original float32
samples; velocities and tracked-body kinematics are then calculated using the
existing formulas. The previous interpolation behavior remains for other
source rates.

## Preserve active normalizer storage

`RunningNorm.update` retains the running mean/variance formula, expression order,
sample threshold and update schedule. The new variance is copied into the
existing buffer when its dtype matches; the count is also updated in place.
This retains FP32/FP64 tensor storage addresses used by compiled CUDA graphs.

When a low-precision normalizer's old arithmetic promotes variance to FP32,
the new code preserves that promotion. The class, registered buffers, model
dimensions, forward calculation, optimizer setup and checkpoint keys are
unchanged.

This source repair avoids the fixed-source-hash overlay needed by the isolated
teleoperation experiment. It does not alter any running trainer or import the
teleoperation Warp runtime.

## Compatibility and evaluation

Existing network state dictionaries retain their loading schema. Goal values
near clip boundaries intentionally differ, and already-50Hz motion libraries
may retain frames previously lost by resampling. Rebuild or revalidate derived
motion caches and their frame-count/source fingerprints before using them with
the repaired loader.

Comparisons between old and new checkpoints should record the code revision
and motion manifest. This change alone establishes neither a cause for prior
training divergence nor an improvement in full-clip failure rate, convergence,
speed or physical-robot performance. Existing published demos and benchmark
scores remain historical measurements.

## CPU regression checks

```bash
python -m unittest discover -s tests -p 'test_motion_tail.py' -v
python -m unittest discover -s tests -p 'test_running_norm_storage.py' -v
python -m unittest discover -s tests -v
```

The motion tests exercise interior/endpoint/synthetic goals, mixed clips,
source-state preservation, generic joint counts and frame-rate behavior.
Normalizer tests compare against the previous formula for values, standard
deviations, outputs and input gradients; they also cover storage identity,
dtype promotion, saturation and checkpoint loading.

The repository update is validated on an isolated checkout of the current
remote revision plus these changes. Independent local robot-development work
is outside this commit.

Validation on 2026-10-28 used Windows, Python 3.12.12, PyTorch 2.9.0+cpu,
NumPy 2.3.4 and SciPy 1.16.3:

```text
Ran 97 tests in 8.917s
OK (skipped=1)
```

96 tests passed, including all 12 new regression tests. One existing
installation test was skipped because symlink creation was unavailable.
Source compilation and staged whitespace checks also passed. The repository's
Linux/Python 3.11 CI remains a separate environment; this source update runs no
GPU training or physical-robot evaluation.

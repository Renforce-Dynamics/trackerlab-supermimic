# A3/T1 Validation, 2026-10-25

The validation used IsaacLab 2.3.2, Isaac Sim 5.1, PyTorch 2.7.0+cu128,
and one RTX PRO 6000 Blackwell Server Edition with 96 GB VRAM.
An unrelated G1 teacher continued running during these checks.

| Check | A3 | T1 |
| --- | ---: | ---: |
| Reference clips | 20 | 1 |
| Raw frames | 58,372 at 120 Hz | 1,635 at 50 Hz |
| Frames after 50 Hz resampling | 24,324 | 1,635 |
| Controlled joints | 29 | 23 |
| Observation dimensions | 99 | 81 |
| PhysX/Torch FK max position difference (m) | 5.07e-7 | 3.58e-7 |
| PhysX/Torch FK max rotation-matrix difference | 1.64e-6 | 8.35e-7 |
| Finite transitions, partial reset, randomization, replay | Passed | Passed |
| Preflight environments | 4,096 | 4,096 |
| Preflight iterations / actual policy updates | 12 / 8 | 12 / 8 |
| Skipped policy updates | 0 | 0 |
| Peak PyTorch allocation (GiB) | 12.80 | 11.66 |
| Preflight eval, policy export, video decode | Passed | Passed |

All 77 CPU regression tests passed on Linux, including existing Go2
downstream behavior, variable robot shapes, resource conversions, task
discovery and the TrackerLab installer. Local Windows installer checks
also passed; its symlink-specific case was skipped because local symlink
creation was unavailable.

The MJCF importer required explicit extension activation. Isaac Sim 5.1
also creates an empty `worldBody` articulation alongside the floating robot;
the loader deactivates that empty prim after checking it contains no rigid
body or children. An empty T1 camera marker with no inertia/geometry is
excluded by the resource converter. Robot inertias and collision shapes are
retained, and the resulting link transforms pass the FK checks above.

The 12-update weights were disposable pipeline validation. They are not a
trained result and are not loaded into the long runs.
Both long runs start from random initialization, use 4,096 environments,
and target 10,000 iterations with 2,000/10,000 evaluation milestones.
The initial 500 iterations train the world model.

Existing Go2 selected policy remains iteration 10,000. The old Go2 follow-up
was disabled. All previous Go2 models and the original stopped G1 model are
preserved. This report makes no claim about A3/T1 convergence, held-out
generalization, SONIC algorithm parity or hardware performance.

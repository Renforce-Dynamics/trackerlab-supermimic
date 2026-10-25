# Multi-Robot Validation, 2026-10-25

The validation used IsaacLab 2.3.2, Isaac Sim 5.1, PyTorch 2.7.0+cu128,
and one RTX PRO 6000 Blackwell Server Edition with 96 GB VRAM.
An unrelated G1 teacher continued running during these checks.

| Check | A3 | T1 | Bumi | Pi+ |
| --- | ---: | ---: | ---: | ---: |
| Reference clips | 20 | 1 | 1 | 4 |
| Source frames / rate | 58,372 / 120 Hz | 1,635 / 50 Hz | 342 / 50 Hz | 1,124 / 30 Hz |
| Frames after 50 Hz conversion | 24,324 | 1,635 | 342 | 1,874 |
| Controlled joints | 29 | 23 | 21 | 22 |
| Observation dimensions | 99 | 81 | 75 | 78 |
| PhysX/Torch FK position max difference (m) | 5.07e-7 | 3.58e-7 | 2.38e-7 | 4.77e-7 |
| PhysX/Torch FK rotation max difference | 1.64e-6 | 8.35e-7 | 9.72e-7 | 1.14e-6 |
| Finite transitions, partial reset, randomization, replay | Passed | Passed | Passed | Passed |
| Preflight environments | 4,096 | 4,096 | 4,096 | 4,096 |
| Preflight iterations / actual policy updates | 12 / 8 | 12 / 8 | 12 / 8 | 12 / 8 |
| Skipped policy updates | 0 | 0 | 0 | 0 |
| Peak PyTorch allocation (GiB) | 12.80 | 11.66 | 11.32 | 11.44 |
| Preflight eval, policy export, video decode | Passed | Passed | Passed | Passed |

All 81 CPU regression tests passed locally, including existing Go2
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

The 12-update weights were disposable pipeline validation and are not loaded
into the fresh long runs. A3 and T1 each completed 10,000 iterations from
random initialization, with 4,096 environments, 2,000/10,000 evaluation
milestones and 500 world-model-only warmup iterations.

Fresh A3/T1 10,000-iteration evaluations completed with returns 0.9844662
(0/20 failures) and 0.8702665 (0/1), respectively. These cover each loaded
training library.

Bumi uses the vendor's one 6.82-second motion. The vendor NPZ's cached body
poses have up to 0.05484 m disagreement with its bundled MJCF; MotionLib
recomputes body targets from root and joint trajectories. Twenty-eight elbow
samples were projected to the pinned model's range; the maximum adjustment
was 0.3301535 rad. Generalization beyond this one clip is unmeasured.
Pi+ uses four vendor CSV clips after named-column conversion; its model has 22
moving joints although its filename says `pi_20dof.xml`.

Fresh Bumi/Pi+ 10,000-iteration runs use the same shared learner and eight
tracking rewards and run sequentially on the shared GPU. Their milestones
will test whether the short source libraries are learnable; the 12-iteration
preflights are pipeline checks only.

Existing Go2 selected policy remains iteration 10,000. The old Go2 follow-up
was disabled. All previous Go2 models and the original stopped G1 model are
preserved. This report makes no claim about A3/T1 convergence, held-out
generalization, SONIC algorithm parity or hardware performance.

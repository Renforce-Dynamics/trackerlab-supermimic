# Release Validation

Date: 2026-10-23, Asia/Shanghai.

## Version 0.2 Refactor

Both layouts were exercised in isolated directories on the existing GPU host:
a standalone checkout and a pristine TrackerLab archive at
`1e5ccc062b445712a0aa7308cfb99edd7296cc88` with the overlay applied.
The active 8,096-environment, 100,000-iteration run continued using its original
code. The isolated venv inherits the installed simulator stack; this is not a
new operating-system or NVIDIA-driver installation.

| Check | Result |
| --- | --- |
| Linux CPU tests | 12 passed, including installer conflicts, backups, idempotence, symlinks and motion configuration |
| Installed wheel | `trackerLab` and `trackerTask` imported from venv site-packages outside the checkout |
| Reference parity | All 40 clips, 441,140 frames, 20 reference tensors identical to version 0.1 |
| Analytic and upstream reward parity | Ten analytic cases; vibed rewards, gradients and MDP heads matched |
| Algorithm parity | Goals, observations, rewards, gradients, reset sampling, policy output, world-model step, training targets and state error matched exactly |
| PhysX smoke | 4 environments x 64 steps; FK, randomization, partial resets, terminal transitions and replay passed |
| Standalone workflow | 5 iterations, all optimizers, 40-clip evaluation, eight reward terms, policy and video export |
| TrackerLab overlay workflow | Same checks using a `data/configs` motion JSON |
| Resume | Host checkpoint advanced from iteration 5 to 7; actor, world model and critic parameters changed |
| Existing exports | Seven original motion-buffer, motion-library and manager exports imported from their original modules |

Generic `ManagerBasedTrackerEnv` and its configuration triggered the same
circular-import error in the pristine host and the overlaid host, after
installing the host's bundled `poselib`. Those two exports and generic task
rollouts are outside the successful compatibility claim. The SuperMimic
environment is a dedicated `InteractiveScene` implementation.

The same trained 32,000-iteration checkpoint was evaluated on all 40 full clips:

| Metric | Before refactor | TrackerLab overlay |
| --- | ---: | ---: |
| Equal-clip normalized return | 0.8694663037746124 | 0.8694663037746124 |
| Frame-weighted normalized return | 0.8701766756758671 | 0.8701766756758671 |
| Failed clips | 0/40 | 0/40 |
| Largest per-motion return difference | - | 0 |

Checkpoint SHA256:
`51c0f4150e6baed82504f5d1ceb5f28332ae9cfb66b9bca9f4884b3be86dd0e0`.
All clips belong to the training library. This validates behavior preservation
for that checkpoint and experiment, not held-out generalization or every
TrackerLab task. Full measurements, hashes and limitations are recorded in
`refactor_validation.json`.

## Historical Version 0.1

The extracted package was tested in a separate directory and Python
environment on the GPU host. The Python environment inherited the already
installed IsaacLab/Isaac Sim runtime, then installed this package with
`pip install --no-deps --no-build-isolation .`. Import from outside the
checkout resolved to the installed `site-packages` package. The active
8,096-environment parent experiment remained running.

This checks installation of the extracted package into the tested simulator
stack. It does not claim that a new operating system and the entire NVIDIA
stack were installed again during this release test.

| Check | Result |
| --- | --- |
| Python syntax | Passed |
| CPU workflow tests | 5 tests passed |
| Pinned resources | 40 original motion files and G1 assets verified |
| Analytic rewards | 10 cases; maximum component error 2.54e-7 |
| vibed reward and gradient parity | Maximum absolute error 0 |
| Observation, termination and critic-head parity | Maximum absolute error 0 |
| Differentiable FK | Finite gradients |
| MuJoCo FK position / rotation error | 7.60e-7 m / 2.78e-7 |
| PhysX FK position / rotation error | 7.16e-7 m / 2.03e-6 |
| PhysX smoke | 4 environments, 64 steps; 5 terminations, 2 truncations |
| Randomization, partial resets, terminal states, replay | Passed |
| Supervised training | 5 iterations, all optimizer paths, zero skipped policy updates |
| Evaluation | All 40 motions, eight reward components |
| Policy export | Native FP32 and serialized inference matched at batches 1 and 7 |
| Video | Rendered from recorded PhysX trajectory; full decode passed; frame inspected |
| Artifact integrity | 22 SHA256 hashes verified |
| Continued training | Iteration 5 -> 7; policy, world model and critic parameters changed |

The smoke preset uses reduced batches and disables compilation. It validates
the publication workflow, not learned tracking quality. Full-batch
8,096-environment execution and compilation were tested in the parent
TrackerLab experiment. The measured 2,000-iteration tracking result belongs
to that parent run, as stated in `results.md`.

The numeric reward and simulation checks are retained in
`release_validation.json`, with machine-specific paths removed. This file
describes version 0.1 before the layout refactor.
The code deliberately excludes credentials, resource files, checkpoints and
machine-specific launch scripts from Git.

The complete resource-download path was also exercised in a fresh local
resource directory, including the pinned upstream MDP. That reference hash
uses GitHub's original LF bytes; the repository enforces LF endings for
source files. The offline resource verification was repeated afterwards.

## Repeating The Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q source scripts tests
python scripts/rls/supermimic/prepare_resources.py --verify_only --with_reference
python scripts/rls/supermimic/validate_rewards.py \
  --upstream_mdp data/supermimic_reference/mdp.py \
  --output logs/validation/rewards.json
python scripts/rls/supermimic/smoke_test.py --headless --output logs/validation/simulation.json
python scripts/rls/supermimic/supervise.py --config scripts/rls/supermimic/configs/smoke.json \
  --run_dir logs/release_smoke --milestones 5 \
  --render_seconds 1 --render_clips walk1_subject1
python scripts/rls/supermimic/train.py --config scripts/rls/supermimic/configs/smoke.json --headless \
  --resume logs/release_smoke/last.pt --iterations 7 \
  --log_dir logs/release_resumed
```

Add the documented Kit root argument for root-owned containers.

For numerical migration checks, retain a separate checkout of version 0.1:

```bash
python scripts/rls/supermimic/validate_refactor.py \
  --baseline /path/to/v0.1-checkout \
  --motion_dir data/retargeted/lafan1_g1 \
  --checkpoint /path/to/trusted/checkpoint.pt \
  --output logs/validation/refactor.json
```

Repeat the smoke supervisor in an isolated host checkout after applying
`scripts/install_trackerlab.py`. A motion JSON can be supplied with
`--motion_config data/configs/supermimic_lafan1.json`. Video decoding and
policy serialization checks are part of milestone packaging.

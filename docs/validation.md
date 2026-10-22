# Release Validation

Date: 2026-10-23, Asia/Shanghai.

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
`release_validation.json`, with machine-specific paths removed.
The code deliberately excludes credentials, resource files, checkpoints and
machine-specific launch scripts from Git.

The complete resource-download path was also exercised in a fresh local
resource directory, including the pinned upstream MDP. That reference hash
uses GitHub's original LF bytes; the repository enforces LF endings for
source files. The offline resource verification was repeated afterwards.

## Repeating The Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/prepare_resources.py --verify_only --with_reference
python scripts/validate_rewards.py \
  --upstream_mdp data/supermimic_reference/mdp.py \
  --output logs/validation/rewards.json
python scripts/smoke_test.py --headless --output logs/validation/simulation.json
python scripts/supervise.py --config configs/smoke.json \
  --run_dir logs/release_smoke --milestones 5 \
  --render_seconds 1 --render_clips walk1_subject1
python scripts/train.py --config configs/smoke.json --headless \
  --resume logs/release_smoke/last.pt --iterations 7 \
  --log_dir logs/release_resumed
```

Add the documented Kit root argument for root-owned containers.

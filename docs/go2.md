# Go2 Gait Task

Go2 extends TrackerLab through a robot specification, task configuration and
motion configuration. MotionLib, MotionBuffer, MotionManager, TrackingSim,
ReplayBuffer, world model, policy, critic, rewards and exports are shared with G1.
The same source runs standalone or inside an existing TrackerLab checkout.

## Public Motion Data

```bash
python scripts/rls/supermimic/prepare_resources.py --task go2
python scripts/rls/supermimic/prepare_resources.py --task go2 --verify_only
python scripts/rls/supermimic/smoke_test.py --task go2 --headless \
  --motion_config data/configs/go2_kine2go.json --output logs/go2_smoke.json
```

Use `--asset_root /path/to/assetslib` and set `SUPERMIMIC_ASSETLIB_ROOT` for
an existing asset checkout. Root containers add `--kit_args=--allow-root` to
simulation commands. Installation into TrackerLab uses
`python scripts/install_trackerlab.py --trackerlab /path/to/trackerLab --apply`.

Default data is **Kine2Go `ai4_dog_trot_00`**, from dataset
`MIMUW-Robotics/kine2go` revision
`7f03a3fb7f6b2e684899530862ba103e81570363`. It contains AI4Animation dog mocap
already retargeted onto Go2 by the publisher. Source metadata attributes
H. Zhang et al., "Mode-Adaptive Neural Networks for Quadruped Motion Control",
SIGGRAPH 2018, under CC BY-NC 4.0. Motions stay outside Git; source README,
licenses, metadata and hashes are retained.

The importer loads numeric NPY with `allow_pickle=False`. It uses the full
548-frame, 60 Hz reference (9.1167 seconds between first and last frame).
Publisher metadata says 571 frames; the manifest records both counts.
Publisher crop settings are not applied. No smoothing or IK rerun is performed.
These references are distinct from the publisher's Genesis/PPO rollout files.

| NPY slice | Meaning |
| --- | --- |
| 0:18 | Raw root DOFs and joint positions |
| 18:36 | Raw DOF velocities |
| 36:48 | Foot link origins in FL, RL, FR, RR order |
| 48:51 | Root XYZ |
| 51:55 | Root quaternion WXYZ |
| 55:61 | Root linear and angular velocity |

Actual columns 6:18 group joints by type: four hips, four thighs, four calves,
each in FL, FR, RL, RR order. This differs from the motor-name list in the
publisher's configuration. Every frame is checked against independently
stored foot link positions using the pinned Go2 MJCF. For trot_00, maximum
position error was 6.984e-7 m. Output CSV has 19 columns: root XYZ, quaternion
XYZW, then hip/thigh/calf for FL, FR, RL, RR. FK validates mapping, not dynamics.

```text
Kine2Go NPY + metadata -> pinned hashes + schema + full-frame FK validation
  -> data/retargeted/go2_kine2go/*.csv + manifest.json
  -> data/configs/go2_kine2go.json (source_fps=60)
  -> MotionLib (50 Hz, velocities and body FK)
  -> MotionBuffer -> MotionManager -> TrackingSim -> ReplayBuffer
  -> world-model fitting / differentiable policy learning -> metrics / export
```

For machines without Hugging Face access, use `prepare_kine2go.py --download_only`
elsewhere, transfer the complete source directory, then run
`prepare_kine2go.py --offline --source_dir /path/to/kine2go`.
`--clips` accepts additional AI4Animation identifiers. Use separate source,
output and motion-config paths for a different selection.

## Train And Evaluate

```bash
python scripts/rls/supermimic/supervise.py \
  --config scripts/rls/supermimic/configs/go2_gait_8096.json \
  --motion_config data/configs/go2_kine2go.json \
  --run_dir logs/go2_kine2go_8096 \
  --milestones 1000 2000 5000 --render_seconds 9
```

The preset uses 8,096 environments and 10,000 iterations; evaluations run at
1,000, 2,000, 5,000 and 10,000. The first 500 iterations fit the world model.
A 4,096-environment preset supports controlled studies. All eight reward terms
remain; Go2 uses position scale 0.15 m, linear-velocity scale 0.5 m/s,
root-position scale 0.2 m and height-error termination at 0.16 m.

Go2 policy actions use `4*tanh(raw/4)` in real and imagined rollouts, including
collection noise clipping. Critic bootstrap and actor tail values have the
upper bound `0.1/(1-gamma)`, or 10 at gamma=0.99. Critic regression remains
unclipped, preserving gradients on overestimates. These options address
observed numerical divergence and leave G1 defaults unchanged.
See [controlled evidence and limitations](go2_divergence.md).

```bash
python scripts/rls/supermimic/evaluate.py --headless \
  --run_dir logs/go2_kine2go_8096 --ckpt logs/go2_kine2go_8096/last.pt \
  --record logs/go2_kine2go_8096/trajectories.npz
python scripts/rls/supermimic/render_demo.py \
  --trajectory logs/go2_kine2go_8096/trajectories.npz \
  --evaluation logs/go2_kine2go_8096/last_eval.json --output outputs/go2_demo.mp4
python scripts/rls/supermimic/export_policy.py \
  --checkpoint logs/go2_kine2go_8096/last.pt --output_dir outputs/go2_policy
```

Evaluation starts at each full clip's first frame. Return is
`sum(reward)/((clip_len-1)*0.1)` with zero contribution after early failure.
Reports include failures, survival duration, world-X/reference velocity,
root-position/velocity RMSE, joint RMSE and eight reward contributions.
Errors cover active frames; survival alone does not demonstrate gait tracking.
Videos replay recorded PhysX poses beside the source reference.

This is reference-conditioned gait imitation. Export requires future reference
goals and observation history. Arbitrary velocity commands, uneven terrain,
held-out motion generalization and hardware performance remain unmeasured.
Scores across G1/Go2 or different motion libraries are not directly comparable.

The public-data run's first 1,000-iteration evaluation on 2026-10-23 gave
return 0.751672 and 0/1 failures over the full 9.10-second resampled clip.
World-X speed was 0.7236 m/s versus reference 0.8377 m/s; root-position RMSE
was 0.3930 m and joint RMSE 0.1393 rad. This is an early result on a training
clip. The 10,000-iteration experiment continues; no final score is claimed.

## Model And Procedural Data

Assets are pinned to assetslib `9df792784e3642163f03390bf6ae16cf88b222a5`,
directory `unitree/unitree_go2`. The model description attributes Unitree URDF
derivation and BSD-3-Clause licensing. External files keep their notices.
Actuation follows IsaacLab v2.3.2 Go2: DC motor torque/saturation 23.5 Nm,
velocity 30 rad/s, stiffness 25 and damping 0.5; simulation is 200 Hz with
50 Hz control. Publisher Genesis gains (30, 1.5) and 60 Hz reference timing
differ. This port requires its own physical evaluation.

The earlier failed run used four **procedural IK** references at 0.35, 0.50,
0.65 and 0.80 m/s, each 20 seconds. These were generated from diagonal foot
trajectories with bounded MuJoCo/SciPy IK. They remain available for comparisons:

```bash
python scripts/rls/supermimic/prepare_resources.py --task go2 --go2_source procedural
```

Their configuration is `data/configs/go2_gaits.json`. IK, joint-limit, velocity
and stance-foot checks validate geometry; physical realizability is measured.

## Extension Contract

| Quantity | G1 | Go2 |
| --- | ---: | ---: |
| Joints/actions | 29 | 12 |
| State/invariant features | 71 | 37 |
| Randomization context | 33 | 16 |
| Relative-orientation observation | 99 | 48 |
| Goal features per frame | 73 | 39 |
| Tracked bodies | 14 | 13 |

`RobotSpec` is immutable. `RobotKinematics` builds differentiable FK from the
selected model and joint order. Networks and replay receive dimensions
explicitly. G1 wrappers and numeric observation aliases preserve checkpoint
support. Another robot requires resource factory, task settings, registration,
matching data and physical validation.

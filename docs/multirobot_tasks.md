# Robot Tasks

Task selection follows TrackerLab's task category, robot and environment/agent
configuration structure. Shared simulation, motion management and algorithms
remain in `source/trackerLab/trackerLab`. Robot tasks live in
`source/trackerTask/trackerTask/supermimic`:

```text
supermimic/
  registry.py
  tracking/
    humanoid/robots/
      g1/{robot.py,env_cfg.py}
      a3/{robot.py,env_cfg.py,agents/supermimic.json}
      t1/{robot.py,env_cfg.py,agents/supermimic.json}
    quadruped/robots/go2/{robot.py,env_cfg.py}
  locomotion/robots/go2/{env_cfg.py,chase_cfg.py,velocity_cfg.py}
```

The registry selects a robot contract, environment factory, algorithm preset
and motion configuration. It loads without Isaac Sim. The runtime is the
existing specialized SuperMimic runner; no Gym wrapper is required.
Legacy `--task g1`, `--task go2` and downstream Go2 commands remain supported.

```bash
python scripts/rls/supermimic/tasks.py list
python scripts/rls/supermimic/tasks.py show --task tracking/a3
python scripts/rls/supermimic/tasks.py prepare --task tracking/a3
python scripts/rls/supermimic/tasks.py prepare --task tracking/t1
python scripts/rls/supermimic/tasks.py smoke --task tracking/a3
python scripts/rls/supermimic/tasks.py smoke --task tracking/t1
python scripts/rls/supermimic/tasks.py supervise --task tracking/a3 \
  --run_dir logs/a3_4096_10k --milestones 2000 10000 \
  --render_clips 001_walk_front_slow 042_squat_shallow_repeated 058_punch_front_alternating
python scripts/rls/supermimic/tasks.py supervise --task tracking/t1 \
  --run_dir logs/t1_4096_10k --milestones 2000 10000
```

Launch the two supervisors as separate processes for concurrent training.
Each preset has 4,096 environments and 10,000 iterations. The first 500
iterations fit the world model; subsequent iterations update the policy too.
Available GPU memory depends on other jobs and model compilation. Override
`--num_envs` after a preflight on the target GPU.

## Data And Actuation

| Robot | Upstream source | Raw format | Training model |
| --- | --- | --- | --- |
| G1 | LAFAN1_Retargeting_Dataset | 40 retargeted clips, 30 Hz | 29 joints |
| Go2 | Kine2Go | Retargeted trot, 60 Hz | 12 joints |
| A3 | AgibotTech/sonic_for_a3 | 20 SOMA example CSVs, 120 Hz | A3 serial rigid-foot model, 29 controlled joints; two zero head joints fixed |
| T1 | BoosterRobotics/booster_assets | t1_dance.csv, 1,635 frames, 50 Hz | T1 23 joints |

Revisions are fixed in `resources.lock.json`. A3 root translation is converted
from centimetres to metres, joint/Euler angles from degrees to radians, and
root Euler angles with SciPy's lowercase `xyz` convention to `xyzw`.
Named columns define the joint order. Nonzero source head angles are rejected.
T1 uses the vendor's metres, radians, `xyzw` and 23-joint ordering unchanged.
The shared MotionLib resamples to 50 Hz and computes velocities and body poses.
These inputs are already robot-space motion; this converter performs format
and unit conversion rather than a new human-to-robot retargeting optimization.

A3 PD gains, motor limits and armatures follow upstream `a3_old.py`.
This preset uses implicit PD and rigid feet; it does not reproduce SONIC's
passive compliant feet or parallel-motor dynamics.
T1 effort/velocity limits and armatures come from its vendor URDF/MJCF.
Its PD gains are an explicit simulation choice: natural frequency 10 Hz,
damping ratio 2, with `kp = armature * (20*pi)^2` and
`kd = 4 * armature * 20*pi`. These are not claimed as vendor tuning.
Both use action scale `0.25 * effort_limit / kp`.

All eight existing tracking reward terms are used. The A3/T1 presets keep the
humanoid runner's unbounded critic and actor, with no transfer from Go2.
Performance on new robots needs measured evaluation and is not implied by
successful model import. Full-clip failures, reward contributions and joint/root
tracking errors are reported by the same evaluator.

The model converter removes source scene ground/lights and source controllers,
retains robot geometry/inertias, fixes the declared A3 head joints, and records
the source and converted hashes. Isaac Sim's MJCF importer generates USD on
first use. Imported PhysX link poses must pass the MuJoCo/Torch FK smoke check.
Resource licenses are retained under each robot's `licenses` directory.

## Existing TrackerLab

`scripts/install_trackerlab.py --trackerlab /path/to/trackerLab --apply`
copies the nested tasks, presets, shared extensions and command scripts.
Existing host exports are preserved by the installer's lazy imports.
Run the same task commands from the host checkout after installation.
Resources are fetched separately, or set `SUPERMIMIC_ASSETLIB_ROOT` to a
prepared shared asset directory. New A3/T1 assets use `agibot/a3` and
`booster/t1` below that root.

To add another robot, define its named joint/body contract and articulation,
add an environment configuration and agent preset under the appropriate
category, register them, and provide a pinned converter producing the shared
motion JSON/CSV contract. Validate units, joint order, FK, physics transitions,
replay windows and actual policy updates before a long experiment.

Measured import, regression and training-preflight results are recorded in
[the validation report](multirobot_validation.md).

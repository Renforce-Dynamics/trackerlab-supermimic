# Go2 Velocity Command and Repeated Dodge

This local downstream task combines body-frame velocity commands with
repeated, collidable horizontal bars. Each command is `(vx, vy, yaw_rate)`.
Bar normals span eight horizontal sectors. High and low encounters use
different randomized heights; bars approach the future commanded path. At the
end of a clear encounter, a new bar appears outside the robot's collision
extent. The robot, episode clock, observation history, and gait-reference phase
continue through the encounter boundary.

The task is a local extension. It does not claim the unreleased objective from
the linked video or an exact reproduction of its training method.

## Data Flow

```text
Kine2Go Go2 reference config
  -> MotionLib (reset states and cyclic gait goals)
  -> VelocityDodgeCfg
  -> VelocityDodgeSim (PhysX bar pose, velocity, contacts)
  -> VelocityReplayBuffer (15-D scenario per transition)
  -> VelocityDodgeMDP (commands, local bar geometry, rewards)
  -> VelocityWorldModel and VelocityPolicy
  -> train_log.csv, run_summary.json and task checkpoints
```

The simulator and imagination share the same deterministic scenario evolution
and obstacle placement. A transition retains both the old encounter's end pose
for reward and the next scenario for the following observation. Continuous
robot state is retained across bar respawns. Scenario randomness is determined
from an episode seed and encounter index, so replay and imagination reproduce
the same future obstacle draw.

The collision cuboid's pose, quaternion, and world velocity come from the
scenario functions used by the geometric collision reward. At each physics
substep, the environment writes pose and velocity explicitly. This path is
used because Isaac Sim 5.1's GPU tensor backend does not implement its
kinematic-target setter. The simulator records position and quaternion error
between the scenario target and PhysX state, as well as physical contact force.

## Reward Bound

Positive per-step maxima are:

| Term | Maximum |
| --- | ---: |
| Body-frame planar velocity command | 0.040 |
| Body-frame yaw-rate command | 0.020 |
| Upright orientation | 0.010 |
| Clearance height | 0.015 |
| Reference gait style | 0.005 |
| Survival | 0.010 |
| **Total** | **0.100** |

Action smoothness, soft joint limits, and bar collision have nonpositive
contributions. Therefore `r <= 0.1`; for `gamma = 0.99`, the infinite-horizon
bootstrap upper bound is `0.1 / (1 - 0.99) = 10`.
The actor's terminal bootstrap and the critic's bootstrap targets use this
upper bound. The critic regression uses its raw output, so an output above the
bound still receives a regression gradient. This bound depends on the reward
scale and discount; revisit it whenever either changes.

Crossing-time features use signed distance and relative velocity along the
bar's local normal. Both physical box contact and the differentiable collision
penalty transform points into the bar-local frame, including its yaw.

## Warm Start

`--warm_start` takes a `go2_chase_dodge` checkpoint. The matching chase actor
and world-model tensors are loaded; new command and bar-orientation residual
heads start at zero. The changed objective receives a new critic and fresh
optimizers. Training starts at velocity/dodge iteration zero and records the
source checkpoint hash and source iteration in `transfer.json`. This is a new
task initialization, not continuation of the chase iteration counter.

`--resume` accepts only a velocity/dodge checkpoint and continues that task's
own optimizer and iteration state.

## Commands

Smoke uses the full-difficulty curriculum endpoint, gathers enough transitions
for repeated obstacle crossings, and emits the same finite-value diagnostics
as the regular runner:

```bash
python scripts/rls/supermimic/train.py \
  --config scripts/rls/supermimic/configs/go2_velocity_dodge_smoke.json \
  --motion_config data/configs/go2_kine2go.json \
  --warm_start /path/to/chase_008000.pt \
  --log_dir logs/go2_velocity_dodge_smoke --headless
```

The staged 8,000-iteration chase checkpoint is the suggested smoke initializer
for repeated encounters. Supply another path with `--warm_start` to select a
different chase iteration. The regular 8,096-environment preset is:

```bash
python scripts/rls/supermimic/train.py \
  --config scripts/rls/supermimic/configs/go2_velocity_dodge_8096.json \
  --motion_config data/configs/go2_kine2go.json \
  --warm_start /path/to/chase_008000.pt \
  --log_dir logs/go2_velocity_dodge_8096 --headless
```

The preset targets 10,000 updates. Its first 100 updates fit the world model.
Curriculum difficulty remains zero for 300 updates and reaches its endpoint
after a further 6,000 updates. Endpoint ranges are:

| Quantity | Range |
| --- | --- |
| Forward command | -0.6 to 1.5 m/s |
| Lateral command | -0.8 to 0.8 m/s |
| Yaw command | -1.5 to 1.5 rad/s |
| Horizontal approach angle | Eight sectors covering 360 degrees |
| Bar speed | 1.8 to 3.6 m/s, subject to minimum approach speed |
| Encounter period | 2.2 to 3.0 s |
| Low-bar center height | 0.08 to 0.18 m |
| High-bar center height | 0.31 to 0.39 m |

These heights are centers; bar thickness is 0.05 m. Encounter periods can extend
while the old bar remains within the robot's clearance region. Commands are
resampled with each encounter. The robot resets on a fall or at the 16-second
episode limit, and remains continuous between encounters within an episode.

Smoke and training logs include completed encounter attempts, successes,
failures and timeouts; maximum encounter index; maximum physical bar position
and orientation error; command-head and orientation-adapter gradient norms;
and WM, actor and critic losses. Falls and timeouts close an active bar attempt
as failures. A successful attempt requires a natural encounter completion,
recorded passage, no measured bar contact, and continued survival.

Short validation runs establish execution, finite gradients and event
accounting. They do not measure trained performance at the new task. The chase
evaluation/render scripts describe the original chase task and its scenario
layout; a velocity/dodge checkpoint requires its own evaluation protocol.

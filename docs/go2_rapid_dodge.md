# Go2 Rapid Directional Bars

This profile extends the velocity-command task within the existing TrackerLab
task, environment, MDP, buffer, algorithm, and metrics layout. It retains a
24-dimensional task input and a 15-dimensional replay scenario.

## Training

```bash
python scripts/rls/supermimic/supervise.py \
  --config scripts/rls/supermimic/configs/go2_rapid_dodge_8096.json \
  --motion_config data/configs/go2_kine2go.json \
  --warm_start checkpoints/chase_008000.pt \
  --run_dir logs/go2_rapid_8096_10k \
  --num_envs 8096 --iterations 10000 --ckpt_every 1000 \
  --milestones 1000 3000 6000 10000
```

The actor and world model initialize from the chase checkpoint; the critic and
optimizers initialize for the new objective. Preflight weights are not used.
The Kine2Go `ai4_dog_trot_00` clip provides reset states, cyclic gait goals, and a
small gait-style term. There is no jump or duck demonstration in this dataset.
Avoidance and command tracking use the task rewards.

The curriculum begins with 1.0-1.2 second encounter periods and 0.38/0.08 metre
high/low bar centers. After 100 world-model-only updates, difficulty advances
over 4000 updates toward 0.5-1.0 second periods, high-bar centers of 0.31-0.39
metres, and low-bar centers of 0.08-0.18 metres. Approach directions span the
full circle at final difficulty. Body commands cover forward speed -0.6 to
1.5 m/s, lateral speed +/-0.8 m/s, and yaw rate +/-1.5 rad/s. A command persists
for four encounters; each encounter draws a new bar.

## Timing And Geometry

Each bar has a fixed world trajectory during its lifetime. Let `T` denote the
scheduled period, `L = 0.45 T` the nominal lead time, `n` its horizontal normal,
and `D(t)` the commanded constant-body-twist displacement. Bar speed `s` is
chosen to satisfy:

```text
s >= (1.1 - dot(D(L), n)) / L
s >= (0.9 - dot(D(T) - D(L), n)) / (T - L)
```

These constraints provide at least 1.1 m initial normal distance and 0.9 m
predicted exit clearance. They can raise speed above the 3.0-4.5 m/s base draw;
the configured command ranges imply a speed below the validated 7 m/s limit.
Nominal warning at final difficulty is 0.225-0.45 seconds.

There is one physical bar, 6 m long and 5 cm thick. Its replacement waits for
the previous bar to clear the robot region. Actual intervals can therefore
exceed the configured period. Forward-side expiry and lateral bypass count as
failed attempts when no qualifying crossing occurred. A bar change preserves
robot state, observation history, and reference phase.

## Evaluation

Milestones run 16 predetermined, continuous 16-second trials at final
difficulty. Trial 0 is chosen for the video before evaluating outcomes; all
trials are saved. Reports include command RMSE, contacts, falls, event outcomes,
actual replacement and root-crossing intervals, deadline delays, and raw
critic quantiles. A low-bar label alone does not establish airborne clearance.
Reported timing is measured from simulator events rather than playback speed.

Positive reward maxima sum to 0.1 per control step. With `gamma = 0.99`, the
critic bootstrap upper bound remains `0.1 / (1 - 0.99) = 10`. Changing obstacle
frequency does not change this reward bound.

The supplied video demonstrates the desired behavior. Its unpublished training
recipe is not a dependency of this local warm-start experiment.

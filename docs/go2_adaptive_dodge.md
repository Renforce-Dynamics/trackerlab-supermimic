# Go2 Autonomous Bar Passage

This downstream SuperMimic task trains one Go2 policy to choose how to pass
moving bars. Height and direction are observations. Neither the actor nor the
learned dynamics receives a jump/duck category. The prior target-height feature
is constant at the nominal standing height; it carries no maneuver instruction.

The `adaptive` profile lives in the existing TrackerLab task / MDP / simulator /
algorithm / evaluation data flow. The `standard`, `rapid` and `jump` profiles
retain their own settings. This is a local task extension, not a result from
the SuperMimic paper.

## Schedule

- 8096 parallel environments, 30000 additional task updates.
- Random 1.0-2.0 second encounter periods throughout training.
- Start with low bar centers around 0.035-0.055 m and high centers around
  0.41-0.45 m. Both groups vary from the first rollout.
- Endpoint center-height ranges are 0.04-0.15 m and 0.30-0.46 m.
- Approach angles expand from a narrow frontal sector to all horizontal
  directions. Each 100-update window must include at least 250 completed
  attempts per height group and at least 65% success in each group to advance.
- The geometry may raise the sampled bar travel speed to meet the requested
  cadence, bounded by 3.0 m/s. The actual interval is measured separately;
  a bar overlapping the robot delays the next spawn.

The 1-2 setting means seconds between encounters, not a 1-2 m/s speed limit.
Full-direction performance remains unproven until evaluated at difficulty 1.

## Objective And Transfer

The reference remains Kine2Go `ai4_dog_trot_00`, used for resets, phase goals
and a small style term. There is no jump demonstration in this experiment.
Actor and dynamics weights transfer from rapid iteration 10000, which has
experience with mixed heights. A new critic and new optimizers match the changed
reward. The jump-only iteration 6000 checkpoint remains archived.

Clearance evaluates body, shin, foot and torso samples against the bar. For
each sample the smooth safety scores for above/below passage are computed.
A soft minimum aggregates the samples within each route; a smooth maximum
chooses a whole-body route. This prevents rewarding a torso above the bar and
feet below it while an intervening leg intersects the obstacle.

The clearance term receives 0.060 of the 0.100 positive per-step reward budget.
Speed, standing height and trot shaping relax near a bar. Geometric collision
penalty doubles to 0.20; PhysX force-based collision and crossing determine the
success metric and the curriculum. The sensor force itself is not a
differentiable reward. Dynamics are learned from physical transitions.

The imagination horizon grows from 32 to 64 control steps (0.64 to 1.28 s).
This provides more time for takeoff and landing inside a rollout, at additional
compute cost and potentially larger model error. Real versus imagined reward,
falls, contact and passage rates must be monitored. No claim of unchanged
training throughput is made.

With gamma=0.99 the positive-return ceiling remains 0.1/(1-0.99)=10. Previous
jump training reported zero value-clipped samples; value clipping alone does
not explain its zero passage success.

## Evaluation

The same sixteen predetermined 16-second PhysX trials are retained.
Reports label scenarios `high_bar` and `low_bar`. Successful observed maneuvers
are classified from recorded geometry as `under`, `airborne_over`, or
`step_over`. Airborne means all four foot centers exceed 0.04 m for at least
two consecutive control samples within 0.45 m of the bar plane; it is not a
measured foot-force event.
Unsuccessful attempts remain explicitly unsuccessful.

The all-attempt success rate includes unfinished horizon attempts as failures.
A separate rate covers naturally completed attempts. Each report records its
actual curriculum difficulty and the measured interval distribution.

```bash
python scripts/rls/supermimic/supervise.py \
  --config scripts/rls/supermimic/configs/go2_adaptive_dodge_8096.json \
  --motion_config data/configs/go2_kine2go.json \
  --warm_start_velocity /path/to/rapid/last.pt \
  --run_dir logs/go2_adaptive_8096_30k \
  --milestones 2000 6000 10000 20000 30000
```

Each milestone evaluates in a separate process while training continues.
Training ends at the documented 30000-update limit, an explicit STOP request,
or a reported failure.

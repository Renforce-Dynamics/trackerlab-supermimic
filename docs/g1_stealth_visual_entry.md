# G1 Visual-First Entry Pilot

## Owner Review Overrides Scalar Improvements

On November 8, 2026, the owner rejected the 5000-iteration LAFAN data
adaptation's startup: the robot must enter its crouched aiming posture
through a natural forward bend, without the current backward/sideways twist.
Visual review has priority over speed, survival, root-pitch numbers,
discriminator scores and world-model loss. A visual rejection holds
continuation and export even if numeric guards pass.

The earlier claim that entry had improved referred only to reduced backward
root pitch. It omitted chest orientation and lateral sway and must not be
treated as visual approval. The full policy remains rejected for startup,
backward foot dragging and arm appearance.

## Measured Failure

Matched seed101 first attempts use identical initial qpos, physical context
and commands. The adapted forward entry at 1.50 s has waist yaw -116.47 deg
and roll -29.21 deg, while the pelvis heading changes about 2 deg.
Across the nine command starts, absolute chest side-lean peaks are
17.84--26.72 deg versus 5.07--11.88 deg in the old32000 policy.
The chest can face sideways/backward even when its up axis projects forward
along the pelvis heading. Root pitch or one projected lean angle cannot
describe the visual posture.

The arm waypoint controller constrains 14 shoulder/elbow/wrist coordinates.
All three waist outputs remained learned and could make the body turn under
those fixed arms. Existing velocity/height entry ramps and an AMP fade-in
do not specify a forward-facing waist path. This explains an unconstrained
mechanism, not an isolated causal proof.

Complete 50 Hz first-entry FK and front/side pose-replay evidence are stored
locally under `artifacts/g1-stealth-entry-visual-20261108/`. A native video
player could not be controlled because the Computer Use host was unavailable.
Actual chronological frames and matched multiview re-renders were inspected;
do not claim uninterrupted realtime video observation from that workflow.

## One Scoped Control Change

The opt-in `--waist-entry-reference` constrains the three named waist joints
using the same scalar entry clock already present in the policy goal.
`prepare_stealth_waist_entry.py` selects one actual forward-expert sample
nearest the componentwise three-waist-joint median in L1 distance. Source
manifest/CSV hashes, frame identity and timing derivation are recorded and
verified at training startup.

Current source:

```text
forward__aiming2_subject5__f02830-02891.csv
waist yaw   -0.03092534938 rad
waist roll  -0.02035302637 rad
waist pitch +0.52 rad
```

A quintic target moves from default standing waist coordinates to this
sample over 1.02 s. That time is the minimum common duration satisfying
the existing 1.2 rad/s and 3 rad/s^2 target bounds, rounded upward to 20 ms.
The controller then holds those waist coordinates through the crouched skill,
including key release. It does not switch abruptly back to unadapted raw
waist outputs after entry. This is explicit persistent waist posture control,
not learned waist motion or a recorded standing-to-crouch demonstration.
All twelve leg channels remain learned. Physical offsets, tracking errors,
base tilt and whole-body balance are still determined by the simulator.

The waist controller composes with the existing arm controller in collection,
imagined rollouts, evaluation and actor export. The actor/world-model/critic
input dimensions are unchanged. Old checkpoints omit this option and retain
their old actions. Resume requires identical saved waist configuration;
changing waist control requires an explicitly new warm-start experiment.

The pilot leaves the expert pool, AMP mode/weight/history, rewards, arm
waypoints, 4.84-second command/height ramp, physical randomization and
normalization settings unchanged. The optional signed-vx AMP feature remains
inactive. The existing recovery warm-start path inherits actor/WM weights,
resets critic and optimizers, and restores AMP only for matching experts.
These transfers remain explicit in its initialization receipt.

## Bounded Pilot And Visual Hold

`run_g1_visual_entry.sh` schedules exactly 1500 training iterations,
including 500 world-model-only and 1000 actor iterations. It waits for the
shared GPU to become idle and never stops an unrelated job. The final
checkpoint receives seed101/103 cycle evaluations with grounded standing
starts and all nine modes recorded. Each first start is rendered in front
and side views. The controller stops at
`awaiting_visual_entry_review_no_extension`.

No further iteration budget or policy export is automatic. The first pilot
tests entry only; it cannot qualify natural reverse steps, command changes
or release behavior. The old backward-drag failure remains outstanding.

`check_stealth_visual_review.py` validates explicit human-review criteria
and binds the decision to checkpoint, trajectory and video SHA256 values.
Entry review requires forward crouch, no lateral sway and natural arms with
front/side video evidence. Full review additionally requires backward steps,
no dragging, command changes and retained crouch on release.
Missing, partial, negative or stale review blocks approval. A utility cannot
prove a reviewer watched a video or automatically judge naturalness.

Kinematic target previews carry a persistent red
`KINEMATIC TARGET ONLY - NOT A POLICY ROLLOUT` label. They are never substituted
for the recorded-physics trial and cannot satisfy physical/visual acceptance.
Even a physically tested pilot remains rejected until its actual front and
side motion looks natural. Preserve all failed predecessors and receipts.

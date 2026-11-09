# G1 Stealth: Learned Arms And Waist

The owner's November10 proposal is feasible with the existing trainer:
learned arms, end-effector AMP features,1.21sentry fade and learned waist.
This is a prepared comparison recipe. The active recovery's5,000additional
actor-update budget and saved-source identity are preserved; this preset
has not been physically trained or visually accepted.

`configs/command_stealth_autonomous.json` disables fixed upper-body targets,
omits any waist-entry reference and uses nominal waist gains. All29raw joint
outputs belong to the actor. The existing differentiable arm filter applies
rate/acceleration limits only to14arm joints, with no pose target. All three
waist outputs and12leg outputs pass through unchanged, including gradients.

`postural_end_effectors` already contains:

- projected gravity and terrain-relative root height;
- joint positions and velocities;
- tracked-body positions relative to the root, including both feet and wrists;
- both wrist-link orientations in6D;
- ten chronological frames per AMP history.

These are discriminator training/scoring features, not reference-motion inputs
to the actor. They use model link origins; measured palm/finger/weapon landmarks
would require a separate physical definition. Source natural-arm channels must
be retained if arm style is to be learned; the old fixed-arm derivative does
not reproduce natural mocap arm motion.

The1.21svalue is the existing quintic fade for command/clearance/style reward.
It supplies no scripted forward bend and guarantees no learned arrival time.
Natural transition, wrist posture, stepping, turning and release require actual
physics videos. Learned waist control alone has not been shown to cause or
eliminate backward lean.

The earlier20kfree-arm trial still had persistent waist targets andPD4.
Consequently its failure does not evaluate this complete combination.
The exact natural-arm/learned-waist/nominal-gain/1.21scombination remains untested.

For a new training experiment, choose a source with learned waist outputs and
the matching nominal motor contract. Use the natural-arm motion pool and a
fresh run, with the existing explicit warm-start behavior: preserve actor
normalization and leg/waist rows, initialize the formerly overridden14arm
output rows, retain compatible WM weights, fit a new AMP if the expert identity
changes, and reset critic/actor-WM optimizers as recorded by the trainer.
This is a controlled adaptation, not an exact full-state resume.

The preset retains the recovery's command/reward coefficients and timing
distribution. Budget5500includes500WM-only plus5000actor-enabled iterations.
Midpoint/final videos must cover allnineWASDmodes and release; early valid
quality failures remain observations under the owner's budget instruction.
No automatic source selection, queued launch or policy export is created by
this configuration-only change.

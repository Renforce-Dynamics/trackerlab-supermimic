# G1 Stealth Turning Reference Extension

The existing G1 stealth AMP pool contains forward, backward and idle references.
This optional extension adds an Aiming2 turning interval and its sagittal
mirror. AMP samples every expert role independently of the command; these
filenames do not select a separate reward for A/D. The actor, world model,
command layout and task reward definitions retain their existing interfaces.

The selected source is `aiming2_subject2.bvh`, source frames `[200, 261)`.
Its retargeted 50 Hz interval has about 20 degrees of root yaw change over
2.0 seconds. The source is retained for `turn_left`; `turn_right` is derived
by a named G1 left/right reflection. The reflection changes world `y` to `-y`,
swaps corresponding left/right joints and applies the signs implied by the
G1 joint axes. It does not alter time sampling or root speed.

Full-source inspection on November 14 found that this is a translating arc:
the original two-second window moves the root 0.636m with mean forward/lateral
velocities about0.245/0.184m/s. The prepared G1 pair retains about0.461m
displacement and0.179/+/-0.131m/s velocities. It supplies no demonstrated
zero-translation pivot. The earlier first500-frame scan was insufficient to
characterize the three full9172-frame Aiming2 recordings. A complete scan found
no windows satisfying the declared combined slow-pivot/style/lift filters;
that result is specific to those filters, not proof of impossibility.

The preparation command also requires the original BVH, the retarget origin
manifest and the selected full-body pool. The generator bundles those source
bytes and their SHA-256 records into `turning-reference-audit.json`, so a
later verifier can reproduce the chronological donor and the mirrored clip.

Each turn clip is passed through the existing full-body solver with the
reviewed selected crouched aiming pose. The solver keeps the arms at that
pose while fitting root height, waist and feet. The output pool is accepted
only after the existing FK, sole, limit, self-penetration and torso checks
pass for every frame.

The new schema requires all five roles, rechecks the actual mirror against its
chronological source, and preserves the original three motions, aiming reward
and heldout bytes. The complete G1 reflection checks include all named bodies,
the eight sole spheres, reordered29joint layouts and temporal velocities.
The construction fixes arm coordinates offline; runtime actions remain learned.

Five ten-frame expert pools provide465valid windows:97forward and92each for
backward, idle and both arcs. Their approximate sampling shares change from
34.5/32.7/32.7percent to20.9/19.8/19.8/19.8/19.8percent. New data also change
the reference-reset distribution and require a fresh discriminator. An ordinary
warm start retains actor/compatible WM weights and frozen actor normalization,
and resets critic and task optimizers. This is not a full-state continuation.

The initial external retry launcher was retired before any trainer started.
The translating-arc experiment remains untrained pending the pivot-source
comparison. `run_g1_stealth_combined.sh` now records its PID, source and input
receipts before waiting for repository/shared locks, honors STOP while waiting,
and treats lock errors as terminal. Its existing budget remains500WM-only plus
5000actor updates and eleven final command/release movies.

Source and reference audit evidence resides in the workspace's
`artifacts/g1-stealth-turn-fullscan-20261114` and
`artifacts/g1-stealth-turn-audit-20261114`. Reference playback is kinematic;
physical learning and visual policy acceptance remain unproved.

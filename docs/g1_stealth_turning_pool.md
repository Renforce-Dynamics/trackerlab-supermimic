# G1 Stealth Turning Reference Extension

The current G1 stealth AMP pool contains forward, backward and idle references.
This extension adds a low-speed Aiming2 turning interval and a sagittal mirror
of the same interval for the two turn commands. The actor, world model,
command layout and task rewards stay unchanged; only the expert history pool
changes.

The selected source is `aiming2_subject2.bvh`, source frames `[200, 261)`.
Its retargeted 50 Hz interval has about 20 degrees of root yaw change over
2.0 seconds. The source is retained for `turn_left`; `turn_right` is derived
by a named G1 left/right reflection. The reflection changes world `y` to `-y`,
swaps corresponding left/right joints and applies the signs implied by the
G1 joint axes. It does not alter time sampling or root speed.

The preparation command also requires the original BVH, the retarget origin
manifest and the selected full-body pool. The generator bundles those source
bytes and their SHA-256 records into `turning-reference-audit.json`, so a
later verifier can reproduce the chronological donor and the mirrored clip.

Each turn clip is passed through the existing full-body solver with the
reviewed selected crouched aiming pose. The solver keeps the arms at that
pose while fitting root height, waist and feet. The output pool is accepted
only after the existing FK, sole, limit, self-penetration and torso checks
pass for every frame.

The extension is deliberately small. It supplies a turning prior for AMP while
leaving command tracking responsible for the requested yaw rate. Physical
training and visual policy acceptance remain separate steps.

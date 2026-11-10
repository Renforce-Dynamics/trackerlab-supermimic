# One Aim Style Across Commands

The owner selected the pose shown at approximately2.27s in the current
six-second Aiming subject5 library preview. This recipe retains velocity-led
WASD control while using one upper-body style for forward, reverse, turning
and crouched idle.

## Source Identity

The current library asset is `motion_9fbbcfe3539e42d0`, action data
`actiondata_58b86b8139a54265`. Its displayed canonical document SHA256 is
`6095c299915e19c8afd04034eb3c4d3b54d8bf60854648e6a2e53e7019980766`;
frame68at30Hz corresponds to2.266667s.

The label retains LAFAN Aiming subject5. The actual display document is a
fixed-body SMPL-X derivative, recorded as `kimodo_smplx`, originating from
inputSHA256 `058fc5f3e41fb0f6889271af07a784982cea950175fcea62b5f342b59d6b982f`.
It differs from the historical centered6s benchmark crop. Original BVH
frame4496..4676and a guessed60..240crop are not the selected training target.
Their diagnostic retargets remain excluded.

`retarget_selected_aim.py` adapts the exact displayed pose using anatomical
hip/shoulder/spine axes and morphology-scaled MinkIK. Arm targets use the
inner90%of joint ranges. Hand-length directions use the canonical SMPL-X
left+X/right-X convention and G1 fixed-hand mesh axes. The G1 model has fixed
fingers; finger shape and a held object are outside this pose representation.
A display-only constant floor shift avoids rendering target foot penetration.
The selected full-body pose is not a policy rollout.

## Data And Objective

`prepare_stealth_shared_aim.py` creates an immutable derivative of the
verified forward/backward/idle pool. All three training clips receive the same
14arm and3waist coordinates from the selected pose. Original root, leg
coordinates and timestamps remain numerically identical. The Aiming1holdout
is copied byte-for-byte.

This is shared upper-body supervision, not three copies of the selected
whole-body motion or a validated reverse-playback gait. Keeping chronological
leg examples preserves the currently learned command behavior while resolving
the contradictory hand poses.

AMPweight0.1replaces0.05; all task/gait/control settings remain the velocity-led
recipe. Changed expert identity initializes a fresh discriminator and its
normalizer/optimizer. Actor, compatibleWMand actor normalization transfer
without changing any output row. Critic and task optimizers reset for the
changed objective.

An additional bounded nonpositive aim cost, at most0.01perstep, uses:

- wrist positions relative to torso, scale0.05m;
- wrist rotations relative to torso;
- fourteen arm joint angles;
- torso up direction with root yaw removed;
- small arm-speed cost.

The same target applies to every command. Its activation fades in over the
existing0.60sentry and remains active on release. The module supplies training
rewards and evaluation metrics only. No action override or runtime reference
is added; all29joint actions remain learned. Soft costs allow deviations and
do not guarantee exact pose invariance.

Physical collection and imagined post-step rewards include the identical
aim module. Checkpoints store the complete calibrated target and source hashes.
Full resume rejects missing or changed aim targets. Evaluation reconstructs
the saved target and reports overall plus per-command aim errors.

## Launch And Acceptance

Use `run_g1_stealth_shared_aim.sh` on main with a verified motion pool containing
`aim-style-reference.json`. It uses the existing locks, fixed source commit
checks, stop receipts and first-attempt review protocol. Budget is500WM-only
steps plus50000actor updates. No new task ID is registered.

Kinematic checks of all303derived training frames found no robot self-contact
penetration exceeding0.5mm. This is a sampled geometric check, not proof of
collision-free policy transitions. Required visual review includes consistent
hands across W/S/A/D/arcs/release, crouched torso, startup, bilateral stepping,
slip and command tracking. Numerical aim errors or AMP scores do not release
an export without reviewing physical videos.

# Selected Aim Palm Correction

The November 10 shared-aim target independently scaled the two arm chains and
constrained a whole-hand length proxy. That changed the selected source's
relative wrist displacement and left palm roll unconstrained. Zero contacts
among enabled robot colliders did not cover the visual-only rubber palms.

`retarget_selected_aim.py` now calibrates full palm frames from the exact
source SMPL-X model's shaped wrist/metacarpal roots and the G1 proximal palm
planes/mesh thumb lobes. It transfers one shoulder-relative wrist midpoint
and one relative displacement using the mean arm-length ratio. Elbows remain
soft preferences. Existing inner 90% arm limits remain in force.

An explicit IK mesh-distance constraint includes the visual-only palms and
restores their original collision masks. This does not change the physical
training model. A candidate that exceeds 20 mm wrist error, 0.15 rad palm
rotation error or falls below 3.5 mm convex palm clearance cannot enter the
shared-aim preparation workflow. Convex mesh clearance is a conservative
geometric test; physical rollout and visual review remain necessary.

The selected canonical is asset `motion_9fbbcfe3539e42d0`, frame 68 at 30 Hz,
SHA256 `6095c299915e19c8afd04034eb3c4d3b54d8bf60854648e6a2e53e7019980766`.
The source body SHA256 is
`376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.
Licensed body assets and motion derivatives remain local research data.

Generate an immutable target and inspect its front, side and oblique views:

```bash
python scripts/rls/supermimic/retarget_selected_aim.py \
  --canonical SELECTED_CANONICAL --body-model LICENSED_SMPLX_NEUTRAL_NPZ \
  --output NEW_TARGET_DIRECTORY/selected-pose.csv
```

The reviewed v5 target has 3.21/3.26 mm wrist position errors,
4.21/3.83 degree rotation errors and 4.00 mm visual palm separation.
Its wrist distance is 68.84 mm, versus 103.27 mm in the rejected target.
G1's fixed open fingers cannot reproduce the human surface's curled fingers.

The new pool replaces the 17 shared waist/arm coordinates only. Source root,
leg motion, timing and held-out bytes remain unchanged. Actor inputs and all
29 learned outputs are unchanged. AMP stays at 0.1, the gait cost at 0.02,
entry at 0.6 seconds, and the source-bound shared-aim cost at at most 0.01.
The source pool and reward metadata have new immutable hashes. Warm-start
keeps the existing actor and compatible WM, with fresh AMP and task optimizers;
full resume of an older reference is rejected. No quality or hardware-use
acceptance follows from this kinematic target.

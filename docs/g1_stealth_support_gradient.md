# G1 Stealth: Sole-Support Gradient Trial

The completed signed-vx AMP trial improved forward speed from0.122to0.229m/s
but still produced zero sustained2cm whole-foot lifts in the recorded
forward/backward2..18s windows. Backward speed was-0.305m/s for a-0.20command.
Its source, checkpoints and videos are preserved; it is not visually accepted.

This independent trial starts from the owner-preferred20kentry baseline.
The owner requested longer training on November9:5000additional iterations,
absolute20001..25000, with a physical midpoint review at22000. The full
5000budget is authorized; the midpoint alone does not certify or extend it.

## Single Change

`differentiate_support=True` retains the derivative of the existing smooth
sole support weight with respect to foot height and slope. The default remains
False for old recipes. Reward values are exactly identical in both modes;
only the actor's imagined gradient through support changes. It also affects
the support-conditioned foot-tilt term, which uses the same support weights.

The existing G1 sole geometry, conditional mean normalization,0.04slide weight,
0.08spacing weight and other penalties are retained. The actor, WM, critic,
AMP and all their optimizers restore from the20kcheckpoint with no resets.
Reference clips, posture-only AMP, frozen actor normalization, WASD commands,
domain randomization,1.21sentry and1.02swaist target ramp remain unchanged.

The isolated production-method gradient test matches finite differences when
enabled. The prior detached mode has zero direct support-height derivative.
Both modes have identical measured reward, unchanged horizontal-speed
derivatives, finite boundary derivatives and zero fully-airborne slide reward.
These contracts do not establish physical gait improvement.

## Resume And Review

Preset: `scripts/rls/supermimic/configs/command_stealth_support_gradient.json`.
Initial adaptation uses `--resume SOURCE_20000 --allow-support-gradient-change`.
It requires a fresh G1 stealth run, g1_sole, and a False-to-True transition.
Further same-objective resumes omit that one-time adaptation flag.

Intermediate22000 and final25000 use seed101,90physical worlds and all nine
held commands, plus18worlds and the42srelease sequence. Each video preserves
its first recorded episode, including failures. Compare whole-sole clearance,
material-point sliding, torso entry, arms, speed/yaw tracking and release.

This gradient can encourage foot slope, brief support removal or hopping.
The conditional mean still has a flat equal-speed region and an all-airborne
boundary. Review those behaviors explicitly. The experiment changes no
contact model and introduces no new positive foot-height reward.

Runtime inference stays a velocity-command policy with the same action
controller; this training-only reward gradient adds no policy observation.
No automatic policy export or training beyond25000 is included.

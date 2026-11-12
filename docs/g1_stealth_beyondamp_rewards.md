# G1 Stealth Clearance And Palm Separation

The completed shared-aim50k trial tracked89/90held-command episodes but retained
asymmetric forward stepping and low backward clearance. A fixed first attempt
over2..18seconds had33/0forward and0/0backward sustained whole-sole lifts above
2cm. The existing support/swing cost could give little height gradient far
below its8mm support transition. The reference's4mm palm gap was also smaller
than observed wrist-pose errors.

This trial adds two training objectives while retaining the existing task,
SuperMimic world-model policy update and AMP prior. It registers no new task
and adds no PPO implementation.

## Verified Upstream

The owner-requested `Renforce-Dynamics/beyondAMP` source was inspected at
`cee88cdc0958c417e316f9452f802e25a71bc289` (master, May29,2026).
Relevant primary source files:

| File | Relevant Functions Or Settings |
| --- | --- |
| `source/amp_tasks/amp_tasks/velocity/mdp/rewards.py` | `foot_clearance_reward`, `feet_height_body`, `feet_gait`, `air_time_variance_penalty` |
| `source/amp_tasks/amp_tasks/velocity/robots/g1/velocity_env_cfg.py` | Period.8s, offsets0/.5, stance.55; feet-slide weight-.2; clearance weight1,target ankle height.1m,std.05,tanh multiplier2 |
| `source/amp_tasks/amp_tasks/velocity/robots/g1/velocity_less_reward_env_cfg.py` | A separate reduced-reward variant; it is not the classic G1 reward table |

The downloaded reward-file SHA256 is
`4d6529b0a736820f27af9099f160d2fc6897911163c1330185084cf75fa09adb`;
the G1 configuration SHA256 is
`fc2bf4b8e262362eff83d49b2bc1cc9c78c743f6cbe6f36de4475042bf4e954b`.
No upstream force/airtime sensor output is claimed to be predicted by our WM.

## Clearance Adaptation

The new opt-in `stealth_gait_version=2` keeps the existing single-swing/support
and flight/slip guards and adds the upstream velocity-weighted height idea.
For each foot, h is its lowest collision-sole height above terrain, and v is
the mean horizontal material velocity of its sole points. Both are normalized
by robot scale. Target d=.04m:

```text
height_cost = (h-d)^2 / ((h-d)^2+d^2)
speed_gate = tanh(2*norm(v))
extra_cost = 2*gait_weight * sum(height_cost*speed_gate)
```

The cost is subtracted and fades in with the existing.6sentry. It is active
for translation or yaw commands and zero for idle. Each moving foot receives
its own target; a planted stationary support foot receives no extra height
cost. Excessive lift is also penalized. Negative/low clearance retains a
usable ascent gradient, unlike a saturated binary-support approximation.
The existing flight penalty remains active at the start of entry.

This is a bounded whole-sole adaptation of BeyondAMP's clearance objective,
not its ankle-origin target or unchanged numerical coefficients. Periodic
contact rewards and contact-airtime variance are not copied: the current
command entry clock clamps at.8s and the WM does not predict contact timers.
An external phase contract would require a separate, explicit change.
Existing foot-slide, foot-tilt, action-rate, joint-acceleration, joint-limit
and body-rotation regularization remain active. No external motion is needed
at policy execution.

## Palm Separation

Optional source-bound palm geometry accompanies the existing aiming reference.
It supplies a nonpositive penalty for insufficient separation of the actual
fixed rubber-hand convex hulls along the selected pose's separating plane.
Its vertices, direction, reference identity and settings are serialized.
It changes reward gradients, with all29joint outputs still learned.
The source aiming pose, finger meshes and simulation collision masks remain.
The fixed-plane test is conservative when another separating plane could exist.

Both new costs are evaluated from FK/state tensors in physical rewards and
imagined rewards. AMP and WM parameters remain frozen during actor gradient
updates, while gradients through predicted states remain enabled. These
changes do not add force/contact predictions or alter actor input dimensions.
Full resume rejects changed gait settings or reference metadata.

## Trial

`command_stealth_beyondamp.json` uses the same reviewed velocity-led10500
checkpoint that initialized the prior hand trial. Actor, compatible WM and
actor normalization are preserved. No arm-output reset or hard waist/arm
override is used. Changed expert identity uses fresh AMP; task optimizers
and critic follow the existing warm-start contract.

AMP remains.1,gaitweight.02,swingtarget.04m,entry.6s. Allnine WASD modes,
domain randomization and command speeds are retained. The controller budgets
500WM-only plus20,000actorupdates and records2500/5500/10500/20500.
The first full quality comparison requires at least5000actorupdates.
Videos must show actual bilateral lift, restrained slip, natural entry,
consistent hand posture and separation across command changes.
Offline reward ordering or finite gradients alone do not establish that result.

An explicitly authorized extension uses `SUPERMIMIC_STEALTH_EXTRA_UPDATES=5000`
with the same wrapper and its full-resume sixth argument. It extends the
cumulative actor budget to25,000 and the total target to25,500. The completed
20,500 checkpoint can resume directly, including its optimizer states and
normalizers; no WM-only warmup is repeated. A missing resume checkpoint or
an incompatible recipe is rejected. Allninecommand/release recordings remain.

## Preflight Evidence

The recorded-state audit compared5,208samples of the reviewed10500policy and
the rejected50500policy. The original gait-plus-slide cost already preferred
the better recordings, while the original gait term alone preferred the
asymmetric final forward recording. With version2, the good-minus-final
mean gait-plus-slide advantage increased from.000543to.005612forward and
from.003135to.005833backward. This strengthens an existing preference rather
than proving the earlier failure was caused by one coefficient. The extra
height term also encourages lower foot speed away from its target; physical
command tracking remains an explicit acceptance check.

Training FK and independent MuJoCo sole geometry agreed within2.59e-9m.
The existing G1 representative-foot offset is zero; no double-counted offset
was found. Recorded root rates were estimated because full root qvel was not
stored; recorded joint rates were retained.

On32recorded-source states, the added cost produced finite nonzero action
gradients through the actual source WM. Whole-sole height prediction MAE was
.244mm at one step,.516mm at8steps and1.441mm at32recorded-action steps.
This checks state prediction and local sensitivity, not full policy-gradient
fidelity or physical learning.

The complete palm hulls contain3191/2966vertices. Exact support witnesses are
selected in bounded batches without retaining projection matrices for backward;
ties select one valid support subgradient. On the actual RTX4090, a131072-state
compiled forward/backward reward batch averaged6.99ms and added.0687GiB peak
allocated memory. No hull vertices were discarded. Total trainer resource use
still requires observing the launched run.

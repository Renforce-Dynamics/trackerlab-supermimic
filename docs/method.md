# Method And Reward Provenance

All eight motion-tracking rewards are in **SuperMimic Appendix A, Table A.2**.
They were already present in the earlier TrackerLab aggregate reward.
The component instrumentation exposes their individual values; it does not
add objectives invented by the public reproduction.

| Term | Weight | Exponential error scale |
| --- | ---: | ---: |
| Body position | 1 | 0.3 m |
| Body orientation | 1 | 0.4 rad |
| Body linear velocity | 1 | 1.0 m/s |
| Body angular velocity | 1 | 3.14 rad/s |
| Root position | 0.5 | 0.3 m |
| Root orientation | 0.5 | 0.4 rad |
| Squared action change, summed over joints | -0.1 | none |
| Linear soft-limit violation, summed over joints | -2 | none |

Each contribution is multiplied by the 0.02-second control period.
Maximum reward per step is 0.1. As specified by the paper, the self-collision
penalty is omitted. Table A.6 concerns velocity-command control and is
outside this full-reference tracking task.

The public implementation supplies concrete choices where the paper leaves
details open. This port follows its XY/yaw alignment for body position and
orientation, world-frame velocity targets, and 90% joint soft limits.
`validate_rewards.py` checks ten analytic cases and optionally compares
actual G1 states, reward gradients and other MDP heads with pinned vibed.

## Actor Input And Reset

The current default uses vibed's 99D input: 93D proprioception plus six
reference-relative root-orientation features. The paper lists 93D
proprioception while leaving the complete goal coordinates less explicit.
The 99D mode is an implementation choice, not an assertion about the
paper authors' unreleased encoding.

`paper_obs93` is retained for older checkpoints. Combining that old
observation with reference-only goals left global heading error unobservable
in the earlier TrackerLab implementation. A diagnostic rotation of all
history frames by 90 degrees left its action unchanged while altering reward.
The current relative-orientation input observes that change.

Training resets follow vibed's perturbations: root position
`+/- (0.05, 0.05, 0.01)` m, rotation vector `+/- (0.1, 0.1, 0.2)` rad,
linear velocity `+/- (0.5, 0.5, 0.2)` m/s, angular velocity
`+/- (0.52, 0.52, 0.78)` rad/s and joint position `+/- 0.1` rad.
Joint resets are clamped to physical limits. Evaluation has no perturbations.

Friction and torso CoM are sampled per environment; joint calibration is
sampled per episode, following vibed's cadence. Height error above 0.25 m or
gravity-component tilt error above 0.8 terminates an episode.
Training truncates after 500 control steps; full-clip evaluation has no
training timeout.

## Training Budget

| Setting | Value |
| --- | ---: |
| Environments | 4096 or 8096 by preset |
| Outer iterations | 320000 or 100000 by preset |
| Collection | 2.5 vector steps per outer iteration on average |
| World model | 10 batches x 2048 eight-step windows |
| Policy | 4096 imagined trajectories, horizon 32 |
| Critic | 5 epochs x 4 minibatches |
| Replay | 1000 steps per environment |
| World-model-only prefix | 500 iterations |
| Learning rate | 0.0002 |
| Schedule | cosine to 0.03x over 250000 iterations |
| Discount / GAE | 0.99 / 0.95 |
| Collection action noise | Gaussian sigma 0.1 |
| Arithmetic | bf16 MLPs, fp32 state/reward computation |

A larger environment count does not increase the policy or world-model
optimizer batches. Running normalizers follow vibed's freeze count of
200 million samples. The learning-rate schedule is not compressed for a
shorter run.

## Scope Of Differences

Physics backend, asset conversion and solver settings differ from vibed.
The TrackerLab reset sampler averages returns when multiple environments
finish in the same bin. Terminal transitions and partial reset histories
are explicitly retained. Multiple settings changed between the historical
93D run and the current run, with one seed each; the observed improvement
does not isolate a causal share for observation, reset noise or environment
count.

Reward parity is a local numeric check. It does not establish equal simulated
trajectories, training speed, hardware transfer or final benchmark results.

# Go2 Divergence Investigation

Observed on 2026-10-23 with the shared SuperMimic learner and PhysX port.
The first Go2 run used four procedural IK trot references, not Kine2Go.
It stopped at iteration 8,174; models and logs were retained. G1 stopped
by user request at 42,638 and has not been resumed.

## Mechanism And Evidence

The actor optimizes rewards along a learned-model rollout plus a critic tail
estimate. Critic targets also bootstrap from that critic. Excessive predictions
can attract the actor into poorly modeled states and increase later targets.
The six positive reward terms sum to at most 0.1 per step; remaining terms
are nonpositive. At gamma=0.99, the discounted value cannot exceed 10.
The old Go2 actor objective exceeded 10 at iteration 700 and reached 3,325.88
at 1,000 despite negative physical reward. That is an optimization failure.

Bootstrapping alone does not mathematically imply amplification: exact
tabular Bellman updates with gamma below one contract value errors. Here
the targets, sampled imagined states, nonlinear function approximator and
actor-induced state distribution evolve together. The observations identify
an unstable approximate actor/critic interaction, not a proof that every
recursive value update will diverge.

Same-iteration historical logs:

| Run / iteration | Actor objective | Critic MSE | Imagined reward / 0.1 | PhysX reward / 0.1 |
| --- | ---: | ---: | ---: | ---: |
| G1 / 700 | 1.6358 | 0.05539 | 0.19836 | 0.25553 |
| Original Go2 / 700 | 13.3939 | 81.2253 | 0.35166 | 0.35765 |
| G1 / 1,000 | 4.0136 | 0.04016 | 0.35808 | 0.39876 |
| Original Go2 / 1,000 | 3,325.88 | 4,815,040 | -3.21778 | -3.03105 |

G1 and Go2 used the same model architecture, learning rate 2e-4, horizon 32,
world-model window 8, batches and 500-iteration model-only warmup.
G1 had 8,096 environments; the original Go2 run had 4,096.
Robot mechanics, reward scales, resets and reference trajectories differ.
Go2 warmup mostly visited the neighborhood of standing poses, with long
episodes; actor updates sharply increased world-model error. G1 warmup had
short episodes and frequent falls. These distribution differences help
explain why a shared algorithm can behave differently, but do not isolate
the initial cause. IK data's dynamic feasibility is not established by IK.

## Controlled Intervention

Two fresh runs kept seed 0, four procedural motions, 4,096 environments,
reward, dynamics and schedules identical, for 1,200 iterations:

- Action-only: smooth action bound 4 for real and imagined policy outputs.
- Action-and-value: identical, with critic bootstrap and actor tail capped
  at the analytic upper bound 10. Regression output remains unclipped.

| Iteration | Action-only critic MSE | Action-and-value critic MSE | Action-only PhysX reward / 0.1 | Action-and-value PhysX reward / 0.1 |
| --- | ---: | ---: | ---: | ---: |
| 650 | 2.49484 | 0.0420498 | 0.461399 | 0.523714 |
| 700 | 14.7728 | 0.0433273 | 0.297262 | 0.546018 |
| 800 | 512.205 | 0.0447987 | 0.191720 | 0.568674 |
| 1,000 | 184,865 | 0.00574815 | -1.22449 | 0.858199 |
| 1,200 | 2,833,590 | 0.00493169 | -1.01649 | 0.896082 |

At 1,200, actor objectives were 3,354.79 versus 8.6027. Both runs had zero
skipped policy updates. This isolates a substantial benefit from the value
bound under these conditions, even without changing the reference data.
It does not establish performance across seeds, robots or longer runs.
Training-step reward above is distinct from deterministic full-clip evaluation.

Deterministic first-frame evaluation on all four 20-second clips gave
full-clip return -0.870403 for action-only (one physical failure), and
0.923566 for action-and-value (zero failures). In the latter, world-X speeds
were 0.3428, 0.5013, 0.6691 and 0.8310 m/s for references 0.35, 0.50, 0.65
and 0.80 m/s. Joint RMSE ranged from 0.0550 to 0.0615 rad. Root-position
RMSE increased to 0.3661 m on the fastest clip; successful survival is not
perfect tracking.

A third run used action limit 4 and gamma=0.97, without a value cap.
At iteration 1,200, critic MSE was 0.0101601, actor objective 2.91023 and
training reward / 0.1 was 0.917156. Deterministic full-clip return was
0.920531 with zero failures. This supports sensitivity to discounted
bootstrapping, but gamma also affects critic targets, actor rewards and
reset-sampling return estimates. It is not an isolated intervention on
the actor tail alone. The public-data run keeps gamma=0.99.

The fourth run retained the original unbounded policy output (real collection
still clips at 20) and changed only value handling to the upper bound. At
1,200, critic MSE was 0.00527406, training reward / 0.1 was 0.888899 and
full-clip return was 0.882540 with zero failures. Thus the value intervention
works without the new action bound in this seed/task; the action bound alone
did not prevent divergence.

| 1,200-iteration condition | Gamma | New action bound | Value cap | Full-clip return | Failed / 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Action only | 0.99 | 4 | None | -0.870403 | 1 |
| Value only | 0.99 | None | 10 | 0.882540 | 0 |
| Action and value | 0.99 | 4 | 10 | 0.923566 | 0 |
| Reduced discount | 0.97 | 4 | None | 0.920531 | 0 |

All four conditions used the same procedural references, seed, 4,096
environments and 12,550,144 simulator transitions. Floating-point GPU
execution is not promised to be bitwise reproducible. The failed initial
value-only launch was a preset-serialization error before training; its
successful retry used an omitted action-limit option, the original default.

## Why G1 And Go2 Differ

The paper's equation (2) weights the terminal critic by `gamma^H * c_H`.
At iteration 600 the logged mean multiplier was 0.27859 for G1 and
0.64277 for original Go2 (2.31x). At 550 it was 0.21710 versus 0.48893.
The initial Go2 policy can remain upright for much longer while following
motion poorly; the current height/tilt termination allows this. It therefore
exposes the still-developing critic more strongly to policy optimization.
G1's early failures attenuate the tail; its tail influence grows more gradually.
By 1,000 G1's multiplier also reaches 0.68442, with critic MSE still 0.04016.
This supports a difference in learning trajectories, not a permanent immunity
of humanoids to overestimation. A multiplier is not the measured fraction of
the total objective contributed by the critic.

The source libraries also have markedly different diversity. G1 has 40 clips
and 8,822 seconds; the original Go2 library has 80 seconds of four repeated
trots. Among sampled future-goal features, 172/390 Go2 dimensions have
standard deviation below 0.01 (130 nearly constant), versus 3/730 G1
dimensions. This measures reference variation, not the full critic-input
distribution. Low variation alone does not prove a normalization bug.
Together with the sharp Go2 world-model error increase after iteration 500,
it motivates testing sensitivity to distribution changes.

Robot support geometry, inertia, actuator response and task scales alter
the dynamics derivatives seen by BPTT. They are not merely cosmetic changes.
However, these factors have not been independently isolated. Single-seed
logs do not identify a unique causal explanation for the G1/Go2 difference.

## Mapping The Changes To Paper Equations

In equation (2), replace terminal `V_psi(s_H,g_H,xi)` by
`min(V_psi(s_H,g_H,xi), 0.1/(1-gamma))`. In equation (3), use the same
bounded estimate for `V_(t+1)` and the initialization `R_H=V_H`.
The left-hand regression prediction `V_t` stays raw. Gradients through the
over-limit tail value become zero, while the actor still receives rollout
reward gradients and survival-weight gradients. The latter still multiply
the clipped value. Values below the limit are unchanged.

Separately, policy output changes from raw `u_theta` to
`a=4*tanh(u_theta/4)`. Its derivative is `1-tanh(u_theta/4)^2`; it is near
one for small output and shrinks near saturation. With Go2 action scale 0.25,
the desired joint offset is bounded by one radian, apart from randomized
calibration offsets. The shared world-model fit (equation 1), imagination
horizon and BPTT algorithm remain intact. Both modifications are extensions,
not clipping rules claimed to appear in the paper.

## Fidelity And Efficiency

Go2 is an extension, not an upstream Go2 experiment. The bound is an explicit
algorithm option added after observing divergence. G1 defaults remain
unbounded; old checkpoints retain their original behavior. The upper bound
follows from the configured reward, while action limiting is a separate
modeling choice that can restrict motion. Both must be recorded.

The extra elementwise work is small relative to network and simulation work,
but an isolated wall-clock benchmark has not been performed. The controlled
runs shared one GPU, so their timings should not be treated as independent
throughput measurements. At equal sample counts, the bounded-value run
improved reward while the other diverged. General sample-efficiency gains
need longer runs and additional seeds. Raw losses alone cannot prove gait
quality; full-clip metrics and recorded physical motion are also inspected.

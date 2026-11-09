# G1 Stealth Gait Regression Recovery

November 9, 2026. The 25k support-gradient continuation remains visually
rejected. This recovery uses the preserved entry-v4 10k stepping policy.
It is a bounded experiment, not an accepted WASD skill.

## Evidence

Fresh frozen-policy evaluations use the same seed101, 24 worlds, a held
forward command of 0.25m/s, physical randomization and 20-second episodes.
Foot measurements cover the fixed first six attempts, seconds5..18.

| Saved policy | Mean vx | Sole slip | Falls | Bilateral sustained lifts |
| --- | ---: | ---: | ---: | --- |
| Entry-v4 10k | .2072 | .0850 | 0/24 | 6/6 recordings |
| WASD-v2 32k | .2097 | .1097 | 0/24 | 3/6 recordings |
| Three-clip reverse adaptation5k | .1852 | .1328 | 0/24 | 1/6 recordings |
| Persistent-waist pilot1500 | .1797 | .1305 | 0/24 | 1/6 recordings |

The late free-arm20k and support25k recordings have zero sustained whole-foot
lifts during straight forward/backward motion. The main loss of stepping is
already visible during expert/learning-state adaptation, before the free-arm
trial. This lineage comparison does not isolate data from learning-state resets.

Separate frozen entry10k counterfactuals with the later subject5 waist target
produced47/49falls with persistent hold/PD4,12/26 with finite release/PD4, and
10/27 with finite release/PD1. Even using a target from the original source
clip produced19/34falls with finite release/PD1. These frozen-controller
changes are rejected. No waist-guidance graft is used in the recovery recipe.

## Narrow Recovery

Source:
`/root/gpufree-data/runs/g1-stealth-entry-v4-pilot-20261106/training/iteration_010000.pt`
SHA256 `55e36fb80141d55fc388c25efd37e04ffcc89bfad9f250dde2e965a7b1c26998`.

Use the original single-forward expert pool
`data/style-research-20261106/g1-stealth-entry-v4/pools/g1/stealth-b70d9df1a19b8e60468c`.
The original actor/WM/AMP transfer, including the matching AMP optimizer.
Actor normalization, fixed arms, leg/motor parameters and observation layout
are preserved. Critic and actor/WM optimizers follow the explicit warm-start
path and reset. All nine existing WASD-v2 commands are trained at actorLR2e-5.
Backward naturalness and idle behavior remain to be evaluated.

The original actor controls all three waist joints. The original4.84s command
clock and4.82s fixed-arm entry remain during this isolation experiment.
This recovery stage therefore retains the historical slow entry; the owner's
preferred faster, forward-bending transition remains a separate open requirement.
No arm rows or world model are freshly initialized; no idle/backward expert,
AMP feature, reward weight, gait phase or positive swing reward is added.

The production controller, AMP implementation and training math are unchanged.
Experimental finite-waist code is excluded after its failed physical tests.
Original and failed checkpoints, data and evidence remain preserved.

## Execution And Review

The initial recovery stopped at1000 total iterations:500 WM-only and only500
actor updates. Held-forward speed was.17487m/s, with0/24falls; the six recorded
first attempts had median sustained lift counts5.5left/0right versus5/3.5 at
original10k. This is an early regression observation, not a completed adaptation.

The owner's subsequent instruction on November9 requires at least5000 iterations
before judging this recovery. That instruction supersedes the early quality stop.

`run_g1_gait_restore.sh PYTHON RUN INITIAL POOL BASELINE SOURCE_COMMIT [RESUME]`
uses `command_stealth_gait_restore.json`. With the saved1000 checkpoint it
restores actor/world/critic/AMP and all optimizer states, trains to3000 for an
intermediate review, then to6000. This adds exactly5000 actor-enabled iterations,
without repeating warmup or resetting networks/optimizers.
Fresh runs instead use500 WM-only plus5000 actor-enabled iterations to5500.
The budget is saved in `training_budget.json`.

All-nine commands, old motion pool, physical randomization, control mappings,
normalizers, learning rates and rewards are unchanged by this continuation.
Repeatedly resetting or changing objectives would prevent measuring whether
the existing adaptation recovers its stepping after adequate training.

`check_stealth_step_retention.py` supplements the existing speed/fall check:
retain at least half of each foot's median sustained-lift count and at least
75% of the baseline's bilateral-stepping fraction across the fixed recordings.
Counts use whole sole above2cm for at least80ms. Early-ending first attempts
count as zero. Its `--diagnostic-only` mode retains failed quality verdicts in
the receipt while continuing the authorized budget. Malformed evidence,
different evaluation protocols, trainer/runtime errors and explicit STOP still
stop the controller. Quality failure is never relabeled as a pass.

Each milestone records held-forward24worlds, all-nine90worlds and42-second
release18worlds, plus original front/side videos. The controller owns both GPU
and paused-scheduler locks. It ends at the saved target, without exporting or
claiming visual acceptance. Entry, gait, hands, turning and release still need
visual review.

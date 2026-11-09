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

`run_g1_gait_restore.sh PYTHON RUN INITIAL POOL BASELINE SOURCE_COMMIT`
uses `command_stealth_gait_restore.json`. It runs1000then2000 total new iterations,
including500 WM-only warmup, and records forward/all-nine/release videos.
It owns both GPU and stopped-scheduler locks and honors STOP while training.
It neither exports a policy nor trains beyond2000.

`check_stealth_step_retention.py` supplements the existing speed/fall check:
retain at least half of each foot's median sustained-lift count and at least
75% of the baseline's bilateral-stepping fraction across the fixed recordings.
Counts use whole sole above2cm for at least80ms. Early-ending first attempts
count as zero. A failed guard prevents the next training stage after preserving
the current recordings. These are regression barriers; human visual review of
entry, gait, hands, turning and release remains necessary.

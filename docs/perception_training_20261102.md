# Perception Training Run

## Code And Scope

The training code was pushed to `main` as
`839dda8ad5d411045a7bf3e7f542a90215f898ec` using the ZouYan93 GitHub
credential. It adds stairs, mountain and parkour profiles for the six
SuperMimic tracking robots: G1, Go2, A3, T1, Bumi and Pi+. Each policy receives
187 idealized MuJoCo-Warp terrain height rays.
There is no RGB/depth encoder or realistic sensor-occlusion model in this
campaign, and these simulation results do not establish hardware transfer.

MicroDuck has a separate native `mjlab` task and a 61-dimensional observation
contract with no terrain-sensor inputs. It is outside this 18-task campaign and
needs a dedicated observation and training integration before claiming
perception coverage for every robot family in the repository.

## Environment And Admission

The isolated server worktree is
`/root/autodl-tmp/work/trackerlab-supermimic-perception-train-20261102`,
detached at the `main` commit above. Its Python 3.11 environment is
`/root/autodl-tmp/envs/supermimic-perception-20261102`. Pinned components are
PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, MuJoCo and MuJoCo-Warp 3.16.0,
Warp 1.18.0 and RSL-RL 2.3.3. `pip check` passed. CUDA matrix execution,
Blackwell device discovery, all six robot model loads and terrain-ray checks
passed.

The server reported zero GPU memory use and zero utilization immediately
before launch. Admission evidence and exact package versions are in
`docs/perception_receipts_20261102/admission.json`.

## Training Campaign

The detached background campaign started on November 2, 2026 at 21:02 China
Standard Time (13:02 UTC). It runs 4,096 environments on the RTX PRO 6000
Blackwell, sequentially covering all 18 robot/profile combinations. Each
combination has a 1,000-update checkpoint followed by a 10,000-update target;
the second stage resumes the first, so the target is 10,000 total updates per
task. Evaluation and TorchScript export follow each completed stage. `--video`
records an evaluation rollout for visual review.

At the November 2, 2026 13:23:00 UTC status snapshot, G1 stairs was training
update 650/1,000, with terrain curriculum mean 4.99/5. The supervising campaign
PID was 133875 and the active trainer PID was 133876. This snapshot is preserved
in `docs/perception_receipts_20261102/status-650.json`. Live checkpoints, logs and campaign
receipts remain on the server under:

`/root/autodl-tmp/work/trackerlab-supermimic-perception-train-20261102/logs/perception-campaign-20261102`

Read `campaign.json` for the active task and queue state. Read each task's
`status.json` and `train_log.csv` for update-level metrics. Create a `STOP` file
in the campaign directory to request a checkpointed stop. The supervisor
preserves completed task outputs and records the stopped state.

The expected campaign duration is several days. It is sequential on one GPU;
the estimate changes with per-robot simulation and evaluation time.

## First Campaign Milestone

G1 stairs completed 1,000 updates at 13:33:51 UTC on November 2, 2026, in
1,852.74 seconds including initialization. Its evaluation and TorchScript
export both exited successfully; the supervisor marked the stage completed
at 13:34:38 UTC and launched G1 mountain. Subsequent profiles for each robot
warm-start from that robot's first completed task; the second training stage
resumes each profile's own checkpoint.

On held-out seed 123, the new G1 checkpoint succeeded in 24/24 normal-ray
episodes, 2/24 zero-ray episodes and 7/24 spatial-reversal episodes. These
are small-sample simulation diagnostics. The source audit recorded 14 files
with no changed or missing files. This stage's video has been generated and
its visual review remains pending. These results concern the first 1,000
updates, while the campaign's per-task target remains 10,000 total updates.

The checkpoint hash, evaluation and completed-stage throughput receipts are
in `docs/perception_receipts_20261102/g1-stairs-1000/`. The training-only
summary retains `evaluation_pending: true`; the separate evaluation receipt
and supervisor exit codes record the subsequent successful evaluation.

## Validation And Throughput

All 18 tasks completed a four-update training, held-out evaluation and
TorchScript export smoke campaign with zero failed jobs before the final
mutually exclusive terminal-outcome fix. At the final code commit, all six
stair robot models passed CPU/GPU terrain-ray comparison,
partial-reset isolation, finite-state and perception-ablation validation.
Maximum measured ray-distance error was below 0.000001 m.
Local tests at that commit reported 162 passed, one skipped and one existing
tensor-to-scalar warning; `git diff --check` passed.

The pre-existing G1 stair calibration checkpoint was evaluated on held-out
seed 123. It succeeded in 22/24 normal-observation episodes, 3/24 zero-ray
episodes and 7/24 spatially shuffled-ray episodes. The calibration checkpoint
predates this campaign; these results are diagnostic evidence, not the outcome
of the new long run. The evaluation receipt records source differences from
that calibration run; this is not a same-source controlled comparison.

For the completed G1 stairs stage, updates 101-1,000 measured 52,899 environment
transitions/second and 0.538 PPO updates/second, excluding the first 100
updates as warmup. The early interval covering updates 101-298 measured
58,423 transitions/second; both receipts are retained. The historical Tele
figure is 2.47115 SuperMimic outer iterations/second. These values describe
distinct workloads: each PPO update includes 24 simulator control steps and
20 optimizer minibatches, while the Tele iteration includes world-model and
imagined policy/critic updates. They do not establish iteration-rate parity.

Detailed receipts, including evaluation metrics and throughput, are stored in
`docs/perception_receipts_20261102/`. Six sampled frames from the G1 calibration
video were reviewed in a contact sheet, rather than reviewing the full video.
The robot and stairs are visible; forward lean and raised/swinging arms remain.
The local workspace evidence directory, outside the Git worktree, is
`artifacts/perception-preflight-20261102/g1-calibration-evaluation-main/`.
The generated evaluation receipt retains its original pending-review field;
this paragraph records the subsequent sampled-frame review.

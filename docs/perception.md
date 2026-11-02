# Terrain-Perception Locomotion

This extension adds 18 velocity locomotion tasks: stairs, mountain and parkour
profiles for G1, Go2, A3, T1, Bumi and Pi+. The actor receives 187 height
measurements from MuJoCo-Warp terrain rays. Training uses RSL-RL PPO with the
existing robot joint, actuator and morphology definitions.

The sensor model is an idealized height scanner. These tasks do not use RGB or
depth cameras, model sensor occlusion, or establish hardware transfer. Evaluation
includes held-out terrain seeds and normal/zero/shuffled-ray ablations.

MicroDuck remains on its separate native locomotion path. Its upstream policy
family requires a shared 61-dimensional observation contract; the current
MicroDuck runtime has no terrain-sensor input slot.

## Prepare And Validate

Install the pinned perception requirements in the dedicated Linux GPU
environment. Install the matching CUDA PyTorch wheel before the requirements:

```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-perception.txt
python scripts/rls/perception/prepare.py --robot all
python scripts/rls/perception/validate.py \
  --tasks locomotion/g1/stairs locomotion/go2/stairs locomotion/a3/stairs \
         locomotion/t1/stairs locomotion/bumi/stairs locomotion/pi_plus/stairs \
  --output logs/perception-validation.json
```

Each task has a four-update, 16-environment smoke run. Full training uses
4,096 environments and staged 1,000/10,000-update checkpoints:

```bash
python scripts/rls/perception/train.py --task locomotion/g1/stairs \
  --smoke --run-dir logs/perception-smoke-g1-stairs
python scripts/rls/perception/campaign.py --run-dir logs/perception-all \
  --num-envs 4096 --stages 1000 10000
```

The campaign owns the GPU lock, records each process and output, evaluates,
exports and measures every completed task. Create a `STOP` file in the campaign
directory to request a checkpointed stop after the current PPO update.

## Throughput

The G1 4,096-environment calibration completed 1,100 updates in 2,048.98
seconds including initialization. A measured interval recorded 2.195 seconds
per PPO update and 44,780 environment transitions per second.

The Tele reference reports 2.47115 SuperMimic outer updates per second, with
world-model and imagined policy/critic updates inside each outer iteration.
Those iteration rates describe different workloads. The campaign writes PPO
updates/second and transitions/second with its timing protocol and preserves the
Tele value as a separate reference; it makes no iteration-parity claim.

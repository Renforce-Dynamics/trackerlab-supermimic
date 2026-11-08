#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
    printf 'usage: run_g1_visual_entry.sh PYTHON RUN POOL SOURCE ARM_REFERENCE WAIST_REFERENCE\n' >&2
    exit 2
fi
python=$1
run=$2
pool=$3
source=$4
arms=$5
waist=$6
repo=$(git rev-parse --show-toplevel)
config=$repo/scripts/rls/supermimic/configs/command_stealth_visual_waist.json
test -x "$python"
for input in "$pool/train.json" "$source" "$arms" "$waist" "$config"; do
    test -f "$input"
done
test ! -e "$run/controller.phase"
mkdir -p "$run" "$repo/logs"
trap 'printf "%s\n" "$?" > "$run/controller.exit"' EXIT
exec 9>"$repo/logs/.supermimic-terrain-gpu0.lock"
flock 9
git diff --exit-code
git diff --cached --exit-code
test -z "$(git ls-files --others --exclude-standard)"
git rev-parse HEAD > "$run/code.receipt"
sha256sum "$source" "$pool/manifest.json" "$arms" "$waist" "$config" > "$run/input.receipt"

# The unrelated shared-GPU queue uses another lock; admit this bounded job only
# after three successive idle observations, without stopping any other job.
printf 'waiting_for_idle_gpu\n' > "$run/controller.phase"
quiet=0
while [ "$quiet" -lt 3 ]; do
    test ! -e "$run/STOP"
    compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    IFS=',' read -r used utilization < <(
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0
    )
    if [ -z "$compute" ] && [ "$used" -lt 1024 ] && [ "$utilization" -lt 10 ]; then
        quiet=$((quiet + 1))
    else
        quiet=0
    fi
    if [ "$quiet" -lt 3 ]; then sleep 15; fi
done
printf 'training_1500_iteration_visual_pilot\n' > "$run/controller.phase"
"$python" -u scripts/rls/supermimic/train_terrain.py \
    --task locomotion/g1/stealth --config "$config" --run-dir "$run/training" \
    --warm-start "$source" --amp-motion-config "$pool/train.json" \
    --upper-body-reference "$arms" --waist-entry-reference "$waist" \
    --iterations 1500 --buffer-steps 500 --keep-every 250 --ckpt-every 250 --log-every 25 \
    > "$run/train.stdout.log" 2>&1
test ! -e "$run/STOP"
test ! -e "$run/training/STOP"
checkpoint=$run/training/iteration_001500.pt
test -f "$checkpoint"
printf 'recording_entry_pilot\n' > "$run/controller.phase"
for seed in 101 103; do
    review=$run/review/seed$seed
    mkdir -p "$review"
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 90 --steps 400 \
        --mode-layout cycle --record-env-ids 0 1 2 3 4 5 6 7 8 \
        --output "$review/evaluation.json" --trajectory-output "$review/trajectories.npz" \
        > "$review/evaluate.stdout.log" 2>&1
    for lane in 0 1 2 3 4 5 6 7 8; do
        "$python" scripts/rls/supermimic/render_stealth_entry_views.py \
            --trajectory "$review/trajectories.npz" --output "$review/lane$lane" \
            --lane "$lane" --seconds 6 \
            > "$review/render-lane$lane.stdout.log" 2>&1
    done
done
# No automatic continuation or export. Entry approval cannot qualify backwards
# gait, command changes, release behavior or a full locomotion policy.
printf 'awaiting_visual_entry_review_no_extension\n' > "$run/controller.phase"

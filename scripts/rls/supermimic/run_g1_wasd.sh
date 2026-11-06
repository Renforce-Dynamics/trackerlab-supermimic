#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    printf 'usage: run_g1_wasd.sh PYTHON RUN_DIRECTORY EXPERT_POOL INITIAL_CHECKPOINT [ITERATIONS...]\n' >&2
    exit 2
fi

python=$1
run=$2
pool=$3
initial=$4
shift 4
milestones=("$@")
if [ "${#milestones[@]}" -eq 0 ]; then
    milestones=(1000 4000 10000 20000)
fi
for target in "${milestones[@]}"; do
    if [[ ! "$target" =~ ^[1-9][0-9]*$ ]] || (( target % 1000 != 0 )); then
        printf 'Milestones must be positive multiples of 1000.\n' >&2
        exit 2
    fi
done
repo=$(git rev-parse --show-toplevel)
config=$repo/scripts/rls/supermimic/configs/command_stealth_wasd_natural.json

test -x "$python"
test -f "$initial"
test -f "$pool/train.json"
mkdir -p "$run" "$repo/logs"
exec 9>"$repo/logs/.supermimic-terrain-gpu0.lock"
flock -n 9

finish() {
    printf '%s\n' "$?" > "$run/controller.exit"
}
trap finish EXIT

git rev-parse HEAD > "$run/code.receipt"
git diff --exit-code
git diff --cached --exit-code
test -z "$(git ls-files --others --exclude-standard)"
sha256sum "$initial" "$pool/manifest.json" \
    > "$run/input.receipt"

train_to() {
    local target=$1
    local source=(--warm-start "$initial")
    if [ -f "$run/training/last.pt" ]; then
        source=(--resume "$run/training/last.pt")
    fi
    printf 'training_to_%s\n' "$target" > "$run/controller.phase"
    "$python" -u scripts/rls/supermimic/train_terrain.py \
        --task locomotion/g1/stealth --config "$config" \
        --run-dir "$run/training" "${source[@]}" \
        --amp-motion-config "$pool/train.json" \
        --iterations "$target" --buffer-steps 500 \
        --keep-every 1000 --ckpt-every 250 --log-every 25 \
        > "$run/train-to-$target.stdout.log" 2>&1
}

review() {
    local target=$1
    local seed=$2
    local checkpoint="$run/training/iteration_$(printf '%06d' "$target").pt"
    local directory="$run/review/iter$(printf '%06d' "$target")/seed$seed"
    printf 'review_%s_seed%s\n' "$target" "$seed" > "$run/controller.phase"
    mkdir -p "$directory/cycle" "$directory/release"
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 90 --steps 1500 \
        --mode-layout cycle --record-env-ids 0 1 2 3 4 5 6 7 8 \
        --output "$directory/cycle/evaluation.json" \
        --trajectory-output "$directory/cycle/trajectories.npz" \
        > "$directory/cycle/evaluate.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 18 --steps 2100 \
        --command-sequence wasd-release-regression --command-stage-seconds 6 \
        --record-env-ids 0 1 2 3 4 5 \
        --output "$directory/release/evaluation.json" \
        --trajectory-output "$directory/release/trajectories.npz" \
        > "$directory/release/evaluate.stdout.log" 2>&1
    for lane in 0 1 2 3 4 5 6 7 8; do
        local view="$directory/cycle/lane$lane"
        mkdir -p "$view"
        cp "$directory/cycle/evaluation.json" "$directory/cycle/trajectories.npz" "$view/"
        "$python" scripts/rls/supermimic/render_terrain.py \
            --directory "$view" --output "$view" --lane "$lane" \
            > "$view/render.stdout.log" 2>&1
    done
    "$python" scripts/rls/supermimic/render_terrain.py \
        --directory "$directory/release" --output "$directory/release" --lane 0 \
        > "$directory/release/render.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/export_command_policy.py \
        --checkpoint "$checkpoint" --output "$run/export-iter$(printf '%06d' "$target")" \
        > "$directory/export.stdout.log" 2>&1
}

for target in "${milestones[@]}"; do
    test ! -e "$run/STOP"
    train_to "$target"
    test ! -e "$run/training/STOP"
    review "$target" 101
done
if [ "${milestones[-1]}" -ge 20000 ]; then
    review "${milestones[-1]}" 103
fi
printf 'completed_pending_visual_review\n' > "$run/controller.phase"

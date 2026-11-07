#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 7 ]; then
    printf 'usage: run_g1_entry_extension.sh PYTHON OUTPUT SOURCE POOL BASELINE BANK BANK_PROTOCOL [MILESTONES...]\n' >&2
    exit 2
fi
python=$1
output=$2
source=$3
pool=$4
baseline=$5
bank=$6
bank_protocol=$7
shift 7
milestones=("$@")
if [ "${#milestones[@]}" -eq 0 ]; then
    milestones=(12000 15000 20000)
fi
previous=10000
for target in "${milestones[@]}"; do
    if [[ ! "$target" =~ ^[1-9][0-9]*$ ]] || (( target % 1000 != 0 || target <= previous )); then
        printf 'Use increasing milestones above 10000, divisible by 1000.\n' >&2
        exit 2
    fi
    previous=$target
done
repo=$(git rev-parse --show-toplevel)
config=$repo/scripts/rls/supermimic/configs/command_stealth_entry.json
test -x "$python"
for file in "$source" "$pool/train.json" "$pool/upper-body-reference.json" "$baseline" "$bank" "$bank_protocol"; do
    test -f "$file"
done
test ! -e "$output/controller.phase"
mkdir -p "$output" "$repo/logs"
trap 'printf "%s\n" "$?" > "$output/controller.exit"' EXIT
printf 'waiting_for_gpu\n' > "$output/controller.phase"
exec 9>"$repo/logs/.supermimic-terrain-gpu0.lock"
flock 9
git diff --exit-code
git diff --cached --exit-code
test -z "$(git ls-files --others --exclude-standard)"
git rev-parse HEAD > "$output/code.receipt"
sha256sum "$source" "$pool/manifest.json" "$pool/upper-body-reference.json" "$baseline" "$bank" "$bank_protocol" \
    > "$output/input.receipt"

review() {
    local checkpoint=$1
    local directory=$2
    local seed=$3
    mkdir -p "$directory/forward" "$directory/cycle"
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 24 --steps 1000 \
        --command-mode forward --record-env-ids 0 1 2 3 4 5 \
        --output "$directory/forward/evaluation.json" \
        --trajectory-output "$directory/forward/trajectories.npz" \
        > "$directory/forward/evaluate.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/render_terrain.py \
        --directory "$directory/forward" --output "$directory/forward" --lane 0 \
        > "$directory/forward/render.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 90 --steps 1500 \
        --mode-layout cycle --record-env-ids 0 1 2 3 4 \
        --output "$directory/cycle/evaluation.json" \
        --trajectory-output "$directory/cycle/trajectories.npz" \
        > "$directory/cycle/evaluate.stdout.log" 2>&1
    for lane in 0 1 2 3 4; do
        local view=$directory/cycle/lane$lane
        mkdir -p "$view"
        cp "$directory/cycle/evaluation.json" "$directory/cycle/trajectories.npz" "$view/"
        "$python" scripts/rls/supermimic/render_terrain.py \
            --directory "$view" --output "$view" --lane "$lane" \
            > "$view/render.stdout.log" 2>&1
    done
}

resume=$source
for target in "${milestones[@]}"; do
    test ! -e "$output/STOP"
    test ! -e "$output/training/STOP"
    printf 'training_to_%s\n' "$target" > "$output/controller.phase"
    "$python" -u scripts/rls/supermimic/train_terrain.py \
        --task locomotion/g1/stealth --config "$config" \
        --run-dir "$output/training" --resume "$resume" \
        --amp-motion-config "$pool/train.json" --upper-body-reference "$pool/upper-body-reference.json" \
        --iterations "$target" --buffer-steps 500 --keep-every 1000 --ckpt-every 250 --log-every 25 \
        > "$output/train-to-$target.stdout.log" 2>&1
    test ! -e "$output/training/STOP"
    tag=$(printf '%06d' "$target")
    checkpoint=$output/training/iteration_$tag.pt
    directory=$output/review/iter$tag
    printf 'review_%s_seed101\n' "$target" > "$output/controller.phase"
    review "$checkpoint" "$directory/seed101" 101
    "$python" scripts/rls/supermimic/validate_command_world.py \
        --checkpoint "$checkpoint" --reference-checkpoint "$source" \
        --bank "$bank" --bank-protocol "$bank_protocol" \
        --output "$directory/world-validation.json" > "$directory/world-validation.stdout.log" 2>&1
    if ! "$python" scripts/rls/supermimic/check_wasd_recovery.py \
            --baseline "$baseline" --candidate "$directory/seed101/forward/evaluation.json" \
            --output "$directory/forward-gate.json" > "$directory/forward-gate.stdout.log" 2>&1; then
        printf 'forward_gate_failed_%s\n' "$target" > "$output/controller.phase"
        exit 2
    fi
    "$python" scripts/rls/supermimic/export_command_policy.py \
        --checkpoint "$checkpoint" --output "$output/export-iter$tag" \
        > "$directory/export.stdout.log" 2>&1
    resume=$output/training/last.pt
done
test ! -e "$output/STOP"
printf 'review_%s_seed103\n' "${milestones[-1]}" > "$output/controller.phase"
review "$checkpoint" "$directory/seed103" 103
printf 'completed_pending_visual_review\n' > "$output/controller.phase"

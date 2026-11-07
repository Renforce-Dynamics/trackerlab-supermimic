#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    printf 'usage: run_g1_command_curriculum.sh PYTHON OUTPUT SOURCE_12000 POOL BASELINE BANK BANK_PROTOCOL\n' >&2
    exit 2
fi
python=$1
output=$2
source=$3
pool=$4
baseline=$5
bank=$6
bank_protocol=$7
repo=$(git rev-parse --show-toplevel)
config=$repo/scripts/rls/supermimic/configs/command_stealth_entry.json
test -x "$python"
for file in "$source" "$pool/train.json" "$pool/manifest.json" "$pool/upper-body-reference.json" "$baseline" "$bank" "$bank_protocol"; do
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
sha256sum "$source" "$pool/manifest.json" "$pool/upper-body-reference.json" "$baseline" "$bank" "$bank_protocol" "$config" > "$output/input.receipt"

stop_requested() {
    if [ -e "$output/STOP" ] || [ -e "$output/training/STOP" ]; then
        printf 'stopped_by_request\n' > "$output/controller.phase"
        exit 0
    fi
}

review() {
    local checkpoint=$1 directory=$2 seed=$3
    mkdir -p "$directory/forward" "$directory/cycle" "$directory/release"
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 24 --steps 1000 \
        --command-mode forward --record-env-ids 0 1 2 3 4 5 \
        --output "$directory/forward/evaluation.json" --trajectory-output "$directory/forward/trajectories.npz" \
        > "$directory/forward/evaluate.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/render_terrain.py \
        --directory "$directory/forward" --output "$directory/forward" --lane 0 \
        > "$directory/forward/render.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 90 --steps 1500 \
        --mode-layout cycle --record-env-ids 0 1 2 3 4 5 6 7 8 \
        --output "$directory/cycle/evaluation.json" --trajectory-output "$directory/cycle/trajectories.npz" \
        > "$directory/cycle/evaluate.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 18 --steps 2100 \
        --command-sequence wasd-release-regression --command-stage-seconds 6 \
        --record-env-ids 0 1 2 3 4 5 \
        --output "$directory/release/evaluation.json" --trajectory-output "$directory/release/trajectories.npz" \
        > "$directory/release/evaluate.stdout.log" 2>&1
    for lane in 0 1 2 3 4 5 6 7 8; do
        local view=$directory/cycle/lane$lane
        mkdir -p "$view"
        cp "$directory/cycle/evaluation.json" "$directory/cycle/trajectories.npz" "$view/"
        "$python" scripts/rls/supermimic/render_terrain.py \
            --directory "$view" --output "$view" --lane "$lane" \
            > "$view/render.stdout.log" 2>&1
    done
    "$python" scripts/rls/supermimic/render_terrain.py \
        --directory "$directory/release" --output "$directory/release" --lane 0 \
        > "$directory/release/render.stdout.log" 2>&1
}

# Retain all earlier modes when adding idle, reverse, then reverse arcs.
modes=(forward turn_left turn_right forward_left forward_right idle)
resume=$source
previous=12000
for target in 13000 14000 15000 17000 19000 21000 23000 25000; do
    stop_requested
    if (( target == 14000 )); then
        modes+=(backward)
    elif (( target == 15000 )); then
        modes+=(backward_left backward_right)
    fi
    printf 'training_to_%s\n' "$target" > "$output/controller.phase"
    "$python" -u scripts/rls/supermimic/train_terrain.py \
        --task locomotion/g1/stealth --config "$config" \
        --run-dir "$output/training" --resume "$resume" \
        --stealth-control-mode wasd --command-reward-mode legacy \
        --command-training-modes "${modes[@]}" --allow-command-sampling-change \
        --amp-motion-config "$pool/train.json" --upper-body-reference "$pool/upper-body-reference.json" \
        --iterations "$target" --buffer-steps 500 --keep-every 1000 --ckpt-every 250 --log-every 25 \
        > "$output/train-to-$target.stdout.log" 2>&1
    stop_requested
    tag=$(printf '%06d' "$target")
    checkpoint=$output/training/iteration_$tag.pt
    directory=$output/review/iter$tag
    mkdir -p "$directory"
    cp "$output/training/run_config.json" "$output/training/run_summary.json" "$directory/"
    "$python" scripts/rls/supermimic/verify_command_continuation.py \
        --source "$resume" --checkpoint "$checkpoint" --source-iteration "$previous" \
        --output "$directory/continuity.json" > "$directory/continuity.stdout.log" 2>&1
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
    if ! "$python" scripts/rls/supermimic/check_command_curriculum.py \
            --cycle "$directory/seed101/cycle/evaluation.json" --training-modes "${modes[@]}" \
            --output "$directory/active-mode-gate.json" > "$directory/active-mode-gate.stdout.log" 2>&1; then
        printf 'active_mode_gate_failed_%s\n' "$target" > "$output/controller.phase"
        exit 2
    fi
    "$python" scripts/rls/supermimic/export_command_policy.py \
        --checkpoint "$checkpoint" --output "$output/export-iter$tag" \
        > "$directory/export.stdout.log" 2>&1
    resume=$checkpoint
    previous=$target
done
stop_requested
printf 'review_25000_seed103\n' > "$output/controller.phase"
review "$checkpoint" "$directory/seed103" 103
printf 'completed_pending_visual_and_command_acceptance\n' > "$output/controller.phase"

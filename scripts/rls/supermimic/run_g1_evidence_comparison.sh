#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    printf 'usage: run_g1_evidence_comparison.sh PYTHON OUTPUT SOURCE POOL BASELINE BANK BANK_PROTOCOL\n' >&2
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

for case in control command_margin_2; do
    test ! -e "$output/STOP"
    run=$output/$case
    mkdir -p "$run"
    options=()
    if [ "$case" = command_margin_2 ]; then
        options=(--command-margin-steps 2)
    fi
    printf 'training_%s_to_10500\n' "$case" > "$output/controller.phase"
    "$python" -u scripts/rls/supermimic/train_terrain.py \
        --task locomotion/g1/stealth --config "$config" \
        --run-dir "$run/training" --resume "$source" \
        --amp-motion-config "$pool/train.json" --upper-body-reference "$pool/upper-body-reference.json" \
        --iterations 10500 --buffer-steps 500 --keep-every 500 --ckpt-every 250 --log-every 25 \
        "${options[@]}" > "$run/train.stdout.log" 2>&1
    test ! -e "$run/training/STOP"
    checkpoint=$run/training/iteration_010500.pt
    printf 'review_%s\n' "$case" > "$output/controller.phase"
    mkdir -p "$run/forward"
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed 101 --num-envs 24 --steps 1000 \
        --command-mode forward --record-env-ids 0 1 2 3 4 5 \
        --output "$run/forward/evaluation.json" --trajectory-output "$run/forward/trajectories.npz" \
        > "$run/forward/evaluate.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/render_terrain.py \
        --directory "$run/forward" --output "$run/forward" --lane 0 \
        > "$run/forward/render.stdout.log" 2>&1
    set +e
    "$python" scripts/rls/supermimic/check_wasd_recovery.py \
        --baseline "$baseline" --candidate "$run/forward/evaluation.json" \
        --output "$run/forward-gate.json" > "$run/forward-gate.stdout.log" 2>&1
    result=$?
    set -e
    if [ "$result" -ne 0 ] && [ "$result" -ne 2 ]; then
        exit "$result"
    fi
    printf '%s\n' "$result" > "$run/forward-gate.exit"
    "$python" scripts/rls/supermimic/validate_command_world.py \
        --checkpoint "$checkpoint" --reference-checkpoint "$source" \
        --bank "$bank" --bank-protocol "$bank_protocol" \
        --output "$run/world-validation.json" > "$run/world-validation.stdout.log" 2>&1
done
printf 'comparison_complete_pending_review\n' > "$output/controller.phase"

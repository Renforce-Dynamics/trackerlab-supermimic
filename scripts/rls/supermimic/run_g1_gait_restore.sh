#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 6 ] || [ "$#" -gt 7 ]; then
    printf 'usage: run_g1_gait_restore.sh PYTHON RUN INITIAL POOL BASELINE SOURCE_COMMIT [RESUME]\n' >&2
    exit 2
fi
python=$1
run=$2
initial=$3
pool=$4
baseline=$5
pin=$6
resume=${7:-}
repo=$(git rev-parse --show-toplevel)
config=$repo/scripts/rls/supermimic/configs/command_stealth_gait_restore.json
test "$(git rev-parse HEAD)" = "$pin"
test -z "$(git status --porcelain)"
test ! -e "$run/controller.phase"
test -x "$python"
test -f "$baseline/checkpoint.receipt"
test "$(sha256sum "$initial" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$baseline/checkpoint.receipt")"
mkdir -p "$run" "$repo/logs"
exec 8>"$run/controller.lock"
flock -n 8
exec 9>"$repo/logs/.supermimic-terrain-gpu0.lock"
flock -n 9
exec 10>>/root/gpufree-data/ultra_ipingpong/queue/.runner.flock
flock -n 10
phase=initializing
set_phase() { phase=$1; printf '%s\n' "$phase" > "$run/controller.phase"; }
finish() {
    local status=$?
    trap - EXIT
    if [ "$status" -ne 0 ]; then set_phase "failed:$phase"; fi
    printf '%s\n' "$status" > "$run/controller.exit"
    exit "$status"
}
trap finish EXIT
check_stop() {
    if [ -e "$run/STOP" ] || [ -e "$run/training/STOP" ]; then
        set_phase stopped
        exit 0
    fi
}
printf '%s\n' "$BASHPID" > "$run/controller.pid"
printf '%s\n' "$pin" > "$run/code.receipt"
plan_options=()
if [ -n "$resume" ]; then plan_options=(--resume "$resume"); fi
budget=$("$python" scripts/rls/supermimic/gait_restore_budget.py \
    "${plan_options[@]}" --output "$run/training_budget.json")
read -r start final midpoint end <<< "$budget"
test "$end" = "$final"
printf '%s\n' "$start" > "$run/starting_iteration"
printf '%s\n' "$final" > "$run/target_iterations"
printf '5000\n' > "$run/additional_actor_updates"
sha256sum "$initial" "$pool/manifest.json" "$pool/train.json" \
    "$pool/upper-body-reference.json" "$config" \
    "$baseline/evaluation.json" "$baseline/trajectories.npz" > "$run/input.receipt"
if [ -n "$resume" ]; then sha256sum "$resume" >> "$run/input.receipt"; fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MKL_THREADING_LAYER=SEQUENTIAL MUJOCO_GL=egl
set_phase checking_original_stepping
"$python" scripts/rls/supermimic/check_stealth_step_retention.py \
    --baseline "$baseline" --candidate "$baseline" --output "$run/baseline-gate.json"
source=(--warm-start "$initial")
if [ -n "$resume" ]; then source=(--resume "$resume"); fi
for target in "$midpoint" "$final"; do
    check_stop
    test "$(git rev-parse HEAD)" = "$pin"
    test -z "$(git status --porcelain)"
    sha256sum -c "$run/input.receipt" > "$run/input-verified.receipt"
    tag=$(printf '%06d' "$target")
    stage=$run/milestones/$tag
    mkdir -p "$stage"
    set_phase "training_to_$target"
    "$python" -u scripts/rls/supermimic/train_terrain.py \
        --task locomotion/g1/stealth --config "$config" --run-dir "$run/training" \
        "${source[@]}" --amp-motion-config "$pool/train.json" \
        --upper-body-reference "$pool/upper-body-reference.json" \
        --iterations "$target" --buffer-steps 500 \
        --keep-every 500 --ckpt-every 250 --log-every 25 --stop-file "$run/STOP" \
        > "$stage/train.stdout.log" 2>&1
    check_stop
    checkpoint=$run/training/iteration_$tag.pt
    sha256sum "$checkpoint" > "$stage/checkpoint.receipt"
    cp "$run/training/run_config.json" "$run/training/run_summary.json" "$stage/"
    set_phase "recording_$target"
    review=$stage/review
    mkdir -p "$review/forward" "$review/cycle" "$review/release"
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed 101 --num-envs 24 --steps 1000 \
        --command-mode forward --record-env-ids 0 1 2 3 4 5 --record-control-trace \
        --output "$review/forward/evaluation.json" --trajectory-output "$review/forward/trajectories.npz" \
        > "$review/forward/evaluate.stdout.log" 2>&1
    check_stop
    "$python" scripts/rls/supermimic/render_stealth_entry_views.py \
        --trajectory "$review/forward/trajectories.npz" --lane 0 --seconds 20 \
        --output "$review/forward/views" > "$review/forward/render.stdout.log" 2>&1
    "$python" scripts/rls/supermimic/check_stealth_step_retention.py \
        --baseline "$baseline" --candidate "$review/forward" \
        --output "$stage/step-retention.json" --diagnostic-only
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed 101 --num-envs 90 --steps 1000 \
        --mode-layout cycle --record-env-ids 0 1 2 3 4 5 6 7 8 --record-control-trace \
        --output "$review/cycle/evaluation.json" --trajectory-output "$review/cycle/trajectories.npz" \
        > "$review/cycle/evaluate.stdout.log" 2>&1
    check_stop
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed 101 --num-envs 18 --steps 2100 \
        --command-sequence wasd-release-regression --command-stage-seconds 6 \
        --record-env-ids 0 1 2 3 4 5 --record-control-trace \
        --output "$review/release/evaluation.json" --trajectory-output "$review/release/trajectories.npz" \
        > "$review/release/evaluate.stdout.log" 2>&1
    check_stop
    for lane in 0 1 2 3 4 5 6 7 8; do
        "$python" scripts/rls/supermimic/render_stealth_entry_views.py \
            --trajectory "$review/cycle/trajectories.npz" --lane "$lane" --seconds 20 \
            --output "$review/cycle/lane$lane" > "$review/cycle/render-lane$lane.stdout.log" 2>&1
        check_stop
    done
    "$python" scripts/rls/supermimic/render_stealth_entry_views.py \
        --trajectory "$review/release/trajectories.npz" --lane 0 --seconds 42 \
        --output "$review/release/views" > "$review/release/render.stdout.log" 2>&1
    check_stop
    printf '%s\n' "$target" > "$run/completed_iterations"
    printf 'recorded_awaiting_visual_review\n' > "$stage/status"
    source=(--resume "$checkpoint")
done
set_phase "completed_${final}_awaiting_visual_review"

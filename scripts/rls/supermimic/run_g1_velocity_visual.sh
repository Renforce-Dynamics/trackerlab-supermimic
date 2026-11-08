#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
    printf 'usage: run_g1_velocity_visual.sh PYTHON RUN POOL SOURCE ARMS WAIST entry|velocity\n' >&2
    exit 2
fi
python=$1
run=$2
pool=$3
source=$4
arms=$5
waist=$6
trial=$7
case "$trial" in
    entry) feature=postural_end_effectors ;;
    velocity) feature=postural_end_effectors_vx ;;
    *) printf 'Trial must be entry or velocity.\n' >&2; exit 2 ;;
esac
repo=$(git rev-parse --show-toplevel)
config=$repo/scripts/rls/supermimic/configs/command_stealth_velocity_visual.json
test -x "$python"
for input in "$pool/train.json" "$source" "$arms" "$waist" "$config"; do
    test -f "$input"
done
test ! -e "$run/controller.phase"
mkdir -p "$run" "$repo/logs"
trap 'printf "%s\n" "$?" > "$run/controller.exit"' EXIT
stop_files=("$run/STOP" "$run/training/STOP")
stop_args=(--stop-file "$run/STOP")
# An enclosing controller may supply its own marker without changing the preset.
if [ -n "${PARENT_STOP_FILE:-}" ]; then
    stop_files+=("$PARENT_STOP_FILE")
    stop_args+=(--stop-file "$PARENT_STOP_FILE")
fi
check_stop() {
    local marker
    for marker in "${stop_files[@]}"; do
        if [ -e "$marker" ]; then
            printf 'stopped\n' > "$run/controller.phase"
            printf 'Stop requested: %s\n' "$marker" >&2
            exit 0
        fi
    done
}
wait_for_lock() {
    local descriptor=$1 phase=$2 lock_status
    printf '%s\n' "$phase" > "$run/controller.phase"
    while true; do
        check_stop
        if flock -n -E 75 "$descriptor"; then
            break
        else
            lock_status=$?
            if [ "$lock_status" -ne 75 ]; then exit "$lock_status"; fi
        fi
        sleep 1
    done
    check_stop
}
check_stop
exec 9>"$repo/logs/.supermimic-terrain-gpu0.lock"
wait_for_lock 9 waiting_for_repository_gpu_lock
git diff --exit-code
git diff --cached --exit-code
test -z "$(git ls-files --others --exclude-standard)"
git rev-parse HEAD > "$run/code.receipt"
sha256sum "$source" "$pool/manifest.json" "$arms" "$waist" "$config" > "$run/input.receipt"
printf '%s\n' "$trial" > "$run/trial.receipt"

shared_gpu_lock=${SHARED_GPU_LOCK:-/root/gpufree-data/ultra_ipingpong/queue/.runner.flock}
exec 10>>"$shared_gpu_lock"
wait_for_lock 10 waiting_for_shared_scheduler_lease

printf 'waiting_for_shared_gpu\n' > "$run/controller.phase"
queue=/root/gpufree-data/ultra_ipingpong/queue
quiet=0
while [ "$quiet" -lt 3 ]; do
    check_stop
    compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    queued=""
    if [ -d "$queue/pending" ] && [ -d "$queue/running" ]; then
        queued=$(find "$queue/pending" "$queue/running" -maxdepth 1 \
            -type f -name '*.job' -print -quit)
    fi
    IFS=',' read -r used utilization < <(
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0
    )
    if [ -z "$queued" ] && [ -z "$compute" ] \
            && [ "$used" -lt 1024 ] && [ "$utilization" -lt 10 ]; then
        quiet=$((quiet + 1))
    else
        quiet=0
    fi
    if [ "$quiet" -lt 3 ]; then sleep 15; fi
done
# Recheck the exact code and inputs after admission; another job may have edited
# the shared checkout while this trial waited.
test "$(git rev-parse HEAD)" = "$(cat "$run/code.receipt")"
git diff --exit-code
git diff --cached --exit-code
test -z "$(git ls-files --others --exclude-standard)"
sha256sum -c "$run/input.receipt" > "$run/input-verified.receipt"
check_stop
printf 'training_1500_visual_comparison\n' > "$run/controller.phase"
"$python" -u scripts/rls/supermimic/train_terrain.py \
    --task locomotion/g1/stealth --config "$config" --run-dir "$run/training" \
    --warm-start "$source" --amp-motion-config "$pool/train.json" \
    --upper-body-reference "$arms" --waist-entry-reference "$waist" \
    "${stop_args[@]}" \
    --style-feature-mode "$feature" --iterations 1500 --buffer-steps 500 \
    --keep-every 250 --ckpt-every 250 --log-every 25 \
    > "$run/train.stdout.log" 2>&1
check_stop
checkpoint=$run/training/iteration_001500.pt
test -f "$checkpoint"
sha256sum "$checkpoint" > "$run/checkpoint.receipt"
printf 'recording_all_commands_and_release\n' > "$run/controller.phase"
for seed in 101 103; do
    check_stop
    review=$run/review/seed$seed
    mkdir -p "$review/cycle" "$review/release"
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 90 --steps 1000 \
        --mode-layout cycle --record-env-ids 0 1 2 3 4 5 6 7 8 \
        --record-control-trace \
        --output "$review/cycle/evaluation.json" \
        --trajectory-output "$review/cycle/trajectories.npz" \
        > "$review/cycle/evaluate.stdout.log" 2>&1
    check_stop
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed "$seed" --num-envs 18 --steps 2100 \
        --command-sequence wasd-release-regression --command-stage-seconds 6 \
        --record-env-ids 0 1 2 3 4 5 \
        --record-control-trace \
        --output "$review/release/evaluation.json" \
        --trajectory-output "$review/release/trajectories.npz" \
        > "$review/release/evaluate.stdout.log" 2>&1
    for lane in 0 1 2 3 4 5 6 7 8; do
        check_stop
        "$python" scripts/rls/supermimic/render_stealth_entry_views.py \
            --trajectory "$review/cycle/trajectories.npz" --output "$review/cycle/lane$lane" \
            --lane "$lane" --seconds 20 \
            > "$review/cycle/render-lane$lane.stdout.log" 2>&1
    done
    check_stop
    "$python" scripts/rls/supermimic/render_stealth_entry_views.py \
        --trajectory "$review/release/trajectories.npz" --output "$review/release/views" \
        --lane 0 --seconds 42 > "$review/release/render.stdout.log" 2>&1
done
check_stop
printf 'awaiting_visual_review_no_automatic_extension\n' > "$run/controller.phase"

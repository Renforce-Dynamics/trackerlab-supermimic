#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
    printf 'usage: run_g1_stealth_free_arms.sh PYTHON RUN POOL SOURCE WAIST\n' >&2
    exit 2
fi
python=$1
run=$2
pool=$3
source=$4
waist=$5
repo=$(git rev-parse --show-toplevel)
config=$repo/scripts/rls/supermimic/configs/command_stealth_free_arms.json
admission=${GPU_ADMISSION:-exclusive}
minimum_free=${GPU_MIN_FREE_MIB:-21000}
case "$admission" in
    exclusive) memory_args=() ;;
    shared_capacity)
        [[ "$minimum_free" =~ ^[0-9]+$ ]] && [ "$minimum_free" -ge 21000 ]
        test -z "${SHARED_GPU_LOCK:-}"
        memory_args=(--cuda-memory-fraction 0.60)
        ;;
    *) printf 'Unknown GPU_ADMISSION: %s\n' "$admission" >&2; exit 2 ;;
esac
test -x "$python"
for input in "$pool/train.json" "$pool/manifest.json" "$source" "$waist" "$config"; do
    test -f "$input"
done
resume_iteration=0
resume_inputs=()
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    test -f "$RESUME_CHECKPOINT"
    resume_iteration=$(CUDA_VISIBLE_DEVICES='' "$python" - "$RESUME_CHECKPOINT" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if (checkpoint.get("task") != "locomotion/g1/stealth"
        or checkpoint.get("training_stage") != "command-style"
        or checkpoint["arguments"].get("entry_duration_s") != 1.21
        or checkpoint.get("stealth_upper_body_constraint") is not None
        or checkpoint.get("upper_body_rate_limit") is None):
    raise ValueError("Resume requires a checkpoint from the free-arm 1.21-second recipe")
iteration = checkpoint["iteration"]
if type(iteration) is not int or not 0 < iteration < 20000:
    raise ValueError("Resume iteration must be between 1 and 19999")
print(iteration)
PY
)
    [[ "$resume_iteration" =~ ^[0-9]+$ ]]
    resume_inputs=("$RESUME_CHECKPOINT")
fi
mkdir -p "$run" "$repo/logs"
exec 8>"$run/controller.lock"
flock -n -E 75 8
test ! -e "$run/controller.phase"
controller_pid=$BASHPID
watcher_pid=
phase=initializing
set_phase() {
    phase=$1
    printf '%s\n' "$phase" > "$run/controller.phase"
}
finish() {
    local status=$?
    trap - EXIT
    if [ -n "$watcher_pid" ]; then
        kill "$watcher_pid" 2>/dev/null || true
        wait "$watcher_pid" 2>/dev/null || true
    fi
    if [ "$status" -ne 0 ]; then set_phase "failed:$phase"; fi
    printf '%s\n' "$status" > "$run/controller.exit"
    exit "$status"
}
trap finish EXIT
printf '%s\n' "$controller_pid" > "$run/controller.pid"
printf '20000\n' > "$run/target_iterations"
printf '%s\n' "$admission" > "$run/gpu_admission.receipt"
stop_files=("$run/STOP" "$run/training/STOP")
stop_args=(--stop-file "$run/STOP")
if [ -n "${PARENT_STOP_FILE:-}" ]; then
    stop_files+=("$PARENT_STOP_FILE")
    stop_args+=(--stop-file "$PARENT_STOP_FILE")
fi
check_stop() {
    local marker
    for marker in "${stop_files[@]}"; do
        if [ -e "$marker" ]; then
            set_phase stopped
            printf 'Stop requested: %s\n' "$marker" >&2
            exit 0
        fi
    done
}
wait_for_lock() {
    local descriptor=$1 lock_phase=$2 status
    set_phase "$lock_phase"
    while true; do
        check_stop
        if flock -n -E 75 "$descriptor"; then break; else status=$?; fi
        if [ "$status" -ne 75 ]; then exit "$status"; fi
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
sha256sum "$source" "$pool/manifest.json" "$pool/train.json" "$waist" "$config" "${resume_inputs[@]}" \
    > "$run/input.receipt"
if [ -n "${SHARED_GPU_LOCK:-}" ]; then
    exec 10>>"$SHARED_GPU_LOCK"
    wait_for_lock 10 waiting_for_shared_scheduler_lease
fi
verify_inputs() {
    test "$(git rev-parse HEAD)" = "$(cat "$run/code.receipt")"
    git diff --exit-code
    git diff --cached --exit-code
    test -z "$(git ls-files --others --exclude-standard)"
    sha256sum -c "$run/input.receipt" > "$run/input-verified.receipt"
}
wait_for_gpu() {
    local quiet=0 snapshot free utilization compute
    set_phase "waiting_for_gpu_$admission"
    while [ "$quiet" -lt 3 ]; do
        check_stop
        snapshot=$(nvidia-smi -i 0 --query-gpu=memory.free,utilization.gpu \
            --format=csv,noheader,nounits)
        IFS=',' read -r free utilization <<< "$snapshot"
        free=${free//[[:space:]]/}
        utilization=${utilization//[[:space:]]/}
        [[ "$free" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
        if [ "$admission" = shared_capacity ]; then
            if [ "$free" -ge "$minimum_free" ]; then quiet=$((quiet + 1)); else quiet=0; fi
        else
            compute=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
            if [ -z "$compute" ] && [ "$utilization" -lt 10 ]; then
                quiet=$((quiet + 1))
            else
                quiet=0
            fi
        fi
        printf '%s free_mib=%s utilization=%s admission=%s\n' \
            "$(date -Is)" "$free" "$utilization" "$admission" >> "$run/gpu_admission.log"
        if [ "$quiet" -lt 3 ]; then sleep 15; fi
    done
    verify_inputs
    check_stop
}
watch_headroom() {
    trap - EXIT
    exec 8>&- 9>&-
    local low=0 free
    while kill -0 "$controller_pid" 2>/dev/null; do
        if ! free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits); then
            printf 'GPU memory query failed\n' > "$run/STOP"
            return
        fi
        free=${free//[[:space:]]/}
        if ! [[ "$free" =~ ^[0-9]+$ ]]; then
            printf 'Invalid GPU memory reading\n' > "$run/STOP"
            return
        fi
        printf '%s free_mib=%s\n' "$(date -Is)" "$free" >> "$run/gpu_headroom.log"
        if [ "$free" -lt 1536 ]; then low=$((low + 1)); else low=0; fi
        if [ "$low" -ge 2 ]; then
            printf 'Shared GPU free memory below 1536MiB for two readings\n' > "$run/STOP"
            return
        fi
        sleep 15
    done
}
start_watch() {
    if [ "$admission" = shared_capacity ]; then
        watch_headroom &
        watcher_pid=$!
    fi
}
stop_watch() {
    if [ -n "$watcher_pid" ]; then
        kill "$watcher_pid" 2>/dev/null || true
        wait "$watcher_pid" 2>/dev/null || true
        watcher_pid=
    fi
}
initialization=(--warm-start "$source")
if [ "$resume_iteration" -gt 0 ]; then
    initialization=(--resume "$RESUME_CHECKPOINT")
    printf '%s\n' "$RESUME_CHECKPOINT" > "$run/resume_checkpoint.receipt"
    printf '%s\n' "$resume_iteration" > "$run/starting_iteration"
fi
for milestone in 2000 5000 10000 15000 20000; do
    if [ "$milestone" -lt "$resume_iteration" ]; then continue; fi
    wait_for_gpu
    start_watch
    label=$(printf '%06d' "$milestone")
    stage=$run/milestones/$label
    mkdir -p "$stage"
    if [ "$milestone" -gt "$resume_iteration" ]; then
        set_phase "training_to_$milestone"
        "$python" -u scripts/rls/supermimic/train_terrain.py \
            --task locomotion/g1/stealth --config "$config" --run-dir "$run/training" \
            "${initialization[@]}" --amp-motion-config "$pool/train.json" \
            --waist-entry-reference "$waist" "${stop_args[@]}" "${memory_args[@]}" \
            --iterations "$milestone" --buffer-steps 500 \
            --keep-every 1000 --ckpt-every 250 --log-every 25 \
            > "$stage/train.stdout.log" 2>&1
        check_stop
        checkpoint=$run/training/iteration_$label.pt
    else
        checkpoint=$RESUME_CHECKPOINT
    fi
    test -f "$checkpoint"
    sha256sum "$checkpoint" > "$stage/checkpoint.receipt"
    for receipt in run_config.json run_summary.json training_speed_summary.json; do
        if [ -f "$run/training/$receipt" ]; then cp "$run/training/$receipt" "$stage/$receipt"; fi
    done
    set_phase "recording_$milestone"
    review=$stage/review
    mkdir -p "$review/cycle" "$review/release"
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed 101 --num-envs 90 --steps 1000 \
        --mode-layout cycle --record-env-ids 0 1 2 3 4 5 6 7 8 --record-control-trace \
        --output "$review/cycle/evaluation.json" \
        --trajectory-output "$review/cycle/trajectories.npz" \
        > "$review/cycle/evaluate.stdout.log" 2>&1
    check_stop
    "$python" scripts/rls/supermimic/evaluate_terrain.py \
        --checkpoint "$checkpoint" --seed 101 --num-envs 18 --steps 2100 \
        --command-sequence wasd-release-regression --command-stage-seconds 6 \
        --record-env-ids 0 1 2 3 4 5 --record-control-trace \
        --output "$review/release/evaluation.json" \
        --trajectory-output "$review/release/trajectories.npz" \
        > "$review/release/evaluate.stdout.log" 2>&1
    check_stop
    for lane in 0 1 2 3 4 5 6 7 8; do
        "$python" scripts/rls/supermimic/render_stealth_entry_views.py \
            --trajectory "$review/cycle/trajectories.npz" --output "$review/cycle/lane$lane" \
            --lane "$lane" --seconds 20 > "$review/cycle/render-lane$lane.stdout.log" 2>&1
        check_stop
    done
    "$python" scripts/rls/supermimic/render_stealth_entry_views.py \
        --trajectory "$review/release/trajectories.npz" --output "$review/release/views" \
        --lane 0 --seconds 42 > "$review/release/render.stdout.log" 2>&1
    check_stop
    printf '%s\n' "$milestone" > "$run/completed_iterations"
    printf 'recorded_awaiting_visual_review\n' > "$stage/status"
    initialization=(--resume "$checkpoint")
    stop_watch
done
set_phase completed_20000_awaiting_visual_review

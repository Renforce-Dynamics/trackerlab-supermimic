#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    printf 'usage: run_g1_stealth_combined.sh PYTHON RUN SOURCE NATURAL_POOL SOURCE_COMMIT [RESUME]\n' >&2
    exit 2
fi
python=$1
run=$2
initial=$3
pool=$4
pin=$5
resume=${6:-}
repo=$(git rev-parse --show-toplevel)
recipe=${SUPERMIMIC_STEALTH_RECIPE:-combined}
extra_updates=${SUPERMIMIC_STEALTH_EXTRA_UPDATES:-0}
phase_profile_updates=${SUPERMIMIC_STEALTH_PHASE_PROFILE_UPDATES:-0}
case "$extra_updates" in
    0|5000) ;;
    *) printf 'Additional actor updates must be 0 or 5000.\n' >&2; exit 2 ;;
esac
if [ "$extra_updates" != 0 ]; then
    case "$recipe" in
        beyondamp|fullbody)
            if [ -z "$resume" ]; then
                printf 'Additional updates require a full resume checkpoint.\n' >&2
                exit 2
            fi
            ;;
        *) printf 'Additional updates require BeyondAMP or fullbody.\n' >&2; exit 2 ;;
    esac
fi
profile_options=()
case "$phase_profile_updates" in
    0) ;;
    1000) profile_options=(--phase-profile-updates "$phase_profile_updates") ;;
    *) printf 'Phase profiling updates must be 0 or 1000.\n' >&2; exit 2 ;;
esac
budget_recipe=()
aim_options=()
expected_final=50500
case "$recipe" in
    combined) config=$repo/scripts/rls/supermimic/configs/command_stealth_combined.json ;;
    velocity-led)
        config=$repo/scripts/rls/supermimic/configs/command_stealth_velocity_led.json
        budget_recipe=(--recipe velocity-led)
        ;;
    shared-aim)
        config=$repo/scripts/rls/supermimic/configs/command_stealth_shared_aim.json
        budget_recipe=(--recipe shared-aim)
        test -f "$pool/aim-style-reference.json"
        aim_options=(--aim-style-reference "$pool/aim-style-reference.json")
        ;;
    beyondamp)
        config=$repo/scripts/rls/supermimic/configs/command_stealth_beyondamp.json
        budget_recipe=(--recipe beyondamp --extra-actor-updates "$extra_updates")
        expected_final=$((20500+extra_updates))
        test -f "$pool/aim-style-reference.json"
        aim_options=(--aim-style-reference "$pool/aim-style-reference.json")
        ;;
    fullbody)
        test "$(git branch --show-current)" = main
        config=$repo/scripts/rls/supermimic/configs/command_stealth_full_body.json
        budget_recipe=(--recipe fullbody --pool "$pool" --extra-actor-updates "$extra_updates")
        expected_final=$((5500+extra_updates))
        test -f "$pool/aim-style-reference.json"
        test -f "$pool/full-body-style.json"
        aim_options=(--aim-style-reference "$pool/aim-style-reference.json")
        ;;
    *) printf 'unknown stealth recipe: %s\n' "$recipe" >&2; exit 2 ;;
esac
test "$(git rev-parse HEAD)" = "$pin"
test -z "$(git status --porcelain)"
test ! -e "$run/controller.phase"
test -x "$python"
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
sha256sum "$initial" "$pool/manifest.json" "$pool/train.json" "$config" > "$run/input.receipt"
if [ "${#aim_options[@]}" -gt 0 ]; then sha256sum "$pool/aim-style-reference.json" >> "$run/input.receipt"; fi
if [ "$recipe" = fullbody ]; then sha256sum "$pool/full-body-style.json" >> "$run/input.receipt"; fi
if [ -n "$resume" ]; then sha256sum "$resume" >> "$run/input.receipt"; fi
plan_options=()
if [ -n "$resume" ]; then plan_options=(--resume "$resume"); fi
budget=$(CUDA_VISIBLE_DEVICES='' "$python" scripts/rls/supermimic/combined_training_budget.py \
    "${plan_options[@]}" "${budget_recipe[@]}" --output "$run/training_budget.json")
read -r -a budget_values <<< "$budget"
start=${budget_values[0]}
final=${budget_values[1]}
milestones=("${budget_values[@]:2}")
test "$final" = "$expected_final"
test "${#milestones[@]}" -gt 0
printf '%s\n' "$start" > "$run/starting_iteration"
printf '%s\n' "$final" > "$run/target_iterations"
printf '%s\n' "$((final-500))" > "$run/actor_update_budget"
printf '%s\n' "$phase_profile_updates" > "$run/phase_profile_updates"
if [ "$phase_profile_updates" != 0 ]; then
    profile_base=$((start+40))
    if [ "$profile_base" -lt 540 ]; then profile_base=540; fi
    if [ "$((profile_base+1000+phase_profile_updates))" -gt "$final" ]; then
        printf 'Phase window does not fit; choose phase profiling 0 for this continuation.\n' >&2
        exit 2
    fi
fi
set_phase waiting_gpu
while true; do
    check_stop
    occupied=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [ -z "$occupied" ]; then break; fi
    printf '%s %s\n' "$(date -Is)" "$occupied" >> "$run/gpu_wait.log"
    sleep 5
done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MKL_THREADING_LAYER=SEQUENTIAL MUJOCO_GL=egl
source=(--warm-start "$initial")
if [ -n "$resume" ]; then source=(--resume "$resume"); fi
for target in "${milestones[@]}"; do
    check_stop
    test "$(git rev-parse HEAD)" = "$pin"
    test -z "$(git status --porcelain)"
    sha256sum -c "$run/input.receipt" > "$run/input-verified.receipt"
    if [ "$recipe" = fullbody ]; then
        set_phase verifying_full_body_pool
        CUDA_VISIBLE_DEVICES='' "$python" scripts/rls/supermimic/prepare_stealth_full_body.py \
            --verify-pool "$pool" > "$run/full-body-pool-verified.json"
        check_stop
        test "$(git rev-parse HEAD)" = "$pin"
        test -z "$(git status --porcelain)"
        sha256sum -c "$run/input.receipt" > "$run/input-verified.receipt"
    fi
    tag=$(printf '%06d' "$target")
    stage=$run/milestones/$tag
    mkdir -p "$stage"
    set_phase "training_to_$target"
    "$python" -u scripts/rls/supermimic/train_terrain.py \
        --task locomotion/g1/stealth --config "$config" --run-dir "$run/training" \
        "${source[@]}" --amp-motion-config "$pool/train.json" \
        "${aim_options[@]}" \
        "${profile_options[@]}" \
        --iterations "$target" --buffer-steps 500 --keep-every 500 --ckpt-every 250 \
        --log-every 25 --stop-file "$run/STOP" > "$stage/train.stdout.log" 2>&1
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

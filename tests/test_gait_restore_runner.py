"""Run the actual gait controller with local checkpoint and review substitutes."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from test_stealth_stop import (
    MOCK_ENV, assert_stopped, bash, running_runner, wait_for_file,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/rls/supermimic/run_g1_gait_restore.sh"
SCHEDULER_LOCK = "/root/gpufree-data/ultra_ipingpong/queue/.runner.flock"
CALL_FIELDS = (
    "mode", "iteration", "warm_start", "resume", "candidate", "output",
    "diagnostic", "checkpoint", "trajectory", "lane", "seconds", "envs", "steps",
)
GIT_MOCK = r'''
git() {
    case "$*" in
        'rev-parse --show-toplevel') printf '%s\n' "$TEST_ROOT" ;;
        'rev-parse HEAD') printf 'fake-code-pin\n' ;;
        'status --porcelain') ;;
        *) return 97 ;;
    esac
}
'''
FAKE_PYTHON = r'''#!/usr/bin/env bash
set -euo pipefail
mode=unknown
iteration= warm= resume= candidate= output= checkpoint= trajectory=
run_dir= lane= seconds= envs= steps=
diagnostic=0
markers=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        *gait_restore_budget.py) mode=budget; shift ;;
        *train_terrain.py) mode=train; shift ;;
        *check_stealth_step_retention.py) mode=check; shift ;;
        *evaluate_terrain.py) mode=evaluate; shift ;;
        *render_stealth_entry_views.py) mode=render; shift ;;
        --run-dir) run_dir=$2; shift 2 ;;
        --iterations) iteration=$2; shift 2 ;;
        --warm-start) warm=$2; shift 2 ;;
        --resume) resume=$2; shift 2 ;;
        --candidate) candidate=$2; shift 2 ;;
        --output) output=$2; shift 2 ;;
        --checkpoint) checkpoint=$2; shift 2 ;;
        --trajectory|--trajectory-output) trajectory=$2; shift 2 ;;
        --stop-file) markers+=("$2"); shift 2 ;;
        --lane) lane=$2; shift 2 ;;
        --seconds) seconds=$2; shift 2 ;;
        --num-envs) envs=$2; shift 2 ;;
        --steps) steps=$2; shift 2 ;;
        --diagnostic-only) diagnostic=1; shift ;;
        *) shift ;;
    esac
done
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$mode" "$iteration" "$warm" "$resume" "$candidate" "$output" \
    "$diagnostic" "$checkpoint" "$trajectory" "$lane" "$seconds" "$envs" "$steps" \
    >> "$TEST_ROOT/calls"
# Every substitute must inherit the controller's three open lease descriptors.
: >&8
: >&9
: >&10
stop_after() {
    if [ "${TEST_STOP_AFTER:-}" = "$1" ]; then
        printf 'owner stop\n' > "$TEST_RUN/STOP"
    fi
}
case "$mode" in
    budget)
        if [ -n "$resume" ]; then
            test -f "$resume"
            start=1000; final=6000; midpoint=3000; warmup=0
        else
            start=0; final=5500; midpoint=2500; warmup=500
        fi
        printf '{"starting_iteration":%s,"target_iteration":%s,"milestones":[%s,%s],"additional_actor_updates":5000,"remaining_world_only_iterations":%s}\n' \
            "$start" "$final" "$midpoint" "$final" "$warmup" > "$output"
        printf '%s %s %s %s\n' "$start" "$final" "$midpoint" "$final"
        ;;
    train)
        # A repeated warm start or mixed initialization must fail the harness.
        if [ -n "$resume" ]; then
            test -z "$warm"
            test -f "$resume"
        else
            test -n "$warm"
            test -f "$warm"
            test ! -e "$TEST_ROOT/child-started"
        fi
        mkdir -p "$run_dir"
        printf '%s\n' "${markers[@]}" > "$TEST_ROOT/forwarded-stops"
        : > "$TEST_ROOT/child-started"
        if [ "${TEST_CHILD_MODE:-complete}" = wait ]; then
            for ((attempt=0; attempt<250; attempt++)); do
                for marker in "$run_dir/STOP" "${markers[@]}"; do
                    if [ -e "$marker" ]; then
                        printf 'checkpoint-on-stop\n' > "$run_dir/last.pt"
                        : > "$TEST_ROOT/child-finished"
                        exit 0
                    fi
                done
                command sleep .02
            done
            exit 98
        fi
        if [ "${TEST_FAIL_AT:-}" = "$iteration" ]; then
            printf 'mock trainer failure at %s\n' "$iteration" >&2
            exit 42
        fi
        printf -v tag '%06d' "$iteration"
        printf 'checkpoint-%s\n' "$iteration" > "$run_dir/iteration_$tag.pt"
        cp "$run_dir/iteration_$tag.pt" "$run_dir/last.pt"
        printf '{"iteration":%s}\n' "$iteration" > "$run_dir/run_config.json"
        printf '{"completed_iterations":%s}\n' "$iteration" > "$run_dir/run_summary.json"
        ;;
    check)
        if [ "$candidate" = "$TEST_BASELINE" ]; then
            test "$diagnostic" = 0
            printf '{"passed":true,"valid_evidence":true}\n' > "$output"
        elif [ "${TEST_QUALITY:-valid-fail}" = invalid ]; then
            printf '{"passed":false,"valid_evidence":false,"error":"malformed trajectory"}\n' > "$output"
            exit 2
        else
            printf '{"passed":false,"valid_evidence":true}\n' > "$output"
            if [ "$diagnostic" -ne 1 ]; then exit 2; fi
            stop_after diagnostic
        fi
        ;;
    evaluate)
        test -f "$checkpoint"
        printf '{"mock_evaluation":true}\n' > "$output"
        printf 'mock trajectory\n' > "$trajectory"
        stop_after "evaluate-$(basename "$(dirname "$output")")"
        ;;
    render)
        test -f "$trajectory"
        mkdir -p "$output"
        printf 'mock video\n' > "$output/entry-front-side.mp4"
        stop_after "render-$(basename "$(dirname "$trajectory")")"
        ;;
    *) exit 97 ;;
esac
'''


@pytest.fixture
def case(tmp_path, bash):
    root = tmp_path / "fake repository"
    root.mkdir()
    run, pool, baseline = root / "trial with spaces", root / "pool", root / "baseline"
    pool.mkdir()
    baseline.mkdir()
    initial, resume = root / "entry 10000.pt", root / "recovery 001000.pt"
    initial.write_bytes(b"original entry checkpoint\n")
    resume.write_bytes(b"full recovery checkpoint at 1000\n")
    digest = hashlib.sha256(initial.read_bytes()).hexdigest()
    (baseline / "checkpoint.receipt").write_text(
        f"{digest}  {initial.as_posix()}\n", encoding="utf-8", newline="\n")
    config = root / "scripts/rls/supermimic/configs/command_stealth_gait_restore.json"
    config.parent.mkdir(parents=True)
    for path in (config, pool / "manifest.json", pool / "train.json",
                 pool / "upper-body-reference.json", baseline / "evaluation.json",
                 baseline / "trajectories.npz"):
        path.write_text("{}\n", encoding="ascii")
    fake = root / "fake-python.sh"
    fake.write_text(FAKE_PYTHON, encoding="ascii", newline="\n")
    fake.chmod(0o755)
    mock = root / "mock-env.sh"
    mock.write_text(MOCK_ENV + GIT_MOCK, encoding="ascii", newline="\n")
    # Redirect only the production scheduler-lock open into the temporary tree.
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count(SCHEDULER_LOCK) == 1
    fixture_runner = root / RUNNER.name
    fixture_runner.write_text(source.replace(
        SCHEDULER_LOCK, shlex.quote((root / "scheduler lock.flock").as_posix())),
        encoding="utf-8", newline="\n")
    env = os.environ.copy()
    for key in ("TEST_CHILD_MODE", "TEST_STOP_AFTER", "TEST_FAIL_AT", "TEST_QUALITY"):
        env.pop(key, None)
    env.update(BASH_ENV=mock.as_posix(), TEST_ROOT=root.as_posix(),
               TEST_RUN=run.as_posix(), TEST_BASELINE=baseline.as_posix())
    command = [bash, "--noprofile", "--norc", fixture_runner.as_posix(), fake.as_posix(),
               run.as_posix(), initial.as_posix(), pool.as_posix(), baseline.as_posix(),
               "fake-code-pin"]
    return SimpleNamespace(root=root, run=run, env=env, command=command,
                           initial=initial, resume=resume, baseline=baseline)


def calls(case, mode=None):
    rows = [dict(zip(CALL_FIELDS, line.split("|"), strict=True)) for line in
            (case.root / "calls").read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if mode is None or row["mode"] == mode]


def run_to_exit(case, expected=0):
    with running_runner(case) as process:
        output, errors = process.communicate(timeout=20)
        assert process.returncode == expected, (output, errors)
    assert (case.run / "controller.exit").read_text().strip() == str(expected)


def assert_inputs_preserved(case):
    assert case.initial.read_bytes() == b"original entry checkpoint\n"
    assert case.resume.read_bytes() == b"full recovery checkpoint at 1000\n"


def test_runner_bash_syntax(bash):
    result = subprocess.run([bash, "-n", RUNNER.as_posix()],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("resumed", [False, True], ids=["six-argument-fresh", "resume-1000"])
def test_budget_and_full_state_handoff_continue_valid_quality_failures(case, resumed):
    if resumed:
        case.command.append(case.resume.as_posix())
    start, midpoint, final = (1000, 3000, 6000) if resumed else (0, 2500, 5500)
    run_to_exit(case)
    plan = json.loads((case.run / "training_budget.json").read_text())
    assert plan == {
        "starting_iteration": start, "target_iteration": final,
        "milestones": [midpoint, final], "additional_actor_updates": 5000,
        "remaining_world_only_iterations": 0 if resumed else 500,
    }
    assert (case.run / "starting_iteration").read_text().strip() == str(start)
    assert (case.run / "target_iterations").read_text().strip() == str(final)
    assert (case.run / "additional_actor_updates").read_text().strip() == "5000"
    assert (case.run / "completed_iterations").read_text().strip() == str(final)
    assert (case.run / "controller.phase").read_text().strip() == (
        f"completed_{final}_awaiting_visual_review")
    budget, = calls(case, "budget")
    assert budget["resume"] == (case.resume.as_posix() if resumed else "")
    trainers = calls(case, "train")
    assert [row["iteration"] for row in trainers] == [str(midpoint), str(final)]
    assert [row["warm_start"] for row in trainers] == [
        "" if resumed else case.initial.as_posix(), ""]
    assert [row["resume"] for row in trainers] == [
        case.resume.as_posix() if resumed else "",
        (case.run / f"training/iteration_{midpoint:06d}.pt").as_posix()]
    reviews = calls(case, "check")
    assert [row["diagnostic"] for row in reviews] == ["0", "1", "1"]
    assert reviews[0]["candidate"] == case.baseline.as_posix()
    for target in (midpoint, final):
        stage = case.run / f"milestones/{target:06d}"
        assert json.loads((stage / "step-retention.json").read_text()) == {
            "passed": False, "valid_evidence": True}
        assert (stage / "status").read_text().strip() == "recorded_awaiting_visual_review"
        assert (stage / "checkpoint.receipt").is_file()
        assert json.loads((stage / "run_config.json").read_text())["iteration"] == target
        assert json.loads((stage / "run_summary.json").read_text())[
            "completed_iterations"] == target
        for relative in ("forward/views", "release/views", *(f"cycle/lane{i}" for i in range(9))):
            assert (stage / "review" / relative / "entry-front-side.mp4").is_file()
    evaluations = calls(case, "evaluate")
    assert [(row["envs"], row["steps"]) for row in evaluations] == [
        ("24", "1000"), ("90", "1000"), ("18", "2100")] * 2
    assert len(calls(case, "render")) == 22
    events = (case.root / "events").read_text().splitlines()
    assert events.index("acquired:8") < events.index("acquired:9") < (
        events.index("acquired:10")) < events.index("gpu-probe")
    assert_inputs_preserved(case)


def test_malformed_quality_evidence_exit_two_stops_before_more_training(case):
    case.command.append(case.resume.as_posix())
    case.env["TEST_QUALITY"] = "invalid"
    run_to_exit(case, expected=2)
    assert [row["iteration"] for row in calls(case, "train")] == ["3000"]
    assert len(calls(case, "evaluate")) == len(calls(case, "render")) == 1
    stage = case.run / "milestones/003000"
    receipt = json.loads((stage / "step-retention.json").read_text())
    assert receipt["passed"] is False and receipt["valid_evidence"] is False
    assert receipt["error"] == "malformed trajectory"
    assert (stage / "review/forward/views/entry-front-side.mp4").is_file()
    assert (stage / "checkpoint.receipt").is_file()
    assert not (case.run / "completed_iterations").exists()
    assert not (case.run / "milestones/006000").exists()
    assert (case.run / "controller.phase").read_text().strip() == "failed:recording_3000"
    assert_inputs_preserved(case)


def test_trainer_failure_preserves_prior_checkpoints_and_reviews(case):
    target = 6000
    case.command.append(case.resume.as_posix())
    case.env["TEST_FAIL_AT"] = str(target)
    run_to_exit(case, expected=42)
    stage = case.run / f"milestones/{target:06d}"
    assert f"mock trainer failure at {target}" in (stage / "train.stdout.log").read_text()
    assert (case.run / "controller.phase").read_text().strip() == f"failed:training_to_{target}"
    assert not (case.run / f"training/iteration_{target:06d}.pt").exists()
    assert not (stage / "review").exists()
    assert (case.run / "completed_iterations").read_text().strip() == "3000"
    assert (case.run / "training/last.pt").read_text().strip() == "checkpoint-3000"
    assert (case.run / "training/iteration_003000.pt").is_file()
    assert (case.run / "milestones/003000/review/release/views/entry-front-side.mp4").is_file()
    assert [row["iteration"] for row in calls(case, "train")] == ["3000", "6000"]
    assert_inputs_preserved(case)


def test_stop_reaches_training_child_and_preserves_its_checkpoint(case):
    case.command.append(case.resume.as_posix())
    case.env["TEST_CHILD_MODE"] = "wait"
    with running_runner(case) as process:
        wait_for_file(case.root / "child-started", process)
        assert (case.root / "forwarded-stops").read_text(encoding="utf-8").splitlines() == [
            (case.run / "STOP").as_posix()]
        marker = case.run / "STOP"
        marker.write_text("owner stop\n", encoding="ascii")
        assert_stopped(case, process)
    assert marker.read_text() == "owner stop\n"
    assert (case.root / "child-finished").is_file()
    assert (case.run / "training/last.pt").read_text().strip() == "checkpoint-on-stop"
    assert [row["mode"] for row in calls(case)] == ["budget", "check", "train"]
    assert calls(case, "train")[0]["resume"] == case.resume.as_posix()
    assert not (case.run / "completed_iterations").exists()
    assert_inputs_preserved(case)



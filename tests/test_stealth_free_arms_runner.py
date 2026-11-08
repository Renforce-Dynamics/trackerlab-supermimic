"""Exercise the bounded controller with fake training, GPU and lock commands."""

import os
from pathlib import Path
import subprocess

import pytest

from test_stealth_stop import (
    bash, running_runner, wait_for_file, assert_stopped, MOCK_ENV,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/rls/supermimic/run_g1_stealth_free_arms.sh"
GPU_MOCK = r'''
nvidia-smi() {
    case "$*" in
        *--query-gpu=memory.free,*) printf '23000, 0\n' ;;
        *--query-gpu=memory.free*)
            if [ -e "$TEST_ROOT/low-memory" ]; then printf '1000\n'; else printf '20000\n'; fi ;;
        *--query-compute-apps=*) ;;
        *) return 96 ;;
    esac
}
'''
FAKE_PYTHON = r'''#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = - ]; then
    cat > /dev/null
    printf '%s\n' "${TEST_RESUME_ITERATION:-680}"
    exit 0
fi
mode=unknown
run_dir=
iteration=
initialization=
cap=
markers=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        *train_terrain.py) mode=train; shift ;;
        *evaluate_terrain.py) mode=evaluate; shift ;;
        *render_stealth_entry_views.py) mode=render; shift ;;
        --run-dir) run_dir=$2; shift 2 ;;
        --iterations) iteration=$2; shift 2 ;;
        --warm-start|--resume) initialization="$1=$2"; shift 2 ;;
        --stop-file) markers+=("$2"); shift 2 ;;
        --cuda-memory-fraction) cap=$2; shift 2 ;;
        --upper-body-reference) exit 97 ;;
        *) shift ;;
    esac
done
printf '%s|%s|%s|%s\n' "$mode" "$iteration" "$initialization" "$cap" >> "$TEST_ROOT/calls"
if [ "$mode" = train ]; then
    mkdir -p "$run_dir"
    : > "$TEST_ROOT/child-started"
    if [ "${TEST_CHILD_MODE:-complete}" = wait ]; then
        for ((attempt=0; attempt<300; attempt++)); do
            for marker in "$run_dir/STOP" "${markers[@]}"; do
                if [ -e "$marker" ]; then
                    printf 'checkpoint-on-stop\n' > "$run_dir/last.pt"
                    exit 0
                fi
            done
            command sleep .02
        done
        exit 98
    fi
    if [ "${TEST_FAIL_AT:-}" = "$iteration" ]; then exit 42; fi
    printf -v label '%06d' "$iteration"
    printf 'checkpoint-%s\n' "$iteration" > "$run_dir/iteration_$label.pt"
    printf '{}\n' > "$run_dir/run_summary.json"
elif [ "$mode" = "${TEST_STOP_AFTER:-}" ]; then
    : > "$TEST_RUN/STOP"
fi
'''


@pytest.fixture
def case(tmp_path, bash):
    from types import SimpleNamespace
    root = tmp_path / "fake repo"
    root.mkdir()
    run, pool = root / "run", root / "pool"
    pool.mkdir()
    config = root / "scripts/rls/supermimic/configs/command_stealth_free_arms.json"
    config.parent.mkdir(parents=True)
    for path in (config, pool / "train.json", pool / "manifest.json",
                 root / "source.pt", root / "waist.json"):
        path.write_text("{}\n", encoding="ascii")
    fake = root / "fake-python.sh"
    fake.write_text(FAKE_PYTHON, encoding="ascii", newline="\n")
    fake.chmod(0o755)
    mock = root / "mock.sh"
    mock.write_text(MOCK_ENV + GPU_MOCK, encoding="ascii", newline="\n")
    env = os.environ.copy()
    for key in ("SHARED_GPU_LOCK", "GPU_ADMISSION", "GPU_MIN_FREE_MIB", "PARENT_STOP_FILE",
                "TEST_CHILD_MODE", "TEST_STOP_AFTER", "TEST_FAIL_AT",
                "RESUME_CHECKPOINT", "TEST_RESUME_ITERATION"):
        env.pop(key, None)
    env.update(BASH_ENV=mock.as_posix(), TEST_ROOT=root.as_posix(), TEST_RUN=run.as_posix())
    command = [bash, RUNNER.as_posix(), fake.as_posix(), run.as_posix(),
               pool.as_posix(), (root / "source.pt").as_posix(), (root / "waist.json").as_posix()]
    return SimpleNamespace(root=root, run=run, env=env, command=command)


def test_five_absolute_milestones_warm_start_once_then_resume_all_state(case):
    with running_runner(case) as process:
        output, errors = process.communicate(timeout=20)
        assert process.returncode == 0, (output, errors)
    calls = (case.root / "calls").read_text(encoding="utf-8").splitlines()
    training = [call.split("|") for call in calls if call.startswith("train|")]
    assert [row[1] for row in training] == ["2000", "5000", "10000", "15000", "20000"]
    assert training[0][2].startswith("--warm-start=")
    for row, previous in zip(training[1:], (2000, 5000, 10000, 15000)):
        assert row[2].endswith(f"--resume={case.run.as_posix()}/training/iteration_{previous:06d}.pt")
    assert all(row[3] == "" for row in training)
    assert sum(call.startswith("evaluate|") for call in calls) == 10
    assert sum(call.startswith("render|") for call in calls) == 50
    assert (case.run / "controller.phase").read_text().strip() == "completed_20000_awaiting_visual_review"
    assert (case.run / "controller.exit").read_text().strip() == "0"
    assert (case.run / "completed_iterations").read_text().strip() == "20000"
    assert len(list((case.run / "milestones").glob("*/checkpoint.receipt"))) == 5


def test_shared_capacity_caps_own_trainer_and_stops_on_low_headroom(case):
    case.env.update(GPU_ADMISSION="shared_capacity", TEST_CHILD_MODE="wait")
    with running_runner(case) as process:
        wait_for_file(case.root / "child-started", process)
        (case.root / "low-memory").touch()
        assert_stopped(case, process)
    calls = (case.root / "calls").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1 and calls[0].endswith("|0.60")
    assert "1536MiB" in (case.run / "STOP").read_text()
    assert (case.run / "training/last.pt").read_text().strip() == "checkpoint-on-stop"
    assert "flock:10" not in (case.root / "events").read_text()


def test_failure_preserves_previous_milestone_and_ends_remaining_queue(case):
    case.env["TEST_FAIL_AT"] = "5000"
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=20)
        assert process.returncode == 42, errors
    assert (case.run / "controller.exit").read_text().strip() == "42"
    assert (case.run / "controller.phase").read_text().strip() == "failed:training_to_5000"
    assert (case.run / "training/iteration_002000.pt").exists()
    assert (case.run / "milestones/002000/checkpoint.receipt").exists()
    assert (case.run / "completed_iterations").read_text().strip() == "2000"


def test_stop_cancels_locked_wait_without_starting_children(case):
    (case.root / "held-9").touch()
    with running_runner(case) as process:
        wait_for_file(case.root / "observed-held-9", process)
        (case.run / "STOP").touch()
        assert_stopped(case, process)
    assert not (case.root / "calls").exists()


def test_changed_source_while_waiting_fails_closed(case):
    case.env["SHARED_GPU_LOCK"] = (case.root / "external.flock").as_posix()
    held = case.root / "held-10"
    held.touch()
    with running_runner(case) as process:
        wait_for_file(case.root / "observed-held-10", process)
        (case.root / "source.pt").write_text("changed")
        held.unlink()
        _, errors = process.communicate(timeout=10)
        assert process.returncode != 0, errors
    assert not (case.root / "calls").exists()
    assert (case.run / "controller.phase").read_text().startswith("failed:")


def test_stop_between_review_stages_preserves_checkpoint(case):
    case.env["TEST_STOP_AFTER"] = "evaluate"
    with running_runner(case) as process:
        assert_stopped(case, process)
    calls = (case.root / "calls").read_text(encoding="utf-8").splitlines()
    assert [call.split("|")[0] for call in calls] == ["train", "evaluate"]
    assert (case.run / "training/iteration_002000.pt").exists()


@pytest.mark.parametrize("iteration,expected", [
    (680, ["2000", "5000", "10000", "15000", "20000"]),
    (5000, ["10000", "15000", "20000"]),
    (7000, ["10000", "15000", "20000"]),
])
def test_full_checkpoint_handoff_never_repeats_warm_start(case, iteration, expected):
    saved = case.root / "saved-full.pt"
    saved.write_bytes(b"saved state")
    case.env.update(RESUME_CHECKPOINT=saved.as_posix(), TEST_RESUME_ITERATION=str(iteration))
    with running_runner(case) as process:
        output, errors = process.communicate(timeout=20)
        assert process.returncode == 0, (output, errors)
    calls = (case.root / "calls").read_text(encoding="utf-8").splitlines()
    training = [call.split("|") for call in calls if call.startswith("train|")]
    assert [row[1] for row in training] == expected
    assert all(row[2].startswith("--resume=") for row in training)
    assert training[0][2].endswith(saved.as_posix())
    assert (case.run / "starting_iteration").read_text().strip() == str(iteration)
    if iteration == 5000:
        assert calls[0].startswith("evaluate|")

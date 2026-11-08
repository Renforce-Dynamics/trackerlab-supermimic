"""CPU stop/checkpoint, mocked Bash controller, and zero-armature export checks."""

import ast
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts/rls/supermimic"
RUNNER = SCRIPTS / "run_g1_velocity_visual.sh"
sys.path.insert(0, str(SCRIPTS))
from bootstrap import configure_paths
configure_paths()
from export_command_policy import command_actuator_metadata
from train_terrain import parse_args, training_stop_requested
from trackerTask.supermimic.terrain import robot_parameters


def test_stop_arguments_append_strings_and_remain_json_serializable():
    args = parse_args(["--task", "locomotion/g1/stealth", "--run-dir", "unused",
                       "--stop-file", "controller/STOP", "--stop-file", "parent with spaces/STOP"])
    assert args.stop_file == ["controller/STOP", "parent with spaces/STOP"]
    payload = json.loads(json.dumps(vars(args) | {"run_dir": str(args.run_dir)}))
    assert payload["stop_file"] == args.stop_file
    assert parse_args(["--task", "locomotion/g1/stealth", "--run-dir", "unused"]).stop_file == []


@pytest.mark.parametrize("which", ["local", "controller", "parent"])
def test_stop_sources_are_independent_and_read_only(tmp_path, which):
    run = tmp_path / "run" / "training"
    markers = {"local": run / "STOP", "controller": run.parent / "STOP",
               "parent": tmp_path / "parent with spaces" / "STOP"}
    external = [str(markers["controller"]), str(markers["parent"])]
    assert not training_stop_requested(run, external)
    marker = markers[which]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("owner stop\n", encoding="ascii")
    assert training_stop_requested(run, external)
    assert marker.read_text(encoding="ascii") == "owner stop\n"
    assert all(not path.exists() for key, path in markers.items() if key != which)
    marker.unlink()
    assert not training_stop_requested(run, external)


def test_external_stop_uses_actual_checkpoint_condition_before_break(tmp_path):
    tree = ast.parse((SCRIPTS / "train_terrain.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "iteration")
    index = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "stop_requested"
                         for target in node.targets))
    tail = copy.deepcopy(loop.body[index:index + 4])
    assert len(tail) == 4 and isinstance(tail[2], ast.If) and isinstance(tail[3], ast.If)
    # Exercise the real predicates with a checkpoint callback, without model setup.
    tail[2].body = ast.parse("checkpoints.append(iteration + 1)").body
    harness = ast.parse("for iteration in range(5):\n    request_stop(iteration)\n")
    harness.body[0].body.extend(tail)
    marker = tmp_path / "parent STOP"
    namespace = {"run": tmp_path / "training", "final": False, "checkpoints": [],
                 "args": SimpleNamespace(stop_file=[str(marker)], ckpt_every=250, keep_every=250),
                 "training_stop_requested": training_stop_requested,
                 "request_stop": lambda i: marker.touch() if i == 2 else None}
    exec(compile(ast.fix_missing_locations(harness), "trainer-stop-tail", "exec"), namespace)
    assert namespace["iteration"] == 2
    assert namespace["checkpoints"] == [3]


@pytest.mark.parametrize("robot", ["g1", "go2", "a3", "t1", "bumi", "pi_plus"])
def test_export_accepts_native_motor_contracts_including_zero_armature(robot):
    spec, _, motors = robot_parameters(robot)
    checkpoint = {"arguments": {"waist_pd_scale": 1.}, "simulator_contract": {
        "joint_names": list(spec.joint_names), "waist_pd_scale": 1.,
        "motors_effort_velocity_kp_kd_armature": motors}}
    result = command_actuator_metadata(checkpoint, spec)
    assert result["nominal_motors"] == motors
    if robot == "bumi":
        assert any(row[4] == 0. for row in result["nominal_motors"])


@pytest.mark.parametrize("column,value", [(i, 0.) for i in range(4)] +
                         [(4, -1.), (4, float("nan")), (4, float("inf")),
                          (4, True), (4, "0")])
def test_export_relaxes_only_nonnegative_numeric_armature(column, value):
    spec = SimpleNamespace(joint_names=("waist_roll_joint",), num_joints=1)
    row = [50., 37., 28.5, 1.8, 0.]
    row[column] = value
    checkpoint = {"arguments": {}, "simulator_contract": {
        "joint_names": list(spec.joint_names),
        "motors_effort_velocity_kp_kd_armature": [row]}}
    with pytest.raises(ValueError, match="motor settings"):
        command_actuator_metadata(checkpoint, spec)


@pytest.fixture(scope="module")
def bash():
    candidates = []
    git = shutil.which("git")
    if os.name == "nt" and git:
        root = Path(git).resolve().parent.parent
        candidates.extend([root / "bin/bash.exe", root / "usr/bin/bash.exe"])
    found = shutil.which("bash")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            try:
                probe = subprocess.run([str(candidate), "--noprofile", "--norc", "-c", "exit 0"],
                                       capture_output=True, timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if probe.returncode == 0:
                return str(candidate)
    pytest.skip("Bash unavailable for mocked controller tests")


MOCK_ENV = r'''
git() {
    case "$*" in
        'rev-parse --show-toplevel') printf '%s\n' "$TEST_ROOT" ;;
        'rev-parse HEAD') printf 'fake-code-pin\n' ;;
        'diff --exit-code'|'diff --cached --exit-code'|'ls-files --others --exclude-standard') ;;
        *) return 97 ;;
    esac
}
flock() {
    local descriptor=${@: -1}
    printf '%s\n' "flock:$descriptor" >> "$TEST_ROOT/events"
    : >&"$descriptor"
    if [ -e "$TEST_ROOT/error-$descriptor" ]; then return 69; fi
    if [ -e "$TEST_ROOT/held-$descriptor" ]; then
        : > "$TEST_ROOT/observed-held-$descriptor"
        return 75
    fi
    printf '%s\n' "acquired:$descriptor" >> "$TEST_ROOT/events"
}
nvidia-smi() {
    printf 'gpu-probe\n' >> "$TEST_ROOT/events"
    case "$*" in
        *--query-gpu=*) printf '0, 0\n' ;;
        *--query-compute-apps=*) ;;
        *) return 96 ;;
    esac
}
sleep() { command sleep .02; }
find() { return 0; }
'''


FAKE_PYTHON = r'''#!/usr/bin/env bash
set -euo pipefail
mode=unknown
run_dir=
markers=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        *train_terrain.py) mode=train; shift ;;
        *evaluate_terrain.py) mode=evaluate; shift ;;
        *render_stealth_entry_views.py) mode=render; shift ;;
        --run-dir) run_dir=$2; shift 2 ;;
        --stop-file) markers+=("$2"); shift 2 ;;
        *) shift ;;
    esac
done
printf '%s\n' "$mode" >> "$TEST_ROOT/calls"
: >&9
: >&10
if [ "$mode" = train ]; then
    mkdir -p "$run_dir"
    printf '%s\n' "${markers[@]}" > "$TEST_ROOT/forwarded-stops"
    : > "$TEST_ROOT/child-started"
    if [ "${TEST_CHILD_MODE:-wait}" = wait ]; then
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
    printf 'fake completed checkpoint\n' > "$run_dir/iteration_001500.pt"
elif [ "$mode" = "${TEST_STOP_AFTER:-}" ]; then
    : > "$TEST_RUN/STOP"
fi
'''


@pytest.fixture
def shell_case(tmp_path, bash):
    root = tmp_path / "fake repository"
    root.mkdir()
    run = root / "trial with spaces"
    pool = root / "pool"
    pool.mkdir()
    config = root / "scripts/rls/supermimic/configs/command_stealth_velocity_visual.json"
    config.parent.mkdir(parents=True)
    inputs = [pool / "train.json", pool / "manifest.json", config,
              root / "source1500.pt", root / "arms.json", root / "waist.json"]
    for path in inputs:
        path.write_text("{}\n", encoding="ascii")
    fake_python = root / "fake-python.sh"
    fake_python.write_text(FAKE_PYTHON, encoding="ascii", newline="\n")
    fake_python.chmod(0o755)
    bash_env = root / "mock-env.sh"
    bash_env.write_text(MOCK_ENV, encoding="ascii", newline="\n")
    env = os.environ.copy()
    for name in ("PARENT_STOP_FILE", "TEST_CHILD_MODE", "TEST_STOP_AFTER"):
        env.pop(name, None)
    env.update(BASH_ENV=bash_env.as_posix(), TEST_ROOT=root.as_posix(), TEST_RUN=run.as_posix(),
               SHARED_GPU_LOCK=(root / "shared scheduler.flock").as_posix())
    command = [bash, "--noprofile", "--norc", RUNNER.as_posix(), fake_python.as_posix(),
               run.as_posix(), pool.as_posix(), (root / "source1500.pt").as_posix(),
               (root / "arms.json").as_posix(), (root / "waist.json").as_posix(), "entry"]
    return SimpleNamespace(root=root, run=run, env=env, command=command)


@contextmanager
def running_runner(case):
    process = subprocess.Popen(case.command, cwd=case.root, env=case.env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield process
    finally:
        if process.poll() is None:
            case.run.mkdir(exist_ok=True)
            (case.run / "STOP").touch()
            try:
                process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=8)


def wait_for_file(path, process):
    deadline = time.monotonic() + 8
    while not path.exists():
        if process.poll() is not None:
            output, errors = process.communicate()
            pytest.fail(f"Controller exited {process.returncode}: {output}\n{errors}")
        if time.monotonic() >= deadline:
            pytest.fail(f"Controller did not create {path.name}")
        time.sleep(.01)


def assert_stopped(case, process):
    output, errors = process.communicate(timeout=8)
    assert process.returncode == 0, (output, errors)
    assert (case.run / "controller.phase").read_text().strip() == "stopped"
    assert (case.run / "controller.exit").read_text().strip() == "0"


def test_runner_bash_syntax(bash):
    result = subprocess.run([bash, "-n", RUNNER.as_posix()], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("marker_kind", ["controller", "parent", "local"])
def test_stop_ends_fake_training_child_and_prevents_evaluation(shell_case, marker_kind):
    case = shell_case
    parent = case.root / "parent stop marker"
    if marker_kind == "parent":
        case.env["PARENT_STOP_FILE"] = parent.as_posix()
    with running_runner(case) as process:
        wait_for_file(case.root / "child-started", process)
        forwarded = (case.root / "forwarded-stops").read_text(encoding="utf-8").splitlines()
        assert forwarded == [(case.run / "STOP").as_posix()] + (
            [parent.as_posix()] if marker_kind == "parent" else [])
        marker = {"controller": case.run / "STOP", "parent": parent,
                  "local": case.run / "training/STOP"}[marker_kind]
        marker.touch()
        assert_stopped(case, process)
    assert (case.root / "child-finished").exists()
    assert (case.run / "training/last.pt").read_text().strip() == "checkpoint-on-stop"
    assert (case.root / "calls").read_text().splitlines() == ["train"]
    events = (case.root / "events").read_text().splitlines()
    assert events.index("acquired:9") < events.index("acquired:10") < events.index("gpu-probe")


@pytest.mark.parametrize("descriptor,phase", [
    (9, "waiting_for_repository_gpu_lock"), (10, "waiting_for_shared_scheduler_lease")])
def test_held_lease_prevents_child_start_and_stop_cancels_wait(shell_case, descriptor, phase):
    case = shell_case
    (case.root / f"held-{descriptor}").touch()
    with running_runner(case) as process:
        wait_for_file(case.root / f"observed-held-{descriptor}", process)
        assert (case.run / "controller.phase").read_text().strip() == phase
        assert not (case.root / "calls").exists()
        assert "gpu-probe" not in (case.root / "events").read_text()
        (case.run / "STOP").touch()
        assert_stopped(case, process)
    assert not (case.root / "calls").exists()
    assert (case.root / f"held-{descriptor}").exists()


def test_parent_stop_cancels_shared_scheduler_lease_wait(shell_case):
    case = shell_case
    marker = case.root / "parent STOP"
    case.env["PARENT_STOP_FILE"] = marker.as_posix()
    (case.root / "held-10").touch()
    with running_runner(case) as process:
        wait_for_file(case.root / "observed-held-10", process)
        marker.touch()
        assert_stopped(case, process)
    assert not (case.root / "calls").exists()


def test_lock_errors_fail_closed_without_starting_training(shell_case):
    case = shell_case
    (case.root / "error-10").touch()
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=8)
        assert process.returncode == 69, errors
    assert (case.run / "controller.exit").read_text().strip() == "69"
    assert not (case.root / "calls").exists()


@pytest.mark.parametrize("stage,expected", [
    ("evaluate", ["train", "evaluate"]),
    ("render", ["train", "evaluate", "evaluate", "render"])])
def test_stop_between_review_stages_prevents_next_child(shell_case, stage, expected):
    case = shell_case
    case.env.update(TEST_CHILD_MODE="complete", TEST_STOP_AFTER=stage)
    with running_runner(case) as process:
        assert_stopped(case, process)
    assert (case.root / "calls").read_text().splitlines() == expected

"""The combined trial completes 5,000 actor updates and preserves visual reviews."""

import shlex

from test_gait_restore_runner import (
    REPO, SCHEDULER_LOCK, case, calls, run_to_exit, assert_inputs_preserved,
)
from test_stealth_stop import bash, assert_stopped, running_runner, wait_for_file


def configure(case):
    runner = REPO / "scripts/rls/supermimic/run_g1_stealth_combined.sh"
    source = runner.read_text(encoding="utf-8")
    fixture = case.root / runner.name
    fixture.write_text(source.replace(
        SCHEDULER_LOCK, shlex.quote((case.root / "scheduler lock.flock").as_posix())),
        encoding="utf-8", newline="\n")
    config = case.root / "scripts/rls/supermimic/configs/command_stealth_combined.json"
    config.write_text("{}\n", encoding="ascii")
    case.command = [*case.command[:3], fixture.as_posix(), *case.command[4:8], "fake-code-pin"]


def test_once_warm_start_then_full_resume_and_all_recordings(case):
    configure(case)
    run_to_exit(case)
    trainers = calls(case, "train")
    assert [row["iteration"] for row in trainers] == ["2500", "5500"]
    assert trainers[0]["warm_start"] == case.initial.as_posix()
    assert trainers[1]["warm_start"] == ""
    assert trainers[1]["resume"].endswith("/training/iteration_002500.pt")
    assert (case.run / "actor_update_budget").read_text().strip() == "5000"
    assert (case.run / "completed_iterations").read_text().strip() == "5500"
    assert (case.run / "controller.phase").read_text().strip() == "completed_5500_awaiting_visual_review"
    assert len(calls(case, "evaluate")) == 6
    assert len(calls(case, "render")) == 22
    assert not calls(case, "check")
    for stage in ("002500", "005500"):
        assert (case.run / f"milestones/{stage}/checkpoint.receipt").exists()
        assert (case.run / f"milestones/{stage}/review/release/views/entry-front-side.mp4").exists()
    assert_inputs_preserved(case)


def test_runtime_failure_preserves_completed_midpoint(case):
    configure(case)
    case.env["TEST_FAIL_AT"] = "5500"
    run_to_exit(case, expected=42)
    assert (case.run / "training/iteration_002500.pt").exists()
    assert (case.run / "completed_iterations").read_text().strip() == "2500"
    assert (case.run / "controller.phase").read_text().strip() == "failed:training_to_5500"
    assert len(calls(case, "evaluate")) == 3
    assert_inputs_preserved(case)


def test_gpu_wait_honors_owner_stop_before_any_training(case):
    configure(case)
    with open(case.env["BASH_ENV"], "a", encoding="utf-8") as stream:
        stream.write("\nnvidia-smi() { : > \"$TEST_ROOT/gpu-wait-observed\"; printf '12345\\n'; }\n")
    with running_runner(case) as process:
        wait_for_file(case.root / "gpu-wait-observed", process)
        (case.run / "STOP").write_text("owner stop\n", encoding="ascii")
        assert_stopped(case, process)
    assert not (case.root / "calls").exists()
    assert (case.run / "gpu_wait.log").is_file()

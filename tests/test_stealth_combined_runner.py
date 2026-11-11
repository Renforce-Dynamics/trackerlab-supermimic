"""The combined trial preserves full-state continuation through 50,000 actor updates."""

import json
import shlex

import pytest

from test_gait_restore_runner import (
    FAKE_PYTHON, REPO, SCHEDULER_LOCK, case, calls, assert_inputs_preserved,
)
from test_stealth_stop import bash, assert_stopped, running_runner, wait_for_file


MOCK_BUDGET = r'''    budget)
        start=0
        if [ -n "$resume" ]; then
            test -f "$resume"
            start=${TEST_RESUME_ITERATION:-1373}
        fi
        final=50500
        actors=50000
        if [ "${SUPERMIMIC_STEALTH_RECIPE:-}" = beyondamp ]; then final=20500; actors=20000; fi
        future=()
        for target in 2500 5500 10500 20500 30500 40500 50500; do
            if ((target > start && target <= final)); then future+=("$target"); fi
        done
        printf '{"starting_iteration":%s,"target_iteration":%s,"actor_update_budget":%s}\n' "$start" "$final" "$actors" > "$output"
        printf '%s %s' "$start" "$final"
        printf ' %s' "${future[@]}"
        printf '\n'
        ;;
'''


def configure(case):
    runner = REPO / "scripts/rls/supermimic/run_g1_stealth_combined.sh"
    source = runner.read_text(encoding="utf-8")
    fixture = case.root / runner.name
    fixture.write_text(source.replace(
        SCHEDULER_LOCK, shlex.quote((case.root / "scheduler lock.flock").as_posix())),
        encoding="utf-8", newline="\n")
    config = case.root / "scripts/rls/supermimic/configs/command_stealth_combined.json"
    config.write_text("{}\n", encoding="ascii")
    fake = FAKE_PYTHON.replace("*gait_restore_budget.py)", "*combined_training_budget.py)")
    beginning, rest = fake.split("    budget)", 1)
    _, tail = rest.split("    train)", 1)
    (case.root / "fake-python.sh").write_text(
        beginning + MOCK_BUDGET + "    train)" + tail, encoding="ascii", newline="\n")
    case.command = [*case.command[:3], fixture.as_posix(), *case.command[4:8], "fake-code-pin"]


def run_to_exit(case, expected=0):
    with running_runner(case) as process:
        output, errors = process.communicate(timeout=60)
        assert process.returncode == expected, (output, errors)
    assert (case.run / "controller.exit").read_text().strip() == str(expected)


@pytest.mark.parametrize("resumed", [False, True])
def test_budget_preserves_resume_and_records_every_pending_milestone(case, resumed):
    configure(case)
    targets = [2500, 5500, 10500, 20500, 30500, 40500, 50500]
    if resumed:
        case.command.append(case.resume.as_posix())
        case.env["TEST_RESUME_ITERATION"] = "5500"
        targets = targets[2:]
    run_to_exit(case)
    trainers = calls(case, "train")
    assert [row["iteration"] for row in trainers] == [str(v) for v in targets]
    assert trainers[0]["warm_start"] == ("" if resumed else case.initial.as_posix())
    assert trainers[0]["resume"] == (case.resume.as_posix() if resumed else "")
    for row, previous in zip(trainers[1:], targets):
        assert row["warm_start"] == ""
        assert row["resume"].endswith(f"/training/iteration_{previous:06d}.pt")
    assert (case.run / "actor_update_budget").read_text().strip() == "50000"
    assert (case.run / "target_iterations").read_text().strip() == "50500"
    assert (case.run / "starting_iteration").read_text().strip() == ("5500" if resumed else "0")
    assert (case.run / "completed_iterations").read_text().strip() == "50500"
    assert (case.run / "controller.phase").read_text().strip() == "completed_50500_awaiting_visual_review"
    assert len(calls(case, "evaluate")) == 3 * len(targets)
    assert len(calls(case, "render")) == 11 * len(targets)
    assert not calls(case, "check")
    plan = json.loads((case.run / "training_budget.json").read_text())
    assert plan["actor_update_budget"] == 50000 and plan["target_iteration"] == 50500
    for target in targets:
        stage = f"{target:06d}"
        assert (case.run / f"milestones/{stage}/checkpoint.receipt").exists()
        assert (case.run / f"milestones/{stage}/review/release/views/entry-front-side.mp4").exists()
    assert_inputs_preserved(case)


def test_runtime_failure_preserves_completed_midpoint(case):
    configure(case)
    case.env["TEST_FAIL_AT"] = "10500"
    run_to_exit(case, expected=42)
    assert (case.run / "training/iteration_005500.pt").exists()
    assert (case.run / "completed_iterations").read_text().strip() == "5500"
    assert (case.run / "controller.phase").read_text().strip() == "failed:training_to_10500"
    assert len(calls(case, "evaluate")) == 6
    assert_inputs_preserved(case)


def test_gpu_wait_honors_owner_stop_before_any_training(case):
    configure(case)
    with open(case.env["BASH_ENV"], "a", encoding="utf-8") as stream:
        stream.write("\nnvidia-smi() { : > \"$TEST_ROOT/gpu-wait-observed\"; printf '12345\\n'; }\n")
    with running_runner(case) as process:
        wait_for_file(case.root / "gpu-wait-observed", process)
        (case.run / "STOP").write_text("owner stop\n", encoding="ascii")
        assert_stopped(case, process)
    assert [row["mode"] for row in calls(case)] == ["budget"]
    assert (case.run / "gpu_wait.log").is_file()


def test_owner_stop_during_resumed_training_keeps_latest_checkpoint(case):
    configure(case)
    case.command.append(case.resume.as_posix())
    case.env["TEST_CHILD_MODE"] = "wait"
    with running_runner(case) as process:
        wait_for_file(case.root / "child-started", process)
        (case.run / "STOP").write_text("owner stop\n", encoding="ascii")
        assert_stopped(case, process)
    assert (case.run / "training/last.pt").read_text().strip() == "checkpoint-on-stop"
    assert calls(case, "train")[0]["resume"] == case.resume.as_posix()
    assert not calls(case, "evaluate")
    assert_inputs_preserved(case)


def test_velocity_led_selects_new_preset_with_same_preserved_review_budget(case):
    configure(case)
    preset = case.root / "scripts/rls/supermimic/configs/command_stealth_velocity_led.json"
    preset.write_text("{}\n", encoding="ascii")
    case.env["SUPERMIMIC_STEALTH_RECIPE"] = "velocity-led"
    run_to_exit(case)
    assert len(calls(case, "train")) == 7
    assert len(calls(case, "evaluate")) == 21
    assert "command_stealth_velocity_led.json" in (case.run / "input.receipt").read_text()


def test_shared_aim_requires_and_hashes_the_reviewed_reference(case):
    configure(case)
    preset = case.root/"scripts/rls/supermimic/configs/command_stealth_shared_aim.json"
    preset.write_text("{}\n", encoding="ascii")
    reference = case.root/"pool/aim-style-reference.json"
    reference.write_text("{}\n", encoding="ascii")
    case.env["SUPERMIMIC_STEALTH_RECIPE"] = "shared-aim"
    run_to_exit(case)
    assert len(calls(case, "train")) == 7
    receipt = (case.run/"input.receipt").read_text()
    assert "command_stealth_shared_aim.json" in receipt
    assert "aim-style-reference.json" in receipt


def test_beyondamp_keeps_geometry_receipts_and_reviews_the_twenty_thousand_budget(case):
    configure(case)
    preset = case.root/"scripts/rls/supermimic/configs/command_stealth_beyondamp.json"
    preset.write_text("{}\n", encoding="ascii")
    (case.root/"pool/aim-style-reference.json").write_text("{}\n", encoding="ascii")
    case.env["SUPERMIMIC_STEALTH_RECIPE"] = "beyondamp"
    run_to_exit(case)
    assert [int(row["iteration"]) for row in calls(case, "train")] == [2500, 5500, 10500, 20500]
    assert (case.run/"actor_update_budget").read_text().strip() == "20000"
    assert (case.run/"completed_iterations").read_text().strip() == "20500"
    assert "aim-style-reference.json" in (case.run/"input.receipt").read_text()
    assert len(calls(case, "render")) == 44

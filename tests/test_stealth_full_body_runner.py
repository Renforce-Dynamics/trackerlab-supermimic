"""Bounded full-body controller checks using local substitutes for all workloads."""

import json
from pathlib import Path
import subprocess

import pytest

from test_gait_restore_runner import REPO, assert_inputs_preserved, bash, calls, case
from test_stealth_combined_runner import configure, run_to_exit
from test_stealth_stop import assert_stopped, running_runner, wait_for_file


SCRIPTS = REPO / "scripts/rls/supermimic"
WRAPPER = SCRIPTS / "run_g1_stealth_full_body.sh"


@pytest.fixture
def fullbody_case(case):
    configure(case)
    for name in ("SUPERMIMIC_STEALTH_RECIPE", "SUPERMIMIC_STEALTH_EXTRA_UPDATES",
                 "TEST_RESUME_ITERATION", "TEST_PREFLIGHT_FAIL", "TEST_PREFLIGHT_MUTATE",
                 "TEST_BRANCH", "TEST_CODE_PIN", "TEST_DIRTY"):
        case.env.pop(name, None)
    preset = case.root / "scripts/rls/supermimic/configs/command_stealth_full_body.json"
    preset.write_bytes((SCRIPTS / "configs" / preset.name).read_bytes())
    for name in ("aim-style-reference.json", "full-body-style.json"):
        (case.root / "pool" / name).write_text("{}\n", encoding="ascii")
    wrapper = case.root / WRAPPER.name
    wrapper.write_text(WRAPPER.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    case.command[3] = wrapper.as_posix()
    with Path(case.env["BASH_ENV"]).open("a", encoding="ascii") as stream:
        stream.write(r'''
git() {
    case "$*" in
        'branch --show-current') printf '%s\n' "${TEST_BRANCH:-main}" ;;
        'rev-parse --show-toplevel') printf '%s\n' "$TEST_ROOT" ;;
        'rev-parse HEAD') printf '%s\n' "${TEST_CODE_PIN:-fake-code-pin}" ;;
        'status --porcelain') printf '%s' "${TEST_DIRTY:-}" ;;
        *) return 97 ;;
    esac
}
''')
    fake = case.root / "fake-python.sh"
    script = fake.read_text(encoding="ascii")
    script = script.replace("mode=unknown", r'''
verify_pool= selected_recipe= budget_pool= aim_reference= config=
printf '%s\n' "$@" >> "$TEST_ROOT/python-arguments"
mode=unknown''')
    script = script.replace("*combined_training_budget.py)", r'''
        *prepare_stealth_full_body.py) mode=preflight; shift ;;
        --verify-pool) verify_pool=$2; shift 2 ;;
        --recipe) selected_recipe=$2; shift 2 ;;
        --pool) budget_pool=$2; shift 2 ;;
        --aim-style-reference) aim_reference=$2; shift 2 ;;
        --config) config=$2; shift 2 ;;
        *combined_training_budget.py)''')
    script = script.replace("    budget)", r'''    preflight)
        test "${CUDA_VISIBLE_DEVICES-unset}" = ""
        test "$verify_pool" = "$TEST_ROOT/pool"
        test -f "$verify_pool/aim-style-reference.json"
        test -f "$verify_pool/full-body-style.json"
        printf 'fullbody-preflight\n' >> "$TEST_ROOT/events"
        if [ -n "${TEST_PREFLIGHT_FAIL:-}" ]; then exit "$TEST_PREFLIGHT_FAIL"; fi
        if [ -n "${TEST_PREFLIGHT_MUTATE:-}" ]; then
            printf 'changed\n' >> "$TEST_PREFLIGHT_MUTATE"
        fi
        printf '{"verified":true,"mock":true}\n'
        stop_after preflight
        ;;
    budget)
        test "${CUDA_VISIBLE_DEVICES-unset}" = ""
        test "$selected_recipe" = fullbody
        test "$budget_pool" = "$TEST_ROOT/pool"
''')
    script = script.replace("        future=()", r'''
        if [ "${SUPERMIMIC_STEALTH_RECIPE:-}" = fullbody ]; then
            final=5500
            actors=5000
            targets=(5500)
        fi
        future=()''')
    script = script.replace("    train)", r'''    train)
        test -f "$TEST_RUN/full-body-pool-verified.json"
        test "$aim_reference" = "$TEST_ROOT/pool/aim-style-reference.json"
        test "$config" = "$TEST_ROOT/scripts/rls/supermimic/configs/command_stealth_full_body.json"
        printf 'fullbody-train\n' >> "$TEST_ROOT/events"
''')
    fake.write_text(script, encoding="ascii", newline="\n")
    return case


def test_fullbody_preset_changes_only_budget_and_measured_height():
    folder = SCRIPTS / "configs"
    beyondamp = json.loads((folder / "command_stealth_beyondamp.json").read_text())
    fullbody = json.loads((folder / "command_stealth_full_body.json").read_text())
    assert fullbody == beyondamp | {"iterations": 5500, "target_clearance_m": 0.6651475726945968}
    assert fullbody["stealth_control_mode"] == "wasd"
    assert "command_training_modes" not in fullbody
    assert fullbody["stealth_upper_body_constraint"] is False
    assert fullbody["waist_pd_scale"] == 1.
    assert fullbody["amp_weight"] == .1


@pytest.mark.parametrize("filename", ["run_g1_stealth_full_body.sh", "run_g1_stealth_combined.sh"])
def test_fullbody_shell_syntax(bash, filename):
    result = subprocess.run([bash, "-n", (SCRIPTS / filename).as_posix()],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("resumed", [False, True])
def test_fullbody_runs_cpu_preflight_then_only_final_review(fullbody_case, resumed):
    case = fullbody_case
    # The wrapper must choose its own bounded recipe regardless of the caller's recipe.
    case.env["SUPERMIMIC_STEALTH_RECIPE"] = "beyondamp"
    case.env["CUDA_VISIBLE_DEVICES"] = "0"
    if resumed:
        case.command.append(case.resume.as_posix())
        case.env["TEST_RESUME_ITERATION"] = "2500"
    run_to_exit(case)
    trainer, = calls(case, "train")
    assert trainer["iteration"] == "5500"
    assert trainer["warm_start"] == ("" if resumed else case.initial.as_posix())
    assert trainer["resume"] == (case.resume.as_posix() if resumed else "")
    assert [row["mode"] for row in calls(case)][:3] == ["budget", "preflight", "train"]
    assert (case.run / "target_iterations").read_text().strip() == "5500"
    assert (case.run / "actor_update_budget").read_text().strip() == "5000"
    assert (case.run / "starting_iteration").read_text().strip() == ("2500" if resumed else "0")
    assert (case.run / "completed_iterations").read_text().strip() == "5500"
    assert (case.run / "controller.phase").read_text().strip() == "completed_5500_awaiting_visual_review"
    assert [p.name for p in (case.run / "milestones").iterdir()] == ["005500"]
    receipt = (case.run / "input.receipt").read_text()
    for name in ("aim-style-reference.json", "full-body-style.json", "manifest.json",
                 "train.json", "command_stealth_full_body.json", case.initial.name):
        assert name in receipt
    if resumed:
        assert case.resume.name in receipt
    assert json.loads((case.run / "full-body-pool-verified.json").read_text())["verified"]
    events = (case.root / "events").read_text().splitlines()
    assert events.index("fullbody-preflight") < events.index("fullbody-train")
    reviews = calls(case, "evaluate")
    assert [(r["envs"], r["steps"]) for r in reviews] == [
        ("24", "1000"), ("90", "1000"), ("18", "2100")]
    renders = calls(case, "render")
    assert len(renders) == 11
    assert [r["lane"] for r in renders[1:10]] == [str(i) for i in range(9)]
    assert [r["seconds"] for r in renders] == ["20"] * 10 + ["42"]
    args = (case.root / "python-arguments").read_text().splitlines()
    for unwanted in ("--allow-style-feature-change", "--allow-command-sampling-change",
                     "--allow-command-reward-change", "--allow-support-gradient-change",
                     "--upper-body-reference", "--waist-entry-reference"):
        assert unwanted not in args
    assert "--command-mode" in args and "forward" in args
    assert "--mode-layout" in args and "cycle" in args
    assert "--command-sequence" in args and "wasd-release-regression" in args
    stage = case.run / "milestones/005500"
    assert (stage / "checkpoint.receipt").exists()
    for relative in ("forward/views", "release/views", *(f"cycle/lane{i}" for i in range(9))):
        assert (stage / "review" / relative / "entry-front-side.mp4").is_file()
    assert_inputs_preserved(case)


@pytest.mark.parametrize("name", ["aim-style-reference.json", "full-body-style.json"])
def test_fullbody_requires_both_reference_documents(fullbody_case, name):
    case = fullbody_case
    (case.root / "pool" / name).unlink()
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=10)
        assert process.returncode != 0, errors
    assert not (case.root / "calls").exists()


@pytest.mark.parametrize("variable,value", [
    ("TEST_BRANCH", "other"), ("TEST_CODE_PIN", "different-pin"),
    ("TEST_DIRTY", " M owned-source.py"),
])
def test_fullbody_refuses_different_branch_pin_or_dirty_source(fullbody_case, variable, value):
    case = fullbody_case
    case.env[variable] = value
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=10)
        assert process.returncode != 0, errors
    assert not (case.root / "calls").exists()


@pytest.mark.parametrize("extra", ["5000", "10000", "-1"])
def test_fullbody_never_accepts_extra_updates_even_with_resume(fullbody_case, extra):
    case = fullbody_case
    case.env["SUPERMIMIC_STEALTH_EXTRA_UPDATES"] = extra
    case.command.append(case.resume.as_posix())
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=10)
        assert process.returncode == 2, errors
    assert not (case.root / "calls").exists()


def test_fullbody_preflight_failure_prevents_training(fullbody_case):
    case = fullbody_case
    case.env["TEST_PREFLIGHT_FAIL"] = "41"
    run_to_exit(case, expected=41)
    assert [r["mode"] for r in calls(case)] == ["budget", "preflight"]
    assert (case.run / "controller.phase").read_text().strip() == "failed:verifying_full_body_pool"
    assert not (case.run / "training").exists()
    assert_inputs_preserved(case)


@pytest.mark.parametrize("relative", [
    "pool/full-body-style.json", "pool/aim-style-reference.json",
    "pool/manifest.json", "pool/train.json",
    "scripts/rls/supermimic/configs/command_stealth_full_body.json",
    "entry 10000.pt", "recovery 001000.pt",
])
def test_fullbody_input_mutation_during_gpu_wait_prevents_training(fullbody_case, relative):
    case = fullbody_case
    case.command.append(case.resume.as_posix())
    with Path(case.env["BASH_ENV"]).open("a", encoding="ascii") as stream:
        stream.write(r'''
nvidia-smi() {
    : > "$TEST_ROOT/gpu-wait-observed"
    if [ ! -e "$TEST_ROOT/release-gpu" ]; then printf '12345\n'; fi
}
''')
    with running_runner(case) as process:
        wait_for_file(case.root / "gpu-wait-observed", process)
        (case.root / relative).write_text("changed\n", encoding="ascii")
        (case.root / "release-gpu").touch()
        _, errors = process.communicate(timeout=10)
        assert process.returncode != 0, errors
    assert [r["mode"] for r in calls(case)] == ["budget"]
    assert not (case.run / "training").exists()


def test_fullbody_rechecks_inputs_after_preflight(fullbody_case):
    case = fullbody_case
    case.env["TEST_PREFLIGHT_MUTATE"] = (case.root / "pool/full-body-style.json").as_posix()
    run_to_exit(case, expected=1)
    assert [r["mode"] for r in calls(case)] == ["budget", "preflight"]
    assert not (case.run / "training").exists()


def test_fullbody_owner_stop_after_preflight_preserves_inputs(fullbody_case):
    case = fullbody_case
    case.env["TEST_STOP_AFTER"] = "preflight"
    with running_runner(case) as process:
        assert_stopped(case, process)
    assert [r["mode"] for r in calls(case)] == ["budget", "preflight"]
    assert_inputs_preserved(case)


def test_fullbody_trainer_failure_is_terminal_without_extra_attempts(fullbody_case):
    case = fullbody_case
    case.env["TEST_FAIL_AT"] = "5500"
    run_to_exit(case, expected=42)
    assert [r["mode"] for r in calls(case)] == ["budget", "preflight", "train"]
    assert (case.run / "controller.phase").read_text().strip() == "failed:training_to_5500"
    assert not calls(case, "evaluate")
    assert not (case.run / "completed_iterations").exists()
    assert_inputs_preserved(case)

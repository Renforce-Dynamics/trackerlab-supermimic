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


def recorded_calls(case):
    return calls(case) if (case.root / "calls").exists() else []


@pytest.fixture(params=["fullbody", "fullbody-contact", "fullbody-sole-speed"])
def fullbody_case(case, request):
    configure(case)
    for name in ("SUPERMIMIC_STEALTH_RECIPE", "SUPERMIMIC_STEALTH_EXTRA_UPDATES",
                 "SUPERMIMIC_STEALTH_PHASE_PROFILE_UPDATES",
                 "TEST_RESUME_ITERATION", "TEST_PREFLIGHT_FAIL", "TEST_PREFLIGHT_MUTATE",
                 "TEST_BRANCH", "TEST_CODE_PIN", "TEST_DIRTY"):
        case.env.pop(name, None)
    case.recipe = request.param
    presets = {
        "fullbody": "command_stealth_full_body.json",
        "fullbody-contact": "command_stealth_full_body_contact.json",
        "fullbody-sole-speed": "command_stealth_full_body_sole_speed.json",
    }
    wrappers = {
        "fullbody": WRAPPER,
        "fullbody-contact": SCRIPTS / "run_g1_stealth_full_body_contact.sh",
        "fullbody-sole-speed": SCRIPTS / "run_g1_stealth_full_body_sole_speed.sh",
    }
    case.preset_name = presets[case.recipe]
    case.env.update(TEST_FULLBODY_RECIPE=case.recipe, TEST_FULLBODY_PRESET=case.preset_name)
    preset = case.root / "scripts/rls/supermimic/configs" / case.preset_name
    preset.write_bytes((SCRIPTS / "configs" / preset.name).read_bytes())
    for name in ("aim-style-reference.json", "full-body-style.json"):
        (case.root / "pool" / name).write_text("{}\n", encoding="ascii")
    source_wrapper = wrappers[case.recipe]
    wrapper = case.root / source_wrapper.name
    wrapper.write_text(source_wrapper.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
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
        test "$selected_recipe" = "$TEST_FULLBODY_RECIPE"
        test "$budget_pool" = "$TEST_ROOT/pool"
''')
    script = script.replace("        future=()", r'''
        if [ "${SUPERMIMIC_STEALTH_RECIPE:-}" = "$TEST_FULLBODY_RECIPE" ]; then
            final=$((5500+${SUPERMIMIC_STEALTH_EXTRA_UPDATES:-0}))
            actors=$((final-500))
            targets=("$final")
        fi
        future=()''')
    script = script.replace("    train)", r'''    train)
        test -f "$TEST_RUN/full-body-pool-verified.json"
        test "$aim_reference" = "$TEST_ROOT/pool/aim-style-reference.json"
        test "$config" = "$TEST_ROOT/scripts/rls/supermimic/configs/$TEST_FULLBODY_PRESET"
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


def test_contact_preset_changes_only_gait_version():
    folder = SCRIPTS / "configs"
    fullbody = json.loads((folder / "command_stealth_full_body.json").read_text())
    contact = json.loads((folder / "command_stealth_full_body_contact.json").read_text())
    assert contact == fullbody | {"stealth_gait_version": 3}
    assert contact["iterations"] == 5500
    assert contact["wm_only_iters"] == 500
    assert contact["stealth_gait_weight"] == .02
    assert contact["stealth_swing_height"] == .04
    assert contact["amp_weight"] == .1
    assert contact["entry_duration_s"] == .6
    assert "command_training_modes" not in contact


def test_sole_speed_preset_changes_only_the_discriminator_feature_mode():
    folder = SCRIPTS / "configs"
    contact = json.loads((folder / "command_stealth_full_body_contact.json").read_text())
    sole = json.loads((folder / "command_stealth_full_body_sole_speed.json").read_text())
    assert sole == contact | {"style_feature_mode": "postural_end_effectors_vx_sole_speed"}
    assert sole["iterations"] - sole["wm_only_iters"] == 5000
    assert sole["amp_weight"] == .1
    assert sole["stealth_gait_version"] == 3


@pytest.mark.parametrize("filename", ["run_g1_stealth_full_body.sh", "run_g1_stealth_combined.sh",
                                      "run_g1_stealth_full_body_contact.sh",
                                      "run_g1_stealth_full_body_sole_speed.sh"])
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
                 "train.json", case.preset_name, case.initial.name):
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
    if case.recipe in ("fullbody-contact", "fullbody-sole-speed"):
        assert "--extra-actor-updates" not in args
    assert "--phase-profile-updates" not in args
    assert (case.run / "phase_profile_updates").read_text().strip() == "0"
    assert [r["mode"] for r in calls(case)] == ["budget", "preflight", "train",
        "evaluate", "render", "evaluate", "evaluate", *(["render"] * 10)]
    events = (case.root / "events").read_text().splitlines()
    assert events.index("acquired:8") < events.index("acquired:9") < (
        events.index("acquired:10")) < events.index("gpu-probe")
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


@pytest.mark.parametrize("extra", ["10000", "-1"])
def test_fullbody_rejects_unbounded_extra_updates(fullbody_case, extra):
    case = fullbody_case
    case.env["SUPERMIMIC_STEALTH_EXTRA_UPDATES"] = extra
    case.command.append(case.resume.as_posix())
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=10)
        assert process.returncode == 2, errors
    assert not (case.root / "calls").exists()


@pytest.mark.parametrize("fullbody_case", ["fullbody"], indirect=True)
def test_fullbody_extension_uses_one_full_resume_and_optional_timing(fullbody_case):
    case = fullbody_case
    case.env.update(SUPERMIMIC_STEALTH_EXTRA_UPDATES="5000",
                    SUPERMIMIC_STEALTH_PHASE_PROFILE_UPDATES="1000",
                    TEST_RESUME_ITERATION="5500")
    case.command.append(case.resume.as_posix())
    run_to_exit(case)
    trainer, = calls(case, "train")
    assert trainer["iteration"] == "10500"
    assert trainer["resume"] == case.resume.as_posix() and trainer["warm_start"] == ""
    assert (case.run / "starting_iteration").read_text().strip() == "5500"
    assert (case.run / "actor_update_budget").read_text().strip() == "10000"
    assert (case.run / "phase_profile_updates").read_text().strip() == "1000"
    assert (case.run / "controller.phase").read_text().strip() == "completed_10500_awaiting_visual_review"
    assert [p.name for p in (case.run / "milestones").iterdir()] == ["010500"]
    arguments = (case.root / "python-arguments").read_text().splitlines()
    position = arguments.index("--phase-profile-updates")
    assert arguments[position + 1] == "1000"
    assert len(calls(case, "evaluate")) == 3 and len(calls(case, "render")) == 11
    assert_inputs_preserved(case)


@pytest.mark.parametrize("fullbody_case", ["fullbody"], indirect=True)
def test_fullbody_extension_requires_full_resume(fullbody_case):
    case = fullbody_case
    case.env["SUPERMIMIC_STEALTH_EXTRA_UPDATES"] = "5000"
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=10)
        assert process.returncode == 2 and "full resume checkpoint" in errors
    assert not (case.root / "calls").exists()


@pytest.mark.parametrize("value", ["-1", "500", "1001"])
def test_fullbody_rejects_invalid_phase_window(fullbody_case, value):
    case = fullbody_case
    case.env["SUPERMIMIC_STEALTH_PHASE_PROFILE_UPDATES"] = value
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=10)
        assert process.returncode == 2, errors
    assert not (case.root / "calls").exists()


@pytest.mark.parametrize("fullbody_case", ["fullbody"], indirect=True)
def test_fullbody_phase_window_must_fit_remaining_updates(fullbody_case):
    case = fullbody_case
    case.env.update(SUPERMIMIC_STEALTH_EXTRA_UPDATES="5000",
                    SUPERMIMIC_STEALTH_PHASE_PROFILE_UPDATES="1000",
                    TEST_RESUME_ITERATION="9000")
    case.command.append(case.resume.as_posix())
    run_to_exit(case, expected=2)
    assert [r["mode"] for r in calls(case)] == ["budget"]


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
    if relative.endswith("command_stealth_full_body.json"):
        relative = "scripts/rls/supermimic/configs/" + case.preset_name
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
    assert [r["mode"] for r in recorded_calls(case)] == ["budget", "preflight", "train"]
    assert (case.run / "controller.phase").read_text().strip() == "failed:training_to_5500"
    assert not calls(case, "evaluate")
    assert not (case.run / "completed_iterations").exists()
    assert_inputs_preserved(case)


@pytest.mark.parametrize("fullbody_case", ["fullbody-contact", "fullbody-sole-speed"], indirect=True)
@pytest.mark.parametrize("resumed", [False, True])
def test_contact_rejects_extra_five_thousand_before_any_workload(fullbody_case, resumed):
    case = fullbody_case
    case.env["SUPERMIMIC_STEALTH_EXTRA_UPDATES"] = "5000"
    if resumed:
        case.command.append(case.resume.as_posix())
    with running_runner(case) as process:
        _, errors = process.communicate(timeout=10)
        assert process.returncode == 2 and "fixed 5000" in errors
    assert not (case.root / "calls").exists()
    assert not (case.root / "events").exists()
    assert_inputs_preserved(case)


@pytest.mark.parametrize("fullbody_case", ["fullbody-contact", "fullbody-sole-speed"], indirect=True)
@pytest.mark.parametrize("start,expected", [(0, 0), (2500, 0), (4000, 2)])
def test_contact_optional_timing_stays_within_original_budget(fullbody_case, start, expected):
    case = fullbody_case
    case.env["SUPERMIMIC_STEALTH_PHASE_PROFILE_UPDATES"] = "1000"
    if start:
        case.env["TEST_RESUME_ITERATION"] = str(start)
        case.command.append(case.resume.as_posix())
    run_to_exit(case, expected=expected)
    assert (case.run / "target_iterations").read_text().strip() == "5500"
    assert (case.run / "actor_update_budget").read_text().strip() == "5000"
    if expected:
        assert [r["mode"] for r in calls(case)] == ["budget"]
    else:
        trainer, = calls(case, "train")
        assert trainer["iteration"] == "5500"
        arguments = (case.root / "python-arguments").read_text().splitlines()
        position = arguments.index("--phase-profile-updates")
        assert arguments[position + 1] == "1000"
        assert len(calls(case, "render")) == 11
    assert_inputs_preserved(case)


@pytest.mark.parametrize("descriptor,phase", [
    (9, "waiting_for_repository_gpu_lock"),
    (10, "waiting_for_shared_scheduler_lease"),
])
def test_fullbody_held_lease_waits_then_runs_after_release(fullbody_case, descriptor, phase):
    case = fullbody_case
    (case.root / f"held-{descriptor}").touch()
    with running_runner(case) as process:
        wait_for_file(case.root / f"observed-held-{descriptor}", process)
        assert (case.run / "controller.phase").read_text().strip() == phase
        assert (case.run / "controller.pid").read_text().strip()
        assert (case.run / "code.receipt").read_text().strip() == "fake-code-pin"
        assert (case.run / "input.receipt").is_file()
        assert not (case.root / "calls").exists()
        (case.root / f"held-{descriptor}").unlink()
        _, errors = process.communicate(timeout=60)
        assert process.returncode == 0, errors
    assert (case.run / "controller.exit").read_text().strip() == "0"
    assert [row["mode"] for row in calls(case)][:3] == ["budget", "preflight", "train"]
    assert (case.root / f"observed-held-{descriptor}").is_file()
    events = (case.root / "events").read_text().splitlines()
    assert events.index("acquired:8") < events.index("acquired:9") < (
        events.index("acquired:10")) < events.index("gpu-probe")
    assert_inputs_preserved(case)


def test_fullbody_controller_lock_rejects_duplicate_without_overwriting_receipts(fullbody_case):
    case = fullbody_case
    case.env["TEST_CHILD_MODE"] = "wait"
    with running_runner(case) as process:
        wait_for_file(case.root / "child-started", process)
        receipts = {
            name: (case.run / name).read_bytes()
            for name in ("controller.phase", "controller.pid", "code.receipt", "input.receipt")
        }
        (case.root / "held-8").touch()
        duplicate = subprocess.Popen(case.command, cwd=case.root, env=case.env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, errors = duplicate.communicate(timeout=10)
        assert duplicate.returncode == 75, (output, errors)
        for name, content in receipts.items():
            assert (case.run / name).read_bytes() == content
        assert not (case.run / "controller.exit").exists()
        (case.root / "held-8").unlink()
        (case.run / "STOP").write_text("owner stop\n", encoding="ascii")
        assert_stopped(case, process)
    assert [row["mode"] for row in calls(case)] == ["budget", "preflight", "train"]
    assert_inputs_preserved(case)


@pytest.mark.parametrize("descriptor,phase", [
    (9, "waiting_for_repository_gpu_lock"),
    (10, "waiting_for_shared_scheduler_lease"),
])
def test_fullbody_owner_stop_during_resource_wait_prevents_training(fullbody_case, descriptor, phase):
    case = fullbody_case
    (case.root / f"held-{descriptor}").touch()
    with running_runner(case) as process:
        wait_for_file(case.root / f"observed-held-{descriptor}", process)
        assert (case.run / "controller.phase").read_text().strip() == phase
        (case.run / "STOP").write_text("owner stop\n", encoding="ascii")
        assert_stopped(case, process)
    assert recorded_calls(case) == []
    assert (case.run / "controller.pid").read_text().strip()
    assert (case.run / "controller.exit").read_text().strip() == "0"
    assert_inputs_preserved(case)


@pytest.mark.parametrize("descriptor,phase", [
    (9, "waiting_for_repository_gpu_lock"),
    (10, "waiting_for_shared_scheduler_lease"),
])
def test_fullbody_resource_lock_error_is_terminal_with_receipts(fullbody_case, descriptor, phase):
    case = fullbody_case
    (case.root / f"error-{descriptor}").touch()
    run_to_exit(case, expected=69)
    assert recorded_calls(case) == []
    assert (case.run / "controller.exit").read_text().strip() == "69"
    assert (case.run / "controller.phase").read_text().strip() == f"failed:{phase}"
    assert (case.run / "controller.pid").read_text().strip()
    assert (case.run / "code.receipt").read_text().strip() == "fake-code-pin"
    assert (case.run / "input.receipt").is_file()
    assert "gpu-probe" not in (case.root / "events").read_text()
    assert_inputs_preserved(case)


@pytest.mark.parametrize("descriptor,phase", [
    (9, "waiting_for_repository_gpu_lock"),
    (10, "waiting_for_shared_scheduler_lease"),
])
@pytest.mark.parametrize("relative", ["entry 10000.pt", "pool/train.json"])
def test_fullbody_input_mutation_during_resource_wait_prevents_workloads(
        fullbody_case, descriptor, phase, relative):
    case = fullbody_case
    (case.root / f"held-{descriptor}").touch()
    with running_runner(case) as process:
        wait_for_file(case.root / f"observed-held-{descriptor}", process)
        assert (case.run / "controller.phase").read_text().strip() == phase
        (case.root / relative).write_text("changed\n", encoding="ascii")
        (case.root / f"held-{descriptor}").unlink()
        _, errors = process.communicate(timeout=10)
        assert process.returncode != 0, errors
    assert recorded_calls(case) == []
    assert (case.run / "controller.exit").read_text().strip() != "0"
    assert (case.run / "controller.phase").read_text().startswith("failed:")
    assert not (case.run / "training").exists()


def test_fullbody_owner_stop_during_gpu_wait_prevents_training(fullbody_case):
    case = fullbody_case
    with Path(case.env["BASH_ENV"]).open("a", encoding="ascii") as stream:
        stream.write(r'''
nvidia-smi() { : > "$TEST_ROOT/gpu-wait-observed"; printf '12345\n'; }
''')
    with running_runner(case) as process:
        wait_for_file(case.root / "gpu-wait-observed", process)
        (case.run / "STOP").write_text("owner stop\n", encoding="ascii")
        assert_stopped(case, process)
    assert [r["mode"] for r in calls(case)] == ["budget"]
    assert (case.run / "gpu_wait.log").is_file()
    assert_inputs_preserved(case)


@pytest.mark.parametrize("marker", ["STOP", "training/STOP"])
def test_fullbody_owner_stop_during_resume_preserves_checkpoint(fullbody_case, marker):
    case = fullbody_case
    case.command.append(case.resume.as_posix())
    case.env.update(TEST_RESUME_ITERATION="2500", TEST_CHILD_MODE="wait")
    with running_runner(case) as process:
        wait_for_file(case.root / "child-started", process)
        (case.run / marker).write_text("owner stop\n", encoding="ascii")
        assert_stopped(case, process)
    assert [r["mode"] for r in calls(case)] == ["budget", "preflight", "train"]
    trainer, = calls(case, "train")
    assert trainer["resume"] == case.resume.as_posix() and trainer["warm_start"] == ""
    assert (case.run / "training/last.pt").read_text().strip() == "checkpoint-on-stop"
    assert not (case.run / "completed_iterations").exists()
    assert_inputs_preserved(case)

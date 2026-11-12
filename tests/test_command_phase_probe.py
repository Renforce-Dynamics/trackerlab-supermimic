"""CPU-only timing/step provenance checks; CUDA smoke is explicitly opt-in."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/rls/supermimic"
sys.path.insert(0, str(SCRIPTS))
from warp_runtime import command_phase_probe as module
from warp_runtime.command_phase_probe import CommandPhaseProbe


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=value))
    return value


@pytest.fixture
def probes(monkeypatch):
    """Avoid exit callbacks escaping tests, including intentionally failed calls."""
    callbacks = []
    monkeypatch.setattr(module, "atexit", SimpleNamespace(
        register=callbacks.append, unregister=lambda callback: callbacks.remove(callback)))
    return callbacks


@pytest.fixture
def cuda(monkeypatch, clock):
    trace, events = [], []
    stream = object()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.at = None
            events.append(self)

        def record(self, current_stream):
            assert current_stream is stream
            self.at = clock()
            trace.append(("event", self.at))

        def elapsed_time(self, other):
            assert trace[-1][0] == "sync"
            return (other.at - self.at) * 1000

        def synchronize(self):
            pytest.fail("per-event synchronization")

    def sync(device):
        assert str(device) == "cuda:3"
        trace.append(("sync", clock()))
        clock.advance(1.)

    def current_stream(device):
        assert str(device) == "cuda:3"
        return stream

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        name="fake-cuda", major=8, minor=9, total_memory=1000))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 456)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        lambda *a, **k: pytest.fail("global peak reset"))
    monkeypatch.setattr(torch.cuda, "set_device",
                        lambda *a, **k: pytest.fail("global device change"))
    return SimpleNamespace(trace=trace, events=events)


def optimizer():
    return torch.optim.Adam([torch.nn.Parameter(torch.ones(2))], lr=.001)


def step(opt, times=1):
    for _ in range(times):
        for group in opt.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        opt.step()


def load(directory):
    def invalid(value):
        pytest.fail(f"nonfinite JSON constant {value}")
    return json.loads((directory / "command_phase_profile.json").read_text(
        encoding="utf-8"), parse_constant=invalid)


def phase(row, name):
    return next(p for p in row["phases"] if p["name"] == name)


def test_exact_cuda_boundaries_nested_phases_and_io(tmp_path, probes, clock, cuda, monkeypatch):
    opt = optimizer()
    step(opt, 7)
    snapshots = []
    original_snapshot = module._optimizer_snapshot

    def snapshot(optimizer):
        snapshots.append(clock())
        assert cuda.trace[-1][0] == "sync"
        return original_snapshot(optimizer)

    monkeypatch.setattr(module, "_optimizer_snapshot", snapshot)
    metadata = {"initialization": "warm-start", "source_iteration": 25500}
    probe = CommandPhaseProbe(tmp_path, first_iteration=20, updates=2, device="cuda",
                              optimizers={"actor": opt}, metadata=metadata)
    metadata["source_iteration"] = 999
    assert len(probes) == 1
    assert not probe.active and not probe.finished
    probe.begin(19)
    probe.start_phase("outside")
    probe.end_phase("outside")
    probe.end(19)
    assert cuda.trace == []
    probe.begin(20)  # initial drain: clock 0 -> 1
    probe.start_phase("collection")
    clock.advance(2)
    probe.start_phase("nested")
    clock.advance(3)
    probe.end_phase("nested")
    clock.advance(1)
    probe.end_phase("collection")
    clock.advance(4)  # normal logs/checkpoint I/O
    step(opt, 2)
    probe.end(20)
    assert not probe.active and not probe.finished
    assert [t for t in cuda.trace if t[0] == "sync"] == [("sync", 0.)]
    assert snapshots == [1.] * 4
    clock.advance(2)  # between-iteration work belongs to whole wall
    probe.begin(21)
    clock.advance(5)
    step(opt, 3)
    probe.end(21)  # terminal drain: clock 18 -> 19
    assert probe.finished and not probe.active and probes == []
    report = load(tmp_path)
    assert [t for t in cuda.trace if t[0] == "sync"] == [("sync", 0.), ("sync", 18.)]
    assert snapshots == [1.] * 4 + [19.] * 4
    assert report["state"] == "full"
    assert report["iteration_indexing"] == "zero-based"
    assert (report["first_iteration"], report["last_iteration"]) == (20, 21)
    assert report["requested_first_iteration"] == 20
    assert report["requested_last_iteration"] == 21
    assert report["wall_seconds"] == 18.
    assert report["completed_updates_per_second"] == pytest.approx(2/18)
    assert phase(report["rows"][0], "whole")["host_ms"] == 10000.
    assert phase(report["rows"][0], "collection")["cuda_ms"] == 6000.
    assert phase(report["rows"][0], "nested")["cuda_ms"] == 3000.
    assert report["phase_statistics"]["whole"]["host_ms"] == {
        "count": 2, "mean": 7500., "median": 7500., "min": 5000., "max": 10000.}
    counter = report["optimizer_counters"]["actor"]
    assert counter["source"]["uniform_step"] == 7
    assert counter["end"]["uniform_step"] == 12
    assert counter["uniform_delta"] == 5  # not 2 iteration-derived updates
    assert report["metadata"]["source_iteration"] == 25500
    assert "initialization" not in report
    assert report["device"]["requested"] == "cuda"
    assert report["device"]["resolved"] == "cuda:3"
    assert report["device"]["name"] == "fake-cuda"
    assert report["runtime"]["torch"] == torch.__version__
    assert report["memory_peaks"]["end"] == {"allocated_bytes": 123, "reserved_bytes": 456}
    assert any("sums must not be interpreted as total kernel time" in n for n in report["notes"])
    before = (tmp_path / "command_phase_profile.json").read_bytes()
    probe.finish()
    probe.begin(22)
    probe.start_phase("ignored")
    probe.end_phase("ignored")
    probe.end(22)
    assert (tmp_path / "command_phase_profile.json").read_bytes() == before
    assert len([t for t in cuda.trace if t[0] == "sync"]) == 2


def test_actual_cpu_optimizers_independent_resumed_steps_and_no_mutation(
        tmp_path, probes, clock, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU probe called CUDA")
    for name in ("synchronize", "Event", "current_stream", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    opts = {name: optimizer() for name in ("world_model", "actor", "critic", "amp")}
    initial = dict(zip(opts, (21, 13, 37, 9)))
    increments = dict(zip(opts, (3, 1, 4, 2)))
    for name, opt in opts.items():
        step(opt, initial[name])
    snapshots = {name: deepcopy(opt.state_dict()) for name, opt in opts.items()}
    rng = torch.get_rng_state().clone()
    probe = CommandPhaseProbe(tmp_path, first_iteration=301, updates=2, device="cpu",
                              optimizers=opts, metadata={"initialization": "resume"})
    probe.begin(301)
    for name, opt in opts.items():
        torch.testing.assert_close(opt.state_dict(), snapshots[name], atol=0, rtol=0)
    for iteration in (301, 302):
        if iteration == 302:
            probe.begin(iteration)
        for name, opt in opts.items():
            probe.start_phase(name)
            step(opt, increments[name])
            clock.advance(.25)
            probe.end_phase(name)
        probe.end(iteration)
    for name, opt in opts.items():
        expected = deepcopy(opt.state_dict())
        probe.finish()
        torch.testing.assert_close(opt.state_dict(), expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    report = load(tmp_path)
    for name, counter in report["optimizer_counters"].items():
        assert counter["source"]["uniform_step"] == initial[name]
        assert counter["end"]["uniform_step"] == initial[name] + 2*increments[name]
        assert counter["uniform_delta"] == 2*increments[name]
        assert counter["delta_status"] == "valid"
    assert report["wall_seconds"] == 2.
    assert report["memory_peaks"]["source"] is None
    assert report["phase_statistics"]["whole"]["cuda_ms"]["count"] == 0


def test_finish_mid_iteration_separates_partial_work_and_total_counters(
        tmp_path, probes, clock, cuda):
    opt = optimizer()
    step(opt, 4)
    probe = CommandPhaseProbe(tmp_path, first_iteration=10, updates=3, device="cuda:3",
                              optimizers={"amp": opt})
    probe.begin(10)
    step(opt, 2)
    clock.advance(2)
    probe.end(10)
    probe.begin(11)
    probe.start_phase("outer")
    probe.start_phase("done")
    step(opt, 3)
    clock.advance(3)
    probe.end_phase("done")
    probe.start_phase("interrupted")
    step(opt)
    clock.advance(4)
    probes[0]()  # the actual registered normal-exit callback
    report = load(tmp_path)
    assert report["state"] == "partial"
    assert report["completed_updates"] == 1 and len(report["rows"]) == 1
    assert report["last_iteration"] == 10 and report["last_started_iteration"] == 11
    partial = report["partial_iteration"]
    assert partial["iteration"] == 11 and not partial["completed"]
    assert phase(partial, "done")["completed"]
    assert not phase(partial, "outer")["completed"]
    assert not phase(partial, "interrupted")["completed"]
    assert phase(partial, "interrupted")["cuda_ms"] == 4000.
    assert set(report["phase_statistics"]) == {"whole"}
    assert report["optimizer_counters"]["amp"]["uniform_delta"] == 6
    assert report["optimizer_counter_scope"] == {
        "completed_rows": 1, "includes_partial_iteration": True,
        "partial_iteration": 11, "per_completed_row_attribution_available": False}
    assert report["completed_updates_per_second"] is None
    assert len([t for t in cuda.trace if t[0] == "sync"]) == 2
    assert probe.finished and not probe.active


def test_finish_with_no_completed_rows(tmp_path, probes, clock):
    probe = CommandPhaseProbe(tmp_path, first_iteration=5, updates=2, device="cpu",
                              optimizers={})
    probe.begin(5)
    clock.advance(1)
    probe.finish()
    report = load(tmp_path)
    assert report["state"] == "partial" and report["completed_updates"] == 0
    assert report["first_iteration"] == 5 and report["last_iteration"] is None
    assert report["partial_iteration"]["iteration"] == 5
    assert report["phase_statistics"] == {}
    assert report["wall_seconds"] == 1


@pytest.mark.parametrize("cross_window", [False, True])
def test_not_started_never_touches_cuda_or_optimizer(tmp_path, probes, monkeypatch, cross_window):
    def forbidden(*args, **kwargs):
        pytest.fail("not-started probe touched CUDA/optimizer state")
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(module, "_optimizer_snapshot", forbidden)
    probe = CommandPhaseProbe(tmp_path, first_iteration=10, updates=3, device="cuda",
                              optimizers={"actor": optimizer()})
    probe.begin(9)
    if cross_window:
        probe.begin(13)
    probe.finish()
    report = load(tmp_path)
    assert report["state"] == "not-started"
    assert report["rows"] == [] and report["wall_seconds"] is None
    assert report["first_iteration"] is report["last_iteration"] is None
    assert report["partial_iteration"] is None
    assert report["optimizer_counters"]["actor"] == {
        "source": None, "end": None, "uniform_delta": None, "delta_status": "unavailable"}
    assert probe.finished


def test_missing_lazy_steps_are_unknown_and_reading_does_not_create_state(
        tmp_path, probes, clock):
    opt = optimizer()
    parameter = opt.param_groups[0]["params"][0]
    assert len(opt.state) == 0
    probe = CommandPhaseProbe(tmp_path, first_iteration=0, updates=1, device="cpu",
                              optimizers={"actor": opt})
    probe.begin(0)
    assert len(opt.state) == 0
    step(opt, 3)
    clock.advance(1)
    probe.end(0)
    counter = load(tmp_path)["optimizer_counters"]["actor"]
    assert counter["source"]["steps"][0]["step"] is None
    assert counter["source"]["steps"][0]["status"] == "missing"
    assert counter["end"]["uniform_step"] == opt.state[parameter]["step"].item() == 3
    assert counter["uniform_delta"] is None


@pytest.mark.parametrize("case,delta,status", [
    ("different_starts", 2, "valid"), ("different_deltas", None, "nonuniform_delta"),
    ("decrease", None, "steps_decreased"),
    ("reorder", None, "parameter_layout_changed"),
    ("replace", None, "parameter_layout_changed"),
    ("new_group", None, "parameter_layout_changed"),
    ("nan", None, "missing_or_invalid_steps"),
    ("nonscalar", None, "missing_or_invalid_steps"),
    ("negative", None, "missing_or_invalid_steps"),
    ("fractional", None, "missing_or_invalid_steps"),
    ("boolean", None, "missing_or_invalid_steps"),
])
def test_counter_delta_requires_comparable_actual_parameter_steps(
        tmp_path, probes, clock, case, delta, status):
    a, b = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.Adam([a, b], lr=.001)
    step(opt, 2)
    b.grad = None
    a.grad = torch.ones_like(a)
    opt.step()  # actual nonuniform starting steps: [3, 2]
    if case in ("nan", "negative", "fractional", "boolean", "nonscalar"):
        opt.state[a]["step"] = {
            "nan": torch.tensor(float("nan")), "negative": torch.tensor(-1.),
            "fractional": torch.tensor(1.5), "boolean": torch.tensor(True),
            "nonscalar": torch.ones(2),
        }[case]
    probe = CommandPhaseProbe(tmp_path, first_iteration=0, updates=1, device="cpu",
                              optimizers={"world_model": opt})
    probe.begin(0)
    if case == "different_starts":
        step(opt, 2)
    elif case == "different_deltas":
        step(opt)
        b.grad = None
        opt.step()
    elif case == "decrease":
        opt.state[a]["step"].zero_()
    elif case == "reorder":
        opt.param_groups[0]["params"].reverse()
    elif case == "replace":
        opt.param_groups[0]["params"][0] = torch.nn.Parameter(torch.ones(1))
    elif case == "new_group":
        opt.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))]})
    clock.advance(1)
    probe.end(0)
    counter = load(tmp_path)["optimizer_counters"]["world_model"]
    assert counter["uniform_delta"] == delta
    assert counter["delta_status"] == status
    assert counter["source"]["uniform_step"] is None


def test_skipped_indices_stay_partial_and_repeated_phase_calls_are_counted(
        tmp_path, probes, clock):
    probe = CommandPhaseProbe(tmp_path, first_iteration=10, updates=4, device="cpu",
                              optimizers={})
    for iteration in (11, 13):
        probe.begin(iteration)
        for duration in (1, 3):
            probe.start_phase("wm")
            clock.advance(duration)
            probe.end_phase("wm")
        probe.end(iteration)
    report = load(tmp_path)
    assert report["state"] == "partial"
    assert (report["first_iteration"], report["last_iteration"]) == (11, 13)
    assert report["phase_statistics"]["wm"]["calls"] == 4
    assert report["phase_statistics"]["wm"]["host_ms"]["mean"] == 2000


def test_partial_finish_between_rows(tmp_path, probes, clock):
    probe = CommandPhaseProbe(tmp_path, first_iteration=0, updates=3, device="cpu",
                              optimizers={})
    probe.begin(0)
    clock.advance(1)
    probe.end(0)
    probe.finish()
    report = load(tmp_path)
    assert report["state"] == "partial"
    assert report["partial_iteration"] is None
    assert report["optimizer_counter_scope"]["includes_partial_iteration"] is False


@pytest.mark.parametrize("updates", [0, -1, 1001, True, 2.0, "2", None])
def test_invalid_update_budget(tmp_path, probes, updates):
    with pytest.raises(ValueError, match="updates"):
        CommandPhaseProbe(tmp_path, first_iteration=0, updates=updates, device="cpu",
                          optimizers={})
    assert probes == []


@pytest.mark.parametrize("first", [-1, True, 2.5, "0", None])
def test_invalid_first_iteration(tmp_path, probes, first):
    with pytest.raises(ValueError, match="first_iteration"):
        CommandPhaseProbe(tmp_path, first_iteration=first, updates=1, device="cpu",
                          optimizers={})


@pytest.mark.parametrize("options,error", [
    ({"device": "meta"}, ValueError), ({"device": 0}, ValueError),
    ({"device": "garbage"}, ValueError),
    ({"optimizers": []}, ValueError), ({"optimizers": {"unknown": None}}, ValueError),
    ({"optimizers": {"actor": None}}, TypeError),
    ({"metadata": []}, TypeError), ({"metadata": {"nonfinite": float("inf")}}, ValueError),
    ({"metadata": {"nested": [float("nan")]}}, ValueError),
    ({"metadata": {"unserializable": object()}}, TypeError),
])
def test_invalid_inputs(tmp_path, probes, options, error):
    arguments = {"first_iteration": 0, "updates": 1, "device": "cpu", "optimizers": {}}
    arguments.update(options)
    with pytest.raises(error):
        CommandPhaseProbe(tmp_path, **arguments)
    assert probes == [] and not list(tmp_path.iterdir())


def test_iteration_and_phase_lifecycle_validation(tmp_path, probes, clock):
    probe = CommandPhaseProbe(tmp_path, first_iteration=1, updates=3, device="cpu",
                              optimizers={})
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError):
            probe.begin(invalid)
    probe.begin(1)
    with pytest.raises(RuntimeError, match="active iteration"):
        probe.begin(2)
    with pytest.raises(ValueError, match="match begin"):
        probe.end(2)
    for invalid in ("", " ", "whole", None, 3):
        with pytest.raises(ValueError):
            probe.start_phase(invalid)
    with pytest.raises(ValueError, match="not active"):
        probe.end_phase("unknown")
    probe.start_phase("wm")
    with pytest.raises(ValueError, match="already active"):
        probe.start_phase("wm")
    with pytest.raises(RuntimeError, match="close named phases"):
        probe.end(1)
    probe.end_phase("wm")
    clock.advance(1)
    probe.end(1)
    for index in (0, 1):
        with pytest.raises(ValueError, match="increase"):
            probe.begin(index)
    probe.finish()


def test_maximum_window_is_bounded_at_1000_rows(tmp_path, probes, clock):
    probe = CommandPhaseProbe(tmp_path, first_iteration=7, updates=1000, device="cpu",
                              optimizers={})
    for iteration in range(7, 1007):
        probe.begin(iteration)
        clock.advance(.01)
        probe.end(iteration)
    assert probe.finished
    report = load(tmp_path)
    assert report["state"] == "full" and len(report["rows"]) == 1000
    assert report["last_iteration"] == report["requested_last_iteration"] == 1006
    probe.begin(1007)
    probe.end(1007)
    assert load(tmp_path) == report


def test_nonfinite_timing_is_null_with_diagnostic(tmp_path, probes, clock, cuda, monkeypatch):
    probe = CommandPhaseProbe(tmp_path, first_iteration=0, updates=1, device="cuda:3",
                              optimizers={})
    probe.begin(0)
    monkeypatch.setattr(cuda.events[0], "elapsed_time", lambda event: float("nan"))
    clock.now = float("inf")
    probe.end(0)
    report = load(tmp_path)
    assert report["state"] == "full" and not report["timing_valid"]
    assert report["wall_seconds"] is None
    assert phase(report["rows"][0], "whole")["host_ms"] is None
    assert phase(report["rows"][0], "whole")["cuda_ms"] is None
    assert len(report["timing_issues"]) == 3


def test_atomic_publication_and_retry_do_not_repeat_boundary_reads(
        tmp_path, probes, clock, cuda, monkeypatch):
    output = tmp_path / "new"
    output.mkdir()
    final = output / "command_phase_profile.json"
    final.write_text('{"previous": true}', encoding="utf-8")
    original_replace = Path.replace
    failures = [True]

    def replace(source, target):
        assert source.parent == output and Path(target) == final
        assert json.loads(source.read_text(encoding="utf-8"))["completed_updates"] == 1
        assert json.loads(final.read_text(encoding="utf-8")) == {"previous": True}
        if failures:
            failures.pop()
            raise OSError("simulated atomic replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    probe = CommandPhaseProbe(output, first_iteration=0, updates=1, device="cuda:3",
                              optimizers={})
    probe.begin(0)
    clock.advance(1)
    with pytest.raises(OSError, match="replace failure"):
        probe.end(0)
    assert not probe.finished and not probe.active
    assert list(output.iterdir()) == [final]
    assert len([t for t in cuda.trace if t[0] == "sync"]) == 2
    probe.finish()
    assert probe.finished
    assert len([t for t in cuda.trace if t[0] == "sync"]) == 2
    assert load(output)["state"] == "full"
    assert list(output.iterdir()) == [final]


def test_normal_process_exit_flushes_incomplete_cpu_iteration(tmp_path):
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import torch
from warp_runtime.command_phase_probe import CommandPhaseProbe
p = torch.nn.Parameter(torch.ones(1))
opt = torch.optim.Adam([p], lr=.001)
p.grad = torch.ones_like(p)
opt.step()
probe = CommandPhaseProbe(sys.argv[2], first_iteration=40, updates=3,
                          device='cpu', optimizers={'actor': opt})
probe.begin(40)
probe.start_phase('actor')
opt.step()
raise SystemExit(0)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(SCRIPTS), str(tmp_path)],
        capture_output=True, text=True, timeout=45, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    report = load(tmp_path)
    assert report["state"] == "partial" and report["completed_updates"] == 0
    assert report["optimizer_counters"]["actor"]["uniform_delta"] == 1
    assert not phase(report["partial_iteration"], "actor")["completed"]


@pytest.mark.skipif(os.environ.get("COMMAND_PHASE_PROBE_CUDA_TEST") != "1",
                    reason="explicit CUDA opt-in required")
def test_optional_cuda_nondefault_current_stream(tmp_path, probes):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda:0")
    p = torch.nn.Parameter(torch.ones(32, device=device))
    opt = torch.optim.Adam([p], lr=.001)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        p.grad = torch.ones_like(p)
        opt.step()
        probe = CommandPhaseProbe(tmp_path, first_iteration=8, updates=1,
                                  device=device, optimizers={"actor": opt})
        probe.begin(8)
        probe.start_phase("actor")
        p.grad = torch.ones_like(p)
        opt.step()
        probe.end_phase("actor")
        probe.end(8)
    report = load(tmp_path)
    assert report["state"] == "full" and report["timing_valid"]
    assert report["optimizer_counters"]["actor"]["uniform_delta"] == 1
    assert phase(report["rows"][0], "actor")["cuda_ms"] >= 0

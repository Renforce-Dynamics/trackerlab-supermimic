"""Prepare one slow forward G1 stealth AMP clip and a disjoint Aiming1 test."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np

from bootstrap import configure_paths

REPO = configure_paths()
from prepare_gait_references import digest
from prepare_lafan_stealth_windows import (
    _require_ignored_output,
    _verify_base_pool,
    validate_windows,
)
from prepare_style_campaign import (
    _build_pool,
    _json_bytes,
    _write_styles,
    load_style,
    select_bounded,
    verify_pool,
)
from prepare_style_references import (
    OFFICIAL_ARCHIVE_SHA256,
    OFFICIAL_REVISION,
    ORIGINAL_AIMING_SHA256,
    prepare_style_references,
)
from retarget_lafan_aiming import convert
from trackerTask.supermimic.terrain import ROBOT_HEIGHTS

SOURCE_ROOT = REPO / "data/raw/lafan1-original"
BASE_POOL = REPO / "data/aiming-g1-v2"
OUTPUT = REPO / "data/style-research-20261104/g1-stealth-single-v1"
TRAIN_WINDOW = ("aiming2_subject3.bvh", 340, 460, "forward")
TRAIN_ORIGINAL = TRAIN_WINDOW[0]
TEST_ORIGINAL = "aiming1_subject1.bvh"
COMMAND_SPEED_MPS = 0.30
COMMAND_CLEARANCE_RATIO = 0.80
COMMAND_JUMP_HEIGHT_M = 0.0
FPS = 50


def validate_train_window(window=TRAIN_WINDOW):
    """Keep this experiment on the existing pinned forward-only source range."""
    pinned = validate_windows()
    if tuple(window) != TRAIN_WINDOW or TRAIN_WINDOW not in pinned:
        raise ValueError("Single-clip training range differs from the pinned forward Aiming2 window")
    return TRAIN_WINDOW


def validate_single_pair(clips):
    """Require exactly one original Aiming2 train clip and one Aiming1 test clip."""
    clips = list(clips)
    train = [clip for clip in clips if clip.split == "train"]
    test = [clip for clip in clips if clip.split == "test"]
    if len(clips) != 2 or len(train) != 1 or len(test) != 1:
        raise ValueError("Single stealth pool requires exactly one train clip and one test clip")
    train, test = train[0], test[0]
    train_origin = train.record.get("origin") or {}
    test_origin = test.record.get("origin") or {}
    if (train.group != "aiming2" or train_origin.get("source_file") != TRAIN_ORIGINAL
            or train_origin.get("source_sha256") != ORIGINAL_AIMING_SHA256[TRAIN_ORIGINAL]):
        raise ValueError("The sole training clip must retain the pinned Aiming2 source identity")
    if (test.group != "aiming1" or test_origin.get("source_file") != TEST_ORIGINAL
            or test_origin.get("source_sha256") != ORIGINAL_AIMING_SHA256[TEST_ORIGINAL]):
        raise ValueError("The held-out clip must retain the pinned Aiming1 source identity")
    if (train.record["source_sha256"] == test.record["source_sha256"]
            or train_origin["source_sha256"] == test_origin["source_sha256"]):
        raise ValueError("Training and held-out source identities must be disjoint")
    if any(clip.begin != 0 or clip.end != len(clip.values) for clip in (train, test)):
        raise ValueError("Training and held-out candidates must retain their complete prepared intervals")
    return train, test


def source_range_receipt(conversion_record, prepared_record):
    """Map prepared CSV frames back to the original BVH sample coordinates."""
    source_begin, source_end = conversion_record["source_frame_range"]
    output_frames = conversion_record["output_frames"]
    prepared_begin, prepared_end = prepared_record["source_window_frames"]
    source_fps = conversion_record["source_fps"]
    output_fps = conversion_record["output_fps"]
    if (type(source_begin) is not int or type(source_end) is not int
            or source_begin < 0 or source_end <= source_begin
            or type(output_frames) is not int or output_frames < 2
            or type(prepared_begin) is not int or type(prepared_end) is not int
            or prepared_begin < 0 or prepared_end > output_frames
            or prepared_end <= prepared_begin
            or source_fps <= 0 or output_fps <= 0):
        raise ValueError("Source frame mapping contains an invalid range or frame rate")
    scale = source_fps / output_fps
    return {
        "source_file": conversion_record["source_file"],
        "source_sha256": conversion_record["source_sha256"],
        "retarget_source_frame_range": [source_begin, source_end],
        "retargeted_csv_frame_range": [prepared_begin, prepared_end],
        "original_sample_frame_positions": [
            source_begin + prepared_begin * scale,
            source_begin + (prepared_end - 1) * scale,
        ],
        "coordinate_note": (
            "Zero-based original BVH sample positions; fractional positions come "
            "from 30-to-50 Hz retarget resampling."
        ),
    }


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_bytes(_json_bytes(value))


def _command_recommendation(train_clip):
    values = train_clip.selected
    velocity = np.gradient(values[:, :2], axis=0) * FPS
    speed = np.linalg.norm(velocity, axis=1)
    root_height = values[:, 2]
    robot_height = float(ROBOT_HEIGHTS["g1"])
    return {
        "mode": "constant_forward",
        "body_vx_m_s": COMMAND_SPEED_MPS,
        "body_vy_m_s": 0.0,
        "body_yaw_rate_rad_s": 0.0,
        "jump_height_m": COMMAND_JUMP_HEIGHT_M,
        "reference_horizontal_speed_m_s": {
            "mean": float(np.mean(speed)),
            "p05": float(np.quantile(speed, 0.05)),
            "p50": float(np.quantile(speed, 0.50)),
            "p95": float(np.quantile(speed, 0.95)),
        },
        "reference_root_height_m": {
            "p05": float(np.quantile(root_height, 0.05)),
            "p50": float(np.quantile(root_height, 0.50)),
            "p95": float(np.quantile(root_height, 0.95)),
        },
        "target_clearance_ratio": COMMAND_CLEARANCE_RATIO,
        "target_clearance_m": robot_height * COMMAND_CLEARANCE_RATIO,
        "robot_height_m": robot_height,
        "height_note": (
            "target_clearance_m is the command MDP ground-clearance target; "
            "reference_root_height_m is the retargeted root Z."
        ),
    }


def prepare(source_root=SOURCE_ROOT, base_pool=BASE_POOL, output=OUTPUT):
    source_root, base_pool, output = (
        Path(path).expanduser().resolve() for path in (source_root, base_pool, output)
    )
    validate_train_window()
    if output.exists():
        raise FileExistsError(f"Use a fresh single-stealth output directory: {output}")
    _require_ignored_output(output)
    if source_root == base_pool or output in source_root.parents or output in base_pool.parents:
        raise ValueError("Source, base pool, and output paths must remain separate")

    base = _verify_base_pool(base_pool, source_root)
    source_dir = output / "source"
    prepared_dir = output / "prepared"
    pool_stage = output / "pool-stage"
    source_dir.mkdir(parents=True)
    train_csv = source_dir / "aiming2_subject3__f00340-00460.csv"
    conversion_report = source_dir / "aiming2_subject3-retarget.json"
    converted = convert(
        [source_root / TRAIN_ORIGINAL],
        train_csv,
        task_group="aiming",
        start_frame=TRAIN_WINDOW[1],
        max_frames=TRAIN_WINDOW[2] - TRAIN_WINDOW[1],
        report=conversion_report,
    )
    train_origin = converted["files"][0]
    if (train_origin["source_sha256"] != ORIGINAL_AIMING_SHA256[TRAIN_ORIGINAL]
            or train_origin["source_frame_range"] != [340, 460]
            or train_origin["source_group"] != "aiming2"
            or train_origin["split"] != "train"):
        raise ValueError("Retarget receipt differs from the pinned Aiming2 training window")

    base_rows = {row["source_file"]: row for row in base["files"]}
    test_origin = copy.deepcopy(base_rows.get(TEST_ORIGINAL))
    if (test_origin is None or test_origin["split"] != "test"
            or test_origin["source_sha256"] != ORIGINAL_AIMING_SHA256[TEST_ORIGINAL]):
        raise ValueError("The audited Aiming1 test reference is absent or has different provenance")
    test_csv = base_pool / test_origin["file"]
    test_copy = source_dir / test_origin["file"]
    if digest(test_csv) != test_origin["sha256"]:
        raise ValueError("Audited Aiming1 reference hash differs from its origin manifest")
    test_copy.write_bytes(test_csv.read_bytes())
    if digest(test_copy) != test_origin["sha256"]:
        raise ValueError("Copied Aiming1 reference hash differs from its audited source")

    origin_manifest = {
        "schema": base["schema"],
        "dataset": base["dataset"],
        "revision": base["revision"],
        "archive_sha256": base["archive_sha256"],
        "license": base["license"],
        "task_group": "aiming",
        "source_fps": base["source_fps"],
        "output_fps": base["output_fps"],
        "files": [train_origin, test_origin],
        "single_clip_selection": {
            "training_source_file": TRAIN_ORIGINAL,
            "training_source_frame_range": [340, 460],
            "training_direction": "forward",
            "held_out_source_file": TEST_ORIGINAL,
            "held_out_source_sha256": test_origin["source_sha256"],
        },
    }
    origin_path = source_dir / "origin_manifest.json"
    _write_json(origin_path, origin_manifest)

    import torch

    torch.set_num_threads(1)
    prepared = prepare_style_references(
        source_dir,
        prepared_dir,
        (train_csv.name, test_copy.name),
        "stealth",
        robot="g1",
        source_fps=float(FPS),
        holdout_groups=("aiming1",),
        minimum_frames=100,
        origin_manifest=origin_path,
    )
    if (prepared.get("status") != "ready"
            or set(prepared.get("held_out_source_groups", ())) != {"aiming1"}):
        raise ValueError("Prepared references must retain the Aiming1 test take")

    candidates, source_receipt = load_style(prepared_dir, "stealth", clip_frames=1000)
    validate_single_pair(candidates)
    selected, selection_report = select_bounded(
        candidates, max_per_source=1, frame_budget=1000)
    train_clip, test_clip = validate_single_pair(selected)
    training_source_range = source_range_receipt(train_origin, train_clip.record)
    recommendation = _command_recommendation(train_clip)
    selection_report.update({
        "training_clips": 1,
        "held_out_clips": 1,
        "training_source_range": training_source_range,
        "command_recommendation": recommendation,
        "source_selection": source_receipt,
    })
    recipe = {
        "version": 1,
        "purpose": "single slow forward G1 stealth AMP reference",
        "robot": "g1",
        "dataset_revision": OFFICIAL_REVISION,
        "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "source_manifest_sha256": digest(prepared_dir / "manifest.json"),
        "origin_manifest_sha256": digest(origin_path),
        "train_source_file": TRAIN_ORIGINAL,
        "train_source_sha256": ORIGINAL_AIMING_SHA256[TRAIN_ORIGINAL],
        "train_source_range": training_source_range,
        "test_source_file": TEST_ORIGINAL,
        "test_source_sha256": ORIGINAL_AIMING_SHA256[TEST_ORIGINAL],
        "command_recommendation": recommendation,
        "selected": [
            {
                "source_sha256": clip.record["source_sha256"],
                "prepared_sha256": digest(clip.path),
                "prepared_range": [clip.begin, clip.end],
                "split": clip.split,
                "group": clip.group,
            }
            for clip in selected
        ],
        "implementation_sha256": digest(__file__),
    }
    pool, _ = _build_pool(
        pool_stage,
        "g1",
        "stealth",
        recipe,
        lambda directory: _write_styles(
            directory, selected, "g1", selection_report),
    )
    verified = verify_pool(pool, "g1", "stealth")
    train_config = _read_json(pool / "train.json")
    test_config = _read_json(pool / "test.json")
    if (len(train_config["files"]) != 1 or len(test_config["files"]) != 1
            or train_config["files"][0] == test_config["files"][0]
            or len([row for row in verified["files"] if row["split"] == "train"]) != 1
            or len([row for row in verified["files"] if row["split"] == "test"]) != 1):
        raise ValueError("Final AMP pool must contain one training file and one held-out file")

    print(json.dumps({
        "status": "ready",
        "train_config": str((pool / "train.json").resolve()),
        "test_config": str((pool / "test.json").resolve()),
        "manifest": str((pool / "manifest.json").resolve()),
        "train_clip": train_config["files"][0],
        "test_clip": test_config["files"][0],
        "train_source_sha256": ORIGINAL_AIMING_SHA256[TRAIN_ORIGINAL],
        "train_source_range": training_source_range,
        "test_source_sha256": ORIGINAL_AIMING_SHA256[TEST_ORIGINAL],
        "command_recommendation": recommendation,
    }, indent=2, sort_keys=True))
    return pool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--base-pool", type=Path, default=BASE_POOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    prepare(args.source_root, args.base_pool, args.output)


if __name__ == "__main__":
    main()

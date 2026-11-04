"""Extend audited LAFAN1 stealth references and publish humanoid-only pools."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from bootstrap import configure_paths

REPO = configure_paths()

from prepare_style_campaign import (  # noqa: E402
    _atomic_json,
    _build_pool,
    _campaign_lock,
    _require_ignored_output,
    _write_styles,
    digest,
    load_style,
    select_bounded,
    verify_pool,
)
from prepare_style_references import (  # noqa: E402
    OFFICIAL_ARCHIVE_SHA256,
    OFFICIAL_REVISION,
    ORIGINAL_AIMING_FILES,
    ORIGINAL_AIMING_SHA256,
    prepare_style_references,
)
from retarget_lafan_aiming import convert  # noqa: E402


BASE_ORIGIN_MANIFEST_SHA256 = (
    "697cecc8a9ad83c96fc711374b40910ba84a1c648c448b83ad4c30af2ab40e29"
)
HUMANOIDS = ("g1", "a3", "t1", "bumi", "pi_plus")
SOURCE_TOTAL_FRAMES = {
    "aiming2_subject3.bvh": 9172,
    "aiming2_subject5.bvh": 9172,
}

# Disjoint, four-second windows selected from source-space low-profile motion.
# Scaled G1 speed proxies are approximately 0.07-0.71 m/s across these ranges.
WINDOWS = (
    ("aiming2_subject3.bvh", 340, 460, "forward"),
    ("aiming2_subject3.bvh", 5030, 5150, "right"),
    ("aiming2_subject5.bvh", 3600, 3720, "backward"),
    ("aiming2_subject5.bvh", 6700, 6820, "left"),
)


def validate_windows(windows=WINDOWS):
    previous = {}
    directions = set()
    for row in windows:
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise ValueError("Each Aiming window needs a source, range and direction")
        name, begin, end, direction = row
        if (name not in SOURCE_TOTAL_FRAMES
                or type(begin) is not int or type(end) is not int
                or begin < 0 or end - begin < 100
                or end > SOURCE_TOTAL_FRAMES[name]
                or direction not in ("forward", "backward", "left", "right")):
            raise ValueError(f"Invalid Aiming2 window: {row!r}")
        if begin < previous.get(name, 0):
            raise ValueError("Aiming windows must be sorted and non-overlapping per source")
        previous[name] = end
        directions.add(direction)
    if directions != {"forward", "backward", "left", "right"}:
        raise ValueError("Selected windows must cover forward, backward and both side directions")
    return tuple(tuple(row) for row in windows)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_base_pool(base_pool, source_root):
    manifest_path = base_pool / "origin_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing audited base origin manifest: {manifest_path}")
    if digest(manifest_path) != BASE_ORIGIN_MANIFEST_SHA256:
        raise ValueError("Base Aiming origin manifest differs from the reviewed v2 receipt")
    manifest = _read_json(manifest_path)
    if (manifest.get("schema") != "lafan1-g1-origin-v1"
            or manifest.get("dataset") != "ubisoft/ubisoft-laforge-animation-dataset"
            or manifest.get("revision") != OFFICIAL_REVISION
            or manifest.get("archive_sha256") != OFFICIAL_ARCHIVE_SHA256
            or manifest.get("task_group") != "aiming"
            or manifest.get("source_fps") != 30.0
            or manifest.get("output_fps") != 50.0):
        raise ValueError("Base origin manifest schema, source pin or frame rates differ")

    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Base origin manifest needs converted Aiming files")
    seen = set()
    for row in rows:
        name = row.get("file")
        source_name = row.get("source_file")
        if (name in seen or source_name not in ORIGINAL_AIMING_SHA256
                or row.get("source_sha256") != ORIGINAL_AIMING_SHA256[source_name]
                or row.get("source_group") != "aiming" + source_name.split("aiming", 1)[1][0]
                or row.get("split") != ("test" if source_name.startswith("aiming1_") else "train")):
            raise ValueError(f"Invalid base Aiming provenance row: {name!r}")
        seen.add(name)
        csv_path = base_pool / name
        if not csv_path.is_file() or digest(csv_path) != row.get("sha256"):
            raise ValueError(f"Base Aiming CSV hash differs: {name}")

    for name in ORIGINAL_AIMING_FILES:
        source = source_root / name
        expected = ORIGINAL_AIMING_SHA256[name]
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f"Pinned original Aiming BVH hash check failed: {name}")
    return manifest


def _source_configs(directory, manifest):
    from prepare_gait_references import motion_config

    names = [row["file"] for row in manifest["files"]]
    for split in ("train", "test"):
        split_names = [row["file"] for row in manifest["files"] if row["split"] == split]
        if not split_names:
            raise ValueError(f"Extended Aiming source has an empty {split} split")
        _write_json(directory / f"{split}.json", motion_config("g1", split_names))
    _write_json(directory / "motion_config.json", motion_config("g1", names))


def extend_source_pool(source_root, base_pool, output):
    source_root, base_pool, output = (
        Path(path).expanduser().resolve() for path in (source_root, base_pool, output)
    )
    _require_ignored_output(output)
    if output.exists():
        base = _verify_base_pool(base_pool, source_root)
        extended = _read_json(output / "origin_manifest.json")
        expected_added = [
            {"source_file": name, "frames": [begin, end], "direction_screen": direction,
             "split": "train"}
            for name, begin, end, direction in validate_windows()
        ]
        if (extended.get("extension", {}).get("base_origin_manifest_sha256")
                != digest(base_pool / "origin_manifest.json")
                or extended.get("extension", {}).get("source_ranges") != expected_added):
            raise ValueError("Existing Aiming extension uses different source ranges")
        expected = {row["file"]: row["sha256"] for row in base["files"]}
        rows = extended.get("files", [])
        for name, begin, end, _ in validate_windows():
            filename = f"{Path(name).stem}__f{begin:05d}-{end:05d}.csv"
            selected = [row for row in rows if row.get("file") == filename]
            if len(selected) != 1:
                raise ValueError(f"Existing Aiming extension lacks the requested crop: {filename}")
            expected[filename] = selected[0]["sha256"]
        if {row.get("file") for row in rows} != set(expected):
            raise ValueError("Existing Aiming extension has unexpected file membership")
        for name, expected_hash in expected.items():
            path = output / name
            if not path.is_file() or digest(path) != expected_hash:
                raise ValueError(f"Existing Aiming extension hash differs: {name}")
        _source_configs(output, extended)
        return output, extended
    if (source_root == output or base_pool == output
            or source_root in output.parents or base_pool in output.parents
            or output in source_root.parents or output in base_pool.parents):
        raise ValueError("Raw BVHs, base pool and extension output must be disjoint")
    windows = validate_windows()
    base = _verify_base_pool(base_pool, source_root)
    output.mkdir(parents=True)

    rows = copy.deepcopy(base["files"])
    for row in rows:
        source = base_pool / row["file"]
        target = output / row["file"]
        target.write_bytes(source.read_bytes())
        if digest(target) != row["sha256"]:
            raise ValueError(f"Base Aiming copy hash differs: {row['file']}")

    for index, (source_name, begin, end, direction) in enumerate(windows):
        source_stem = source_name.removesuffix(".bvh")
        target_name = f"{source_stem}__f{begin:05d}-{end:05d}.csv"
        target = output / target_name
        report = output / f".retarget-window-{index:02d}.json"
        document = convert(
            [source_root / source_name],
            target,
            task_group="aiming",
            start_frame=begin,
            max_frames=end - begin,
            report=report,
        )
        row = document["files"][0]
        report.unlink()
        expected_source_hash = ORIGINAL_AIMING_SHA256[source_name]
        if (row.get("source_file") != source_name
                or row.get("source_sha256") != expected_source_hash
                or row.get("source_group") != "aiming2"
                or row.get("split") != "train"
                or row.get("source_frame_range") != [begin, end]
                or row.get("output_frames") != 199
                or digest(target) != row.get("sha256")):
            raise ValueError(f"Retargeted range receipt differs from request: {target_name}")
        rows.append(row)
        print(
            f"Retargeted {source_name}[{begin}:{end}) direction={direction} "
            f"frames={row['output_frames']} sha256={row['sha256']}",
            flush=True,
        )

    extended = copy.deepcopy(base)
    extended["files"] = rows
    extended["motion_config"] = copy.deepcopy(base["motion_config"])
    extended["motion_config"]["files"] = [row["file"] for row in rows]
    extended["extension"] = {
        "method": "Four source-range-pinned Aiming2 BVH crops; original frames retained in each receipt.",
        "source_root": str(source_root),
        "source_ranges": [
            {"source_file": name, "frames": [begin, end], "direction_screen": direction,
             "split": "train"}
            for name, begin, end, direction in windows
        ],
        "base_origin_manifest_sha256": digest(base_pool / "origin_manifest.json"),
        "retargeter": "retarget_lafan_aiming.py / morphology-scaled Mink IK",
    }
    _write_json(output / "origin_manifest.json", extended)
    _source_configs(output, extended)
    return output, extended


def _resolve_map_path(map_path, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (map_path.parent / path).resolve()


def verify_source_map(map_path):
    from supervise_style import ROBOTS

    map_path = Path(map_path).expanduser().resolve()
    mapping = _read_json(map_path)
    if mapping.get("version") != 1:
        raise ValueError("Base style map must declare version=1")
    if set(mapping.get("robots", {})) != set(ROBOTS):
        raise ValueError("Base style map robot set differs from the campaign")
    for robot, styles in mapping["robots"].items():
        for style, config_name in styles.items():
            if config_name is None:
                continue
            config_path = _resolve_map_path(map_path, config_name)
            if not config_path.is_file():
                raise FileNotFoundError(f"Base style config is missing: {config_path}")
            receipt = mapping["pools"][robot][style]
            declared_manifest = _resolve_map_path(map_path, receipt["manifest"])
            manifest_path = (declared_manifest if declared_manifest.is_file()
                             else config_path.parent / "manifest.json")
            manifest = verify_pool(config_path.parent, robot, style)
            if (not manifest_path.is_file()
                    or digest(manifest_path) != receipt.get("sha256")
                    or manifest_path != config_path.parent / "manifest.json"
                    or digest(config_path) != receipt["config_sha256"]["train"]["sha256"]
                    or digest(config_path.parent / "test.json")
                    != receipt["config_sha256"]["test"]["sha256"]
                    or manifest.get("robot") != robot or manifest.get("style") != style):
                raise ValueError(f"Base style-map receipt differs: {robot}/{style}")
            mapping["robots"][robot][style] = str(config_path)
            mapping["evaluations"][robot][style] = str(config_path.parent / "test.json")
            mapping["pools"][robot][style]["manifest"] = str(manifest_path)
    return map_path, mapping


def _pool_summary(manifest):
    files = manifest["files"]
    result = {}
    for split in ("train", "test"):
        rows = [row for row in files if row["split"] == split]
        result[split] = {
            "clips": len(rows),
            "frames": sum(row["frames"] for row in rows),
            "source_groups": sorted({row["source_group"] for row in rows}),
            "source_ranges": [
                {
                    "source_file": row["upstream_record"].get("source_file")
                    or row["upstream_record"].get("origin", {}).get("source_file"),
                    "frames": row["upstream_record"].get("source_frame_range")
                    or row["upstream_record"].get("origin", {}).get("source_frame_range"),
                    "output_frames": row["frames"],
                    "foot_ik_error_m_p95": (
                        row.get("retarget") or {}
                    ).get("foot_ik_error_m_p95"),
                }
                for row in rows
            ],
        }
    return result


def verify_published(output, source_map):
    output = Path(output).expanduser().resolve()
    source_map, original = verify_source_map(source_map)
    map_path = output / "style-map.json"
    result = _read_json(map_path)
    if (result.get("preparation", {}).get("source_map_sha256") != digest(source_map)
            or set(result.get("robots", {})) != set(original["robots"])):
        raise ValueError("Published style map source receipt or robot set differs")
    for robot in HUMANOIDS:
        config = Path(result["robots"][robot]["stealth"])
        manifest_path = Path(result["pools"][robot]["stealth"]["manifest"])
        manifest = verify_pool(config.parent, robot, "stealth")
        receipt = result["pools"][robot]["stealth"]
        if (digest(manifest_path) != receipt["sha256"]
                or digest(config) != receipt["config_sha256"]["train"]["sha256"]
                or digest(config.parent / "test.json") != receipt["config_sha256"]["test"]["sha256"]):
            raise ValueError(f"Published stealth pool receipt failed: {robot}")
        if (result["robots"][robot]["walk"] != original["robots"][robot]["walk"]
                or result["robots"][robot]["parkour"] != original["robots"][robot]["parkour"]):
            raise ValueError(f"Unrelated style pool changed: {robot}")
        print(json.dumps({
            "robot": robot, "pool": str(manifest_path),
            "manifest_sha256": digest(manifest_path), **_pool_summary(manifest),
        }, sort_keys=True), flush=True)
    print(json.dumps({"status": "ready", "style_map": str(map_path),
                      "style_map_sha256": digest(map_path)}, sort_keys=True), flush=True)
    return map_path


def prepare(source_root, base_pool, source_map, extension_output, prepared_output, output):
    source_root, base_pool, extension_output, prepared_output, output = (
        Path(path).expanduser().resolve()
        for path in (source_root, base_pool, extension_output, prepared_output, output)
    )
    source_map, mapping = verify_source_map(source_map)
    for path in (extension_output, prepared_output, output):
        _require_ignored_output(path)
    if output.exists():
        raise FileExistsError(f"Use a fresh v8 style-map output: {output}")
    if extension_output.exists() or prepared_output.exists():
        raise FileExistsError("Use fresh Aiming extension and prepared-reference directories")

    extension, _ = extend_source_pool(source_root, base_pool, extension_output)
    source_manifest_path = extension / "origin_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    files = tuple(row["file"] for row in source_manifest["files"])
    from prepare_style_references import prepare_style_references as prepare_g1_references

    prepared = prepare_g1_references(
        extension,
        prepared_output,
        files,
        "stealth",
        robot="g1",
        source_fps=50.0,
        holdout_groups=("aiming1",),
        origin_manifest=source_manifest_path,
        minimum_frames=100,
    )
    if (prepared.get("status") != "ready"
            or set(prepared.get("held_out_source_groups", ())) != {"aiming1"}
            or not prepared.get("style_training_files")
            or not prepared.get("style_evaluation_files")):
        raise ValueError("Prepared stealth pool must have Aiming2 train and Aiming1 held-out data")

    candidates, source_receipt = load_style(prepared_output, "stealth")
    selected, selection_report = select_bounded(candidates, max_per_source=2, frame_budget=4000)
    if (set(selection_report["training_families"]) != {"aiming"}
            or set(selection_report["held_out_groups"]) != {"aiming1"}
            or not any(clip.split == "train" and clip.group == "aiming2" for clip in selected)):
        raise ValueError("Selected stealth crops violate Aiming take split policy")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    _require_ignored_output(stage)
    created = {}
    try:
        with _campaign_lock(stage):
            for robot in HUMANOIDS:
                recipe = {
                    "version": 1,
                    "purpose": "slower low-profile LAFAN1 aiming locomotion",
                    "robot": robot,
                    "source_manifest_sha256": digest(prepared_output / "manifest.json"),
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
                pool, receipt = _build_pool(
                    stage,
                    robot,
                    "stealth",
                    recipe,
                    lambda directory, target=robot: _write_styles(
                        directory, selected, target, selection_report),
                )
                created[robot] = (pool.relative_to(stage), receipt)

            result = copy.deepcopy(mapping)
            for robot, (relative_pool, receipt) in created.items():
                final_pool = output / relative_pool
                result["robots"][robot]["stealth"] = str(final_pool / "train.json")
                result["evaluations"][robot]["stealth"] = str(final_pool / "test.json")
                result["pools"][robot]["stealth"] = {
                    "manifest": str(final_pool / "manifest.json"),
                    "sha256": digest(stage / relative_pool / "manifest.json"),
                    "config_sha256": receipt["configs"],
                }
            result["preparation"] = {
                "source_map": str(source_map),
                "source_map_sha256": digest(source_map),
                "source_origin_manifest": str(source_manifest_path),
                "source_origin_manifest_sha256": digest(source_manifest_path),
                "prepared_g1_manifest": str(prepared_output / "manifest.json"),
                "prepared_g1_manifest_sha256": digest(prepared_output / "manifest.json"),
                "scope": "Humanoid stealth pools extended with four source-pinned Aiming2 windows; Aiming1 remains held out.",
                "added_windows": [
                    {"source_file": name, "frames": [begin, end], "direction_screen": direction}
                    for name, begin, end, direction in validate_windows()
                ],
                "selection": selection_report,
                "source_selection": source_receipt,
            }
            _atomic_json(stage / "style-map.json", result)

        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        os.replace(stage, output)
    except Exception:
        # Preserve partial evidence for inspection; never alter the base pools.
        raise

    final_map = output / "style-map.json"
    published = _read_json(final_map)
    for robot in HUMANOIDS:
        config = Path(published["robots"][robot]["stealth"])
        manifest_path = Path(published["pools"][robot]["stealth"]["manifest"])
        manifest = verify_pool(config.parent, robot, "stealth")
        if (digest(manifest_path) != published["pools"][robot]["stealth"]["sha256"]
                or digest(config) != published["pools"][robot]["stealth"]["config_sha256"]["train"]["sha256"]):
            raise ValueError(f"Published stealth pool receipt failed: {robot}")
        if published["robots"][robot]["walk"] != mapping["robots"][robot]["walk"]:
            raise ValueError(f"Walk config changed unexpectedly: {robot}")
        if published["robots"][robot]["parkour"] != mapping["robots"][robot]["parkour"]:
            raise ValueError(f"Parkour config changed unexpectedly: {robot}")
        print(json.dumps({
            "robot": robot,
            "pool": str(manifest_path),
            "manifest_sha256": digest(manifest_path),
            **_pool_summary(manifest),
        }, sort_keys=True), flush=True)
    print(json.dumps({
        "status": "ready",
        "style_map": str(final_map),
        "style_map_sha256": digest(final_map),
        "source_extension": str(extension / "origin_manifest.json"),
        "prepared_g1_manifest": str(prepared_output / "manifest.json"),
        "train_selected_frames": selection_report["selected_frames"],
        "selection_report": selection_report,
    }, sort_keys=True), flush=True)
    return final_map


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=REPO / "data/raw/lafan1-original")
    parser.add_argument("--base-pool", type=Path,
                        default=REPO / "data/aiming-g1-v2")
    parser.add_argument("--source-map", type=Path,
                        default=REPO / "data/style-references-v7/style-map.json")
    parser.add_argument("--extension-output", type=Path,
                        default=REPO / "data/aiming-g1-stealth-v3")
    parser.add_argument("--prepared-output", type=Path,
                        default=REPO / "data/style-research-20261104/g1-stealth-v4")
    parser.add_argument("--output", type=Path,
                        default=REPO / "data/style-references-v9-stealth")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_existing:
        verify_published(args.output, args.source_map)
        return
    try:
        import torch
    except ImportError as error:
        parser.error(f"PyTorch is required for CPU retargeting: {error}")
    torch.set_num_threads(1)
    print(json.dumps({
        "python": os.sys.executable,
        "torch_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "windows": validate_windows(),
    }, sort_keys=True), flush=True)
    prepare(
        args.source_root,
        args.base_pool,
        args.source_map,
        args.extension_output,
        args.prepared_output,
        args.output,
    )


if __name__ == "__main__":
    main()

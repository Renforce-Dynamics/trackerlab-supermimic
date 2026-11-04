"""Publish immutable command-style pools with sustained locomotion windows."""

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from bootstrap import configure_paths
REPO = configure_paths()
from prepare_style_campaign import (
    _atomic_json, _build_pool, _campaign_lock, _require_ignored_output, digest, verify_pool,
)

ANOMALOUS_T1_BUMI_WALK = "clip_002__walk1__00091-00491.csv"
ANOMALOUS_SOURCE_SHA256 = "68c7887438e94f1f3d9ebacd3a8ce59251a22c7b05129a0f5d7cacaedc48c28d"


def is_anomalous_walk_source(robot, row):
    if robot not in ("t1", "bumi"):
        return False
    candidates = [row.get("file"), row.get("source_file")]
    for key in ("source", "provenance", "upstream_record"):
        value = row.get(key)
        if isinstance(value, dict):
            candidates.extend((value.get("file"), value.get("source_file")))
    return any(isinstance(value, str) and Path(value).name == ANOMALOUS_T1_BUMI_WALK
               for value in candidates)


def is_anomalous_parkour_source(robot, row):
    source = row.get("source")
    return (robot in ("t1", "bumi") and isinstance(source, dict)
            and source.get("source_sha256") == ANOMALOUS_SOURCE_SHA256
            and source.get("prepared_frame_range") == [91, 491]
            and source.get("source_file") == "walk1_subject1.csv")


def moving_segments(values, minimum_speed=.15, history=10):
    rotation = Rotation.from_quat(values[:, 3:7]).as_matrix()
    velocity = np.gradient(values[:, :3], .02, axis=0)
    local = (rotation.transpose(0, 2, 1) @ velocity[..., None])[..., 0]
    good = ((local[:, 0] > minimum_speed) & (abs(velocity[:, 2]) < .3)
            & (rotation[:, 2, 2] > .8))
    edges = np.flatnonzero(np.diff(np.r_[False, good, False].astype(np.int8)))
    return [(int(begin), int(end)) for begin, end in edges.reshape(-1, 2)
            if end - begin >= history]


def prepare(source_map, output):
    source_map, output = source_map.resolve(), output.resolve()
    _require_ignored_output(output)
    if output == source_map.parent or source_map.parent.is_relative_to(output):
        raise ValueError("Use a distinct command style output directory")
    mapping = json.loads(source_map.read_text(encoding="utf-8"))
    for section in ("robots", "evaluations"):
        for robot, styles in mapping[section].items():
            for style, name in styles.items():
                if name:
                    styles[style] = str((source_map.parent / name).resolve())
    for styles in mapping["pools"].values():
        for receipt in styles.values():
            receipt["manifest"] = str((source_map.parent / receipt["manifest"]).resolve())
    output.mkdir(parents=True, exist_ok=True)

    with _campaign_lock(output):
        for robot in mapping["robots"]:
            style = "parkour" if robot == "go2" else "walk"
            source = Path(mapping["robots"][robot][style]).parent
            manifest = verify_pool(source, robot, style)
            selected, excluded = [], []
            for row in manifest["files"]:
                if row.get("clip_type") == "jump":
                    continue
                if is_anomalous_walk_source(robot, row):
                    excluded.append(row["file"])
                    continue
                values = np.loadtxt(source / row["file"], delimiter=",", ndmin=2)
                for begin, end in moving_segments(values):
                    selected.append((row, begin, end, values[begin:end].copy()))
            if {row["split"] for row, *_ in selected} != {"train", "test"}:
                raise ValueError(f"{robot}: sustained-motion pool needs both disjoint splits")

            def write_pool(directory):
                records = []
                for index, (row, begin, end, values) in enumerate(selected):
                    record = copy.deepcopy(row)
                    path = source / row["file"]
                    if digest(path) != row["sha256"]:
                        raise ValueError("Style source changed during publication")
                    name = f"moving_{index:03d}__{path.stem}__{begin}-{end}.csv"
                    np.savetxt(directory / name, values, delimiter=",", fmt="%.17g")
                    record.update(file=name, sha256=digest(directory / name), frames=len(values),
                                  duration_seconds=(len(values) - 1) * .02,
                                  source_pool=str(source), source_csv_sha256=row["sha256"],
                                  source_pool_range=[begin, end],
                                  source_pool_manifest=digest(source / "manifest.json"))
                    records.append(record)
                return records, {"scope_source": "Sustained forward locomotion; no static windows or jumps",
                                 "upstream_pool": str(source), "excluded_anomalous_clips": excluded}

            pool, receipt = _build_pool(output, robot, "walk", {
                "version": 3, "purpose": "sustained command locomotion",
                "source_manifest_sha256": digest(source / "manifest.json"),
                "selected": [[row["sha256"], begin, end] for row, begin, end, _ in selected],
                "implementation_sha256": digest(__file__),
            }, write_pool)
            mapping["robots"][robot]["walk"] = str(pool / "train.json")
            mapping["evaluations"][robot]["walk"] = str(pool / "test.json")
            mapping["pools"][robot]["walk"] = {
                "manifest": str(pool / "manifest.json"), "sha256": digest(pool / "manifest.json"),
                "config_sha256": receipt["configs"],
            }
        for robot in mapping["robots"]:
            if robot == "go2":
                continue
            source = Path(mapping["robots"][robot]["parkour"]).parent
            manifest = verify_pool(source, robot, "parkour")
            retained, excluded = [], []
            for row in manifest["files"]:
                if is_anomalous_parkour_source(robot, row):
                    excluded.append({"file": row["file"], "sha256": row["sha256"]})
                else:
                    retained.append(row)

            def write_parkour(directory):
                records = []
                for row in retained:
                    original = source / row["file"]
                    if digest(original) != row["sha256"]:
                        raise ValueError("Parkour source changed during publication")
                    shutil.copyfile(original, directory / row["file"])
                    records.append(copy.deepcopy(row))
                return records, {
                    "upstream_pool": str(source),
                    "excluded_anomalous_sources": excluded,
                }

            pool, receipt = _build_pool(output, robot, "parkour", {
                "version": 2, "purpose": "parkour style with audited source anomaly removed",
                "source_manifest_sha256": digest(source / "manifest.json"),
                "retained_files": [[row["file"], row["sha256"], row["split"]]
                                   for row in retained],
                "excluded_files": excluded,
                "implementation_sha256": digest(__file__),
            }, write_parkour)
            mapping["robots"][robot]["parkour"] = str(pool / "train.json")
            mapping["evaluations"][robot]["parkour"] = str(pool / "test.json")
            mapping["pools"][robot]["parkour"] = {
                "manifest": str(pool / "manifest.json"), "sha256": digest(pool / "manifest.json"),
                "config_sha256": receipt["configs"],
            }
        mapping["preparation"] = {
            "source_map": str(source_map), "source_map_sha256": digest(source_map),
            "scope": "Moving walk windows; exact T1/Bumi anomaly excluded from walk and parkour pools; Go2 native pool retained",
        }
        _atomic_json(output / "style-map.json", mapping)
    return output / "style-map.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source_map, args.output))


if __name__ == "__main__":
    main()

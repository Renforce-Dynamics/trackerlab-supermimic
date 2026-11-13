"""Prepare an auditable chronological G1 walk pivot donor for stealth IK."""

import argparse
import json
from pathlib import Path

from prepare_stealth_turning_pool import (
    WALK_PIVOT_SHA256, WALK_REVISION, _load_origin, walk_pivot_values,
)
from prepare_style_campaign import _json_bytes, _require_ignored_output, digest
from trackerTask.supermimic.robots import get_robot


def prepare(original, begin, end, output):
    original, output = Path(original).resolve(), Path(output).resolve()
    _require_ignored_output(output)
    if output.is_relative_to(original.parent) or original.is_relative_to(output):
        raise ValueError("Pivot source output must remain outside the original input tree")
    if output.exists():
        raise FileExistsError(output)
    if (original.name not in WALK_PIVOT_SHA256
            or digest(original) != WALK_PIVOT_SHA256[original.name]):
        raise ValueError("Expected the pinned walk3subject4 recording")
    values, floor, mirrored, total = walk_pivot_values(original, begin, end)
    output.mkdir(parents=True)
    csv = output / f"{original.stem}__f{begin:05d}-{end:05d}.csv"
    import numpy as np
    np.savetxt(csv, values, delimiter=",", fmt="%.17g")
    record = {
        "file": csv.name, "sha256": digest(csv),
        "source_file": original.name, "source_sha256": digest(original),
        "source_group": "walk3", "split": "train", "source_fps": 30., "output_fps": 50.,
        "source_frame_range": [begin, end], "source_total_frames": total,
        "output_frames": len(values), "target_model_sha256": digest(get_robot("g1").mjcf_path),
        "joint_names": list(get_robot("g1").joint_names), "reflected_to_left": mirrored,
        "constant_floor_shift_m": floor["root_z_shift_m"],
        "floor_alignment": floor, "implementation_sha256": digest(__file__),
    }
    origin = output / "origin.json"
    origin.write_bytes(_json_bytes({
        "schema": "g1-stealth-walk-pivot-origin-v1", "revision": WALK_REVISION,
        "dataset": "lvhaidong/LAFAN1_Retargeting_Dataset", "files": [record],
        "scope": "Recorded walk pivot, unchanged timestamps; stealth pose applied separately",
    }))
    _load_origin(origin, csv, original)
    return {"csv": str(csv), "origin": str(origin), "record": record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--begin", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.original, args.begin, args.end, args.output), indent=2))


if __name__ == "__main__":
    main()

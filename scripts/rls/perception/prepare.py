"""Prepare pinned robot geometry without downloading imitation motion datasets."""

import argparse
from pathlib import Path

from common import REPO, save
from prepare_resources import LOCK, prepare_assets
from prepare_humanoids import checkout, prepare_model
from trackerTask.supermimic.robots import TRACKING_TASKS


def prepare(robot, offline=False):
    assets = REPO / "data/assetslib"
    if robot in ("g1", "go2"):
        return prepare_assets(assets, offline, robot)
    cfg = dict(LOCK[robot])
    cfg["sparse_paths"] = [
        p for p in cfg["sparse_paths"]
        if not any(part in p.lower() for part in ("motion", "dataset", "a3_data", "scripts"))
    ]
    source = REPO / "data/raw" / robot
    checkout(source, cfg, offline)
    target = assets / cfg["vendor"] / robot
    prepare_model(source, target, cfg, robot)
    return {"robot": robot, "asset": str(target), "revision": cfg["revision"],
            "motion_data_required": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=(*TRACKING_TASKS, "all"), default="all")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    robots = TRACKING_TASKS if args.robot == "all" else (args.robot,)
    for robot in robots:
        receipt = prepare(robot, args.offline)
        save(REPO / "data/perception_resources" / f"{robot}.json", receipt)
        print(f"Prepared {robot}: revision={receipt['revision']}", flush=True)


if __name__ == "__main__":
    main()

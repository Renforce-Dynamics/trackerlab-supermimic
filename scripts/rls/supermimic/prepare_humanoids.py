"""Pinned A3/T1 assets and reference motions for the shared TrackerLab pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.robots import get_robot

LOCK = json.loads(Path(__file__).with_name("resources.lock.json").read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def checkout(source, cfg, offline):
    if not source.exists():
        if offline:
            raise FileNotFoundError(source)
        source.mkdir(parents=True)
        git(source, "init", "--quiet")
        git(source, "remote", "add", "origin", cfg["repository"])
        git(source, "sparse-checkout", "init", "--cone")
        git(source, "sparse-checkout", "set", *cfg["sparse_paths"])
        git(source, "fetch", "--depth", "1", "origin", cfg["revision"])
        git(source, "checkout", "--quiet", "--detach", cfg["revision"])
    if git(source, "rev-parse", "HEAD") != cfg["revision"]:
        raise ValueError("Resource checkout revision differs from resources.lock.json")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Resource checkout contains tracked modifications")


def convert_a3(path, joints):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if columns is None or len(columns) != len(set(columns)):
            raise ValueError("A3 CSV needs unique named columns")
        rows = list(reader)
    fields = ["Frame", *(f"root_translate{axis}" for axis in "XYZ"),
              *(f"root_rotate{axis}" for axis in "XYZ"), *joints,
              "head_yaw_joint", "head_pitch_joint"]
    if not set(fields).issubset(columns):
        raise ValueError("A3 CSV missing required root/joint columns")
    values = np.array([[float(row[key]) for key in fields] for row in rows], dtype=np.float64)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("A3 CSV must contain finite motion frames")
    if not np.allclose(np.diff(values[:, 0]), 1):
        raise ValueError("A3 frame indices must be consecutive")
    if not np.allclose(values[:, -2:], 0, atol=1e-5):
        raise ValueError("This A3 preset fixes head joints; source head angles must be zero")
    quaternion = Rotation.from_euler("xyz", values[:, 4:7], degrees=True).as_quat()
    return np.column_stack((values[:, 1:4] * .01, quaternion,
                            np.deg2rad(values[:, 7:7 + len(joints)])))


def validate_motion(values, joints):
    if values.ndim != 2 or values.shape[1] != 7 + len(joints) or len(values) < 3:
        raise ValueError("Motion shape differs from the robot contract")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite motion values")
    if not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, atol=2e-4):
        raise ValueError("Root quaternion must have unit norm")
    if np.min(values[:, 2]) <= 0 or np.max(values[:, 2]) > 3:
        raise ValueError("Root height is outside the expected metre range")


def prepare_model(source, target, cfg, robot):
    tree = ET.parse(source / cfg["model"])
    root = tree.getroot()
    compiler = root.find("compiler")
    meshdir = (source / cfg["model"]).parent / compiler.get("meshdir", ".")
    compiler.set("meshdir", "../meshes")
    compiler.set("angle", "radian")
    for section in ("actuator", "sensor", "keyframe", "equality", "contact", "visual", "statistic"):
        for node in root.findall(section):
            root.remove(node)
    world = root.find("worldbody")
    for node in list(world):
        if node.tag != "body":
            world.remove(node)
    for body in world.iter("body"):
        for leaf in list(body.findall("body")):
            if len(leaf) == 0:
                body.remove(leaf)
        for joint in list(body.findall("joint")):
            if joint.get("name") in cfg["fixed_joints"]:
                body.remove(joint)
    asset = root.find("asset")
    (target / "meshes").mkdir(parents=True, exist_ok=True)
    for node in list(asset):
        if node.tag == "texture" or node.get("name") in ("groundplane", "matplane"):
            asset.remove(node)
        elif node.tag == "mesh":
            original = (meshdir / node.get("file")).resolve()
            destination = target / "meshes" / original.name
            if not original.is_file() or original.read_bytes()[:80].startswith(b"version https://git-lfs"):
                raise ValueError(f"Missing mesh: {original}")
            shutil.copy2(original, destination)
            node.set("file", original.name)
    path = target / "mjcf" / f"{robot}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    license_dir = target / "licenses"
    license_dir.mkdir(exist_ok=True)
    for p in source.glob("*"):
        if p.is_file() and p.name.upper().startswith(("LICENSE", "NOTICE")):
            shutil.copy2(p, license_dir / p.name)
    if (source / "legal").is_dir():
        shutil.copytree(source / "legal", license_dir / "legal", dirs_exist_ok=True)
    save(target / "manifest.json", {
        "source": cfg["repository"], "revision": cfg["revision"],
        "source_model": cfg["model"], "source_model_sha256": digest(source / cfg["model"]),
        "fixed_joints": cfg["fixed_joints"],
        "modifications": ["remove scene ground/lights", "remove external control/sensors",
                          "relocate meshes", "fix declared zero head joints",
                          "remove empty fixed marker bodies with no mass/geometry"],
        "sha256": {p.relative_to(target).as_posix(): digest(p) for p in target.rglob("*")
                   if p.is_file() and p.name != "manifest.json" and "usd" not in p.parts},
    })


def verify_hashes(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Hash mismatch: {root / relative}")
    return manifest


def prepare(args):
    cfg = LOCK[args.task]
    output = (args.motion_dir or REPO / "data/retargeted" / cfg["data_name"]).resolve()
    motion_config = (args.motion_config or REPO / "data/configs" / (cfg["data_name"] + ".json")).resolve()
    asset = args.asset_root.resolve() / cfg["vendor"] / args.task
    if args.verify_only:
        verify_hashes(asset)
        result = verify_hashes(output)
        config = json.loads(motion_config.read_text(encoding="utf-8"))
        if (motion_config.parent / config["root"]).resolve() != output:
            raise ValueError("Motion configuration points to another dataset")
        if config["joint_names"] != list(get_robot(args.task).joint_names):
            raise ValueError("Motion configuration joint order differs")
        if (config["version"] != 1 or config["quaternion_order"] != "xyzw"
                or config["source_fps"] != cfg["source_fps"]
                or config["files"] != list(result["clips"])
                or result["revision"] != cfg["revision"]):
            raise ValueError("Motion configuration or provenance differs from the locked source")
        return result
    source = (args.source_dir or REPO / "data/raw" / args.task).resolve()
    checkout(source, cfg, args.offline)
    spec = get_robot(args.task)
    prepare_model(source, asset, cfg, args.task)
    paths = sorted((source / cfg["motions"]).glob("*.csv"))
    if not paths:
        raise ValueError("Source dataset contains no CSV motions")
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for path in paths:
        values = (convert_a3(path, spec.joint_names) if args.task == "a3"
                  else np.loadtxt(path, delimiter=",", ndmin=2))
        validate_motion(values, spec.joint_names)
        np.savetxt(output / path.name, values, delimiter=",", fmt="%.9g")
        records[path.name] = {"frames": len(values), "duration_s": (len(values) - 1) / cfg["source_fps"],
                              "source_sha256": digest(path), "sha256": digest(output / path.name)}
    manifest = {
        "source": cfg["repository"], "dataset": cfg["repository"].removesuffix(".git"),
        "revision": cfg["revision"], "robot": args.task, "source_fps": cfg["source_fps"],
        "control_fps": 50, "motion_count": len(paths),
        "total_frames": sum(r["frames"] for r in records.values()),
        "retargeting": "Robot-space motions supplied by upstream; no local human retarget optimization",
        "conversion": ("centimetres/degrees -> metres/radians; scipy xyz Euler -> xyzw; heads fixed"
                       if args.task == "a3" else "preserve vendor metres/radians/xyzw and named joint order"),
        "joint_names": spec.joint_names, "clips": records,
        "sha256": {name: info["sha256"] for name, info in records.items()},
    }
    save(output / "manifest.json", manifest)
    save(motion_config, {
        "version": 1, "root": os.path.relpath(output, motion_config.parent).replace("\\", "/"),
        "source_fps": cfg["source_fps"], "quaternion_order": "xyzw",
        "joint_names": list(spec.joint_names), "files": list(records),
    })
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("a3", "t1"), required=True)
    parser.add_argument("--source_dir", type=Path)
    parser.add_argument("--asset_root", type=Path, default=Path(
        os.environ.get("SUPERMIMIC_ASSETLIB_ROOT", REPO / "data/assetslib")))
    parser.add_argument("--motion_dir", type=Path)
    parser.add_argument("--motion_config", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    result = prepare(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("robot", "motion_count", "total_frames", "source_fps")}))


if __name__ == "__main__":
    main()

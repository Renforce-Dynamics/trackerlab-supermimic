"""Derive an immutable single-clip G1 stealth pool with explicit static arms.

Only the training CSV's 14 shoulder/elbow/wrist fields are replaced. Other
fields and line endings retain their original bytes; the Aiming1 holdout is
copied verbatim. The reference may be a named joint mapping or a document with
``reference_q`` and additional entry/evidence metadata. The complete reference
document is preserved, but this tool only applies its static pose to AMP data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np

from bootstrap import configure_paths
configure_paths()
from prepare_style_campaign import (
    _build_pool, _json_bytes, _require_ignored_output, verify_pool,
)
from trackerTask.supermimic.robots import get_robot


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_reference(payload, joint_names):
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("Reference must be a named joint mapping or a reference_q document")
    reference = document.get("reference_q", document)
    arms = [name for name in joint_names
            if any(part in name.split("_") for part in ("shoulder", "elbow", "wrist"))]
    if len(arms) != 14 or not isinstance(reference, dict) or set(reference) != set(arms):
        raise ValueError("reference_q must map exactly the 14 named G1 arm joints")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or abs(value) > float(np.finfo(np.float32).max) for value in reference.values()):
        raise ValueError("Reference joint angles must be finite float32-compatible numbers")
    return {name: float(reference[name]) for name in arms}


def _replace_arm_fields(payload, reference, joint_names, frames):
    replacements = {7 + joint_names.index(name): format(value, ".17g").encode("ascii")
                    for name, value in reference.items()}
    width = 7 + len(joint_names)
    output, count = [], 0
    for line in payload.splitlines(keepends=True):
        body = line.rstrip(b"\r\n")
        ending = line[len(body):]
        if not body.strip() or body.lstrip().startswith(b"#"):
            output.append(line)
            continue
        fields = next(csv.reader([body.decode("ascii")], strict=True))
        if len(fields) != width or b'"' in body:
            raise ValueError("Source CSV must contain plain numeric fields in the declared joint order")
        if not all(math.isfinite(float(value)) for value in fields):
            raise ValueError("Source CSV contains nonfinite coordinates")
        # Split the validated numeric row to retain untouched field bytes exactly.
        tokens = body.split(b",")
        for column, value in replacements.items():
            tokens[column] = value
        output.append(b",".join(tokens) + ending)
        count += 1
    if count != frames:
        raise ValueError("Source CSV frame count differs from its manifest")
    result = b"".join(output)
    before = np.loadtxt(BytesIO(payload), delimiter=",", ndmin=2)
    after = np.loadtxt(BytesIO(result), delimiter=",", ndmin=2)
    preserved = [column for column in range(width) if column not in replacements]
    if not np.array_equal(before[:, preserved], after[:, preserved]):
        raise ValueError("Static correction changed nonarm values")
    return result, list(replacements)


def _write_new(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)


def _with_entry_profile(reference_bytes, profile_bytes, joint_names, provenance):
    document = json.loads(reference_bytes)
    reference = _read_reference(reference_bytes, joint_names)
    profile = json.loads(profile_bytes)
    if not isinstance(profile, dict):
        raise ValueError("Entry profile must be a waypoint document")
    names = profile.get("joint_names")
    times, poses = profile.get("waypoint_times_s"), profile.get("waypoint_q_rad")
    if (not isinstance(names, list) or len(names) != 14
            or any(not isinstance(name, str) for name in names) or set(names) != set(reference)
            or not isinstance(times, list) or len(times) < 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in times)
            or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:]))
            or not isinstance(poses, list) or len(poses) != len(times)
            or any(not isinstance(row, list) or len(row) != len(names) for row in poses)):
        raise ValueError("Entry profile requires named arm waypoints and increasing times from zero")
    references = [
        _read_reference(_json_bytes(dict(zip(names, row))), joint_names) for row in poses
    ]
    if any(abs(references[-1][name] - value) > 1e-6 for name, value in reference.items()):
        raise ValueError("Entry profile final pose differs from the static reference")
    wrapped = copy.deepcopy(document) if "reference_q" in document else {"reference_q": reference}
    for name, value in (("entry_times_s", times), ("entry_reference_q", references)):
        if name in wrapped and wrapped[name] != value:
            raise ValueError("Existing reference schedule differs from the entry profile")
        wrapped[name] = value
    wrapped["source"] = {
        "kind": "local-static-arm-correction-with-procedural-entry",
        "provenance": provenance,
        "reference_input_sha256": _sha(reference_bytes),
        "entry_profile_sha256": _sha(profile_bytes),
        "reference_source": document.get("source"),
        "scope": "Fixed local waypoints; separate kinematic evidence and physical pilot required.",
    }
    return _json_bytes(wrapped)


def prepare(source_pool, reference, output, provenance, *, entry_profile=None):
    """Build from one explicit local source pool; never edit or reuse an output tree."""
    source_pool, reference, output = (
        Path(path).expanduser().resolve() for path in (source_pool, reference, output)
    )
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if output.is_relative_to(source_pool) or source_pool.is_relative_to(output):
        raise ValueError("Source pool and output must be separate trees")
    _require_ignored_output(output)
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must describe the static correction source")
    provenance = provenance.strip()
    manifest_bytes = (source_pool / "manifest.json").read_bytes()
    manifest = verify_pool(source_pool, "g1", "stealth")
    training = [row for row in manifest["files"] if row["split"] == "train"]
    heldout = [row for row in manifest["files"] if row["split"] == "test"]
    if (len(training) != 1 or len(heldout) != 1
            or training[0].get("source_group") != "aiming2"
            or heldout[0].get("source_group") != "aiming1"):
        raise ValueError("Expected one Aiming2 training clip and one Aiming1 held-out clip")
    train, test = training[0], heldout[0]
    reference_input = reference.read_bytes()
    joint_names = list(get_robot("g1").joint_names)
    profile_bytes = None
    if entry_profile is not None:
        entry_profile = Path(entry_profile).expanduser().resolve()
        profile_bytes = entry_profile.read_bytes()
        reference_bytes = _with_entry_profile(reference_input, profile_bytes, joint_names, provenance)
    else:
        reference_bytes = reference_input
    pose = _read_reference(reference_bytes, joint_names)
    inputs = {source_pool / "manifest.json": manifest_bytes, reference: reference_input}
    if entry_profile is not None:
        inputs[entry_profile] = profile_bytes
    for name in ("train.json", "test.json", *(row["file"] for row in manifest["files"])):
        inputs[source_pool / name] = (source_pool / name).read_bytes()
    for row in manifest["files"]:
        if _sha(inputs[source_pool / row["file"]]) != row["sha256"]:
            raise ValueError("Source CSV changed after pool verification")
    derived, columns = _replace_arm_fields(
        inputs[source_pool / train["file"]], pose, joint_names, train["frames"])
    correction = {
        "kind": "local-static-arm-correction",
        "provenance": provenance,
        "raw_lafan_motion": False,
        "source_upper_body_posture_retained": False,
        "description": "Local static arm correction of a prepared Aiming2 clip, not raw LAFAN motion.",
        "joint_names": list(pose),
        "csv_columns_zero_based": columns,
        "reference_q": pose,
        "reference_input_sha256": _sha(reference_input),
        "reference_document_sha256": _sha(reference_bytes),
        "entry_profile_sha256": _sha(profile_bytes) if profile_bytes is not None else None,
        "nonarm_handling": "Original root, quaternion, leg and waist CSV fields and line endings retained verbatim.",
        "heldout_handling": "Aiming1 CSV and record copied verbatim.",
        "validation_scope": "Content integrity only; collision margins, entry paths and physical tracking require separate evidence.",
    }
    recipe = {
        "version": 1, "purpose": "single-clip G1 stealth static arm derivative",
        "source_pool_recipe_sha256": manifest["recipe_sha256"],
        "source_pool_manifest_sha256": _sha(manifest_bytes),
        "source_training_csv_sha256": train["sha256"],
        "derived_training_csv_sha256": _sha(derived),
        "heldout_csv_sha256": test["sha256"],
        "correction": correction,
        # Universal-newline reading keeps the implementation pin portable across Git checkouts.
        "implementation_sha256": _sha(Path(__file__).read_text(encoding="utf-8").encode("utf-8")),
    }

    def unchanged():
        if any(path.read_bytes() != payload for path, payload in inputs.items()):
            raise ValueError("An input changed during pool preparation")

    def write_pool(directory):
        unchanged()
        records = copy.deepcopy(manifest["files"])
        for record in records:
            if record["split"] == "train":
                _write_new(directory / record["file"], derived)
                record.update(
                    sha256=_sha(derived),
                    transform="Local static arm correction; root, legs, waist and timing retained.",
                    upper_body="14 arm coordinates replaced with an explicit local static pose.",
                    local_correction=copy.deepcopy(correction),
                    parent_pool_record=copy.deepcopy(train),
                    source_identity_scope="Inherited source hashes describe ancestry; output sha256 identifies the corrected derivative.",
                )
            else:
                _write_new(directory / record["file"], inputs[source_pool / record["file"]])
        _write_new(directory / "upper-body-reference.json", reference_bytes)
        auxiliary = {"reference": {"file": "upper-body-reference.json", "sha256": _sha(reference_bytes)}}
        if profile_bytes is not None:
            _write_new(directory / "entry-profile.json", profile_bytes)
            auxiliary["entry_profile"] = {"file": "entry-profile.json", "sha256": _sha(profile_bytes)}
        return records, {
            "source": correction["description"],
            "source_pipeline": "verified source pool -> named static arm correction -> immutable derived pool",
            "pipeline_revision": recipe["implementation_sha256"],
            "local_correction": copy.deepcopy(correction),
            "upstream_selection": copy.deepcopy(manifest.get("selection", {})),
            "selection": {
                "training_clips": 1, "training_frames": train["frames"],
                "held_out_clips": 1, "held_out_frames": test["frames"],
                "held_out_groups": ["aiming1"],
                "policy": "Preserve source-pool intervals, FPS and train/test membership.",
            },
            "auxiliary_files": auxiliary,
        }

    unchanged()
    output.mkdir(parents=True, exist_ok=False)
    pool, built = _build_pool(output, "g1", "stealth", recipe, write_pool)
    verified = verify_pool(pool, "g1", "stealth")
    if verified != built or (pool / "upper-body-reference.json").read_bytes() != reference_bytes:
        raise ValueError("Derived pool or embedded reference differs from its receipt")
    if (pool / test["file"]).read_bytes() != inputs[source_pool / test["file"]]:
        raise ValueError("Held-out CSV changed during publication")
    unchanged()
    _write_new(output / "preparation.json", _json_bytes({
        "status": "ready", "scope": correction["validation_scope"],
        "pool": pool.relative_to(output).as_posix(),
        "recipe_sha256": built["recipe_sha256"],
        "manifest_sha256": _sha((pool / "manifest.json").read_bytes()),
        "train_csv_sha256": _sha(derived), "test_csv_sha256": test["sha256"],
        "reference_sha256": _sha(reference_bytes),
        "nonarm_csv_bytes_preserved": True, "heldout_bytes_identical": True,
        "source_files_unchanged": True, "verify_pool_passed": True,
    }))
    return pool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Named static q JSON, optionally wrapped in reference_q with entry/evidence metadata")
    parser.add_argument("--entry-profile", type=Path,
                        help="Optional joint_names/waypoint_times_s/waypoint_q_rad JSON to embed as named entry poses")
    parser.add_argument("--output", type=Path, required=True, help="Fresh ignored repository data/ tree")
    parser.add_argument("--provenance", required=True, help="Explicit description of the local static correction source")
    args = parser.parse_args(argv)
    pool = prepare(args.source_pool, args.reference, args.output, args.provenance,
                   **({"entry_profile": args.entry_profile} if args.entry_profile else {}))
    print(json.dumps({
        "pool": str(pool), "train_config": str(pool / "train.json"),
        "test_config": str(pool / "test.json"), "reference": str(pool / "upper-body-reference.json"),
        "manifest": str(pool / "manifest.json"),
        "receipt": str(args.output.resolve() / "preparation.json"),
        "recipe_sha256": verify_pool(pool, "g1", "stealth")["recipe_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()

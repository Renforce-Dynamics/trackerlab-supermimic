"""Validate human review metadata and bytes; this cannot prove videos were viewed.

visual_review.json requires review_scope (entry/full), passed, reviewer, notes,
reviewed_at (ISO timestamp), checkpoint_sha256, trajectory_sha256, criteria,
videos ([{"path": "side.mp4", "sha256": "..."}]), and camera_views
({"front": "front.mp4", "side": "side.mp4", "commands": "commands.mp4"}).
Entry uses the first three criteria and front/side views. Full uses all seven
criteria and adds the commands view. Output must be a new file.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PureWindowsPath

CRITERIA = (
    "entry_forward_crouch", "entry_no_side_sway", "arms_natural",
    "backward_steps", "feet_no_drag", "command_changes", "release_crouch_hold",
)
ENTRY_CRITERIA = CRITERIA[:3]
SCHEMA = "trackerlab-stealth-visual-review-v1"
FILE_ERRORS = (OSError, ValueError, RuntimeError)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_file(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or PureWindowsPath(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError(f"invalid relative artifact path: {relative!r}")
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"artifact escapes review directory or is not a file: {relative}")
    return resolved


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate(directory, checkpoint, scope="full"):
    root, checkpoint = Path(directory).resolve(), Path(checkpoint).resolve()
    reserved = {checkpoint, root / "visual_review.json", root / "trajectories.npz"}
    inputs = set(reserved)
    result = {"schema": SCHEMA, "passed": False, "review_scope": None,
              "requested_scope": scope, "files": {}, "videos": {}, "reasons": []}
    reasons = result["reasons"]
    try:
        receipt = local_file(root, "visual_review.json")
        inputs.add(receipt)
        raw = receipt.read_bytes()
        result["visual_review_sha256"] = hashlib.sha256(raw).hexdigest()
        review = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        if not isinstance(review, dict):
            raise ValueError("visual_review.json must contain an object")
    except (*FILE_ERRORS, UnicodeError) as exc:
        review = {}
        reasons.append(f"review receipt unreadable: {exc}")
    result["review_scope"] = review.get("review_scope")
    if scope not in ("entry", "full") or review.get("review_scope") != scope:
        reasons.append(f"review_scope must be exactly {scope!r}; scope cannot be promoted")
    if review.get("passed") is not True:
        reasons.append("review receipt passed must be explicitly true")
    for field in ("reviewer", "notes", "reviewed_at"):
        value = review.get(field)
        result[field] = value if isinstance(value, str) else ""
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"{field} must be a non-empty string")
    try:
        timestamp = result["reviewed_at"]
        if "T" not in timestamp:
            raise ValueError
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        reasons.append("reviewed_at must be an ISO-8601 timestamp")
    required = ENTRY_CRITERIA if scope == "entry" else CRITERIA
    criteria = review.get("criteria")
    criteria = criteria if isinstance(criteria, dict) else {}
    result["criteria"] = criteria
    if set(criteria) != set(required):
        reasons.append(f"criteria must contain exactly: {', '.join(required)}")
    for name in required:
        if criteria.get(name) is not True:
            reasons.append(f"visual criterion must be boolean true: {name}")

    def bind(label, path, expected):
        path = path.resolve()
        inputs.add(path)
        actual = sha256(path)
        matched = isinstance(expected, str) and expected == actual
        result["files"][label] = {"path": str(path), "sha256": actual,
                                  "expected_sha256": expected, "matched": matched}
        if not matched:
            reasons.append(f"SHA-256 mismatch: {label}")
        return actual, matched

    for path, key in ((checkpoint, "checkpoint"), (root / "trajectories.npz", "trajectory")):
        try:
            if key == "trajectory":
                path = local_file(root, "trajectories.npz")
            actual, _ = bind(key, path, review.get(f"{key}_sha256"))
            result[f"{key}_sha256"] = actual
        except FILE_ERRORS as exc:
            reasons.append(f"invalid {key}: {exc}")
    videos = review.get("videos")
    if not isinstance(videos, list) or not 1 <= len(videos) <= 32:
        reasons.append("videos must list between 1 and 32 local files")
        videos = []
    verified, seen, seen_paths = set(), set(), set()
    for item in videos:
        if not isinstance(item, dict):
            reasons.append("each video entry must contain path and sha256")
            continue
        relative = item.get("path")
        try:
            if isinstance(relative, str):
                inputs.add((root / relative).resolve())
            path = local_file(root, relative)
            if relative in seen or path in seen_paths or path in reserved:
                raise ValueError("duplicate video or alias of another reviewed input")
            seen.add(relative)
            seen_paths.add(path)
            actual, matched = bind(f"video:{relative}", path, item.get("sha256"))
            result["videos"][relative] = actual
            if matched:
                verified.add(relative)
        except FILE_ERRORS as exc:
            reasons.append(f"invalid video {relative!r}: {exc}")
    views = review.get("camera_views")
    views = views if isinstance(views, dict) else {}
    result["camera_views"] = views
    for view in ("front", "side") + (("commands",) if scope == "full" else ()):
        relative = views.get(view)
        if not isinstance(relative, str) or relative not in verified:
            reasons.append(f"camera view {view!r} must reference a hash-verified video")
    result["passed"] = not reasons
    return result, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("entry", "full"), default="full")
    args = parser.parse_args(argv)
    try:
        result, inputs = validate(args.directory, args.checkpoint, args.scope)
    except FILE_ERRORS as exc:
        result = {"schema": SCHEMA, "passed": False, "review_scope": None,
                  "requested_scope": args.scope, "reasons": [str(exc)]}
        inputs = set()
    try:
        output = args.output.resolve()
        if output in inputs:
            raise ValueError("output must not overwrite a reviewed input")
        # Exclusive creation also protects hardlink aliases and malformed-review inputs.
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    except FILE_ERRORS as exc:
        result["passed"] = False
        result["reasons"].append(f"cannot write review result: {exc}")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

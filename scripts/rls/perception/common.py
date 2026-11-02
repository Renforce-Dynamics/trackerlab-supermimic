"""Paths and durable receipts for the isolated native locomotion workflow."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
for root in ("source/trackerLab", "source/trackerTask", "scripts/rls/supermimic"):
    sys.path.insert(0, str(REPO / root))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_audit(recorded, root=REPO):
    """Compare a run's recorded sources without modifying its original receipt."""
    changed, missing = {}, []
    for relative, expected in recorded.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        current = digest(path)
        if current != expected:
            changed[relative] = {"training_sha256": expected, "current_sha256": current}
    return {"recorded_files": len(recorded), "changed": changed, "missing": missing}

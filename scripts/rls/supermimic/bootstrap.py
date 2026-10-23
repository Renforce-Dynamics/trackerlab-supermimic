"""Use the packages belonging to this standalone or host TrackerLab checkout."""

from pathlib import Path
import os
import sys


def configure_paths():
    root = Path(__file__).resolve().parents[3]
    for name in ("trackerLab", "trackerTask"):
        directory = root / "source" / name
        loaded = sys.modules.get(name)
        if loaded is not None:
            actual = Path(loaded.__file__).resolve()
            if not actual.is_relative_to(directory.resolve()):
                raise RuntimeError(f"{name} already imported from a different installation: {actual}")
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    os.environ.setdefault("SUPERMIMIC_REPO_ROOT", str(root))
    return root

"""JSON training presets; explicit command-line options override preset values."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_config(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Training configuration must be a JSON object")
    return value


def apply_config(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> None:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    path = probe.parse_known_args(argv)[0].config
    if path is None:
        return
    try:
        config = load_config(path)
        actions = {action.dest: action for action in parser._actions}
        for key, value in config.items():
            if key not in actions or key in ("help", "config", "resume", "log_dir", "stop_file"):
                raise ValueError(f"Unsupported preset key: {key}")
            action = actions[key]
            if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)):
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a JSON boolean")
                continue
            values = value if isinstance(value, list) else [value]
            if isinstance(value, list) and action.nargs is None:
                raise ValueError(f"{key} must be a scalar")
            if isinstance(action.nargs, int) and len(values) != action.nargs:
                raise ValueError(f"{key} needs {action.nargs} values")
            for item in values:
                if action.type is int and (not isinstance(item, int) or isinstance(item, bool)):
                    raise ValueError(f"{key} must be an integer")
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError(f"{key} must be finite")
                if action.type is not None:
                    converted = action.type(item)
                    if isinstance(item, bool) or converted != item:
                        raise ValueError(f"Invalid {key}: {item!r}")
                if action.choices is not None and item not in action.choices:
                    raise ValueError(f"Invalid {key}: {item!r}; expected {action.choices}")
        parser.set_defaults(**config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

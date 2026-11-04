"""Render fixed first physical attempts from a command/style evaluation."""

import argparse
from pathlib import Path

from render_terrain import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()
    render(args.run_dir.resolve(), level=args.level)


if __name__ == "__main__":
    main()

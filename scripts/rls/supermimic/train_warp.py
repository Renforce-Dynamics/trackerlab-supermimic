"""Start the isolated G1 MuJoCo-Warp scratch-training runtime."""

from pathlib import Path
import runpy
import sys


def main():
    runtime = Path(__file__).resolve().with_name("warp_runtime")
    sys.path.insert(0, str(runtime))
    runpy.run_path(str(runtime / "train_warp.py"), run_name="__main__")


if __name__ == "__main__":
    main()

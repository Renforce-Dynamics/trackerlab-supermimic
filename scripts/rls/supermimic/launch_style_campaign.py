"""Launch one detached Linux training campaign after checking its live owner."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bootstrap import configure_paths
REPO = configure_paths()
from supervise_style import resolve_styles
from supervise_terrain import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style-map", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=8096)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Detached training launch requires Linux")
    resolve_styles(args.style_map.resolve())
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "STOP").exists():
        parser.error("Campaign has an explicit STOP marker")
    receipt = root / "campaign.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        pid = previous.get("pid")
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists():
            identity = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "supervise_style.py" in identity and str(root) in identity:
                print(json.dumps({"status": "already_running", "pid": pid, "run_dir": str(root)}))
                return
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("supervise_style.py")),
        "--style-map", str(args.style_map.resolve()), "--run-dir", str(root),
        "--iterations", str(args.iterations), "--max-rounds", str(args.max_rounds),
        "--num-envs", str(args.num_envs),
    ]
    with (root / "supervisor_stdout.log").open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    result = {"status": "launched", "pid": child.pid, "run_dir": str(root),
              "command": command, "launcher_pid": os.getpid()}
    save(root / "launch.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

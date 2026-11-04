"""Render source-style poses for visual review, explicitly separate from policies."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
if Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json").exists():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from bootstrap import configure_paths
configure_paths()
from render_demo import font
from trackerTask.supermimic.mujoco_terrain import build_model
from trackerTask.supermimic.terrain import generate, ROBOT_HEIGHTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = (args.config.resolve().parent / config["root"]).resolve()
    height = ROBOT_HEIGHTS[args.robot]
    model, _, _, qids, *_ = build_model(
        args.robot, generate("velocity", height), render_size=(640, 360))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 360, 640)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    sheet = Image.new("RGB", (1280, 820), "#eef2f2")
    draw = ImageDraw.Draw(sheet)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5 * height / .76, 130, -15
    names = config["files"][:2]
    try:
        for row, name in enumerate(names):
            values = np.loadtxt(root / name, delimiter=",", ndmin=2)
            for column, fraction in enumerate((.25, .7)):
                frame = min(len(values) - 1, int(fraction * len(values)))
                q = values[frame]
                data.qpos[:3] = (0, 0, q[2])
                data.qpos[3:7], data.qpos[qids] = q[[6, 3, 4, 5]], q[7:]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = (0, 0, q[2] - .1 * height / .76)
                renderer.update_scene(data, camera=camera, scene_option=option)
                pixels = renderer.render().copy()
                if pixels.std() < 5 or (pixels[50:330, 180:460].min(-1) < 60).sum() < 100:
                    raise ValueError("Reference render is blank")
                x, y = column * 640, row * 410
                sheet.paste(Image.fromarray(pixels), (x, y + 40))
                draw.text((x + 12, y + 9), f"REFERENCE ONLY | {args.robot} | frame {frame}",
                          font=font(18), fill="#243a41")
                draw.text((x + 12, y + 382), name[:70], font=font(13), fill="#243a41")
    finally:
        renderer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(str(args.output))


if __name__ == "__main__":
    main()

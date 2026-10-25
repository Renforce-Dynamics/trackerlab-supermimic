"""Replay recorded PhysX target/bar/robot states with MuJoCo graphics."""

import argparse
import json
from pathlib import Path
import subprocess

from render_demo import render_model, font
from bootstrap import configure_paths
configure_paths()
from trackerTask.supermimic.robots import get_robot
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.directory / "trajectories.npz")
    evaluation = json.loads((args.directory / "evaluation.json").read_text())
    spec = get_robot("go2")
    model = render_model(1280, 620, spec)
    state = mujoco.MjData(model)
    qids = [int(model.jnt_qposadr[model.joint(name).id]) for name in spec.joint_names]
    renderer = mujoco.Renderer(model, 620, 1280)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 4., -18, 125
    output = args.directory / "demo.mp4"
    encoder = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "25", "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    frames, thumbnails = 0, []
    def add_geom(kind, size, position, color):
        idx = renderer.scene.ngeom
        mujoco.mjv_initGeom(renderer.scene.geoms[idx], kind, np.array(size, dtype=float),
                           np.array(position, dtype=float), np.eye(3).ravel(), np.array(color, dtype=np.float32))
        renderer.scene.ngeom += 1
    try:
        for trial, name in enumerate(data["names"]):
            for step in range(0, len(data["root_p"]), 2):
                p, c = data["root_p"][step, trial], data["scenario"][step, trial]
                state.qpos[:3] = p
                state.qpos[3:7] = data["root_quat"][step, trial]
                state.qpos[qids] = data["q"][step, trial]
                mujoco.mj_forward(model, state)
                camera.lookat[:] = [p[0]+.4, p[1], .3]
                renderer.update_scene(state, camera=camera)
                target = c[:2] + c[2:4]*c[9] + [0, .3*c[10]*np.sin(.7*c[9])]
                bar = [c[4]+c[6]*c[9], c[5], c[7]]
                if c[8] > 0:
                    add_geom(mujoco.mjtGeom.mjGEOM_BOX, [.025, 3., .025], bar, [0.85, .15, .05, 1])
                add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.09, 0, 0], [*target, .1], [.05, .7, .3, 1])
                image = Image.new("RGB", (1280, 720), "#e4ebed")
                image.paste(Image.fromarray(renderer.render().copy()), (0, 70))
                draw = ImageDraw.Draw(image)
                draw.text((16, 10), f"Go2 CHASE + DODGE | task iteration {int(data['iteration'])} | {name}", font=font(23), fill="#183037")
                metrics = evaluation["per_mode"][str(name)]
                draw.text((16, 42), f"Difficulty {evaluation['difficulty']:.2f} | t={step*.02:.2f}s | contact={data['contact_force'][step, trial]:.1f} N | height={p[2]:.2f} m",
                          font=font(18), fill="#183037")
                if not data["alive"][step, trial]:
                    draw.text((16, 90), "Trial terminated; subsequent poses are outside evaluation", font=font(20), fill="#aa2030")
                draw.text((16, 695), f"Recorded PhysX trajectory | 1x speed | {name} success: {metrics['successes']}/{metrics['trials']} | preselected trial",
                          font=font(16), fill="#183037")
                encoder.stdin.write(np.asarray(image).tobytes())
                frames += 1
                if step == 150:
                    thumbnails.append(image.resize((640, 360)))
    finally:
        encoder.stdin.close()
        code = encoder.wait()
        renderer.close()
    if code:
        raise RuntimeError("Video encoding failed")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output), "-f", "null", "-"], check=True)
    sheet = Image.new("RGB", (640, len(thumbnails)*360))
    for index, image in enumerate(thumbnails):
        sheet.paste(image, (0, index*360))
    sheet.save(args.directory / "demo_contact_sheet.png")
    thumbnails[0].save(args.directory / "demo.png")
    (args.directory / "demo.json").write_text(json.dumps({
        "frames": frames, "fps": 25, "source": "Recorded PhysX trajectory",
        "renderer": "MuJoCo pose replay; bar and target match stored scenario", "decode": "passed",
        "selection": "first trial of flat, duck and jump; no outcome-based selection",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

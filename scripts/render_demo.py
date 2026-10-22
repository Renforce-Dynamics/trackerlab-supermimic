"""Render recorded PhysX policy trajectories beside their LAFAN1 references."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
if Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json").exists():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from trackerlab_supermimic.robot import G1_JOINT_NAMES, MJCF_PATH

DEFAULT_CLIPS = [
    "walk1_subject1", "run2_subject1", "dance2_subject1",
    "jumps1_subject1", "fallAndGetUp1_subject1", "fight1_subject2",
]


def render_model(width, height):
    source = Path(MJCF_PATH)
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source.parent / compiler.get("meshdir", ".")).resolve()))
    # Keep the original robot materials while making the pose readable against a light floor.
    for material in root.findall("./asset/material"):
        if material.get("name") == "black":
            material.set("rgba", "0.16 0.18 0.20 1")
        elif material.get("name") == "metal":
            material.set("rgba", "0.62 0.67 0.70 1")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    settings = visual.find("global")
    if settings is None:
        settings = ET.SubElement(visual, "global")
    settings.set("offwidth", str(width))
    settings.set("offheight", str(height))
    assets = root.find("asset")
    ET.SubElement(assets, "texture", name="demo_checker", type="2d", builtin="checker",
                  rgb1="0.75 0.81 0.82", rgb2="0.54 0.63 0.65", width="512", height="512")
    ET.SubElement(assets, "texture", name="demo_sky", type="skybox", builtin="flat",
                  rgb1="0.84 0.90 0.92", width="256", height="1536")
    ET.SubElement(assets, "material", name="demo_floor", texture="demo_checker",
                  texrepeat="1 1", texuniform="true", reflectance="0")
    world = root.find("worldbody")
    for light in world.findall("light"):
        world.remove(light)
    ET.SubElement(world, "geom", name="demo_ground", type="plane", size="0 0 0.1",
                  material="demo_floor", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos="0 0 6", dir="0 0 -1", directional="true",
                  diffuse="0.7 0.7 0.7", ambient="0.18 0.18 0.18")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.vis.headlight.ambient[:] = 0.14
    model.vis.headlight.diffuse[:] = 0.42
    return model


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--fps", type=int, default=25, choices=(10, 25, 50))
    args = parser.parse_args()
    archive = np.load(args.trajectory)
    trajectory = {name: archive[name] for name in archive.files}
    evaluation = json.loads(args.evaluation.read_text())
    iteration = args.iteration if args.iteration is not None else int(trajectory["iteration"])
    if evaluation.get("iteration", iteration) != iteration:
        raise ValueError("Trajectory and evaluation checkpoint iterations differ")
    if args.seconds <= 0 or args.start < 0:
        parser.error("--seconds must be positive and --start must be nonnegative")
    names = trajectory["names"].tolist()
    missing = set(args.clips).difference(names)
    if missing:
        raise ValueError(f"Unknown clips: {sorted(missing)}")
    width, height, top, bottom = 1280, 720, 84, 64
    pane_width, pane_height = width // 2, height - top - bottom
    model = render_model(pane_width, pane_height)
    data = mujoco.MjData(model)
    joint_qpos_ids = [int(model.jnt_qposadr[model.joint(name).id]) for name in G1_JOINT_NAMES]
    renderer = mujoco.Renderer(model, pane_height, pane_width)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.7, 135, -15
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(args.output),
    ], stdin=subprocess.PIPE)
    output_frames = 0
    selections = []
    contact_frames = []

    def pose_image(position, quaternion, joints):
        data.qpos[:3], data.qpos[3:7], data.qpos[joint_qpos_ids] = position, quaternion, joints
        mujoco.mj_forward(model, data)
        camera.lookat[:] = [position[0], position[1], max(0.5, float(position[2]))]
        renderer.update_scene(data, camera=camera)
        return Image.fromarray(renderer.render().copy())

    try:
        for clip in args.clips:
            index = names.index(clip)
            start = int(trajectory["clip_start"][index])
            clip_length = int(trajectory["clip_len"][index])
            recorded_steps = len(trajectory["alive"])
            alive_indices = np.flatnonzero(trajectory["alive"][:, index])
            if len(alive_indices) == 0:
                raise ValueError(f"No active policy frames for {clip}")
            last_alive = int(alive_indices[-1])
            failed = bool(evaluation["per_motion"][clip]["failed"])
            cutoff = min(last_alive + 1, recorded_steps - 1)
            start_step = round(args.start * 50)
            start_step -= start_step % (50 // args.fps)
            end_step = min(clip_length, start_step + max(1, round(args.seconds * 50)))
            if start_step >= end_step:
                raise ValueError(f"Requested excerpt starts after clip end: {clip}")
            duration_frames = end_step - start_step
            snapshot_step = min(end_step - 1, start_step + 150)
            snapshot_step -= snapshot_step % (50 // args.fps)
            selections.append({"clip": clip, "start_seconds": start_step / 50, "duration_seconds": duration_frames / 50,
                               "failed": failed, "observed_active_steps": last_alive + 1})
            for step in range(start_step, end_step, 50 // args.fps):
                actual_step = min(step, cutoff)
                reference_frame = min(start + step, start + clip_length - 1)
                reference_quat = np.empty(4)
                mujoco.mju_mat2Quat(reference_quat, trajectory["ref_root_R"][reference_frame].reshape(-1).astype(np.float64))
                actual_position = trajectory["root_p"][actual_step, index]
                frame = Image.new("RGB", (width, height), "#dce5e7")
                frame.paste(pose_image(
                    actual_position, trajectory["root_quat"][actual_step, index],
                    trajectory["q"][actual_step, index],
                ), (0, top))
                frame.paste(pose_image(
                    trajectory["ref_root_p"][reference_frame], reference_quat,
                    trajectory["ref_q"][reference_frame],
                ), (pane_width, top))
                draw = ImageDraw.Draw(frame)
                draw.text((18, 10), f"TrackerLab SuperMimic | iteration {iteration} | {clip}", font=font(23), fill="#193337")
                draw.text((18, 46), "POLICY: recorded PhysX rollout", font=font(19), fill="#08786f")
                draw.text((pane_width + 18, 46), "REFERENCE: G1 LAFAN1 motion", font=font(19), fill="#4b5094")
                if failed and step > last_alive:
                    draw.rectangle((18, top + 16, pane_width - 18, top + 62), fill="#a92d3b")
                    draw.text((30, top + 24), f"Terminated at {(last_alive + 1) / 50:.2f}s; pose held",
                              font=font(20), fill="white")
                root_error = float(np.linalg.norm(actual_position - trajectory["ref_root_p"][reference_frame]))
                score = evaluation["per_motion"][clip]["normalized_return"]
                draw.text((18, height - bottom + 6),
                          f"t = {step / 50:.2f}s   |   root error = {root_error:.2f} m   |   full-clip return = {score:.4f}",
                          font=font(18), fill="#193337")
                draw.text((18, height - 25),
                          "1x playback | Independent camera centering | Recorded poses rendered with MuJoCo",
                          font=font(15), fill="#506469")
                ffmpeg.stdin.write(np.asarray(frame).tobytes())
                if step == snapshot_step:
                    if not contact_frames:
                        frame.save(args.output.with_suffix(".png"))
                    contact_frames.append(frame.resize((640, 360), Image.Resampling.LANCZOS))
                output_frames += 1
    finally:
        ffmpeg.stdin.close()
        returncode = ffmpeg.wait()
        renderer.close()
    if returncode:
        raise RuntimeError(f"Video encoder exited with code {returncode}")
    import cv2
    video = cv2.VideoCapture(str(args.output))
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    good, first = video.read()
    video.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 3))
    next_good, later = video.read()
    video.release()
    if not good or not next_good or first.std() < 5 or np.mean(np.abs(first.astype(float) - later.astype(float))) < 0.01:
        raise AssertionError("Video pixel verification failed")
    if count != output_frames:
        raise AssertionError(f"Expected {output_frames} frames, decoded {count}")
    contact_sheet = Image.new("RGB", (1280, 360 * ((len(contact_frames) + 1) // 2)), "#dce5e7")
    for i, frame in enumerate(contact_frames):
        contact_sheet.paste(frame, ((i % 2) * 640, (i // 2) * 360))
    contact_sheet.save(args.output.with_name(args.output.stem + "_contact_sheet.png"))
    metadata = {
        "iteration": iteration,
        "source_trajectory": str(args.trajectory),
        "policy_simulator": "IsaacLab / PhysX",
        "renderer": "MuJoCo EGL; pose replay only",
        "cameras": "independently centered; numerical root error retains world-coordinate difference",
        "fps": args.fps,
        "frames": output_frames,
        "duration_seconds": output_frames / args.fps,
        "clip_selection": "fixed representative set, selected before the 2000-iteration evaluation",
        "clips": selections,
        "pixel_and_frame_count_checks": "passed",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

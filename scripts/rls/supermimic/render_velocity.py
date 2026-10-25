"""Render a full preselected velocity trial from stored PhysX states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def render(directory, trial=0, include_failure=False, output_directory=None,
           camera_distance=5.2, camera_elevation=-24.0, camera_azimuth=125.0):
    from render_demo import render_model, font
    from bootstrap import configure_paths

    configure_paths()
    from trackerTask.supermimic.robots import get_robot
    from trackerLab.tracker_env.mdp.supermimic_velocity_dodge import bar_pose
    import imageio_ffmpeg
    import mujoco
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    directory = Path(directory)
    output_directory = Path(output_directory) if output_directory is not None else directory
    if not all(math.isfinite(value) for value in
               (camera_distance, camera_elevation, camera_azimuth)) or camera_distance <= 0:
        raise ValueError("Camera values must be finite and distance must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if int(data["iteration"]) != evaluation["iteration"]:
        raise ValueError("Trajectory and evaluation iterations differ")
    count = data["alive"].shape[1]
    if not 0 <= trial < count:
        raise ValueError(f"Trial must lie in [0, {count})")
    dt = float(data["control_dt"])
    stride = round(1 / (25 * dt))
    if stride < 1 or abs(stride * dt - .04) > 1e-6 or len(data["alive"]) != round(16 / dt):
        raise ValueError("Expected full 16-second, 50 Hz PhysX trajectories")
    selections = [(trial, "preselected trial 0" if trial == 0 else "explicitly selected trial")]
    if include_failure:
        failed = next((row["trial"] for row in evaluation["per_trial"]
                       if not row["sequence_success"] and row["trial"] != trial), None)
        if failed is not None:
            selections.append((failed, "failure example selected after evaluation"))
    task = evaluation["protocol"]["task_config"]
    poses = bar_pose(torch.from_numpy(data["scenario"])).numpy()
    if not np.allclose(poses, data["bar_pose"], atol=1e-5):
        raise ValueError("Stored scenario and shared bar_pose disagree")
    spec = get_robot("go2")
    model = render_model(1280, 600, spec)
    state = mujoco.MjData(model)
    qids = [int(model.jnt_qposadr[model.joint(name).id]) for name in spec.joint_names]
    renderer = mujoco.Renderer(model, 600, 1280)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = (
        camera_distance, camera_elevation, camera_azimuth)
    output = output_directory / "demo.mp4"
    encoder = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "25", "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-crf", "20", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    frames, thumbnails, visual_changes = 0, [], 0
    previous_pixels = None
    selection_metadata = []
    events = {(event["trial"], event["encounter"]): event for event in evaluation["encounters"]}
    snapshot_steps = {round(second / dt) // stride * stride for second in (1, 5, 9, 13)}
    try:
        for selected, label in selections:
            active_indices = np.flatnonzero(data["alive"][:, selected])
            if not len(active_indices):
                raise ValueError(f"Trial {selected} has no active transitions")
            last = int(active_indices[-1])
            trial_result = evaluation["per_trial"][selected]
            selection_metadata.append({
                "trial": selected, "selection": label,
                "sequence_success": trial_result["sequence_success"],
                "survived_full_episode": trial_result["survived_full_episode"],
                "completed_events_all_success": trial_result["completed_events_all_success"],
                "censored_final_encounters": trial_result["censored_final_encounters"],
                "first_fall_s": trial_result["first_fall_s"], "duration_seconds": 16.0,
            })
            for step in range(0, len(data["alive"]), stride):
                held = step > last
                index = min(step, last)
                scenario = data["scenario"][index, selected]
                p = data["root_p"][index, selected]
                quat = data["root_quat"][index, selected]
                q = data["q"][index, selected]
                pose = poses[index, selected]
                if held:
                    terminal = data["s1"][last, selected]
                    p, quat, q = terminal[:3], terminal[3:7], terminal[13:25]
                    pose = bar_pose(torch.from_numpy(data["reward_scenario"][last, selected])).numpy()
                state.qpos[:3], state.qpos[3:7], state.qpos[qids] = p, quat, q
                mujoco.mj_forward(model, state)
                camera.lookat[:] = [p[0], p[1], .30]
                renderer.update_scene(state, camera=camera)
                if scenario[8] > 0:
                    rotation = np.empty(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(rotation, pose[3:].astype(np.float64))
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom, mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([task["bar_half_thickness"], task["bar_half_width"],
                                  task["bar_half_thickness"]], dtype=float),
                        pose[:3].astype(float), rotation, np.array([.85, .15, .05, 1], dtype=np.float32),
                    )
                    renderer.scene.ngeom += 1
                pixels = renderer.render().copy()
                if pixels.std() < 5:
                    raise AssertionError("Rendered scene is blank")
                if previous_pixels is not None and np.mean(np.abs(
                        pixels.astype(float) - previous_pixels.astype(float))) > .01:
                    visual_changes += 1
                previous_pixels = pixels
                image = Image.new("RGB", (1280, 720), "#e4ebed")
                image.paste(Image.fromarray(pixels), (0, 82))
                draw = ImageDraw.Draw(image)
                labels = (("flat", "high bar", "low bar") if evaluation["task_profile"] == "adaptive"
                          else ("flat", "duck", "jump"))
                mode = labels[int(scenario[8])]
                command = data["command"][index, selected]
                measured = data["measured_body_velocity"][index, selected]
                draw.text((16, 8), f"Go2 VELOCITY + DODGE | {evaluation['task_profile']} | "
                          f"iteration {evaluation['iteration']} | trial {selected}", font=font(23), fill="#183037")
                draw.text((16, 39), f"t={step * dt:.2f}s | {mode} | command vx/vy/wz: "
                          f"{command[0]:+.2f} / {command[1]:+.2f} / {command[2]:+.2f} | "
                          f"measured: {measured[0]:+.2f} / {measured[1]:+.2f} / {measured[2]:+.2f}",
                          font=font(18), fill="#183037")
                draw.text((16, 63), f"Encounter {int(scenario[14])} | contact peak "
                          f"{data['contact_force'][index, selected]:.1f} N | {label}",
                          font=font(15), fill="#183037")
                if held:
                    draw.rectangle((16, 98, 990, 136), fill="#a92d3b")
                    draw.text((28, 104), f"Trial ended at {(last + 1) * dt:.2f}s; terminal PhysX pose held",
                              font=font(20), fill="white")
                event = events[(selected, int(scenario[14]))]
                if event["failure_reasons"]:
                    outcome = "Recorded outcome: " + ", ".join(event["failure_reasons"])
                    if event["censored"]:
                        outcome += " | horizon-censored; counted as failed attempt"
                    outcome += f" | clearance delay {event['clearance_delay_s']:.2f}s"
                    text_font = font(17)
                    for size in range(17, 10, -1):
                        text_font = font(size)
                        if draw.textbbox((0, 0), outcome, font=text_font)[2] <= 1216:
                            break
                    draw.rectangle((16, 146, 1264, 178), fill="#e4ebed")
                    draw.text((28, 152), outcome, font=text_font, fill="#8b2635")
                draw.text((16, 691), f"Recorded PhysX | 1x playback | successes "
                          f"{trial_result['successes']}/{trial_result['attempts']} attempts | "
                          f"fall: {trial_result['fall']} | full 16s display", font=font(17), fill="#183037")
                encoder.stdin.write(np.asarray(image).tobytes())
                frames += 1
                if step in snapshot_steps:
                    if not thumbnails:
                        image.save(output_directory / "demo.png")
                    thumbnails.append(image.resize((640, 360), Image.Resampling.LANCZOS))
    finally:
        try:
            encoder.stdin.close()
        finally:
            code = encoder.wait()
            renderer.close()
    if code:
        raise RuntimeError(f"Video encoder exited with code {code}")
    if frames != 400 * len(selections) or visual_changes == 0:
        raise AssertionError("Video frame count or scene-motion verification failed")
    decoded = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output),
        "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ], check=True, capture_output=True, text=True)
    decoded_frames = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines()
                      if line.startswith("frame=")]
    if not decoded_frames or decoded_frames[-1] != frames:
        raise AssertionError("Decoded video frame count differs from rendered frames")
    sheet = Image.new("RGB", (1280, 360 * ((len(thumbnails) + 1) // 2)), "#e4ebed")
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % 2) * 640, (index // 2) * 360))
    sheet.save(output_directory / "demo_contact_sheet.png")
    source_hashes = {}
    for name in ("evaluation.json", "trajectories.npz"):
        with (directory / name).open("rb") as stream:
            source_hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "iteration": evaluation["iteration"], "task_profile": evaluation["task_profile"],
        "frames": frames, "fps": 25, "duration_seconds": frames / 25, "trials": selection_metadata,
        "source": "Recorded PhysX robot states and per-step scenarios",
        "renderer": "MuJoCo pose replay; shared bar_pose with full yaw quaternion",
        "terminal_display": "Hold terminal robot and bar pose after a fall; mark the hold explicitly",
        "censoring": "Unfinished horizon attempts remain failures; completion and survival are separate",
        "command_measurement": "Post-transition body velocity for the displayed interval",
        "decode": "passed", "pixel_motion_and_frame_count_checks": "passed",
        "camera": {"distance": camera_distance, "elevation": camera_elevation,
                   "azimuth": camera_azimuth},
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "source_sha256": source_hashes,
    }
    (output_directory / "demo.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--include_failure", action="store_true")
    parser.add_argument("--output_directory", type=Path)
    parser.add_argument("--camera_distance", type=float, default=5.2)
    parser.add_argument("--camera_elevation", type=float, default=-24.0)
    parser.add_argument("--camera_azimuth", type=float, default=125.0)
    args = parser.parse_args()
    render(args.directory, args.trial, args.include_failure, args.output_directory,
           args.camera_distance, args.camera_elevation, args.camera_azimuth)


if __name__ == "__main__":
    main()

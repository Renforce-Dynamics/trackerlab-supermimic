"""Local CPU FK replay of the first recorded G1 entry, with heading-follow views."""

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")

import numpy as np

FPS, WIDTH, HEIGHT, TOP, BOTTOM = 25, 960, 600, 72, 28
WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
TARGET_LABEL = "KINEMATIC TARGET ONLY - NOT A POLICY ROLLOUT"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_trajectory(path, lane=0, seconds=6.):
    from audit_stealth_trajectory import load_record

    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    poses, dt, metadata = load_record(path, lane=lane)
    stride = round(1. / (FPS * dt))
    if stride < 1 or not math.isclose(stride * dt, 1. / FPS, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Recorded cadence must divide 25fps with an integer stride")
    duration = min(seconds, (len(poses) - 1) * dt)
    count = min(math.floor(duration * FPS + 1e-9), (len(poses) - 1) // stride + 1)
    if count < 1:
        raise ValueError("Pre-terminal window is too short for a 25fps frame")
    indices = np.arange(count, dtype=np.int64) * stride
    with np.load(path, allow_pickle=False) as archive:
        iterations = archive["solver_iterations"] if "solver_iterations" in archive else np.array(50)
        if iterations.ndim or iterations.dtype.kind not in "iu" or iterations < 1:
            raise ValueError("solver_iterations must be a positive integer")
        metadata["solver_iterations"] = int(iterations)
        scale = archive["waist_pd_scale"] if "waist_pd_scale" in archive else np.array(1.)
        if scale.ndim or scale.dtype.kind not in "fiu" or not np.isfinite(scale) or scale <= 0:
            raise ValueError("waist_pd_scale must be a finite positive scalar")
        metadata["waist_pd_scale"] = float(scale)
        for name, default in (("self_collisions", False), ("collision_mask_version", 2)):
            value = archive[name] if name in archive else np.array(default)
            if value.ndim or (name == "self_collisions" and value.dtype.kind != "b") or (
                    name == "collision_mask_version" and (value.dtype.kind not in "iu" or value not in (1, 2))):
                raise ValueError(f"Invalid scalar {name}")
            metadata[name] = value.item()
    metadata.update(control_dt=dt, stride=stride, last_displayed_source_frame=int(indices[-1]),
                    source_duration_s=(len(poses) - 1) * dt, requested_seconds=seconds)
    return poses, indices, metadata


def heading_degrees(pose):
    w, x, y, z = pose[3:7]
    forward_x, forward_y = 1 - 2 * (y*y + z*z), 2 * (x*y + w*z)
    if math.hypot(forward_x, forward_y) < 1e-8:
        raise ValueError("Root heading is undefined")
    return math.degrees(math.atan2(forward_y, forward_x))


def waist_targets(poses, dt, reference, model):
    """Use the planned class's quintic schedule; leave all other qpos untouched."""
    config = reference.get("constraint")
    if (reference.get("schema") != "g1-stealth-waist-entry-v1"
            or not isinstance(config, dict)):
        raise ValueError("Require the prepared g1-stealth-waist-entry-v1 reference")
    names = config.get("joint_names")
    if (not isinstance(names, list) or len(names) != 29 or any(not isinstance(n, str) for n in names)
            or len(set(names)) != 29 or not all(name in names for name in WAIST_NAMES)):
        raise ValueError("Reference must contain 29 unique named joints including the waist")
    default = np.asarray(config.get("default_q"), dtype=np.float64)
    targets = config.get("reference_q")
    duration = config.get("ramp_duration_s")
    if (default.shape != (29,) or not np.isfinite(default).all()
            or not isinstance(targets, dict) or set(targets) != set(WAIST_NAMES)
            or isinstance(duration, bool) or not isinstance(duration, (float, int))
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Invalid finite waist reference/default/ramp duration")
    start = default[[names.index(name) for name in WAIST_NAMES]]
    end = np.asarray([targets[name] for name in WAIST_NAMES], dtype=np.float64)
    if end.shape != (3,) or not np.isfinite(end).all():
        raise ValueError("Waist reference must contain three finite target positions")
    addresses = [int(model.joint(name).qposadr[0]) for name in WAIST_NAMES]
    if len(set(addresses)) != 3 or any(address < 7 or address >= poses.shape[1] for address in addresses):
        raise ValueError("Waist joints must map to exactly three hinge qpos entries")
    phase = np.clip(np.arange(len(poses)) * dt / duration, 0., 1.)
    blend = phase**3 * (10. + phase * (-15. + 6. * phase))
    changed = poses.copy()
    changed[:, addresses] = start + blend[:, None] * (end - start)
    return changed, {"changed_joint_names": list(WAIST_NAMES), "changed_qpos_addresses": addresses,
                     "ramp_duration_s": duration, "start_q": start.tolist(), "target_q": end.tolist()}


def nominal_model(metadata):
    from bootstrap import configure_paths
    configure_paths()
    from trackerTask.supermimic.mujoco_terrain import build_model
    from trackerTask.supermimic.terrain import generate

    return build_model("g1", generate("stealth", .76, metadata["seed"]),
                       metadata["solver_iterations"], render_size=(WIDTH, HEIGHT),
                       waist_pd_scale=metadata.get("waist_pd_scale", 1.),
                       self_collisions=metadata.get("self_collisions", False),
                       collision_mask_version=metadata.get("collision_mask_version", 2))


def replay_pose(model, data, pose, timestamp):
    import mujoco

    data.qpos[:] = pose
    data.time = timestamp
    mujoco.mj_forward(model, data)


def render(trajectory, output, lane=0, seconds=6., waist_reference=None):
    trajectory, output = Path(trajectory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(output))

    import imageio_ffmpeg
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    source_hash = digest(trajectory)
    poses, indices, metadata = load_trajectory(trajectory, lane, seconds)
    commands = None
    with np.load(trajectory, allow_pickle=False) as archive:
        if "command_sequence" in archive:
            values = archive["command_sequence"]
            if (values.ndim != 3 or values.shape[1] <= lane or values.shape[2] != 7
                    or len(values) < len(poses) or not np.isfinite(values[:len(poses), lane]).all()):
                raise ValueError("Expected finite command_sequence [T, N, 7] aligned with qpos")
            commands = values[:len(poses), lane].copy()
    model, *_, contract = nominal_model(metadata)
    xml = Path(contract["source_mjcf"]).resolve()
    target, target_info, reference_hash = waist_reference is not None, None, None
    if target:
        waist_reference = Path(waist_reference).resolve()
        reference_hash = digest(waist_reference)
        reference = json.loads(waist_reference.read_text(encoding="utf-8"))
        if not isinstance(reference, dict):
            raise ValueError("waist reference must contain an object")
        poses, target_info = waist_targets(poses, metadata["control_dt"], reference, model)
    # A fresh directory and exclusive file creation preserve every existing input/output.
    output.mkdir(parents=True, exist_ok=False)
    stem = "entry-target-only-front-side" if target else "entry-front-side"
    video, receipt = output / f"{stem}.mp4", output / f"{stem}.json"
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=21)
    data, camera, option = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    option.geomgroup[:] = 1
    camera.distance, camera.elevation = 2.8, -8.
    model.vis.headlight.ambient[:] = .32
    model.vis.headlight.diffuse[:] = .55
    frame_size = (WIDTH * 2, TOP + HEIGHT + BOTTOM)
    minimum_std = {"front": math.inf, "side": math.inf}
    snapshots = {}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        def image_for(index):
            pose, t = poses[index], index * metadata["control_dt"]
            replay_pose(model, data, pose, t)
            heading = heading_degrees(pose)
            camera.lookat[:] = (pose[0], pose[1], .82)
            frame = Image.new("RGB", frame_size, "#f3f5f5")
            draw = ImageDraw.Draw(frame)
            title = TARGET_LABEL if target else "RECORDED PHYSICS / CPU POSE REPLAY"
            draw.rectangle((0, 0, frame_size[0], 36), fill="#8b1525" if target else "#203930")
            draw.text((16, 4), title, font=font, fill="white")
            if commands is not None:
                vx, vy, yaw = commands[index, 1:4]
                draw.text((1120, 4), f"COMMAND vx={vx:+.2f} vy={vy:+.2f} m/s  yaw={yaw:+.2f} rad/s",
                          font=small, fill="white")
            # MuJoCo azimuth 0 puts the eye at -X; +180 views the robot's +X front.
            for pane, (view, offset) in enumerate((("front", 180.), ("side", 90.))):
                camera.azimuth = heading + offset
                renderer.update_scene(data, camera=camera, scene_option=option)
                pixels = renderer.render().copy()
                deviation = float(pixels.std())
                minimum_std[view] = min(minimum_std[view], deviation)
                if deviation < 5:
                    raise ValueError(f"Blank {view} render at t={t:.2f}")
                frame.paste(Image.fromarray(pixels), (pane * WIDTH, TOP))
                draw.text((pane * WIDTH + 16, 42),
                          f"{view.upper()} / root heading {heading:.1f} deg", font=small, fill="#18262b")
            draw.text((16, TOP + HEIGHT + 2),
                      f"t={t:.2f}s | checkpoint iteration {metadata['iteration']} (NPZ)"
                      f" | envId {metadata['environment_id']} | first attempt | 1x | no physics steps",
                      font=small, fill="#18262b")
            return frame

        encoder = subprocess.Popen([
            ffmpeg, "-nostdin", "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{frame_size[0]}x{frame_size[1]}", "-r", str(FPS), "-i", "-",
            "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for index in indices:
                encoder.stdin.write(np.asarray(image_for(int(index))).tobytes())
        finally:
            encoder.stdin.close()
            error = encoder.stderr.read().decode("utf-8", errors="replace")
            encoder.stderr.close()
            returncode = encoder.wait()
        if returncode:
            raise RuntimeError(f"ffmpeg encode failed ({returncode}): {error}")
        for timestamp in (.4, 1.5):
            index = round(timestamp / metadata["control_dt"])
            if timestamp < len(indices) / FPS and index < len(poses):
                name = f"{stem}-{timestamp:.2f}s.png"
                with (output / name).open("xb") as stream:
                    image_for(index).save(stream, format="PNG")
                snapshots[name] = {"source_time_s": index * metadata["control_dt"],
                                   "sha256": digest(output / name)}

    decoded = subprocess.run([
        ffmpeg, "-nostdin", "-v", "error", "-threads", "2", "-i", str(video),
        "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ], check=True, capture_output=True, text=True)
    counts = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines() if line.startswith("frame=")]
    if not counts or counts[-1] != len(indices):
        raise ValueError("Full decode frame count differs from the recorded video")
    if digest(trajectory) != source_hash or (target and digest(waist_reference) != reference_hash):
        raise ValueError("Source changed during rendering")
    result = {
        "schema": "g1-stealth-entry-views-v1", "trajectory": str(trajectory),
        "trajectory_sha256": source_hash, "xml": str(xml), "xml_sha256": digest(xml),
        "video": video.name, "video_sha256": digest(video), "fps": FPS, "frames": len(indices),
        "duration_s": len(indices) / FPS, "frame_size": frame_size, "source": metadata,
        "first_attempt": True, "no_physics_steps": True,
        "renderer": f"MuJoCo CPU FK / {os.environ['MUJOCO_GL']}",
        "physical_evidence": not target, "target_only": target, "visual_acceptance": "unreviewed",
        "camera_views": {"front": "root heading + 180deg", "side": "root heading + 90deg"},
        "camera_convention": "MuJoCo azimuth 0 eye is -X (rear); azimuth 180 eye is +X (front).",
        "decode_check": {"frames": counts[-1], "completed": True},
        "pixel_nonblank": {"minimum_std_per_view": minimum_std, "completed": True},
        "model_contract": contract, "snapshots": snapshots,
        "command_overlay": commands is not None,
        "waist_reference": str(waist_reference) if target else None,
        "waist_reference_sha256": reference_hash, "waist_target": target_info,
        "scope": TARGET_LABEL if target else "Recorded first-episode qpos; no inference, reference substitution or physics steps.",
    }
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=6.)
    parser.add_argument("--waist-reference", type=Path)
    args = parser.parse_args()
    render(args.trajectory, args.output, args.lane, args.seconds, args.waist_reference)


if __name__ == "__main__":
    main()

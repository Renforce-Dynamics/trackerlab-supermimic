"""Record completed terrain policies and render their fixed first attempts."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
if Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json").exists():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

from bootstrap import configure_paths
REPO = configure_paths()
from trackerTask.supermimic.terrain import get_task, generate


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render(directory, level=5, fps=25):
    import imageio_ffmpeg
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from render_demo import font
    from trackerTask.supermimic.mujoco_terrain import build_model

    directory = Path(directory)
    evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    task = get_task(str(trajectory["task"]))
    if task.id != evaluation["task"] or int(trajectory["iteration"]) != evaluation["trained_iterations"]:
        raise ValueError("Evaluation and recorded trajectory differ")
    if not 0 <= level < 6 or fps not in (10, 25, 50):
        raise ValueError("Use a difficulty in 0..5 and a divisor of the 50 Hz control rate")
    if not np.isfinite(trajectory["qpos"]).all() or not np.isfinite(trajectory["scan"]).all():
        raise ValueError("Recorded poses or scans contain nonfinite values")
    width, height, top, bottom = 1280, 720, 70, 52
    scene_height = height - top - bottom
    seed = int(trajectory["seed"])
    terrain = generate(task.profile, task.height, seed)
    model, *_ = build_model(task.robot, terrain, int(trajectory["solver_iterations"]),
                            render_size=(width, scene_height))
    model.vis.headlight.ambient[:] = .32
    model.vis.headlight.diffuse[:] = .55
    model.vis.headlight.specular[:] = .12
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, scene_height, width)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    camera = mujoco.MjvCamera()
    scale = task.height / .76
    camera.distance, camera.elevation, camera.azimuth = 3.5 * scale, -19, 130
    dt = float(trajectory["control_dt"])
    flat_command_task = task.profile in ("velocity", "stealth")
    stride = round(1 / (fps * dt))
    if not math.isclose(stride * dt, 1 / fps, abs_tol=1e-8):
        raise ValueError("Video cadence does not match recorded control time")
    terminal = trajectory["terminal_steps"]
    outcomes = trajectory["outcomes"]
    last = int(terminal[level]) if terminal[level] >= 0 else len(trajectory["qpos"]) - 1
    total_frames = math.ceil(last / stride) + 1 + fps
    output = directory / "demo.mp4"
    encoder = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    previous = None
    changing_frames = 0
    thumbnail = None

    def image_for(index, lane):
        pose = trajectory["qpos"][index, lane]
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        camera.lookat[:] = (pose[0] + .45 * scale, pose[1], pose[2] - .14 * scale)
        renderer.update_scene(data, camera=camera, scene_option=option)
        pixels = renderer.render().copy()
        if pixels.std() < 5:
            raise AssertionError("Rendered terrain scene is blank")
        return Image.fromarray(pixels)

    try:
        for frame_number in range(total_frames):
            index = min(frame_number * stride, last)
            held = frame_number * stride > last
            scene = image_for(index, level)
            pixels = np.asarray(scene)
            if previous is not None and np.abs(pixels.astype(float) - previous).mean() > .01:
                changing_frames += 1
            previous = pixels.astype(float)
            frame = Image.new("RGB", (width, height), "#f2f5f5")
            frame.paste(scene, (0, top))
            draw = ImageDraw.Draw(frame)
            draw.text((18, 9), f"{task.robot.upper()} / {task.profile.upper()}  |  SuperMimic",
                      font=font(25), fill="#18262b")
            draw.text((18, 42), f"Checkpoint {int(trajectory['iteration']):,}  |  seed {seed}"
                      f"  |  difficulty {level + 1}/6  |  first attempt", font=font(17), fill="#42545b")
            scan = trajectory["scan"][index, level].reshape(17, 11)
            value = np.clip((.8 - scan / 2) / 1.6, 0, 1)
            rgb = np.stack((30 + 210 * value, 90 + 120 * value, 180 - 130 * value), -1).astype("uint8")
            panel = Image.fromarray(rgb).resize((88, 136), Image.Resampling.NEAREST)
            draw.rectangle((width - 124, top + 14, width - 16, top + 194), fill="#f2f5f5")
            frame.paste(panel, (width - 114, top + 45))
            draw.text((width - 116, top + 22), "Height scan", font=font(13), fill="#243a41")
            outcome = str(outcomes[level]) if index >= last else "running"
            color = "#197a51" if outcome == "success" else "#b33344" if outcome == "fall" else "#42545b"
            progress = float(trajectory["qpos"][index, level, 0])
            if flat_command_task:
                commands = trajectory.get("command_sequence")
                current = (commands[min(index, len(commands) - 1), level]
                           if commands is not None else trajectory["commands"][level])
                status_line = (f"t={index * dt:.2f}s  |  vx={current[1]:.2f} "
                               f"vy={current[2]:.2f} wz={current[3]:.2f}  |  {outcome}")
            else:
                status_line = f"t={index * dt:.2f}s  |  x={progress:.2f}/{terrain.goal_x:.2f} m  |  {outcome}"
            draw.text((18, height - 45),
                      status_line,
                      font=font(19), fill=color)
            draw.text((18, height - 21), "1x playback  |  MuJoCo-Warp physics  |  fixed attempt; no outcome filtering",
                      font=font(14), fill="#42545b")
            if held:
                draw.rectangle((18, top + 16, 584, top + 54), fill=color)
                draw.text((30, top + 24), f"{outcome.upper()} at {last * dt:.2f}s; terminal pose held",
                          font=font(20), fill="white")
            encoder.stdin.write(np.asarray(frame).tobytes())
            if frame_number == max(1, math.ceil(last / stride) // 2):
                thumbnail = frame.copy()
        commands = trajectory.get("command_sequence")
        stop_command = (flat_command_task and commands is not None
                        and np.abs(commands[:, level, 1:4]).max() < 1e-6)
        if changing_frames < 2 and not stop_command:
            raise AssertionError("Rendered policy has no visible motion")
        if thumbnail is None:
            thumbnail = frame
        thumbnail.save(directory / "demo.png")
        sheet = Image.new("RGB", (1280, 1080), "#f2f5f5")
        for lane in range(6):
            end = int(terminal[lane]) if terminal[lane] >= 0 else len(trajectory["qpos"]) - 1
            index = min(round(4 / dt), end)
            tile = image_for(index, lane).resize((640, 299), Image.Resampling.LANCZOS)
            row = Image.new("RGB", (640, 360), "#f2f5f5")
            row.paste(tile, (0, 40))
            label = ImageDraw.Draw(row)
            label.text((12, 9), f"Difficulty {lane + 1}/6 | {str(outcomes[lane])}"
                       f" | snapshot t={index * dt:.2f}s", font=font(18), fill="#243a41")
            sheet.paste(row, ((lane % 2) * 640, (lane // 2) * 360))
        sheet.save(directory / "demo_contact_sheet.png")
    finally:
        try:
            encoder.stdin.close()
        finally:
            returncode = encoder.wait()
            renderer.close()
    if returncode:
        raise RuntimeError(f"Video encoder exited with {returncode}")
    decoded = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(output),
        "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats",
    ], check=True, capture_output=True, text=True)
    decoded_frames = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines()
                      if line.startswith("frame=")]
    if not decoded_frames or decoded_frames[-1] != total_frames:
        raise AssertionError("Decoded frame count differs from the recorded video")
    result = {
        "task": task.id, "robot": task.robot, "profile": task.profile,
        "checkpoint": evaluation["checkpoint"], "iteration": int(trajectory["iteration"]),
        "checkpoint_sha256": digest(evaluation["checkpoint"]),
        "trajectory_sha256": digest(directory / "trajectories.npz"),
        "terrain_seed": seed, "terrain_seed_held_out": evaluation["terrain_seed_held_out"],
        "physics": "MuJoCo-Warp", "renderer": "MuJoCo EGL pose replay of recorded physical states",
        "fps": fps, "frames": total_frames, "duration_seconds": total_frames / fps,
        "difficulty": level + 1, "attempt_outcome": str(outcomes[level]),
        "attempt_seconds": last * dt, "all_difficulty_outcomes": outcomes.tolist(),
        "selection": f"First episode of environment ID {level}; no outcome filtering",
        "perception": "187 analytic heightfield samples; visual inset uses the noise-free scan",
        "terminal_display": "Pre-reset terminal state held for one second and explicitly labelled",
        "decode_and_pixel_motion_checks": "passed",
        "static_stop_command": bool(stop_command),
    }
    (directory / "demo.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def write_gallery(root, entries, campaign):
    payload = {"created_at": datetime.now(timezone.utc).isoformat(),
               "campaign": campaign, "demos": entries}
    (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    labels = {"stairs": "上楼梯", "mountain": "山地", "parkour": "跑酷"}
    tiles = []
    for row in entries:
        relative = row["directory"]
        score = row.get("test_macro_success")
        baseline_score = row.get("baseline_single_seed_macro_success")
        score_label = (f"{score * 100:.2f}%" if score is not None
                       else f"{baseline_score * 100:.2f}%（旧单种子）"
                       if baseline_score is not None else "尚无独立测试")
        outcome = {"success": "完成", "fall": "摔倒", "timeout": "超时", "horizon": "达到观察时限"}[row["attempt_outcome"]]
        tiles.append(
            f'<article data-robot="{html.escape(row["robot"])}">'
            f'<div class="caption"><h2>{html.escape(row["robot"].upper())} · {labels[row["profile"]]}</h2>'
            f'<span>测试成功率 {score_label}</span></div>'
            f'<video controls playsinline preload="none" poster="{relative}/demo.png">'
            f'<source src="{relative}/demo.mp4" type="video/mp4"></video>'
            f'<div class="detail"><span>最高难度 · 本次{outcome} · {row["attempt_seconds"]:.2f} 秒</span>'
            f'<span>最佳权重 {row["iteration"]:,} 轮</span></div>'
            f'<div class="links"><a href="{relative}/demo.mp4">MP4</a>'
            f'<a href="{relative}/demo_contact_sheet.png">六档难度截图</a>'
            f'<a href="{relative}/demo.json">录制记录</a></div></article>'
        )
    document = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SuperMimic 地形任务 Demo</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f7f8f9;color:#20272b;font:15px/1.55 system-ui,sans-serif;letter-spacing:0}
header{border-bottom:1px solid #d8dddf;padding:24px max(20px,calc((100% - 1440px)/2))}
h1{font-size:26px;line-height:1.3;margin:0 0 9px}header p{margin:5px 0;color:#526167;font-size:14px}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}
select{font:inherit;padding:6px 28px 6px 10px;border:1px solid #c3cdd0;border-radius:4px;background:white}
main{max-width:1480px;margin:auto;padding:0 20px 32px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
article{min-width:0;background:#fff;border:1px solid #d4dcdf;border-radius:6px;overflow:hidden}
.caption{padding:12px 14px;display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap}
h2{font-size:18px;margin:0}.caption span{color:#146849;font-variant-numeric:tabular-nums}
video{display:block;width:100%;aspect-ratio:16/9;background:#e3e9e9;object-fit:contain}
.detail,.links{padding:9px 14px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
.detail{color:#56656a;font-size:13px}.links{border-top:1px solid #edf0f1;justify-content:flex-start;gap:20px}
a{color:#16647d;text-underline-offset:3px}article[hidden]{display:none}
@media(max-width:800px){.grid{grid-template-columns:1fr;gap:18px}h1{font-size:23px}header{padding:20px}
.caption span{font-size:13px}}
</style>
<header><h1>SuperMimic 地形任务 Demo</h1>
<p>真实物理仿真 · 1 倍速 · 固定种子 44 · 最高难度的完整首轮尝试</p>
<p>成绩为两个独立测试种子的分难度平均成功率；A3 上楼梯沿用旧单种子结果。录制尝试保留摔倒与超时，结尾停留一秒。</p></header>
<main><div class="toolbar"><label for="robot">机器人</label>
<select id="robot"><option value="">全部</option><option>g1</option><option>go2</option>
<option>a3</option><option>t1</option><option>bumi</option><option>pi_plus</option></select>
<span id="count"></span></div><div class="grid">__TILES__</div></main>
<script>
const robot=document.querySelector('#robot'),items=[...document.querySelectorAll('article')];
function filter(){let count=0;items.forEach(item=>{item.hidden=Boolean(robot.value&&item.dataset.robot!==robot.value);
if(item.hidden)item.querySelector('video').pause();else count++});document.querySelector('#count').textContent=count+' 个视频'}
robot.addEventListener('change',filter);filter();
document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{if(other!==v)other.pause()})));
</script></html>"""
    document = document.replace("__TILES__", "\n".join(tiles))
    if entries:
        seeds = ", ".join(str(seed) for seed in sorted({row["terrain_seed"] for row in entries}))
        document = document.replace("固定种子 44", f"固定种子 {seeds}")
    (root / "index.html").write_text(document, encoding="utf-8")


def write_portable_gallery(root):
    document = (root / "index.html").read_text(encoding="utf-8")
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["demos"]
    for row in entries:
        for filename, mime in (
            ("demo.mp4", "video/mp4"), ("demo.png", "image/png"),
            ("demo_contact_sheet.png", "image/png"), ("demo.json", "application/json"),
        ):
            relative = f"{row['directory']}/{filename}"
            content = base64.b64encode((root / relative).read_bytes()).decode("ascii")
            uri = f"data:{mime};base64,{content}"
            download = f"{row['robot']}_{row['profile']}_{filename}"
            if filename == "demo.mp4":
                attributes = (f"""download="{download}" href="#" """
                              """onclick="this.href=this.closest('article').querySelector('source').src" """)
                document = document.replace(f'href="{relative}"', attributes)
            else:
                document = document.replace(f'href="{relative}"', f'download="{download}" href="{uri}"')
            document = document.replace(f'"{relative}"', f'"{uri}"')
    (root / "SuperMimic_Terrain_Demos.html").write_text(document, encoding="utf-8")


def write_overview(root):
    import imageio_ffmpeg

    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["demos"]
    if not entries:
        return
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error"]
    for row in entries:
        command.extend(("-i", str(root / row["directory"] / "demo.mp4")))
    inputs = "".join(f"[{index}:v]" for index in range(len(entries)))
    command.extend((
        "-filter_complex_threads", "2",
        "-filter_complex", f"{inputs}concat=n={len(entries)}:v=1:a=0[out]",
        "-map", "[out]", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(root / "SuperMimic_Terrain_Demos.mp4"),
    ))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--portable", action="store_true", help="Embed every video in a standalone HTML file")
    parser.add_argument("--overview", action="store_true", help="Concatenate all demos at their recorded 1x speed")
    parser.add_argument("--gallery-only", action="store_true", help="Rebuild the gallery from an existing manifest")
    parser.add_argument("--directory", type=Path, help="Render an existing recording only")
    args = parser.parse_args()
    if args.directory:
        render(args.directory)
        return
    if args.gallery_only:
        root = args.output.resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        write_gallery(root, manifest["demos"], manifest["campaign"])
        if args.portable:
            write_portable_gallery(root)
        if args.overview:
            write_overview(root)
        return
    if not args.campaign:
        parser.error("--campaign is required for recording")
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    jobs = [job for job in campaign["jobs"] if job.get("phase") == "completed"]
    plan_path = args.campaign.parent / "retrain_plan.json"
    if plan_path.is_file():
        retained = json.loads(plan_path.read_text(encoding="utf-8")).get("retained", {})
        for task_id, row in retained.items():
            jobs.append({"task": task_id, "best_checkpoint": row["checkpoint"],
                         "baseline_single_seed_macro_success": row["metrics"]["macro_success"]})
    if args.tasks:
        jobs = [job for job in jobs if job["task"] in args.tasks]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for job in jobs:
        task = get_task(job["task"])
        directory = root / task.robot / task.profile
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "demo.json"
        if receipt.is_file():
            previous = json.loads(receipt.read_text(encoding="utf-8"))
            if (previous["checkpoint"] != str(Path(job["best_checkpoint"]).resolve())
                    or previous["terrain_seed"] != args.seed
                    or previous["checkpoint_sha256"] != digest(job["best_checkpoint"])):
                raise ValueError(f"Saved demo belongs to different weights or seed: {directory}; use a new output")
        if not receipt.is_file() or not (directory / "demo.mp4").is_file():
            checkpoint = job["best_checkpoint"]
            print(f"Recording {task.id} from {checkpoint}", flush=True)
            subprocess.run([
                sys.executable, "-u", str(Path(__file__).with_name("evaluate_terrain.py")),
                "--checkpoint", checkpoint, "--output", str(directory / "evaluation.json"),
                "--trajectory-output", str(directory / "trajectories.npz"),
                "--num-envs", "96", "--steps", "1500", "--seed", str(args.seed),
            ], cwd=REPO, check=True)
            render(directory)
        result = json.loads(receipt.read_text(encoding="utf-8"))
        result.update(directory=str(directory.relative_to(root)).replace("\\", "/"),
                      test_macro_success=job.get("validation_metrics", {}).get("test_macro_success"),
                      baseline_single_seed_macro_success=job.get("baseline_single_seed_macro_success"))
        entries.append(result)
        write_gallery(root, entries, str(args.campaign.resolve()))
        print(f"Demo completed: {task.id}; {len(entries)}/{len(jobs)}", flush=True)
    if args.portable and entries:
        write_portable_gallery(root)
    if args.overview and entries:
        write_overview(root)


if __name__ == "__main__":
    main()

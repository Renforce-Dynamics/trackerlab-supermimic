"""Evaluate continuous body-command and directional-bar trials in PhysX."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import time


MODES = ("flat", "duck", "jump")
DIRECTIONS = ("front", "front_left", "left", "rear_left",
              "rear", "rear_right", "right", "front_right")
TIMING_FIELDS = (
    "encounter_elapsed_s", "encounter_period_s", "encounter_clearance_delay_s",
    "encounter_deadline_miss", "crossing", "crossing_interval_s", "crossing_interval_valid",
    "encounter_index", "next_encounter_index", "encounter_expired_ahead", "encounter_lateral_bypass",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_task(checkpoint):
    """Restore saved values using the profile's dataclass, without defaulting rapid."""
    from trackerTask.supermimic import go2_velocity_dodge_cfg as configuration

    arguments = checkpoint["args"]
    if arguments.get("task") != "go2_velocity_dodge":
        raise ValueError("Expected a go2_velocity_dodge checkpoint")
    profile = arguments.get("task_profile", "standard")
    if profile not in ("standard", "rapid", "jump", "adaptive"):
        raise ValueError(f"Unknown task profile: {profile}")
    factory = getattr(configuration, "get_task_cfg", None)
    if factory is not None:
        template = factory(profile)
    else:
        name = "RapidDodgeCfg" if profile == "rapid" else "VelocityDodgeCfg"
        template = getattr(configuration, name)()
    saved = checkpoint.get("task_cfg")
    if saved is None:
        raise ValueError("Checkpoint is missing its task_cfg")
    task = type(template)(**(asdict(saved) if is_dataclass(saved) else saved))
    if task.task_feature_dim != 24 or task.scenario_dim != 15:
        raise ValueError("Expected 24 task features and 15 scenario fields")
    if not math.isclose(task.episode_steps * task.control_dt, 16.0, abs_tol=1e-8):
        raise ValueError("This evaluation protocol requires a full 16-second episode")
    return profile, task


def distribution(values):
    import numpy as np

    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values):
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None,
                "max": None, "mean": None}
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite evaluation samples")
    result = dict(zip(("min", "p50", "p90", "p99", "max"),
                      np.quantile(values, (0, .5, .9, .99, 1)).tolist()))
    return {"count": int(len(values)), **result, "mean": float(values.mean())}


def command_metrics(data, mask, dt):
    import numpy as np

    error = data["command_error"][mask]
    if not len(error):
        return {"active_steps": 0, "active_seconds": 0.0, "vx_rmse_m_s": None,
                "vy_rmse_m_s": None, "velocity_rmse_m_s": None, "yaw_rmse_rad_s": None}
    square = np.square(error.astype(float)).mean(axis=0)
    return {"active_steps": int(len(error)), "active_seconds": float(len(error) * dt),
            "vx_rmse_m_s": float(np.sqrt(square[0])), "vy_rmse_m_s": float(np.sqrt(square[1])),
            "velocity_rmse_m_s": float(np.sqrt(square[:2].sum())),
            "yaw_rmse_rad_s": float(np.sqrt(square[2]))}


def encounter_metrics(events):
    obstacles = [event for event in events if event["mode"] != "flat"]
    count = len(obstacles)
    successes = sum(event["success"] for event in obstacles)
    contacts = sum(event["contact"] for event in obstacles)
    falls = sum(event["fall"] for event in obstacles)
    return {
        "segments": len(events), "attempts": count, "successes": successes,
        "failures": count - successes, "success_rate": successes / count if count else None,
        "natural_completions": sum(event["natural_end"] for event in obstacles),
        "timeouts": sum(event["timeout"] for event in obstacles),
        "contacts": contacts, "encounter_contact_rate": contacts / count if count else None,
        "falls": falls, "encounter_fall_rate": falls / count if count else None,
        "segment_falls": sum(event["fall"] for event in events),
        "deadline_misses": sum(event["deadline_miss"] for event in obstacles),
        "terminal_deadline_misses": sum(event["deadline_miss"] and event["terminal"] for event in obstacles),
        "expired_ahead": sum(event["expired_ahead"] for event in obstacles),
        "lateral_bypasses": sum(event["lateral_bypass"] for event in obstacles),
        "censored_attempts": sum(event["censored"] for event in obstacles),
        "failure_reason_counts": dict(Counter(reason for event in obstacles for reason in event["failure_reasons"])),
        "successful_observed_maneuvers": dict(Counter(
            event.get("observed_maneuver", "unmeasured") for event in obstacles if event["success"])),
        "natural_completion_success_rate": (
            successes / sum(event["natural_end"] for event in obstacles)
            if any(event["natural_end"] for event in obstacles) else None),
        "natural_encounter_elapsed_seconds": distribution(
            [event["elapsed_s"] for event in obstacles if event["natural_end"]]),
        "clearance_delay_seconds": distribution(
            [event["clearance_delay_s"] for event in obstacles if event["finished"]]),
        "terminal_attempt_elapsed_seconds": distribution(
            [event["elapsed_s"] for event in obstacles if event["terminal"]]),
        "terminal_attempt_clearance_delay_seconds": distribution(
            [event["clearance_delay_s"] for event in obstacles if event["terminal"]]),
        "minimum_foot_calf_clearance_m": distribution(
            [event["minimum_foot_calf_clearance_m"] for event in obstacles
             if event["minimum_foot_calf_clearance_m"] is not None]),
        "events_with_clearance_samples": sum(event["clearance_sample_count"] > 0 for event in obstacles),
    }


def summarize(data, task, gamma):
    """Use active transitions and count terminal obstacle attempts as failures."""
    import numpy as np

    dt = task.control_dt
    active = data["alive"].astype(bool)
    scenarios = data["scenario"]
    if not np.array_equal(data["encounter_index"][active], scenarios[:, :, 14][active]):
        raise ValueError("Encounter outcome indices differ from the evaluated scenario")
    if not np.array_equal(data["next_encounter_index"][active], data["scenario1"][:, :, 14][active]):
        raise ValueError("Next encounter indices differ from the next scenario")
    direction_ids = np.full(active.shape, -1, dtype=np.int8)
    mode_names = ("flat", "high_bar", "low_bar") if task.autonomous_passage else MODES
    events, trials = [], []
    for trial in range(active.shape[1]):
        active_indices = np.flatnonzero(active[:, trial])
        trial_events = []
        for encounter in np.unique(scenarios[active_indices, trial, 14]).astype(int):
            indices = active_indices[scenarios[active_indices, trial, 14] == encounter]
            first, last = int(indices[0]), int(indices[-1])
            initial = scenarios[first, trial]
            heading = math.atan2(data["root_R"][first, trial, 1, 0],
                                 data["root_R"][first, trial, 0, 0])
            angle = (float(initial[6]) - heading + math.pi) % (2 * math.pi) - math.pi
            direction = int(math.floor((angle + math.pi / 8) / (math.pi / 4))) % 8
            direction_ids[indices, trial] = direction
            mode = mode_names[int(initial[8])]
            natural = bool(data["encounter_natural_end"][last, trial])
            fall = bool(data["terminated"][indices, trial].any())
            contact = bool(data["collision"][indices, trial].any())
            success = bool(data["encounter_success"][last, trial]) if mode != "flat" else None
            timeout = bool(data["encounter_timeout"][last, trial])
            expired_ahead = bool(data["encounter_expired_ahead"][last, trial])
            lateral_bypass = bool(data["encounter_lateral_bypass"][last, trial])
            deadline_miss = bool(data["encounter_deadline_miss"][indices, trial].any())
            crossed = np.flatnonzero(data["crossing"][indices, trial])
            reasons = []
            if mode != "flat" and not success:
                if contact:
                    reasons.append("physical_contact")
                if fall:
                    reasons.append("fall")
                if timeout:
                    reasons.append("episode_timeout")
                if expired_ahead:
                    reasons.append("expired_ahead")
                if lateral_bypass:
                    reasons.append("lateral_bypass")
                if deadline_miss:
                    reasons.append("deadline_miss")
                if not len(crossed):
                    reasons.append("no_crossing")
                if not data["encounter_end"][last, trial]:
                    reasons.append("unfinished_attempt")
            event = {
                "trial": trial, "encounter": int(encounter), "mode": mode,
                "direction": DIRECTIONS[direction], "normal_yaw_relative_to_spawn_heading_deg": math.degrees(angle),
                "start_s": first * dt, "end_s": (last + 1) * dt,
                "elapsed_s": float(data["encounter_elapsed_s"][last, trial]),
                "period_s": float(data["encounter_period_s"][last, trial]),
                "clearance_delay_s": float(data["encounter_clearance_delay_s"][last, trial]),
                "deadline_miss": deadline_miss,
                "command_body_vx_vy_wz": initial[:3].tolist(),
                "bar_height_m": float(initial[5]), "bar_speed_m_s": float(initial[7]),
                "natural_end": natural, "success": success, "contact": contact, "fall": fall,
                "timeout": timeout, "failure_reasons": reasons,
                "finished": bool(data["encounter_end"][last, trial]),
                "terminal": bool(data["done"][last, trial]),
                "expired_ahead": expired_ahead, "lateral_bypass": lateral_bypass,
                "censored": bool(timeout and not natural and not fall),
                "next_encounter": int(data["next_encounter_index"][last, trial]),
                "crossing_s": (int(indices[crossed[0]]) + 1) * dt if len(crossed) else None,
                "peak_bar_contact_N": float(data["contact_force"][indices, trial].max()),
                "control_intervals_with_contact_peak": int(data["collision"][indices, trial].sum()),
                "minimum_foot_calf_clearance_m": (
                    float(data["jump_clearance_m"][indices, trial][
                        data["jump_clearance_valid"][indices, trial].astype(bool)].min())
                    if data["jump_clearance_valid"][indices, trial].any() else None),
                "clearance_sample_count": int(data["jump_clearance_valid"][indices, trial].sum()),
            }
            if "min_foot_height_m" in data:
                near_bar = np.abs(data["root_bar_normal_distance_m"][indices, trial]) <= 0.45
                airborne = (data["min_foot_height_m"][indices, trial] > 0.04) & near_bar
                sustained_airborne = bool((airborne[1:] & airborne[:-1]).any())
                center_step = int(indices[np.argmin(
                    np.abs(data["root_bar_normal_distance_m"][indices, trial]))])
                top_at_crossing = float(data["robot_top_height_m"][center_step, trial])
                under = top_at_crossing < float(initial[5]) - task.bar_half_thickness
                event.update(
                    airborne_geometric_seconds=float(airborne.sum() * dt),
                    peak_min_foot_height_m=float(data["min_foot_height_m"][indices, trial].max()),
                    root_height_min_m=float(data["root_p"][indices, trial, 2].min()),
                    root_height_max_m=float(data["root_p"][indices, trial, 2].max()),
                    robot_top_at_root_crossing_m=top_at_crossing,
                    observed_maneuver=("under" if under else
                                       "airborne_over" if sustained_airborne else
                                       "step_over") if success else "unsuccessful",
                )
            events.append(event)
            trial_events.append(event)
        mask = np.zeros_like(active)
        mask[:, trial] = active[:, trial]
        obstacle_events = [event for event in trial_events if event["mode"] != "flat"]
        fallen = bool(data["terminated"][:, trial].any())
        completed = len(active_indices) == task.episode_steps and not fallen
        completed_events = [event for event in obstacle_events if event["natural_end"]]
        censored = [event["encounter"] for event in obstacle_events if event["censored"]]
        first_fall = np.flatnonzero(data["terminated"][:, trial])
        trials.append({
            "trial": trial, "completed_episode": completed, "fall": fallen,
            "survived_full_episode": completed,
            "completed_events_all_success": (
                all(event["success"] for event in completed_events) if completed_events else None),
            "completed_event_count": len(completed_events), "censored_final_encounters": censored,
            "first_fall_s": (int(first_fall[0]) + 1) * dt if len(first_fall) else None,
            "contact": bool(data["collision"][:, trial].any()),
            "sequence_success": bool(completed and obstacle_events and
                                     all(event["success"] for event in obstacle_events)),
            **command_metrics(data, mask, dt), **encounter_metrics(trial_events),
        })

    per_mode, per_direction, per_direction_mode = {}, {}, {}
    for mode, name in enumerate(mode_names):
        mask = active & (scenarios[:, :, 8] == mode)
        per_mode[name] = {**command_metrics(data, mask, dt),
                          **encounter_metrics([event for event in events if event["mode"] == name])}
    for direction, name in enumerate(DIRECTIONS):
        selected = [event for event in events if event["direction"] == name]
        per_direction[name] = {
            **command_metrics(data, active & (direction_ids == direction), dt),
            **encounter_metrics(selected),
        }
        per_direction_mode[name] = {
            mode: encounter_metrics([event for event in selected if event["mode"] == mode])
            for mode in mode_names[1:]
        }
    raw_values = data["raw_value"][active]
    bound = task.value_bound(gamma)
    crossings = active & data["crossing_interval_valid"].astype(bool) & data["crossing"].astype(bool)
    finished = active & data["encounter_natural_end"].astype(bool)
    return {
        "overall": {
            "trials": len(trials), "completed_episodes": sum(t["completed_episode"] for t in trials),
            "survived_full_episode_trials": sum(t["survived_full_episode"] for t in trials),
            "completed_events_all_success_trials": sum(t["completed_events_all_success"] is True for t in trials),
            "trials_with_completed_events": sum(t["completed_event_count"] > 0 for t in trials),
            "sequence_successes": sum(t["sequence_success"] for t in trials),
            "sequence_success_rate": sum(t["sequence_success"] for t in trials) / len(trials),
            "fall_rate": sum(t["fall"] for t in trials) / len(trials),
            "contact_rate": sum(t["contact"] for t in trials) / len(trials),
            **command_metrics(data, active, dt), **encounter_metrics(events),
            "root_crossing_interval_seconds": distribution(data["crossing_interval_s"][crossings]),
            "actual_natural_encounter_interval_seconds": distribution(data["encounter_elapsed_s"][finished]),
            "max_bar_pose_error_m": float(data["bar_pose_error"][active].max()),
            "max_bar_rotation_error": float(data["bar_rotation_error"][active].max()),
        },
        "per_mode": per_mode, "per_direction": per_direction,
        "per_direction_mode": per_direction_mode, "per_trial": trials, "encounters": events,
        "raw_value_quantiles": distribution(raw_values),
        "theoretical_value_bound": bound,
        "raw_value_above_theoretical_bound_fraction": float((raw_values > bound).mean()),
        "mean_trial_velocity_rmse_m_s": float(np.mean([t["velocity_rmse_m_s"] for t in trials])),
        "mean_trial_yaw_rmse_rad_s": float(np.mean([t["yaw_rmse_rad_s"] for t in trials])),
    }


def evaluate(args, app_launcher):
    import numpy as np
    import torch
    import trackerLab.algorithms.supermimic.networks as networks
    from trackerLab.algorithms.supermimic.networks import Critic
    from trackerLab.algorithms.supermimic.velocity_networks import VelocityPolicy
    from trackerLab.kinematics.supermimic.robot import RobotKinematics
    from trackerLab.kinematics.supermimic.rotations import mat_to_quat
    from trackerLab.motion_buffer.motion_lib.supermimic import MotionLib
    from trackerLab.tracker_env.mdp.supermimic import RewardCfg, normalize_xi, unflatten_state
    from trackerLab.tracker_env.mdp.supermimic_velocity_dodge import (
        VelocityDodgeMDP, bar_pose, make_scenario, reward_term_names,
    )
    from trackerLab.tracker_env.supermimic.env import make_sim, launch_simulation
    from trackerLab.tracker_env.supermimic.velocity_dodge import VelocityDodgeSim
    from trackerTask.supermimic.go2_velocity_dodge_cfg import make_task_cfg
    from trackerTask.supermimic.robots import get_robot

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    profile, task = checkpoint_task(checkpoint)
    TERMS = reward_term_names(task)
    config = checkpoint["args"]
    iteration = int(checkpoint["iteration"])
    gamma = float(config["gamma"])
    networks.AMP_DTYPE = torch.bfloat16 if config.get("amp") == "bf16" else None
    spec, cfg = get_robot("go2"), make_task_cfg()
    cfg.num_envs, cfg.max_episode_steps = args.trials, task.episode_steps
    cfg.env_spacing, cfg.reset_noise = 64.0, False
    kin = RobotKinematics(args.device, spec)
    motion_path = Path(args.motion_config or config["motion_config"])
    motion = MotionLib.from_config(str(motion_path), kin, args.device)
    mdp = VelocityDodgeMDP(kin, motion, torch.tensor(spec.default_q, device=args.device),
                          cfg=RewardCfg(**cfg.reward_overrides), obs_mode=config["obs_mode"],
                          task_cfg=task)
    goal_dim = spec.goal_frame_dim * config["goal_frames"]
    pi = VelocityPolicy(mdp.obs_dim, goal_dim, hist=config["hist"],
                        action_limit=config["action_limit"]).to(args.device).eval()
    critic = Critic(spec.state_dim + spec.context_dim + task.task_feature_dim + goal_dim).to(args.device).eval()
    for network, name in ((pi, "pi"), (critic, "critic")):
        network.load_state_dict(checkpoint[name])
        if not all(bool(torch.isfinite(value).all()) for value in network.state_dict().values()):
            raise FloatingPointError(f"Nonfinite {name} checkpoint")
    transfer = checkpoint.get("transfer")
    del checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    sim_cfg = make_sim(args.device, task_cfg=cfg)
    trajectory = {}
    started = time.monotonic()
    with launch_simulation(sim_cfg, args, app_launcher=app_launcher), torch.no_grad():
        env = VelocityDodgeSim(sim_cfg, args.trials, motion, mdp, cfg, eval_mode=True, gamma=gamma)
        env.difficulty = args.difficulty
        ids = torch.arange(args.trials, device=args.device)
        clip_length = motion.clip_end[0] - motion.clip_start[0] + 1
        frame_ids = motion.clip_start[0] + (ids % 4) * clip_length // 4
        seeds = ids + (args.seed % (2**20 - 32)) + 1
        env.reset(frame_ids=frame_ids.long())
        env.scenario[:] = make_scenario(env.state(), args.difficulty, seeds, task)
        env._teleport_bar(env.scenario, ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)
        trial_manifest = [
            {"trial": i, "seed": int(seeds[i]), "reference_frame": int(frame_ids[i]),
             "initial_scenario": env.scenario[i].cpu().tolist(),
             "initial_root_position": env.state()[0][i].cpu().tolist()}
            for i in range(args.trials)
        ]
        save_json(args.output / "trial_manifest.json", {
            "seed": args.seed, "trial_count": args.trials, "episode_seconds": 16.0,
            "task_profile": profile, "task_config": asdict(task), "trials": trial_manifest,
            "selection": "Fixed IDs, seeds and reference phases declared before policy rollout",
            "schedule": "Deterministic encounter draws; world placement depends on simulated state",
        })
        history = None
        for step in range(task.episode_steps):
            S = env.state()
            active = ~env.done_eval.clone()
            scenario = env.scenario.clone()
            observation = mdp.obs(S, env.a_prev, env.xi, env.f, scenario)
            history = (observation[:, None].repeat(1, config["hist"], 1) if history is None else
                       torch.cat((history[:, 1:], observation[:, None]), 1))
            goal = motion.goal(env.f, config["goal_frames"])
            action = pi(history, goal, mdp.task_features(S, scenario))
            raw_value = critic(torch.cat((
                mdp.task_critic_features(S, normalize_xi(env.xi), scenario), goal), -1))
            actual_bar = env.scene["bar"].data
            row = {
                "root_p": S[0].clone(), "root_R": S[1].clone(), "root_quat": mat_to_quat(S[1]),
                "root_v": S[2].clone(), "root_w": S[3].clone(), "q": S[4].clone(), "qd": S[5].clone(),
                "scenario": scenario, "alive": active, "raw_value": raw_value,
                "bar_pose": bar_pose(scenario),
                "bar_pose_actual": torch.cat((actual_bar.root_link_pos_w - env.env_origins,
                                              actual_bar.root_link_quat_w), -1).clone(),
            }
            out = env.step(action)
            S1 = unflatten_state(out["s1"])
            body_v = (S1[1].transpose(-1, -2) @ S1[2].unsqueeze(-1)).squeeze(-1)
            body_w = (S1[1].transpose(-1, -2) @ S1[3].unsqueeze(-1)).squeeze(-1)
            measured = torch.cat((body_v[:, :2], body_w[:, 2:3]), -1)
            row.update(
                command=out["reward_scenario"][:, :3], measured_body_velocity=measured,
                command_error=measured - out["reward_scenario"][:, :3],
                contact_force=out["bar_contact_force"],
                reward_terms=torch.stack([out["reward_terms"][name] for name in TERMS], -1),
            )
            clearance, clearance_valid = mdp.jump_clearance_metric(S1, out["reward_scenario"])
            row["jump_clearance_m"] = clearance
            row["jump_clearance_valid"] = clearance_valid
            if task.autonomous_passage:
                from trackerLab.tracker_env.mdp.supermimic_velocity_dodge import crossing_state
                points, radii, feet = mdp.passage_geometry(S1)
                row["min_foot_height_m"] = feet[..., 2].amin(-1)
                row["robot_top_height_m"] = (points[..., 2] + radii).amax(-1)
                row["root_bar_normal_distance_m"] = crossing_state(S1, out["reward_scenario"])[0]
            for name in (
                "s", "s1", "a", "a_prev", "xi", "f", "r", "scenario1", "reward_scenario",
                "terminated", "truncated", "done", "collision", "encounter_end",
                "encounter_success", "encounter_natural_end", "encounter_failure",
                "encounter_timeout", "bar_pose_error", "bar_rotation_error", *TIMING_FIELDS,
            ):
                row[name] = out[name]
            for name, value in row.items():
                sample = value.detach().float().cpu().numpy() if value.dtype == torch.bfloat16 else value.detach().cpu().numpy()
                if not np.isfinite(sample).all():
                    raise FloatingPointError(f"Nonfinite {name} at evaluation step {step}")
                trajectory.setdefault(name, []).append(sample.copy())
            if (step + 1) % 200 == 0:
                print(f"Velocity evaluation {step + 1}/{task.episode_steps} steps; "
                      f"{int((~env.done_eval).sum())}/{args.trials} active", flush=True)
        data = {name: np.stack(values) for name, values in trajectory.items()}
        result = {
            "task": "go2_velocity_dodge", "task_profile": profile, "iteration": iteration,
            "difficulty": args.difficulty, "checkpoint_sha256": digest(args.checkpoint),
            "transfer": transfer, **summarize(data, task, gamma),
            "evaluation_seconds": time.monotonic() - started,
            "protocol": {
                "simulator": "IsaacLab / PhysX", "episode_seconds": 16.0,
                "control_dt": task.control_dt, "physics_dt": env.physics_dt,
                "control_decimation": env.control_decimation, "seed": args.seed,
                "task_config": asdict(task), "task_config_class": type(task).__name__,
                "gamma": gamma, "reward_terms": list(TERMS),
                "value_bound_derivation": "sum(positive reward maxima) / (1 - gamma)",
                "action_noise": False, "observation_noise": False, "domain_randomization": False,
                "friction": 1.0, "environment_spacing_m": cfg.env_spacing,
                "robot_resets_after_initialization": 0, "history_resets_at_encounters": 0,
                "command_frame": "Full root-body rotation: R.T @ root world linear/angular velocity",
                "contact_rule": "Peak bar net-force norm over physics substeps > 1 N; obstacle modes only",
                "contact_sensor_limit": "Net force can combine or cancel simultaneous contacts",
                "success": "Natural encounter end, normal-axis root crossing > 0.45 m within bar span, "
                           "no contact above threshold and no fall; terminal incomplete attempts fail",
                "sequence_success": "Survive all 16 seconds and succeed on every attempted obstacle",
                "censoring": "An unfinished bar at the 16-second horizon remains a failed attempt. "
                             "completed_events_all_success covers naturally ended events only (null if none); "
                             "survived_full_episode is reported independently",
                "jump_label": "Jump category; per-frame signed foot/calf clearance near crossing is recorded",
                "action_selection": ("Policy sees bar geometry; no jump/duck category or prescribed "
                                     "maneuver height is supplied" if task.autonomous_passage else
                                     "Policy receives scenario category and desired root height"),
                "maneuver_measurement": (
                    "Successful events only: under uses robot-top height at nearest root crossing; "
                    "airborne_over uses all four kinematic foot centers above 0.04 m for two "
                    "consecutive control samples within 0.45 m of the bar plane; "
                    "other successful events are step_over. "
                    "This is geometric classification, not foot-force measurement."),
                "direction_frame": "Bar normal relative to robot heading at encounter spawn",
                "intervals": "Actual natural encounter duration and valid within-episode first-crossing intervals",
                "inactive_states": "Retained for audit; metrics exclude steps after termination",
                "recorded_trials": list(range(args.trials)), "video_trial": 0,
                "video_selection": "Trial 0 selected before outcomes; failure example is optional and labeled",
                "trial_manifest_sha256": digest(args.output / "trial_manifest.json"),
                "motion_config": str(motion_path), "motion_config_sha256": digest(motion_path),
                "reference_motion_names": motion.names, "evaluator_sha256": digest(__file__),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(args.device)
                if str(args.device).startswith("cuda") else "cpu",
            },
        }
        np.savez_compressed(
            args.output / "trajectories.npz", **data, iteration=np.array(iteration),
            task_profile=np.array(profile), difficulty=np.array(args.difficulty),
            control_dt=np.array(task.control_dt), trial_ids=np.arange(args.trials),
            time_s=np.arange(task.episode_steps) * task.control_dt,
        )
        save_json(args.output / "evaluation.json", result)
        print(json.dumps({"iteration": iteration, "task_profile": profile,
                          "overall": result["overall"]}, indent=2), flush=True)


def main():
    from bootstrap import configure_paths

    configure_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion_config", type=Path)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--trials", type=int, choices=(8, 16), default=16)
    parser.add_argument("--seed", type=int, default=20261023)
    # Keep dependency-free help available on machines without IsaacLab.
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, device="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.difficulty <= 1:
        parser.error("--difficulty must lie in [0, 1]")
    evaluate(args, AppLauncher(args))


if __name__ == "__main__":
    main()

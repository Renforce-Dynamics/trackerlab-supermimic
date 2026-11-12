"""Plan cumulative stealth budgets and an explicit 5,000-update continuation."""

import argparse
import hashlib
import json
from pathlib import Path


WARMUP_ITERATIONS = 500
ACTOR_UPDATES = 50000
TARGET_ITERATION = WARMUP_ITERATIONS + ACTOR_UPDATES
REVIEW_ITERATIONS = (2500, 5500, 10500, 20500, 30500, 40500, TARGET_ITERATION)


def plan(start_iteration=0, *, actor_updates=ACTOR_UPDATES):
    if type(actor_updates) is not int or actor_updates not in (5000, 20000, 25000, ACTOR_UPDATES):
        raise ValueError("Supported actor budgets are 5000, 20000, 25000 and 50000 updates")
    final = WARMUP_ITERATIONS + actor_updates
    if (type(start_iteration) is not int or start_iteration < 0
            or start_iteration >= final):
        raise ValueError(f"Combined run start must be an integer before iteration {final}")
    completed_actor = max(0, start_iteration - WARMUP_ITERATIONS)
    reviews = (final,) if actor_updates == 5000 else (*REVIEW_ITERATIONS, final)
    return {
        "starting_iteration": start_iteration,
        "target_iteration": final,
        "actor_update_budget": actor_updates,
        "actor_updates_already_completed": completed_actor,
        "remaining_actor_updates": actor_updates - completed_actor,
        "remaining_world_only_iterations": max(0, WARMUP_ITERATIONS - start_iteration),
        "milestones": [value for value in sorted(set(reviews))
                       if start_iteration < value <= final],
        "quality_failure_action": "record_and_continue_authorized_budget",
        "visual_acceptance": False,
    }


def actor_budget(recipe, extra_actor_updates=0):
    if type(extra_actor_updates) is not int or extra_actor_updates not in (0, 5000):
        raise ValueError("Additional actor updates must be 0 or 5000")
    if extra_actor_updates and recipe != "beyondamp":
        raise ValueError("The extra 5000 updates require the BeyondAMP recipe")
    if recipe == "fullbody":
        return 5000
    return (20000 if recipe == "beyondamp" else ACTOR_UPDATES) + extra_actor_updates


def validate_fullbody_resume(checkpoint, pool):
    """Bind a bounded continuation to its preset, learned controller and pool."""
    preset_path = Path(__file__).parent / "configs/command_stealth_full_body.json"
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    args = checkpoint["arguments"]
    expected = preset | {
        "task": "locomotion/g1/stealth",
        "command_training_modes": None, "command_reward_mode": None,
        "stealth_command": None, "upper_body_reference": None, "waist_entry_reference": None,
        "allow_command_sampling_change": False, "allow_command_reward_change": False,
        "allow_style_feature_change": False, "allow_support_gradient_change": False,
        "warm_start_normalizer_prior_count": 0.,
    }
    if (checkpoint.get("robot") != "g1"
            or Path(args.get("config") or "").name != preset_path.name
            or any(name not in args or args[name] != value for name, value in expected.items())):
        raise ValueError("Fullbody resume requires the matching preset and target height")
    from bootstrap import configure_paths
    configure_paths()
    from trackerTask.supermimic.robots import get_robot
    from trackerLab.algorithms.supermimic.stealth_action_filter import UpperBodyRateLimiter
    spec = get_robot("g1")
    controller = UpperBodyRateLimiter(
        spec.joint_names, spec.action_scale,
        max_joint_rate_rad_s=preset["upper_body_rate"],
        max_joint_acceleration_rad_s2=preset["upper_body_acceleration"],
        control_dt=.02, action_limit=preset["action_limit"]).to_dict()
    if checkpoint["upper_body_rate_limit"] != controller:
        raise ValueError("Fullbody resume requires the matching learned 29-action controller")
    if pool is None:
        raise ValueError("Fullbody resume requires --pool to verify saved source state")
    pool = Path(pool).resolve()
    # The entry guidance already saves a digest of the complete source manifest.
    # The dedicated pool preflight and trainer validate its actual motion bytes.
    manifest_bytes = (pool / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    aim_bytes = (pool / "aim-style-reference.json").read_bytes()
    aim = json.loads(aim_bytes)
    auxiliary = manifest.get("auxiliary_files", {}).get("aim_style", {})
    entry, style = checkpoint["entry_style_guidance"], checkpoint["amp_style"]
    source = entry.get("source")
    expert_hash = style.get("expert_hash")
    if (not (pool / "full-body-style.json").is_file()
            or not isinstance(source, dict)
            or source.get("pool_manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
            or entry.get("weight") != preset["entry_style_weight"]
            or entry.get("duration") != preset["entry_duration_s"]
            or not isinstance(expert_hash, str) or len(expert_hash) != 64
            or any(char not in "0123456789abcdef" for char in expert_hash)
            or entry.get("expert_hash") != expert_hash
            or style.get("feature_mode") != preset["style_feature_mode"]
            or style.get("history_steps") != preset["amp_history"]
            or checkpoint["aim_style_guidance"] != aim
            or auxiliary.get("file") != "aim-style-reference.json"
            or auxiliary.get("sha256") != hashlib.sha256(aim_bytes).hexdigest()
            or aim != manifest.get("recipe", {}).get("guidance")
            or aim.get("source") != manifest.get("recipe", {}).get("source")):
        raise ValueError("Fullbody resume source or saved style state differs from the pool")
    forward = [row for row in manifest.get("files", ())
               if row.get("split") == "train" and row.get("clip_role") == "forward"]
    if (len(forward) != 1 or source.get("forward_csv") != forward[0].get("file")
            or source.get("forward_csv_sha256") != forward[0].get("sha256")):
        raise ValueError("Fullbody resume entry source differs from the forward expert")


def checkpoint_start(checkpoint, recipe="combined", *, extra_actor_updates=0, pool=None):
    budget = actor_budget(recipe, extra_actor_updates)
    args = checkpoint.get("arguments", {})
    expected = {
        "wm_only_iters": WARMUP_ITERATIONS,
        "stealth_control_mode": "wasd",
        "style_feature_mode": "postural_end_effectors_vx",
        "entry_duration_s": 1.21,
        "entry_style_weight": .008,
        "waist_pd_scale": 1.,
        "freeze_actor_normalization": True,
    }
    if recipe in ("velocity-led", "shared-aim", "beyondamp", "fullbody"):
        expected.update(entry_duration_s=.60, wasd_reward_version=3,
                        stealth_gait_weight=.02, amp_weight=.05 if recipe == "velocity-led" else .1)
        if recipe in ("beyondamp", "fullbody"):
            expected.update(stealth_gait_version=2, stealth_swing_height=.04)
    elif recipe != "combined":
        raise ValueError("Unknown stealth recipe")
    if (checkpoint.get("task") != "locomotion/g1/stealth"
            or checkpoint.get("training_stage") != "command-style"
            or not isinstance(args, dict)
            or any(args.get(name) != value for name, value in expected.items())
            or checkpoint.get("stealth_upper_body_constraint") is not None
            or checkpoint.get("stealth_waist_constraint") is not None
            or not isinstance(checkpoint.get("upper_body_rate_limit"), dict)
            or not isinstance(checkpoint.get("entry_style_guidance"), dict)):
        raise ValueError("Resume requires the matching autonomous combined-stealth checkpoint")
    if recipe in ("shared-aim", "beyondamp", "fullbody") and not isinstance(
            checkpoint.get("aim_style_guidance"), dict):
        raise ValueError("Shared aim resume requires saved aim guidance")
    if recipe in ("beyondamp", "fullbody") and not isinstance(
            checkpoint["aim_style_guidance"].get("palm_separation"), dict):
        raise ValueError("BeyondAMP recipe requires saved palm separation geometry")
    for key in ("actor", "world", "critic", "opt_pi", "opt_wm", "opt_v", "amp_style"):
        if not isinstance(checkpoint.get(key), dict):
            raise ValueError(f"Full combined resume is missing {key}")
    if recipe == "fullbody":
        validate_fullbody_resume(checkpoint, pool)
    iteration = checkpoint.get("iteration")
    plan(iteration, actor_updates=budget)
    if iteration < 1:
        raise ValueError("Resume requires a saved completed iteration")
    return iteration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recipe", choices=("combined", "velocity-led", "shared-aim", "beyondamp",
                                             "fullbody"),
                        default="combined")
    parser.add_argument("--pool", type=Path,
                        help="Current fullbody pool for saved-source validation on resume")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-actor-updates", type=int, choices=(0, 5000), default=0)
    args = parser.parse_args(argv)
    budget = actor_budget(args.recipe, args.extra_actor_updates)
    if args.extra_actor_updates and args.resume is None:
        parser.error("Extra actor updates require a full --resume checkpoint")
    start = 0
    if args.resume is not None:
        import torch
        start = checkpoint_start(
            torch.load(args.resume, map_location="cpu", weights_only=False), args.recipe,
            extra_actor_updates=args.extra_actor_updates, pool=args.pool)
    result = plan(start, actor_updates=budget)
    if args.extra_actor_updates:
        result["additional_actor_update_budget"] = args.extra_actor_updates
        result["continuation"] = "full checkpoint resume; unchanged training objective"
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(start, result["target_iteration"], *result["milestones"])


if __name__ == "__main__":
    main()

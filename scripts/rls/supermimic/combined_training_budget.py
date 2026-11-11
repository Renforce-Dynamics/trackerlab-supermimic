"""Plan cumulative 20,000 or 50,000 actor-update stealth experiments."""

import argparse
import json
from pathlib import Path


WARMUP_ITERATIONS = 500
ACTOR_UPDATES = 50000
TARGET_ITERATION = WARMUP_ITERATIONS + ACTOR_UPDATES
REVIEW_ITERATIONS = (2500, 5500, 10500, 20500, 30500, 40500, TARGET_ITERATION)


def plan(start_iteration=0, *, actor_updates=ACTOR_UPDATES):
    if type(actor_updates) is not int or actor_updates not in (20000, ACTOR_UPDATES):
        raise ValueError("Supported actor budgets are 20000 and 50000 updates")
    final = WARMUP_ITERATIONS + actor_updates
    if (type(start_iteration) is not int or start_iteration < 0
            or start_iteration >= final):
        raise ValueError(f"Combined run start must be an integer before iteration {final}")
    completed_actor = max(0, start_iteration - WARMUP_ITERATIONS)
    return {
        "starting_iteration": start_iteration,
        "target_iteration": final,
        "actor_update_budget": actor_updates,
        "actor_updates_already_completed": completed_actor,
        "remaining_actor_updates": actor_updates - completed_actor,
        "remaining_world_only_iterations": max(0, WARMUP_ITERATIONS - start_iteration),
        "milestones": [value for value in REVIEW_ITERATIONS if start_iteration < value <= final],
        "quality_failure_action": "record_and_continue_authorized_budget",
        "visual_acceptance": False,
    }


def checkpoint_start(checkpoint, recipe="combined"):
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
    if recipe in ("velocity-led", "shared-aim", "beyondamp"):
        expected.update(entry_duration_s=.60, wasd_reward_version=3,
                        stealth_gait_weight=.02, amp_weight=.05 if recipe == "velocity-led" else .1)
        if recipe == "beyondamp":
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
    if recipe in ("shared-aim", "beyondamp") and not isinstance(checkpoint.get("aim_style_guidance"), dict):
        raise ValueError("Shared aim resume requires saved aim guidance")
    if recipe == "beyondamp" and not isinstance(
            checkpoint["aim_style_guidance"].get("palm_separation"), dict):
        raise ValueError("BeyondAMP recipe requires saved palm separation geometry")
    for key in ("actor", "world", "critic", "opt_pi", "opt_wm", "opt_v", "amp_style"):
        if not isinstance(checkpoint.get(key), dict):
            raise ValueError(f"Full combined resume is missing {key}")
    iteration = checkpoint.get("iteration")
    plan(iteration, actor_updates=20000 if recipe == "beyondamp" else ACTOR_UPDATES)
    if iteration < 1:
        raise ValueError("Resume requires a saved completed iteration")
    return iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recipe", choices=("combined", "velocity-led", "shared-aim", "beyondamp"),
                        default="combined")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = 0
    if args.resume is not None:
        import torch
        start = checkpoint_start(
            torch.load(args.resume, map_location="cpu", weights_only=False), args.recipe)
    result = plan(start, actor_updates=20000 if args.recipe == "beyondamp" else ACTOR_UPDATES)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(start, result["target_iteration"], *result["milestones"])


if __name__ == "__main__":
    main()

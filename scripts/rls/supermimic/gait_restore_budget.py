"""Require five thousand actor updates after the selected recovery state."""

import argparse
import json
from pathlib import Path


ADDITIONAL_ACTOR_UPDATES = 5000
WARMUP_ITERATIONS = 500


def plan(start_iteration=0):
    if (type(start_iteration) is not int or start_iteration < 0
            or start_iteration % 500 or (0 < start_iteration < WARMUP_ITERATIONS)):
        raise ValueError("Recovery starts must be zero or a saved post-warmup 500-iteration milestone")
    warmup = WARMUP_ITERATIONS if start_iteration == 0 else 0
    return {
        "starting_iteration": start_iteration,
        "remaining_world_only_iterations": warmup,
        "additional_actor_updates": ADDITIONAL_ACTOR_UPDATES,
        "target_iteration": start_iteration + warmup + ADDITIONAL_ACTOR_UPDATES,
        "milestones": [start_iteration + warmup + 2000,
                       start_iteration + warmup + ADDITIONAL_ACTOR_UPDATES],
        "quality_failure_action": "record_all_results_and_continue_authorized_budget",
        "invalid_evidence_action": "stop_with_receipts",
        "visual_acceptance": False,
    }


def checkpoint_start(checkpoint):
    if (checkpoint.get("task") != "locomotion/g1/stealth"
            or checkpoint.get("training_stage") != "command-style"
            or checkpoint.get("arguments", {}).get("wm_only_iters") != WARMUP_ITERATIONS):
        raise ValueError("Require the matching G1 recovery objective and completed warmup")
    iteration = checkpoint.get("iteration")
    if type(iteration) is not int or iteration < WARMUP_ITERATIONS:
        raise ValueError("Resume requires completed world-model warmup")
    for key in ("actor", "world", "critic", "opt_pi", "opt_wm", "opt_v", "amp_style"):
        if not isinstance(checkpoint.get(key), dict):
            raise ValueError(f"Full-state resume is missing {key}")
    plan(iteration)
    return iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = 0
    if args.resume is not None:
        import torch
        start = checkpoint_start(torch.load(args.resume, map_location="cpu", weights_only=False))
    result = plan(start)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(start, result["target_iteration"], *result["milestones"])


if __name__ == "__main__":
    main()

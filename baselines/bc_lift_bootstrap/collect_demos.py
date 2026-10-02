"""Collect prefixes ending at sustained native success; never relabel transients."""
from common import default_source_model, model_path, package_versions, sha256, git_sha, verify_stack, device
from env import DEFAULT_ASSIST_DISTANCE, environment_contract, make_lift_env
from dataset import pack_episodes, save_dataset, validate_dataset, extract_demo, first_sustained_interval
from rollout import run_episode

import argparse
import hashlib
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from tqdm import tqdm


def collect(args):
    verify_stack()
    if args.successful_episodes <= 0 or args.max_attempts <= 0:
        raise ValueError("Episode/attempt counts must be positive")
    first_sustained_interval([], args.success_hold_steps)
    path = model_path(args.model)
    if args.output.suffix != ".npz":
        raise ValueError("Output must be .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    env = make_lift_env(args.seed, assist_distance=args.assist_distance)
    try:
        contract = environment_contract(env, assist_distance=args.assist_distance)
        model = SAC.load(path, env=env, device=device(args.device))
    finally:
        env.close()
    source_config = path.parent / "config.json"
    if not source_config.exists():
        raise FileNotFoundError("Source model needs its saved config.json to verify stage-1 assistance")
    import json
    config = json.loads(source_config.read_text())
    if config.get("curriculum_stage") != 1 or config.get("assistance", {}).get("assist_distance_m") != args.assist_distance:
        raise ValueError("Source policy assistance configuration does not match collection")
    episodes, seeds, attempts, results, descriptions = [], [], [], [], []
    rejected_transient = 0
    transitions = 0
    with tqdm(total=args.successful_episodes, desc="Collect demos", unit="successful", dynamic_ncols=True) as progress:
        for attempt in range(args.max_attempts):
            trajectory, result = run_episode(model, args.seed+attempt, assist_distance=args.assist_distance)
            extracted = extract_demo(trajectory, args.success_hold_steps)
            result["sustained_success"] = extracted is not None
            if extracted is not None:
                demo, sustained_start, length = extracted
                episode_id = len(episodes)
                demo["episode_ids"] = np.full(length, episode_id, dtype=np.int32)
                demo["ever_success"] = np.ones(length, dtype=bool)
                descriptions.append({"episode_id": episode_id, "seed": args.seed+attempt, "source_attempt": attempt,
                                     "initial_success": result["initial_success"], "first_success_step": result["first_success_step"],
                                     "sustained_success_start": sustained_start, "success_hold_steps": args.success_hold_steps,
                                     "saved_transition_count": length, "verified_rollout_length": result["episode_length"],
                                     "saved_final_observation_sha256": hashlib.sha256(np.asarray(demo["next_observations"][-1], dtype=np.float32).tobytes()).hexdigest(),
                                     "source_final_success": result["final_success"],
                                     "saved_end_terminated": bool(demo["terminated"][-1]), "saved_end_truncated": bool(demo["truncated"][-1])})
                episodes.append(demo); seeds.append(args.seed+attempt); attempts.append(attempt)
                transitions += length
                progress.update(1)
            elif result["ever_success"]:
                rejected_transient += 1
            results.append(result)
            progress.set_postfix(attempts=attempt+1, success_rate=f"{100*len(episodes)/(attempt+1):.1f}%", transitions=transitions, rejected_transient=rejected_transient)
            if len(episodes) == args.successful_episodes:
                break
    complete = len(episodes) == args.successful_episodes
    metadata = {"schema_version": 2, "status": "COLLECTION_COMPLETE" if complete else "COLLECTION_INCOMPLETE",
                "packages": package_versions(), "git_commit": git_sha(), "seed": args.seed,
                "assist_distance": args.assist_distance, "obs_dim": contract["obs_dim"], "action_dim": contract["action_dim"],
                "environment": contract, "number_of_successful_episodes": len(episodes), "number_of_transitions": transitions,
                "requested_successful_episodes": args.successful_episodes, "attempts": len(results), "max_attempts": args.max_attempts,
                "source_policy_success_rate": len(episodes)/len(results),
                "source_policy_success_rate_definition": "sustained native success",
                "source_policy_ever_success_rate": sum(r["ever_success"] for r in results)/len(results),
                "success_hold_steps": args.success_hold_steps, "rejected_transient_success": rejected_transient,
                "demonstrations": descriptions, "episode_seeds": seeds, "source_attempts": attempts,
                "source_model_path": str(path), "source_model_sha256": sha256(path), "source_config": config,
                "collection_results": results, "labels": "actual actions captured immediately below assistance wrapper",
                "environment_source_sha256": sha256(Path(__file__).parent / "assisted_env.py")}
    data = pack_episodes(episodes, metadata)
    validate_dataset(data, metadata, allow_empty=True)
    save_dataset(args.output, data, metadata)
    print(f"{metadata['status']} episodes={len(episodes)}/{args.successful_episodes} attempts={len(results)} transitions={metadata['number_of_transitions']} output={args.output}")
    return 0 if complete else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=default_source_model())
    p.add_argument("--success-hold-steps", type=int, default=10)
    p.add_argument("--successful-episodes", type=int, default=50)
    p.add_argument("--max-attempts", type=int, default=1000)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--output", type=Path, default=Path("data/lift_sustained_50_hold10.npz"))
    p.add_argument("--assist-distance", type=float, default=DEFAULT_ASSIST_DISTANCE)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    raise SystemExit(collect(p.parse_args()))


if __name__ == "__main__":
    main()

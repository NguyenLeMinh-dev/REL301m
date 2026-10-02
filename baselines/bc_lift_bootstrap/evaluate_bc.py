"""Fixed new-seed real-environment deterministic BC evaluation."""
from common import device, model_path, verify_stack, write_json, package_versions, git_sha, sha256
from rollout import run_episode
from env import DEFAULT_ASSIST_DISTANCE

import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from tqdm import tqdm


def evaluate(args):
    verify_stack()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    path = model_path(args.model)
    model = SAC.load(path, device=device(args.device))
    metadata = getattr(model, "bc_bootstrap_metadata", None)
    if metadata is None:
        raise ValueError("Checkpoint is not a BC bootstrap SAC model")
    collection_seeds = set(metadata["collection_metadata"]["episode_seeds"])
    eval_seeds = list(range(args.seed, args.seed+args.episodes))
    if collection_seeds.intersection(eval_seeds):
        raise ValueError("Evaluation seeds overlap demonstration episodes")
    assist_distance = metadata["environment"]["assist_distance"]
    results = []
    with tqdm(eval_seeds, desc="BC eval", unit="episode", dynamic_ncols=True) as progress:
        for seed in progress:
            _, result = run_episode(model, seed, assist_distance=assist_distance)
            results.append(result)
            progress.set_postfix(ever_success=f"{sum(r['ever_success'] for r in results)}/{len(results)}")
    first = [r["first_success_step"] for r in results if r["first_success_step"] is not None]
    returns = [r["episode_return"] for r in results]
    evaluation = {"episodes": args.episodes, "seeds": eval_seeds, "mean_return": float(np.mean(returns)),
                  "std_return": float(np.std(returns)), "ever_success_rate": float(np.mean([r["ever_success"] for r in results])),
                  "final_success_rate": float(np.mean([r["final_success"] for r in results])),
                  "mean_first_success_step": float(np.mean(first)) if first else None,
                  "first_success_mean_denominator": "successful episodes only; null when none",
                  "mean_episode_length": float(np.mean([r["episode_length"] for r in results])),
                  "grasp_rate": float(np.mean([r["ever_grasp"] for r in results])),
                  "premature_close_rate": sum(r["premature_close_count"] for r in results)/sum(r["episode_length"] for r in results),
                  "premature_close_definition": "raw policy gripper>0 while pre-step distance exceeds assistance threshold",
                  "episode_results": results, "evaluation_mode": "stage1_assisted", "assist_distance": assist_distance,
                  "checkpoint_path": str(path), "checkpoint_sha256": sha256(path), "package_versions": package_versions(),
                  "git_commit": git_sha(), "deterministic": True, "demonstrations_used": False}
    output = args.output or path.parent.parent / "evaluation.json"
    write_json(output, evaluation)
    summary_path = path.parent.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["evaluation_success"] = {key: evaluation[key] for key in ("ever_success_rate", "final_success_rate", "episodes", "seeds", "checkpoint_sha256")}
        summary["evaluation_path"] = str(output.resolve())
        write_json(summary_path, summary)
    print(f"BC EVAL episodes={args.episodes} return={evaluation['mean_return']:.6f} ever_success={evaluation['ever_success_rate']:.3f} final_success={evaluation['final_success_rate']:.3f}")
    print(f"EVALUATION_SAVED={output}")
    return evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--output", type=Path)
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()

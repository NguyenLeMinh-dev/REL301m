#!/usr/bin/env python3
"""Verify random full-horizon episodes with five independent episode metrics."""

import argparse
import json
from pathlib import Path

import numpy as np

from rel301m.envs.robosuite_factory import DEFAULT_CONFIG, PROJECT_ROOT, make_two_arm_lift
from rel301m.evaluation.rollout import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, help="Override the environment and random-action seed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments/phase1/random_rollout.json")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    env = make_two_arm_lift(args.config, seed=args.seed)
    episodes = []
    try:
        rng = np.random.default_rng(env.seed)
        for index in range(args.episodes):
            metrics = run_episode(env, rng)
            episodes.append(metrics)
            print(f"Episode {index + 1}/{args.episodes}: " + json.dumps(metrics, allow_nan=False), flush=True)
        output = {"status": "PASS", "seed": env.seed, "horizon": env.horizon, "episodes": episodes}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("PASS: reset/step, finite data, valid actions, and full configured horizons")
        print("Saved", args.output.resolve())
    finally:
        env.close()


if __name__ == "__main__":
    main()

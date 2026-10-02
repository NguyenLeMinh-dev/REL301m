#!/usr/bin/env python3
"""Print the complete live contract and write JSON/Markdown artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np

from rel301m.envs.robosuite_factory import DEFAULT_CONFIG, PROJECT_ROOT, load_env_config, make_two_arm_lift
from rel301m.envs.contract import build_environment_spec, write_specs
from rel301m.envs.spec_documentation import write_environment_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--write-spec", type=Path, default=PROJECT_ROOT / "docs/environment_spec.md")
    args = parser.parse_args()
    env = make_two_arm_lift(args.config)
    try:
        observations = env.reset()
        specs = build_environment_spec(env, observations, load_env_config(args.config))
        env_spec, observation_spec, action_spec = specs
        print("All observation keys, shapes, and dtypes:")
        for name, value in observations.items():
            array = np.asarray(value)
            print(f"  {name}: shape={array.shape}, dtype={array.dtype}")
        print("Enabled observables:", json.dumps(observation_spec["enabled_observables"]))
        print("Active observables:", json.dumps(observation_spec["active_observables"]))
        print("Complete action_spec.low:", action_spec["low"])
        print("Complete action_spec.high:", action_spec["high"])
        print("Total action dimension:", action_spec["action_dim"])
        print("Per-robot controllers and action layout:")
        print(json.dumps(action_spec["robots"], indent=2))
        print("Reward/success/termination contract:")
        print(json.dumps(env_spec["reward_contract"], indent=2))
        write_specs(args.output_dir, *specs)
        write_environment_spec(args.write_spec, *specs)
        print("Wrote env_spec.json, observation_spec.json, action_spec.json to", args.output_dir.resolve())
        print("Wrote", args.write_spec.resolve())
    finally:
        env.close()


if __name__ == "__main__":
    main()

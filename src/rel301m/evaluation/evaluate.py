"""Evaluate decentralized actors; success is the upstream task predicate."""

import argparse
from copy import deepcopy
import hashlib
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml

from rel301m.algorithms.masac import MASAC
from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
from rel301m.envs.robosuite_factory import make_two_arm_lift


from .metrics import EPISODE_FIELDS, episode_record, summarize_episodes


def evaluate_policy(env, model, episodes=10, seed=0, random_policy=False, *, initial_rng_state=None,
                    episode_callback=None):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if initial_rng_state is not None:
        # Keep the generator object: native robot/placement samplers reference it.
        # Restore only the dedicated eval environment, never global/train RNGs.
        env._env.rng.bit_generator.state = deepcopy(initial_rng_state)
    rng = np.random.default_rng(seed)
    bounds = [env.action_specs[agent] for agent in env.agent_ids]
    rows = []
    for episode in range(episodes):
        observations = env.reset()
        initial_hash = hashlib.sha256(env.critic_state.tobytes()).hexdigest()
        success = env.check_success()
        ever_success, first_success = success, 0 if success else None
        total = 0.0
        for step in range(1, env.horizon + 1):
            if random_policy:
                actions = tuple(rng.uniform(low, high) for low, high in bounds)
            else:
                actions = model.act([observations[agent]['actor_obs'] for agent in env.agent_ids], deterministic=True)
            observations, reward, done, _ = env.step(*actions)
            total += float(reward)
            success = env.check_success()
            ever_success |= success
            if success and first_success is None:
                first_success = step
            if bool(done) != (step == env.horizon):
                raise RuntimeError('Evaluation termination differs from the locked horizon contract')
            if done:
                break
        rows.append(episode_record(episode, total, ever_success, first_success, success,
                                   step, env.control_freq, initial_hash))
        if episode_callback is not None:
            episode_callback(episode + 1, episodes)
    summary = summarize_episodes(rows)
    return rows, summary


def write_episodes(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    model, metadata = MASAC.load(args.checkpoint, device=device, load_optimizers=False)
    if 'env_config' not in metadata:
        raise ValueError("Checkpoint lacks the environment config required for evaluation")
    output = args.output_dir or args.checkpoint.parent / f'evaluation_seed{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='rel301m-eval-') as temp:
        config_path = Path(temp) / 'env.yaml'
        config_path.write_text(yaml.safe_dump(metadata['env_config']), encoding='utf-8')
        env = MultiAgentWrapper(make_two_arm_lift(config_path, seed=args.seed))
        try:
            if tuple(env.observation_dims[a]['actor_obs'] for a in env.agent_ids) != model.obs_dims or env.critic_state_dim != model.state_dim:
                raise ValueError("Checkpoint observation dimensions differ from live contract")
            for i, agent in enumerate(env.agent_ids):
                if not all(np.array_equal(a, b) for a, b in zip(env.action_specs[agent], model.action_specs[i])):
                    raise ValueError("Checkpoint action bounds differ from live contract")
            initial_rng_state = deepcopy(env._env.rng.bit_generator.state)
            rows, summary = evaluate_policy(env, model, args.episodes, args.seed,
                                            initial_rng_state=initial_rng_state)
        finally:
            env.close()
    write_episodes(output / 'episodes.csv', rows)
    summary.update(checkpoint=str(args.checkpoint.resolve()), seed=args.seed, policy='deterministic', device=device,
                   communication_cost='N/A', evaluation_protocol='fixed_rng_state')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

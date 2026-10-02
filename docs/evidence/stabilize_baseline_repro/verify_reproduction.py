"""Compare saved learner tensors and real Adam counters without executing updates."""
import argparse
import json
from pathlib import Path

import torch


def adam_steps(optimizer):
    return max((int(state['step']) for state in optimizer['state'].values()), default=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-dir', type=Path, required=True)
    parser.add_argument('--reproduction-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[1000, 2000, 3000])
    args = parser.parse_args()
    results = {}
    for step in args.steps:
        filename = f'checkpoint_{step:07d}.pt'
        old = torch.load(args.historical_dir / filename, map_location='cpu', weights_only=True)
        new = torch.load(args.reproduction_dir / filename, map_location='cpu', weights_only=True)
        if old['model'].keys() != new['model'].keys():
            raise ValueError(f'Model keys differ at step {step}')
        mismatches = [name for name, tensor in old['model'].items()
                      if not torch.equal(tensor, new['model'][name])]
        actual = dict(critic_optimizer_updates=adam_steps(new['q_optimizer']),
                      actor_optimizer_updates_0=adam_steps(new['actor_optimizers'][0]),
                      actor_optimizer_updates_1=adam_steps(new['actor_optimizers'][1]),
                      alpha_optimizer_updates=adam_steps(new['alpha_optimizer']))
        recorded = new['optimizer_update_counts']
        if any(recorded[name] != value for name, value in actual.items()):
            raise ValueError(f'Recorded counters differ from Adam states at step {step}')
        if recorded['actor_optimizer_steps_total'] != sum(actual[f'actor_optimizer_updates_{i}'] for i in range(2)):
            raise ValueError(f'Total actor counter differs at step {step}')
        results[step] = dict(bitwise_equal_all_model_tensors=not mismatches,
                             mismatches=mismatches, optimizer_update_counts=recorded)
    print(json.dumps(results, indent=2))
    return 0 if all(result['bitwise_equal_all_model_tensors'] for result in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())

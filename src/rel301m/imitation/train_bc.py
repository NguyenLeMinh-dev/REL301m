"""Train two decentralized actors on held-out-episode BC and export native MASAC checkpoints."""

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import yaml

from rel301m.algorithms.masac import MASAC
from rel301m.envs.robosuite_factory import PROJECT_ROOT, make_two_arm_lift
from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
from rel301m.evaluation.evaluate import evaluate_policy, write_episodes
from rel301m.utils.diagnostics import require_finite
from rel301m.utils.logger import RunLogger, parameter_sha256
from rel301m.utils.seed import seed_everything
from rel301m.training.train import select_device

from .bc import fit_normalization, fold_normalization, normalized_action_loss, imitation_mse
from .demonstrations import file_sha256, load_dataset, split_episodes, stack_actor_data


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_bc(config):
    seed_everything(config['seed'])
    torch.set_num_threads(config['torch_threads'])
    device = select_device(config['device'])
    for key in ('epochs', 'batch_size', 'validation_episodes', 'test_episodes', 'eval_episodes'):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['learning_rate'] <= 0:
        raise ValueError('BC learning rate must be positive')
    dataset_path, output = resolve(config['dataset']), resolve(config['output_dir'])
    manifest, episodes = load_dataset(dataset_path)
    split = split_episodes(len(episodes), config['validation_episodes'], config['test_episodes'], config['seed'])
    output.mkdir(parents=True, exist_ok=False)
    algo = yaml.safe_load(resolve(config['algo_config']).read_text(encoding='utf-8'))
    model = MASAC(manifest['observation_dims'], manifest['critic_state_dim'], manifest['action_specs'], algo, device)
    normalizations, tensors = [], []
    for i in range(2):
        raw = {name: stack_actor_data(episodes, indices, i) for name, indices in split.items()}
        mean, std = fit_normalization(raw['train'][0], config['observation_std_floor'])
        normalizations.append((mean, std))
        tensors.append({name: (torch.as_tensor((obs - mean) / std, device=device),
                               torch.as_tensor(actions, device=device)) for name, (obs, actions) in raw.items()})
    metadata = dict(training_stage='behavior_cloning', seed=config['seed'], config=config,
                    env_config=manifest['env_config'], actor_input_transform='folded_into_first_linear',
                    demonstration_manifest_sha256=file_sha256(dataset_path / 'manifest.json'),
                    demonstration_episodes=len(episodes), demonstration_steps=sum(len(e['o0']) for e in episodes),
                    episode_split=split, observation_dims=manifest['observation_dims'],
                    action_dims=manifest['action_dims'], critic_state_dim=manifest['critic_state_dim'],
                    critics_trained=False, rl_updates=0,
                    objective='bounded normalized-action MSE; Gaussian log_std has no supervised target',
                    observation_normalization=[dict(mean=mean.tolist(), std=std.tolist()) for mean, std in normalizations],
                    teacher=manifest['teacher'], dataset_packages=manifest['packages'])
    git = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=PROJECT_ROOT, capture_output=True, text=True)
    metadata['git_sha'] = git.stdout.strip() if git.returncode == 0 else 'uncommitted (no HEAD yet)'
    metadata['source_sha256'] = {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in
                               (PROJECT_ROOT / 'src/rel301m/imitation').glob('*.py')}
    freeze = subprocess.run([sys.executable, '-I', '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (output / 'requirements.freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    (output / 'episode_split.json').write_text(json.dumps(split, indent=2)+'\n', encoding='utf-8')
    fields = ['epoch', 'training_mse_0', 'training_mse_1', 'validation_mse_0', 'validation_mse_1', 'mean_validation_mse']
    logger = RunLogger(output, dict(bc_losses=fields))
    optimizers = [torch.optim.Adam(actor.parameters(), lr=config['learning_rate']) for actor in model.actors]
    rng = np.random.default_rng(config['seed'])
    best_loss, best_epoch, best_state = float('inf'), 0, None
    start = time.perf_counter()
    try:
        for epoch in range(1, config['epochs'] + 1):
            count = len(tensors[0]['train'][0])
            order = rng.permutation(count)
            for offset in range(0, count, config['batch_size']):
                indices = torch.as_tensor(order[offset:offset + config['batch_size']], device=device)
                for i, (actor, optimizer) in enumerate(zip(model.actors, optimizers)):
                    obs, actions = tensors[i]['train']
                    loss = normalized_action_loss(actor, obs[indices], actions[indices])
                    require_finite('BC loss', loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            metrics = dict(epoch=epoch)
            for i, actor in enumerate(model.actors):
                require_finite('BC parameters', *actor.parameters())
                for name in ('train', 'validation'):
                    mse = imitation_mse(actor, *tensors[i][name])
                    metrics[f'{"training" if name == "train" else name}_mse_{i}'] = mse
                    logger.tensorboard.add_scalar(f'bc/{name}_mse_{i}', mse, epoch)
            metrics['mean_validation_mse'] = (metrics['validation_mse_0'] + metrics['validation_mse_1']) / 2
            logger.writers['bc_losses'].writerow(metrics)
            if metrics['mean_validation_mse'] < best_loss:
                best_loss, best_epoch = metrics['mean_validation_mse'], epoch
                best_state = deepcopy(model.state_dict())
            if epoch % 25 == 0 or epoch == config['epochs']:
                logger.flush()
                print(json.dumps(metrics), flush=True)
        model.load_state_dict(best_state)
        exported = fold_normalization(model, normalizations)
        metadata.update(best_epoch=best_epoch, best_validation_mse=best_loss,
                        bc_optimizer_updates_per_agent=config['epochs'] * int(np.ceil(count / config['batch_size'])),
                        actor_parameter_sha256=[parameter_sha256(actor) for actor in exported.actors])
        heldout, folding = {}, []
        for i in range(2):
            test_obs, test_actions = stack_actor_data(episodes, split['test'], i)
            raw_test = torch.as_tensor(test_obs, device=device)
            heldout[f'agent_{i}_test_mse'] = imitation_mse(exported.actors[i], raw_test, torch.as_tensor(test_actions, device=device))
            with torch.no_grad():
                normalized_actions = model.actors[i].sample(tensors[i]['test'][0], deterministic=True)[0]
                raw_actions = exported.actors[i].sample(raw_test, deterministic=True)[0]
                difference = (normalized_actions - raw_actions).abs().max().item()
                if difference > 1e-4:
                    raise RuntimeError(f'Normalization folding mismatch: {difference}')
                folding.append(difference)
        exported.save(output / 'best.pt', metadata)
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        # Actual closed-loop evaluation is separate from held-out imitation MSE.
        with tempfile.TemporaryDirectory(prefix='rel301m-bc-eval-') as temp:
            env_yaml = Path(temp) / 'env.yaml'
            env_yaml.write_text(yaml.safe_dump(manifest['env_config']), encoding='utf-8')
            env = MultiAgentWrapper(make_two_arm_lift(env_yaml, seed=config['eval_seed']))
            try:
                initial_rng = deepcopy(env._env.rng.bit_generator.state)
                rows, evaluation = evaluate_policy(env, exported, config['eval_episodes'], config['eval_seed'],
                                                   initial_rng_state=initial_rng)
                write_episodes(output / 'evaluation_episodes.csv', rows)
            finally:
                env.close()
        summary = dict(status='completed', training_stage='behavior_cloning', epochs=config['epochs'],
                       best_epoch=best_epoch, validation_mse=best_loss, heldout_imitation=heldout,
                       folding_max_action_error=folding, deterministic_evaluation=evaluation,
                       wall_seconds=time.perf_counter() - start, critic_training=False,
                       checkpoint=str((output / 'best.pt').resolve()))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except BaseException as error:
        import traceback
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(error), traceback=traceback.format_exc()), indent=2)+'\n')
        raise
    finally:
        logger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/imitation/bc.yaml'))
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding='utf-8'))
    for value, key in ((args.dataset, 'dataset'), (args.output_dir, 'output_dir'), (args.epochs, 'epochs')):
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    train_bc(config)


if __name__ == '__main__':
    main()

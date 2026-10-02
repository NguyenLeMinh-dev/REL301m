"""Retained train-only demonstrations for short BC-assisted MASAC experiments."""
from copy import deepcopy
import numpy as np
import torch

from rel301m.algorithms.replay_buffer import ReplayBuffer
from .demonstrations import load_dataset, file_sha256, stack_actor_data
from .bc import imitation_mse


class DemonstrationReplay:
    def __init__(self, dataset, checkpoint, model, env_config, seed):
        manifest, episodes = load_dataset(dataset)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        metadata = payload['metadata']
        if metadata.get('training_stage') != 'behavior_cloning':
            raise ValueError('Demonstrations must match a BC checkpoint')
        if metadata.get('demonstration_manifest_sha256') != file_sha256(dataset / 'manifest.json'):
            raise ValueError('BC checkpoint and demonstration manifest differ')
        if manifest['env_config'] != env_config or metadata['env_config'] != env_config:
            raise ValueError('Demonstration environment contract differs')
        if tuple(manifest['observation_dims']) != model.obs_dims or manifest['critic_state_dim'] != model.state_dim:
            raise ValueError('Demonstration dimensions differ from live environment')
        if tuple(manifest['action_dims']) != model.action_dims:
            raise ValueError('Demonstration action dimensions differ')
        if len(manifest['action_specs']) != 2:
            raise ValueError('Demonstrations require two action-bound pairs')
        for actual, expected in zip(manifest['action_specs'], model.action_specs):
            if any(not np.array_equal(a, b) for a, b in zip(actual, expected)):
                raise ValueError('Demonstration action bounds differ from live environment')
        split = metadata['episode_split']
        indices = [i for group in ('train', 'validation', 'test') for i in split[group]]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(episodes))):
            raise ValueError('Invalid or overlapping demonstration episode split')
        if any(not split[group] for group in ('train', 'validation', 'test')):
            raise ValueError('Empty demonstration split')
        capacity = sum(len(episodes[i]['r']) for i in split['train'])
        self.replay = ReplayBuffer(capacity, model.obs_dims, model.state_dim, model.action_dims, seed)
        for i in split['train']:
            episode = episodes[i]
            for row in range(len(episode['r'])):
                self.replay.add(**{key: episode[key][row] for key in self.replay.arrays})
        self.validation = [tuple(torch.as_tensor(a, device=model.device) for a in
                                 stack_actor_data(episodes, split['validation'], agent)) for agent in range(2)]
        self.metadata = dict(manifest_sha256=metadata['demonstration_manifest_sha256'],
                             episode_split=deepcopy(split), retained_training_transitions=capacity,
                             validation_transitions=len(self.validation[0][0]), test_used=False)

    def prefill(self, replay):
        if replay.capacity < len(self.replay):
            raise ValueError('Replay capacity cannot hold training demonstration prefill')
        for row in range(len(self.replay)):
            replay.add(**{key: array[row] for key, array in self.replay.arrays.items()})

    def sample_mixed(self, replay, batch_size, ratio, device):
        if not np.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError('Demo ratio must be within [0, 1]')
        count = int(round(batch_size * ratio))
        pieces = []
        if count:
            pieces.append(self.replay.sample(count, device))
        if batch_size - count:
            pieces.append(replay.sample(batch_size - count, device))
        return {key: torch.cat([piece[key] for piece in pieces], dim=0) for key in pieces[0]}

    def validation_metrics(self, model):
        return {f'validation_mse_{i}': imitation_mse(actor, *self.validation[i])
                for i, actor in enumerate(model.actors)}


def validate_fine_tune(config):
    settings = config['algo'].get('fine_tune')
    if settings is None:
        return
    if not isinstance(settings, dict):
        raise ValueError('fine_tune must be a mapping')
    if not config.get('bc_checkpoint') or not settings.get('dataset'):
        raise ValueError('Fine-tune requires a BC checkpoint and demonstration dataset')
    allowed = {'dataset', 'lambda_bc', 'actor_init_log_std', 'freeze_log_std_updates',
               'critic_pretrain_updates', 'critic_pretrain_lr', 'actor_freeze_updates',
               'demo_ratio_initial', 'demo_ratio_final', 'demo_ratio_decay_steps', 'prefill_replay', 'alpha_mode'}
    if set(settings) != allowed:
        raise ValueError(f'Fine-tune keys mismatch: {sorted(set(settings) ^ allowed)}')
    if type(settings['prefill_replay']) is not bool or settings['alpha_mode'] not in ('auto', 'fixed'):
        raise ValueError('Invalid prefill or alpha mode')
    for key in ('freeze_log_std_updates', 'critic_pretrain_updates', 'actor_freeze_updates', 'demo_ratio_decay_steps'):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('lambda_bc', 'critic_pretrain_lr', 'actor_init_log_std', 'demo_ratio_initial', 'demo_ratio_final'):
        if isinstance(settings[key], bool) or not isinstance(settings[key], (int, float)) or not np.isfinite(settings[key]):
            raise ValueError(f'{key} must be finite numeric')
    if settings['lambda_bc'] < 0 or settings['critic_pretrain_lr'] <= 0:
        raise ValueError('Invalid BC coefficient or pretraining learning rate')
    if not -20 <= settings['actor_init_log_std'] <= 2:
        raise ValueError('Invalid initial log_std')
    if any(not 0 <= settings[key] <= 1 for key in ('demo_ratio_initial', 'demo_ratio_final')):
        raise ValueError('Demo ratios must be within [0, 1]')
    if not 0 < config['total_steps'] <= 30000:
        raise ValueError('Fine-tune budget is capped at 30k; long collision-safe pilot is not established')


def demo_ratio(settings, step):
    duration = settings['demo_ratio_decay_steps']
    fraction = min(step / duration, 1.0) if duration else 1.0
    return settings['demo_ratio_initial'] + fraction * (settings['demo_ratio_final'] - settings['demo_ratio_initial'])

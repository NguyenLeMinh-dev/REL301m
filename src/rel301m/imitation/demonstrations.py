"""Validate episode-grouped state/action data and avoid transition-level leakage."""

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_trajectory(data, metadata):
    horizon = metadata['env_config']['horizon']
    sizes = dict(o0=metadata['observation_dims'][0], o1=metadata['observation_dims'][1],
                 s=metadata['critic_state_dim'], a0=metadata['action_dims'][0], a1=metadata['action_dims'][1], r=1,
                 next_o0=metadata['observation_dims'][0], next_o1=metadata['observation_dims'][1],
                 next_s=metadata['critic_state_dim'], done=1, timeout=1)
    for key, width in sizes.items():
        array = data[key]
        if array.shape != (horizon, width) or not np.isfinite(array).all():
            raise ValueError(f'Invalid demonstration {key} shape or NaN/Inf')
    expected_done = np.zeros((horizon, 1))
    expected_done[-1] = 1
    if not np.array_equal(data['done'], expected_done) or not np.array_equal(data['timeout'], expected_done):
        raise ValueError('Demonstration done/timeout violates the full-horizon contract')
    for i, bounds in enumerate(metadata['action_specs']):
        low, high = np.asarray(bounds[0]), np.asarray(bounds[1])
        if np.any(data[f'a{i}'] < low) or np.any(data[f'a{i}'] > high):
            raise ValueError('Demonstration action exceeds measured bounds')
    for key in ('o0', 'o1', 's'):
        if not np.array_equal(data[key][1:], data[f'next_{key}'][:-1]):
            raise ValueError('Demonstration current/next transition alignment mismatch')
    success, grasps = data['success'], data['grasps']
    if success.shape != (horizon + 1,) or grasps.shape != (horizon + 1, 2):
        raise ValueError('Success/grasp trace shape mismatch')
    if not np.isin(success, (0, 1)).all() or not np.isin(grasps, (0, 1)).all():
        raise ValueError('Success/grasp traces must be binary')
    if not success[-1] or not np.any(success.astype(bool) & grasps.astype(bool).all(axis=1)):
        raise ValueError('Demos require native final success and simultaneous two-handle grasp during success')
    for key in ('qpos', 'qvel'):
        if data[key].ndim != 2 or data[key].shape[0] != horizon + 1 or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid demonstration simulator state {key}')


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1 or manifest.get('status') != 'completed':
        raise ValueError('Need a completed REL301m demonstration dataset')
    episodes = []
    ids = set()
    for record in manifest['episodes']:
        if record['episode'] in ids:
            raise ValueError('Duplicate demonstration episode id')
        ids.add(record['episode'])
        relative = Path(record['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Demonstration path must stay inside dataset directory')
        path = directory / relative
        if file_sha256(path) != record['sha256']:
            raise ValueError('Demonstration file SHA-256 mismatch')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        validate_trajectory(arrays, manifest)
        episodes.append(arrays)
    if len(episodes) != manifest['successful_episodes'] or len(episodes) < 3:
        raise ValueError('Dataset count mismatch or insufficient independent episodes')
    return manifest, episodes


def split_episodes(count, validation_episodes, test_episodes, seed):
    if min(validation_episodes, test_episodes) < 1 or count <= validation_episodes + test_episodes:
        raise ValueError('Train, validation and test must each contain independent episodes')
    indices = np.random.default_rng(seed).permutation(count)
    return dict(train=indices[validation_episodes + test_episodes:].tolist(),
                validation=indices[:validation_episodes].tolist(),
                test=indices[validation_episodes:validation_episodes + test_episodes].tolist())


def stack_actor_data(episodes, indices, agent):
    return (np.concatenate([episodes[index][f'o{agent}'] for index in indices]).astype(np.float32),
            np.concatenate([episodes[index][f'a{agent}'] for index in indices]).astype(np.float32))

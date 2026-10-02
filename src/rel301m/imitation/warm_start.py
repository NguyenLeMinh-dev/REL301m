"""Copy BC actors only, preserving fresh critics, optimizers, alpha and RNGs."""

from pathlib import Path

import numpy as np
import torch

from .demonstrations import file_sha256


def warm_start_from_bc(model, checkpoint, *, expected_env_config=None, initial_log_std=None):
    if initial_log_std is not None and (isinstance(initial_log_std, bool) or
            not isinstance(initial_log_std, (int, float)) or not np.isfinite(initial_log_std) or
            not -20 <= initial_log_std <= 2):
        raise ValueError('initial_log_std must be finite within [-20, 2]')
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = payload.get('metadata', {})
    if payload.get('format_version') != 1 or metadata.get('training_stage') != 'behavior_cloning':
        raise ValueError('Actor initialization requires an explicit behavior_cloning checkpoint')
    if metadata.get('actor_input_transform') != 'folded_into_first_linear':
        raise ValueError('BC checkpoint must accept raw Phase 2 actor observations')
    if expected_env_config is not None:
        source_env = {k: v for k, v in metadata.get('env_config', {}).items() if k != 'seed'}
        target_env = {k: v for k, v in expected_env_config.items() if k != 'seed'}
        if source_env != target_env:
            raise ValueError('BC checkpoint environment contract mismatch')
    if tuple(payload['obs_dims']) != model.obs_dims or payload['state_dim'] != model.state_dim:
        raise ValueError('BC checkpoint observation/state dimensions mismatch')
    for source, expected in zip(payload['action_specs'], model.action_specs):
        if not all(np.array_equal(a, b) for a, b in zip(source, expected)):
            raise ValueError('BC checkpoint action bounds mismatch')
    if len(payload['action_specs']) != len(model.action_specs):
        raise ValueError('BC checkpoint agent count mismatch')
    states = []
    for i, actor in enumerate(model.actors):
        prefix = f'actors.{i}.'
        state = {key[len(prefix):]: value for key, value in payload['model'].items() if key.startswith(prefix)}
        expected = actor.state_dict()
        if state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in state):
            raise ValueError('BC actor architecture mismatch')
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Nonfinite BC actor weights')
        states.append(state)
    for actor, state in zip(model.actors, states):
        actor.load_state_dict(state)
    if initial_log_std is not None:
        model.initialize_log_std(initial_log_std)
    return dict(log_std_override=initial_log_std, method='behavior_cloning', checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=file_sha256(checkpoint),
                demonstration_manifest_sha256=metadata['demonstration_manifest_sha256'],
                bc_seed=metadata['seed'], bc_best_epoch=metadata['best_epoch'],
                critic_optimizer_alpha_replay='fresh', exact_resume=False)

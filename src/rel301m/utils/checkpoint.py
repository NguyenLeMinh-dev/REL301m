"""Atomic model/optimizer checkpoints; replay/RNG resume is not implemented."""

from pathlib import Path

import torch


def save_checkpoint(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, obs_dims=model.obs_dims, state_dim=model.state_dim,
                   action_specs=[(low.tolist(), high.tolist()) for low, high in model.action_specs],
                   config=model.config, model=model.state_dict(), updates=model.updates,
                   global_step=(metadata or {}).get('step'),
                   actor_optimizers=[opt.state_dict() for opt in model.actor_optimizers],
                   q_optimizer=model.q_optimizer.state_dict(), alpha_optimizer=model.alpha_optimizer.state_dict(),
                   metadata=metadata or {})
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu', load_optimizers=True):
    from rel301m.algorithms.masac import MASAC

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError("Unsupported MASAC checkpoint")
    model = MASAC(payload['obs_dims'], payload['state_dim'], payload['action_specs'], payload['config'], device)
    model.load_state_dict(payload['model'])
    model.updates = payload['updates']
    model.configure_log_std()
    if load_optimizers:
        for optimizer, state in zip(model.actor_optimizers, payload['actor_optimizers']):
            optimizer.load_state_dict(state)
        model.q_optimizer.load_state_dict(payload['q_optimizer'])
        model.alpha_optimizer.load_state_dict(payload['alpha_optimizer'])
    return model, payload['metadata']

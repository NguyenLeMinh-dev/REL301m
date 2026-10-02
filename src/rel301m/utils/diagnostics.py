"""Numeric guards and explicitly defined diagnostics; never sample extra actions."""

import math

import numpy as np
import torch


class NonfiniteTrainingError(FloatingPointError):
    def __init__(self, label, nan_count, inf_count):
        self.nan_count, self.inf_count = nan_count, inf_count
        super().__init__(f'{label}: {nan_count} NaN, {inf_count} Inf')


def require_finite(label, *values):
    if not values:
        return
    if isinstance(values[0], torch.Tensor):
        flat = torch.cat([value.detach().reshape(-1) for value in values])
        if torch.isfinite(flat).all().item():
            return
        nan_count = int(torch.isnan(flat).sum().item())
        inf_count = int(torch.isinf(flat).sum().item())
    else:
        flat = np.concatenate([np.asarray(value).reshape(-1) for value in values])
        if np.isfinite(flat).all():
            return
        nan_count, inf_count = int(np.isnan(flat).sum()), int(np.isinf(flat).sum())
    raise NonfiniteTrainingError(label, nan_count, inf_count)


@torch.no_grad()
def td_error_metrics(q1, q2, target):
    # Pool both critics' absolute Bellman residuals, before the optimizer step.
    errors = torch.cat(((q1 - target).abs().reshape(-1), (q2 - target).abs().reshape(-1)))
    quantiles = torch.quantile(errors, errors.new_tensor([.5, .95, .99]))
    return dict(td_abs_mean=errors.mean().item(),
                **{f'td_abs_p{p}': value.item() for p, value in zip((50, 95, 99), quantiles)})


@torch.no_grad()
def action_metrics(actor, observations, sampled_actions):
    _, log_std = actor(observations)
    normalized = (sampled_actions.detach() - actor.action_bias) / actor.action_scale
    return dict(action_std=log_std.exp().mean().item(),
                action_sample_std=sampled_actions.detach().std(dim=0, unbiased=False).mean().item(),
                action_saturation=(normalized.abs() >= .95).float().mean().item())


def tensorboard_tag(key):
    if key in ('q1_mean', 'q2_mean', 'target_q_mean') or key.startswith('td_abs_'):
        return f'critic/{key}'
    if key == 'q_grad_norm':
        return 'grad/q_norm'
    for i in range(2):
        if key == f'actor_{i}_grad_norm':
            return f'grad/actor_{i}_norm'
        for field, tag in (('action_std', 'std'), ('action_sample_std', 'sample_std'),
                           ('action_saturation', 'saturation')):
            if key == f'{field}_{i}':
                return f'action/{tag}_{i}'
        if key == f'actor_{i}_loss':
            return f'train/actor_loss_{i}'
    return f'train/{key}'


def finite_json(value):
    """Keep failure reports valid JSON even if parameters were corrupted."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

"""Independent Gaussian-actor BC; fold train-only normalization into Linear weights."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn


def fit_normalization(observations, std_floor=.01):
    if std_floor <= 0 or observations.ndim != 2 or not np.isfinite(observations).all():
        raise ValueError('Need finite training observations and a positive std floor')
    mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(observations.std(axis=0, dtype=np.float64), std_floor).astype(np.float32)
    return mean, std


@torch.no_grad()
def fold_normalization(model, normalizations):
    exported = deepcopy(model)
    for actor, (mean, std) in zip(exported.actors, normalizations):
        layer = actor.net[0]
        if not isinstance(layer, nn.Linear):
            raise ValueError('Normalization folding requires a first Linear layer')
        mean = torch.as_tensor(mean, device=layer.weight.device, dtype=layer.weight.dtype)
        std = torch.as_tensor(std, device=layer.weight.device, dtype=layer.weight.dtype)
        if mean.shape != (layer.in_features,) or std.shape != mean.shape or not torch.all(std > 0):
            raise ValueError('Normalization shape/scale mismatch')
        # W ((x - mean) / std) + b == (W / std) x + (b - (W / std) mean).
        layer.weight.div_(std)
        layer.bias.sub_(layer.weight @ mean)
    return exported


def normalized_action_loss(actor, observations, expert_actions):
    mean, _ = actor(observations)
    expert = (expert_actions - actor.action_bias) / actor.action_scale
    return (mean.tanh() - expert).square().mean()


@torch.no_grad()
def imitation_mse(actor, observations, expert_actions):
    mean, _ = actor(observations)
    actions = actor.action_bias + actor.action_scale * mean.tanh()
    return (actions - expert_actions).square().mean().item()

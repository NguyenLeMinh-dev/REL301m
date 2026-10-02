"""Decentralized Gaussian actors and centralized scalar Q functions."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, low, high, hidden_dims=(256, 256)):
        super().__init__()
        low, high = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        if low.ndim != 1 or low.shape != high.shape or not np.isfinite([low, high]).all() or np.any(high <= low):
            raise ValueError("Actor requires finite, strictly ordered vector action bounds")
        self.obs_dim, self.action_dim = obs_dim, len(low)
        self.net = mlp(obs_dim, hidden_dims, 2 * self.action_dim)
        self.register_buffer('action_scale', torch.from_numpy((high - low) / 2))
        self.register_buffer('action_bias', torch.from_numpy((high + low) / 2))

    def forward(self, observations):
        if observations.shape[-1] != self.obs_dim:
            raise ValueError("Actor observation dimension mismatch")
        mean, log_std = self.net(observations).chunk(2, dim=-1)
        fixed = getattr(self, "fixed_log_std", None)
        if fixed is not None:
            log_std = torch.full_like(log_std, fixed)
        return mean, log_std.clamp(-20, 2)

    def sample(self, observations, deterministic=False):
        mean, log_std = self(observations)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        z = mean if deterministic else distribution.rsample()
        actions = self.action_bias + self.action_scale * z.tanh()
        # Stable log(1 - tanh(z)^2), including the affine action transform.
        correction = 2 * (np.log(2) - z - F.softplus(-2 * z))
        log_prob = (distribution.log_prob(z) - correction - self.action_scale.log()).sum(-1, keepdim=True)
        return actions, log_prob


class CentralizedQ(nn.Module):
    def __init__(self, state_dim, joint_action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.state_dim, self.joint_action_dim = state_dim, joint_action_dim
        self.net = mlp(state_dim + joint_action_dim, hidden_dims, 1)

    def forward(self, state, joint_action):
        if state.shape[-1] != self.state_dim or joint_action.shape[-1] != self.joint_action_dim:
            raise ValueError("Critic state/action dimension mismatch")
        return self.net(torch.cat((state, joint_action), dim=-1))

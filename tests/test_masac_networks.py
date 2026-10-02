import numpy as np
import pytest
import torch

from rel301m.algorithms.networks import CentralizedQ, GaussianActor


@pytest.mark.parametrize('batch_size', [1, 256])
def test_shapes_and_parameter_counts(batch_size):
    actor = GaussianActor(66, -np.ones(7), np.ones(7))
    critic = CentralizedQ(119, 14)
    actions, log_prob = actor.sample(torch.randn(batch_size, 66))
    assert actions.shape == (batch_size, 7)
    assert log_prob.shape == (batch_size, 1)
    assert critic(torch.randn(batch_size, 119), torch.cat((actions, actions), -1)).shape == (batch_size, 1)
    assert sum(p.numel() for p in actor.parameters()) == 86542
    assert sum(p.numel() for p in critic.parameters()) == 100353


def test_affine_tanh_bounds_and_correct_density():
    low, high = np.array([-2., .1, -1.]), np.array([4., 3., 1.])
    actor = GaussianActor(5, low, high, (16, 16))
    obs = torch.randn(64, 5)
    actions, density = actor.sample(obs)
    assert torch.all(actions >= torch.tensor(low))
    assert torch.all(actions <= torch.tensor(high))
    mean, log_std = actor(obs)
    normalized = (actions - actor.action_bias) / actor.action_scale
    z = torch.atanh(normalized)
    expected = (torch.distributions.Normal(mean, log_std.exp()).log_prob(z)
                - torch.log1p(-normalized.square()) - actor.action_scale.log()).sum(-1, keepdim=True)
    torch.testing.assert_close(density, expected, rtol=2e-3, atol=2e-3)
    deterministic, _ = actor.sample(obs, deterministic=True)
    torch.testing.assert_close(deterministic, actor.action_bias + actor.action_scale * mean.tanh())
    density.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_saturated_actor_still_has_finite_log_probability():
    actor = GaussianActor(66, -np.ones(7), np.ones(7))
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.net[-1].bias[:7].fill_(100)
    actions, density = actor.sample(torch.zeros(3, 66))
    assert torch.isfinite(density).all()
    assert torch.isfinite(actions).all()
    assert (actions <= 1).all()


def test_bad_network_dimensions_rejected():
    actor = GaussianActor(66, -np.ones(7), np.ones(7))
    with pytest.raises(ValueError):
        actor(torch.zeros(1, 119))
    critic = CentralizedQ(119, 14)
    with pytest.raises(ValueError):
        critic(torch.zeros(1, 66), torch.zeros(1, 14))
    with pytest.raises(ValueError):
        GaussianActor(66, np.ones(7), np.ones(7))

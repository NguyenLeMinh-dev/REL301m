from copy import deepcopy

import numpy as np
import pytest
import torch

from rel301m.algorithms.masac import MASAC
from rel301m.algorithms.replay_buffer import ReplayBuffer
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.training.train import load_config


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = load_config(PROJECT_ROOT / 'configs/experiment/smoke.yaml')['algo']
    return MASAC((66, 66), 119, [(-np.ones(7), np.ones(7))] * 2, config)


def batch(size=16):
    result = {key: torch.randn(size, dim) for key, dim in dict(o0=66, o1=66, s=119, a0=7, a1=7,
                                                             r=1, next_o0=66, next_o1=66, next_s=119).items()}
    result.update(done=torch.zeros(size, 1), timeout=torch.zeros(size, 1))
    return result


def test_actor_execution_cannot_receive_critic_or_teammate_state(model):
    observations = [np.ones(66), np.ones(66)]
    initial = model.act(observations, deterministic=True)
    changed = model.act([observations[0], np.full(66, 99)], deterministic=True)
    np.testing.assert_array_equal(initial[0], changed[0])
    assert not np.array_equal(initial[1], changed[1])
    with pytest.raises(ValueError):
        model.act([np.ones(119), np.ones(66)])
    assert model.actors[0].net[0].in_features == 66
    assert model.q1.net[0].in_features == 119 + 14


@pytest.mark.parametrize('index', [0, 1])
def test_actor_gradient_isolation(model, index):
    model.q1.requires_grad_(False)
    model.q2.requires_grad_(False)
    loss, _ = model.actor_loss(index, batch())
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.actors[index].parameters())
    assert all(p.grad is None for p in model.actors[1 - index].parameters())
    assert all(p.grad is None for p in model.q1.parameters())
    assert all(p.grad is None for p in model.q2.parameters())
    assert all(p.grad is None for p in model.target_q1.parameters())
    assert model.log_alpha.grad is None


def test_terminal_timeout_entropy_and_minimum_target(model, monkeypatch):
    data = batch(3)
    data['r'].fill_(1)
    data['done'] = torch.tensor([[0.], [1.], [1.]])
    data['timeout'] = torch.tensor([[0.], [0.], [1.]])
    with torch.no_grad():
        for net, constant in ((model.target_q1, 2.), (model.target_q2, 4.)):
            for parameter in net.parameters():
                parameter.zero_()
            net.net[-1].bias.fill_(constant)
    for i, actor in enumerate(model.actors):
        def sample(obs, log_prob=-(i + 2.)):
            return torch.zeros(len(obs), 7), torch.full((len(obs), 1), log_prob)
        monkeypatch.setattr(actor, 'sample', sample)
    expected = 1 + .99 * (2 + .2 * 5)
    torch.testing.assert_close(model.critic_target(data), torch.tensor([[expected], [1.], [expected]]))
    model.config['bootstrap_time_limits'] = False
    torch.testing.assert_close(model.critic_target(data), torch.tensor([[expected], [1.], [1.]]))
    assert not model.critic_target(data).requires_grad


def test_exact_polyak_update(model):
    model.config['tau'] = .25
    with torch.no_grad():
        for net in (model.q1, model.q2):
            for parameter in net.parameters():
                parameter.fill_(4)
        for net in (model.target_q1, model.target_q2):
            for parameter in net.parameters():
                parameter.fill_(2)
    model.soft_update()
    for net in (model.target_q1, model.target_q2):
        assert all(torch.all(parameter == 2.5) for parameter in net.parameters())


def test_update_changes_online_parameters_and_freezes_targets(model):
    before = deepcopy(model.state_dict())
    metrics = model.update(batch())
    assert model.updates == 1
    assert np.isfinite(list(metrics.values())).all()
    assert metrics['q_grad_norm'] > 0
    for i in range(2):
        assert metrics[f'actor_{i}_grad_norm'] > 0
        assert metrics[f'alpha_{i}'] > 0
    for name in ('actors.0.net.0.weight', 'actors.1.net.0.weight', 'q1.net.0.weight', 'q2.net.0.weight', 'log_alpha'):
        assert not torch.equal(before[name], model.state_dict()[name])
    for online, target in (('q1', 'target_q1'), ('q2', 'target_q2')):
        for name, value in model.state_dict().items():
            if name.startswith(target):
                suffix = name.removeprefix(target)
                expected = .995 * before[name] + .005 * model.state_dict()[online + suffix]
                torch.testing.assert_close(value, expected)
    assert all(p.grad is None for net in (model.target_q1, model.target_q2, model.q1, model.q2) for p in net.parameters())
    assert all(p.requires_grad for net in (model.q1, model.q2) for p in net.parameters())


def test_save_load_model_optimizers_and_deterministic_actions(model, tmp_path):
    model.update(batch())
    path = tmp_path / 'test.pt'
    model.save(path, {'step': 100, 'purpose': 'test'})
    loaded, metadata = MASAC.load(path)
    assert metadata == {'step': 100, 'purpose': 'test'}
    assert loaded.updates == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    observations = [np.ones(66), np.zeros(66)]
    for a, b in zip(model.act(observations, True), loaded.act(observations, True)):
        np.testing.assert_array_equal(a, b)
    assert loaded.q_optimizer.state_dict()['state']
    assert all(optimizer.state_dict()['state'] for optimizer in loaded.actor_optimizers)
    assert loaded.alpha_optimizer.state_dict()['state']
    data = batch()
    torch.manual_seed(2)
    original = model.update(data)
    torch.manual_seed(2)
    restored = loaded.update(data)
    assert original == pytest.approx(restored)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_training_batch(model):
    cuda_model = MASAC(model.obs_dims, model.state_dim, model.action_specs, model.config, 'cuda')
    data = {key: value.cuda() for key, value in batch(256).items()}
    metrics = cuda_model.update(data)
    assert np.isfinite(list(metrics.values())).all()
    assert cuda_model.device.type == 'cuda'

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
import yaml

from rel301m.algorithms.masac import MASAC
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.imitation.bc import fit_normalization, fold_normalization, normalized_action_loss
from rel301m.imitation.collect import collect
from rel301m.imitation.demonstrations import load_dataset, split_episodes, validate_trajectory
from rel301m.imitation.train_bc import train_bc
from rel301m.imitation.warm_start import warm_start_from_bc
from rel301m.training.train import load_config


def model(seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    config = load_config(PROJECT_ROOT / 'configs/experiment/pilot.yaml')['algo']
    return MASAC((66, 66), 119, [(-np.ones(7), np.ones(7))] * 2, config)


def test_episode_split_and_train_only_statistics():
    split = split_episodes(40, 5, 5, 42)
    assert [len(split[k]) for k in ('train', 'validation', 'test')] == [30, 5, 5]
    assert not any(set(split[a]) & set(split[b]) for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')))
    assert split == split_episodes(40, 5, 5, 42)
    train = np.arange(4 * 66, dtype=np.float32).reshape(4, 66)
    mean, std = fit_normalization(train)
    np.testing.assert_allclose(mean, train.mean(0))
    np.testing.assert_allclose(std, train.std(0))
    assert np.all(fit_normalization(np.zeros((10, 66)))[1] >= .009999)
    with pytest.raises(ValueError):
        split_episodes(2, 1, 1, 42)


def test_folding_normalization_matches_both_mean_and_log_std_without_changing_model():
    original = model()
    before = deepcopy(original.state_dict())
    rng = np.random.default_rng(2)
    normalization = [(rng.normal(size=66).astype(np.float32), rng.uniform(.01, 2, 66).astype(np.float32)) for _ in range(2)]
    exported = fold_normalization(original, normalization)
    for i, (mean, std) in enumerate(normalization):
        raw = torch.tensor(rng.normal(size=(100, 66)).astype(np.float32))
        norm = (raw - torch.tensor(mean)) / torch.tensor(std)
        for actual, expected in zip(exported.actors[i](raw), original.actors[i](norm)):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
    for key, value in before.items():
        torch.testing.assert_close(original.state_dict()[key], value)


def test_bc_gradient_has_no_critic_teammate_or_log_std_head_path():
    learner = model()
    loss = normalized_action_loss(learner.actors[0], torch.randn(32, 66), torch.rand(32, 7) * 2 - 1)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner.actors[0].parameters())
    assert all(p.grad is None for p in learner.actors[1].parameters())
    assert all(p.grad is None for net in (learner.q1, learner.q2, learner.target_q1, learner.target_q2) for p in net.parameters())
    assert learner.log_alpha.grad is None
    assert torch.count_nonzero(learner.actors[0].net[-1].weight.grad[7:]) == 0


def bc_checkpoint(path):
    source = model(17)
    source.save(path, dict(training_stage='behavior_cloning', actor_input_transform='folded_into_first_linear',
                          demonstration_manifest_sha256='d' * 64, seed=17, best_epoch=2,
                          env_config={'contract': 'fixture'}))
    return source


def test_warm_start_copies_only_actors_and_preserves_rng_critics_alpha_optimizers(tmp_path):
    path = tmp_path / 'bc.pt'
    source = bc_checkpoint(path)
    target = model(42)
    before = deepcopy(target.state_dict())
    rng_before = torch.get_rng_state().clone()
    info = warm_start_from_bc(target, path, expected_env_config={'contract': 'fixture'})
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[key] if key.startswith('actors.') else before[key], rtol=0, atol=0)
    assert target.updates == 0 and target.q_optimizer.state_dict()['state'] == {}
    assert all(not optimizer.state_dict()['state'] for optimizer in target.actor_optimizers)
    assert info['method'] == 'behavior_cloning' and len(info['checkpoint_sha256']) == 64
    with pytest.raises(ValueError, match='environment'):
        warm_start_from_bc(target, path, expected_env_config={'contract': 'different'})


@pytest.mark.parametrize('corruption', ['stage', 'dimension', 'bounds', 'nonfinite'])
def test_invalid_bc_checkpoint_cannot_initialize_actors(tmp_path, corruption):
    path = tmp_path / 'bc.pt'
    bc_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    if corruption == 'stage':
        payload['metadata']['training_stage'] = 'reinforcement_learning'
    elif corruption == 'dimension':
        payload['obs_dims'] = (119, 119)
    elif corruption == 'bounds':
        payload['action_specs'][0][0][0] = -2.
    else:
        payload['model']['actors.0.net.0.weight'][0, 0] = float('nan')
    torch.save(payload, path)
    target = model()
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError):
        warm_start_from_bc(target, path)
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_bc_pilot_preserves_nominal_hyperparameters_budget_and_observation_contract():
    nominal = load_config(PROJECT_ROOT / 'configs/experiment/pilot.yaml')
    assisted = load_config(PROJECT_ROOT / 'configs/experiment/bc_pilot.yaml')
    assert nominal['algo'] == assisted['algo']
    for key in ('total_steps', 'eval_seed', 'eval_interval', 'eval_episodes', 'final_eval_episodes', 'env_config'):
        assert nominal[key] == assisted[key]
    assert assisted['algo']['warmup_steps'] == 10000
    assert assisted['bc_checkpoint']


def test_collect_bc_export_and_native_evaluation_integration(tmp_path):
    dataset = tmp_path / 'demos'
    manifest = collect(dataset, episodes=3, max_attempts=6, seed=40000, video=False)
    assert manifest['status'] == 'completed' and manifest['successful_episodes'] == 3
    manifest, episodes = load_dataset(dataset)
    assert all(e['success'][-1] and e['grasps'][-1].all() for e in episodes)
    assert all(e['o0'].shape == (200, 66) and e['s'].shape == (200, 119) for e in episodes)
    broken = {key: array.copy() for key, array in episodes[0].items()}
    broken['a0'][0, 0] = 2
    with pytest.raises(ValueError, match='bounds'):
        validate_trajectory(broken, manifest)
    broken['a0'][0, 0] = 0
    broken['o0'][1, 0] += 10
    with pytest.raises(ValueError, match='alignment'):
        validate_trajectory(broken, manifest)
    config = yaml.safe_load((PROJECT_ROOT / 'configs/imitation/bc.yaml').read_text())
    config.update(dataset=str(dataset), output_dir=str(tmp_path / 'bc'), epochs=2,
                  validation_episodes=1, test_episodes=1, batch_size=32, eval_episodes=1, device='cpu')
    summary = train_bc(config)
    assert summary['status'] == 'completed' and summary['critic_training'] is False
    assert all(v < 1e-4 for v in summary['folding_max_action_error'])
    assert summary['deterministic_evaluation']['mean_episode_length'] == 200
    loaded, metadata = MASAC.load(tmp_path / 'bc/best.pt', load_optimizers=False)
    assert metadata['training_stage'] == 'behavior_cloning'
    assert loaded.updates == 0 and not loaded.q_optimizer.state_dict()['state']
    assert torch.allclose(loaded.alpha, torch.full((2,), .2))
    target = model(1)
    warm_start_from_bc(target, tmp_path / 'bc/best.pt', expected_env_config=manifest['env_config'])

import random

import numpy as np
import pytest
import torch

from rel301m.agents.masac import MASAC as PublicMASAC
from rel301m.algorithms.masac import MASAC
from rel301m.agents.networks import GaussianActor as PublicActor
from rel301m.algorithms.networks import GaussianActor
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.training.train import load_config
from rel301m.utils.checkpoint import save_checkpoint, load_checkpoint
from rel301m.utils.seed import seed_everything


def test_public_agents_reuse_one_implementation():
    assert PublicMASAC is MASAC
    assert PublicActor is GaussianActor


def test_coordinator_rng_reproducibility():
    seed_everything(17)
    expected = random.random(), np.random.random(), torch.rand(7)
    seed_everything(17)
    actual = random.random(), np.random.random(), torch.rand(7)
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_critics_and_targets_have_independent_storage():
    config = load_config(PROJECT_ROOT/'configs/experiment/smoke.yaml')['algo']
    model = MASAC((66, 66), 119, [(-np.ones(7), np.ones(7))]*2, config)
    critics = [model.q1, model.q2, model.target_q1, model.target_q2]
    storage_sets = [{p.data_ptr() for p in module.parameters()} for module in critics]
    assert all(not storage_sets[i].intersection(storage_sets[j]) for i in range(4) for j in range(i))
    before = [module.net[0].weight.clone() for module in critics[1:]]
    with torch.no_grad():
        model.q1.net[0].weight.add_(5)
    for module, expected in zip(critics[1:], before):
        torch.testing.assert_close(module.net[0].weight, expected)


def test_checkpoint_global_step_and_legacy_loading(tmp_path):
    config = load_config(PROJECT_ROOT/'configs/experiment/smoke.yaml')['algo']
    model = MASAC((66, 66), 119, [(-np.ones(7), np.ones(7))]*2, config)
    path = tmp_path/'state.pt'
    save_checkpoint(model, path, {'step': 42})
    payload = torch.load(path, weights_only=True)
    assert payload['global_step'] == 42
    loaded, metadata = load_checkpoint(path)
    assert metadata['step'] == 42
    assert loaded.updates == model.updates
    del payload['global_step']
    torch.save(payload, path)
    legacy, metadata = MASAC.load(path)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[name])


def test_presets_freeze_shared_evaluation_and_algorithm():
    configs = [load_config(PROJECT_ROOT/f'configs/experiment/{name}.yaml') for name in ('smoke','pilot','final')]
    assert [c['final_eval_episodes'] for c in configs] == [10,50,100]
    assert [c['total_steps'] for c in configs] == [50000,300000,700000]
    assert all(c['eval_seed'] == 20000 for c in configs)
    assert configs[0]['algo'] == configs[1]['algo'] == configs[2]['algo']
    assert all(not c['predictor']['enabled'] for c in configs)

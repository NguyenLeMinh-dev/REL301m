from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from rel301m.algorithms.masac import MASAC
from rel301m.algorithms.networks import GaussianActor
from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.training.train import load_config, train
from rel301m.utils.diagnostics import (NonfiniteTrainingError, action_metrics,
                                       require_finite, td_error_metrics, finite_json)
from rel301m.utils.logger import initialization_metadata
from rel301m.utils.seed import seed_everything


def new_model(seed=42):
    seed_everything(seed)
    torch.set_num_threads(1)
    config = load_config(PROJECT_ROOT / 'configs/experiment/smoke.yaml')['algo']
    return MASAC((66, 66), 119, [(-np.ones(7), np.ones(7))] * 2, config)


def batch(size=16):
    data = {key: torch.randn(size, dim) for key, dim in dict(o0=66, o1=66, s=119, a0=7, a1=7,
            r=1, next_o0=66, next_o1=66, next_s=119).items()}
    data.update(done=torch.zeros(size, 1), timeout=torch.zeros(size, 1))
    return data


def test_initialization_hash_reproducibility_without_rng_side_effects():
    model = new_model(17)
    before = torch.get_rng_state().clone()
    metadata = initialization_metadata(model)
    torch.testing.assert_close(torch.get_rng_state(), before)
    assert metadata == initialization_metadata(new_model(17))
    assert metadata['initial_parameter_sha256']['model'] != initialization_metadata(new_model(18))['initial_parameter_sha256']['model']
    assert metadata['initialization'] == 'torch.nn.Linear default reset_parameters'
    hashes = metadata['initial_parameter_sha256']
    assert hashes['q1'] == hashes['target_q1'] and hashes['q2'] == hashes['target_q2']
    assert hashes['actor_0'] != hashes['actor_1']
    for group in metadata['optimizer_settings'].values():
        assert group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            bound = 1 / math.sqrt(layer.in_features)
            assert layer.weight.abs().max() <= bound and layer.bias.abs().max() <= bound


def test_instrumentation_preserves_rng_parameters_and_optimizer_state():
    plain = new_model()
    measured = deepcopy(plain)
    data = batch()
    torch.manual_seed(777)
    plain_metrics = plain.update(data, collect_diagnostics=False)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(777)
    metrics = measured.update(data, collect_diagnostics=True)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(value, measured.state_dict()[key], rtol=0, atol=0)
    for key in plain_metrics:
        assert metrics[key] == plain_metrics[key]
    for left, right in zip(plain.actor_optimizers + [plain.q_optimizer, plain.alpha_optimizer],
                           measured.actor_optimizers + [measured.q_optimizer, measured.alpha_optimizer]):
        for index, state in left.state_dict()['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, right.state_dict()['state'][index][key], rtol=0, atol=0)
    for i in range(2):
        assert metrics[f'entropy_{i}'] == -metrics[f'log_pi_mean_{i}']
        assert 0 <= metrics[f'action_saturation_{i}'] <= 1
        assert metrics[f'action_std_{i}'] > 0
    assert metrics['td_abs_p50'] <= metrics['td_abs_p95'] <= metrics['td_abs_p99']


@pytest.mark.parametrize('entropy, increases', [(-10., True), (0., False)])
def test_alpha_update_direction_matches_entropy_target(monkeypatch, entropy, increases):
    model = new_model()
    initial = model.alpha.detach().clone()
    def loss(index, data, *, diagnostics=None):
        differentiable = model.actors[index].net[0].weight.sum() * 0
        return differentiable, torch.full((len(data['o0']), 1), -entropy)
    monkeypatch.setattr(model, 'actor_loss', loss)
    metrics = model.update(batch(), collect_diagnostics=False)
    assert metrics['entropy_0'] == entropy
    assert torch.all(model.alpha > initial) if increases else torch.all(model.alpha < initial)


def test_td_percentiles_pool_both_critics_before_update():
    metrics = td_error_metrics(torch.tensor([[0.], [2.], [4.]]),
                               torch.tensor([[1.], [3.], [5.]]), torch.zeros(3, 1))
    assert metrics == pytest.approx(dict(td_abs_mean=2.5, td_abs_p50=2.5, td_abs_p95=4.75, td_abs_p99=4.95))


def test_action_diagnostics_use_live_affine_bounds_and_pre_tanh_std():
    actor = GaussianActor(66, np.full(7, -2), np.full(7, 4))
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.net[-1].bias[7:] = math.log(.5)
    normalized = torch.stack((torch.full((7,), .99), torch.zeros(7)))
    actions = actor.action_bias + actor.action_scale * normalized
    metrics = action_metrics(actor, torch.zeros(2, 66), actions)
    assert metrics['action_std'] == pytest.approx(.5)
    assert metrics['action_saturation'] == .5
    assert metrics['action_sample_std'] == pytest.approx(1.485)


@pytest.mark.parametrize('tensor', [False, True])
def test_numeric_guard_counts_nan_and_inf(tensor):
    values = np.array([0., np.nan, np.inf, -np.inf])
    with pytest.raises(NonfiniteTrainingError) as error:
        require_finite('test', torch.from_numpy(values) if tensor else values)
    assert error.value.nan_count == 1 and error.value.inf_count == 2
    assert finite_json({'bad': [float('nan'), float('inf')], 'good': 1.}) == {'bad': [None, None], 'good': 1.}


def test_nonfinite_replay_and_target_gradients_stop_before_optimizer_changes():
    model = new_model()
    original = deepcopy(model.state_dict())
    bad = batch()
    bad['r'][0] = float('nan')
    with pytest.raises(NonfiniteTrainingError):
        model.update(bad)
    assert model.updates == 0
    for key, value in original.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    next(model.target_q1.parameters()).grad = torch.zeros_like(next(model.target_q1.parameters()))
    with pytest.raises(RuntimeError, match='Target networks'):
        model.update(batch())


@pytest.mark.parametrize('failure_kind', ['controller', 'numeric'])
def test_failed_run_retains_diagnostics_traceback_and_simulator_state(tmp_path, monkeypatch, failure_kind):
    config = load_config(PROJECT_ROOT / 'configs/experiment/smoke.yaml')
    config.update(total_steps=1, eval_interval=1, eval_episodes=1, final_eval_episodes=1,
                  checkpoint_interval=1, log_interval=1, device='cpu')
    config['algo'].update(warmup_steps=0, batch_size=32, buffer_capacity=32)
    if failure_kind == 'controller':
        original_step = MultiAgentWrapper.step
        calls = 0
        def broken_step(self, *actions):
            nonlocal calls
            calls += 1
            if calls == 201:
                raise SystemError('injected opspace_matrices failure')
            return original_step(self, *actions)
        monkeypatch.setattr(MultiAgentWrapper, 'step', broken_step)
        error_type = SystemError
    else:
        def invalid_action(self, observations, deterministic=False):
            return np.array([np.nan] + [0.] * 6), np.zeros(7)
        monkeypatch.setattr(MASAC, 'act', invalid_action)
        error_type = NonfiniteTrainingError
    run = tmp_path / failure_kind
    with pytest.raises(error_type):
        train(config, run)
    failure = json.loads((run / 'failure.json').read_text())
    diagnostics = json.loads((run / 'diagnostics.json').read_text())
    assert failure['step'] == 1 and failure['completed_env_steps'] == 0
    assert 'Traceback' in failure['traceback']
    assert diagnostics['status'] == 'failed' and diagnostics['global_step'] == 0
    assert diagnostics['numeric']['nan_count'] == int(failure_kind == 'numeric')
    assert not (run / 'summary.json').exists() and not (run / 'final.pt').exists()
    with np.load(run / 'failure_state.npz') as state:
        assert state['joint_action'].shape == (14,) and state['critic_state'].shape == (119,)
        assert state['qpos'].ndim == state['qvel'].ndim == 1

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from rel301m.algorithms.masac import MASAC
from rel301m.algorithms.replay_buffer import ReplayBuffer
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.imitation.demo_finetune import DemonstrationReplay, demo_ratio, validate_fine_tune
from rel301m.imitation.demonstrations import file_sha256
from rel301m.training.train import load_config


def config():
    return load_config(PROJECT_ROOT / 'configs/experiment/bc_finetune.yaml')


def learner():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = config()['algo']
    cfg['hidden_dims'] = [32, 32]
    return MASAC([66, 66], 119, [(-np.ones(7), np.ones(7))] * 2, cfg)


def batch():
    return {key: torch.randn(8, width) for key, width in
            dict(o0=66, o1=66, s=119, a0=7, a1=7, r=1, next_o0=66, next_o1=66,
                 next_s=119, done=1, timeout=1).items()} | {'done': torch.ones(8, 1), 'timeout': torch.ones(8, 1)}


def test_small_std_initialization_preserves_mean_and_no_rng_consumption():
    model = learner()
    obs = torch.randn(8, 66)
    means = [actor(obs)[0].clone() for actor in model.actors]
    rng = torch.get_rng_state().clone()
    model.initialize_log_std(-3)
    model.configure_log_std()
    assert torch.equal(rng, torch.get_rng_state())
    for actor, mean in zip(model.actors, means):
        torch.testing.assert_close(actor(obs)[0], mean, atol=0, rtol=0)
        torch.testing.assert_close(actor(obs)[1], torch.full((8, 7), -3.0))
        sampled, logp = actor.sample(obs)
        assert (sampled.abs() <= 1).all() and torch.isfinite(logp).all()
    with pytest.raises(ValueError):
        model.initialize_log_std(float('nan'))


def test_critic_only_freezes_actor_alpha_and_optimizer_momentum():
    model = learner()
    data = batch()
    # Populate Adam state first: freeze must preserve it too.
    model.update(data, demo_batch=data)
    before = deepcopy(model.state_dict())
    optimizers = deepcopy([optimizer.state_dict() for optimizer in model.actor_optimizers])
    alpha_optimizer = deepcopy(model.alpha_optimizer.state_dict())
    metrics = model.update(data, update_actor=False)
    assert metrics['actor_updated'] == 0 and metrics['actor_0_grad_norm'] == 0
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items() if key.startswith('q1.'))
    for key, value in before.items():
        if key.startswith('actors.') or key == 'log_alpha':
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    for old, optimizer in zip(optimizers + [alpha_optimizer], model.actor_optimizers + [model.alpha_optimizer]):
        actual = optimizer.state_dict()
        assert old['param_groups'] == actual['param_groups']
        for index, state in old['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, actual['state'][index][key], rtol=0, atol=0)


def test_bc_term_changes_actor_gradient_without_teammate_critic_leakage():
    model = learner()
    obs = torch.randn(8, 66)
    mean, _ = model.actors[0](obs)
    loss = (mean.tanh() - .5).square().mean() * model.config['fine_tune']['lambda_bc']
    loss.backward()
    assert model.actors[0].net[-1].weight.grad[:7].abs().sum() > 0
    assert torch.count_nonzero(model.actors[0].net[-1].weight.grad[7:]) == 0
    assert all(p.grad is None for p in model.actors[1].parameters())
    assert all(p.grad is None for net in [model.q1, model.q2, model.target_q1, model.target_q2] for p in net.parameters())
    assert model.log_alpha.grad is None
    with pytest.raises(ValueError, match='demonstration batch'):
        model.update(batch())
    metrics = model.update(batch(), demo_batch=batch())
    assert metrics['bc_loss_0'] > 0 and metrics['bc_loss_1'] > 0
    assert all(np.isfinite(value) for value in metrics.values())


def test_frozen_std_schedule_fixed_alpha_and_checkpoint_roundtrip(tmp_path):
    model = learner()
    model.config['fine_tune'].update(alpha_mode='fixed', freeze_log_std_updates=2)
    model.initialize_log_std(-3)
    before_alpha = model.alpha.detach().clone()
    data = batch()
    for _ in range(2):
        model.update(data, demo_batch=data)
    torch.testing.assert_close(before_alpha, model.alpha, rtol=0, atol=0)
    assert not model.alpha_optimizer.state
    model.save(tmp_path / 'model.pt')
    restored, _ = MASAC.load(tmp_path / 'model.pt')
    assert restored.updates == 2
    restored.act([np.zeros(66), np.zeros(66)])
    assert all(actor.fixed_log_std is None for actor in restored.actors)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def dataset(tmp_path, model):
    directory = tmp_path / 'demos'
    directory.mkdir()
    sizes = dict(o0=66, o1=66, s=119, a0=7, a1=7, r=1, next_o0=66, next_o1=66, next_s=119, done=1, timeout=1)
    manifest = dict(format_version=1, status='completed', successful_episodes=4, env_config={'horizon': 3},
                    observation_dims=[66, 66], critic_state_dim=119, action_dims=[7, 7],
                    action_specs=[[[-1.] * 7, [1.] * 7]] * 2, episodes=[])
    for i in range(4):
        arrays = {key: np.full((3, width), i, np.float32) for key, width in sizes.items()}
        arrays.update(a0=np.zeros((3, 7), np.float32), a1=np.zeros((3, 7), np.float32),
                      done=np.array([[0], [0], [1]], np.float32), timeout=np.array([[0], [0], [1]], np.float32),
                      success=np.ones(4, dtype=bool), grasps=np.ones((4, 2), dtype=bool),
                      qpos=np.zeros((4, 1)), qvel=np.zeros((4, 1)))
        path = directory / f'episode_{i}.npz'
        np.savez_compressed(path, **arrays)
        manifest['episodes'].append(dict(episode=i, file=path.name, sha256=file_sha256(path)))
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    checkpoint = tmp_path / 'bc.pt'
    model.save(checkpoint, dict(env_config=manifest['env_config'], training_stage='behavior_cloning',
        demonstration_manifest_sha256=file_sha256(directory / 'manifest.json'),
        episode_split=dict(train=[0, 1], validation=[2], test=[3])))
    return directory, checkpoint


def test_train_only_prefill_retained_sampling_and_validation_separation(tmp_path):
    model = learner()
    directory, checkpoint = dataset(tmp_path, model)
    demos = DemonstrationReplay(directory, checkpoint, model, {'horizon': 3}, 7)
    assert len(demos.replay) == 6
    assert set(demos.replay.arrays['o0'][:, 0]) == {0, 1}
    assert torch.all(demos.validation[0][0] == 2)
    replay = ReplayBuffer(12, [66, 66], 119, [7, 7])
    demos.prefill(replay)
    assert len(replay) == 6 and set(replay.arrays['o0'][:6, 0]) == {0, 1}
    # Fill and overwrite the mutable replay completely; retained demos survive.
    for _ in range(24):
        replay.add(np.full(66, 100), np.full(66, 100), np.full(119, 100), np.zeros(7), np.zeros(7), 0,
                   np.full(66, 100), np.full(66, 100), np.full(119, 100), False)
    mixed = demos.sample_mixed(replay, 4, .75, 'cpu')
    assert (mixed['o0'][:3, 0] < 2).all() and mixed['o0'][3, 0] == 100
    assert set(demos.replay.arrays['o0'][:, 0]) == {0, 1}
    assert demos.metadata['test_used'] is False
    assert all(np.isfinite(v) for v in demos.validation_metrics(model).values())
    assert len(demos.sample_mixed(replay, 4, 0, 'cpu')['r']) == 4
    assert len(demos.sample_mixed(replay, 4, 1, 'cpu')['r']) == 4
    with pytest.raises(ValueError, match='capacity'):
        demos.prefill(ReplayBuffer(4, [66, 66], 119, [7, 7]))
    payload = torch.load(checkpoint, weights_only=True)
    payload['metadata']['episode_split']['train'].append(2)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='overlapping'):
        DemonstrationReplay(directory, checkpoint, model, {'horizon': 3}, 7)


@pytest.mark.parametrize('change', [dict(lambda_bc=-1), dict(demo_ratio_initial=2),
    dict(actor_init_log_std=float('nan')), dict(actor_freeze_updates=True),
    dict(critic_pretrain_lr=0), dict(prefill_replay=1), dict(alpha_mode='invalid')])
def test_invalid_fine_tune_configuration(change):
    cfg = config()
    cfg['algo']['fine_tune'].update(change)
    with pytest.raises(ValueError):
        validate_fine_tune(cfg)


def test_short_budget_and_ratio_schedule_keep_nominal_baseline_unchanged():
    cfg = config()
    settings = cfg['algo']['fine_tune']
    assert demo_ratio(settings, 0) == .75
    assert demo_ratio(settings, 5000) == .5
    assert demo_ratio(settings, 30000) == .5
    cfg['total_steps'] = 300001
    with pytest.raises(ValueError, match='30k'):
        validate_fine_tune(cfg)
    nominal = load_config(PROJECT_ROOT / 'configs/experiment/bc_pilot.yaml')
    assert 'fine_tune' not in nominal['algo'] and nominal['algo']['warmup_steps'] == 10000
    assert nominal['algo']['initial_alpha'] == .2


def test_auxiliary_loss_is_applied_to_actual_masac_update():
    assisted = learner()
    baseline = deepcopy(assisted)
    baseline.config['fine_tune']['lambda_bc'] = 0
    data, expert = batch(), batch()
    expert['a0'].fill_(.9)
    expert['a1'].fill_(-.9)
    rng = torch.get_rng_state().clone()
    base_metrics = baseline.update(data)
    torch.set_rng_state(rng)
    assisted_metrics = assisted.update(data, demo_batch=expert)
    for i in range(2):
        assert assisted_metrics[f'actor_{i}_loss'] != base_metrics[f'actor_{i}_loss']
        assert any(not torch.equal(a, b) for a, b in zip(assisted.actors[i].parameters(), baseline.actors[i].parameters()))
    for actual, expected in zip(assisted.q1.parameters(), baseline.q1.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(p.grad is None for net in [assisted.q1, assisted.q2, assisted.target_q1, assisted.target_q2] for p in net.parameters())

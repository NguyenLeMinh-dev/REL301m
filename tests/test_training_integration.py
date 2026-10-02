import csv
import json

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rel301m.algorithms.masac import MASAC
from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.training.train import load_config, train


def test_one_thousand_step_training_integration(tmp_path):
    config = load_config(PROJECT_ROOT / 'configs/experiment/smoke.yaml')
    config.update(name='integration', total_steps=1000, eval_interval=500, eval_episodes=2, final_eval_episodes=2,
                  checkpoint_interval=500, log_interval=100,
                  device='cuda' if torch.cuda.is_available() else 'cpu')
    config['algo'].update(warmup_steps=256, batch_size=32, buffer_capacity=2000)
    run_dir = tmp_path / 'integration'
    summary = train(config, run_dir)
    assert summary['status'] == 'completed'
    assert summary['steps'] == 1000
    assert summary['updates'] == 745
    assert np.isfinite(list(summary['last_update_metrics'].values())).all()
    assert summary['last_update_metrics']['q_grad_norm'] > 0
    with (run_dir / 'episodes.csv').open() as stream:
        episodes = list(csv.DictReader(stream))
    assert len(episodes) == 5
    assert all(int(row['episode_length']) == 200 and row['partial'] == 'False' for row in episodes)
    assert all(row['ever_success'] in ('True', 'False') for row in episodes)
    if summary['cuda_peak_allocated_bytes'] is not None:
        assert summary['cuda_peak_allocated_bytes'] < 1024 ** 3
    metadata = json.loads((run_dir / 'metadata.json').read_text())
    assert metadata['observation_dims'] == [66, 66]
    assert metadata['action_dims'] == [7, 7]
    assert metadata['critic_state_dim'] == 119
    assert metadata['target_entropy'] == [-7, -7]
    assert metadata['predictor_enabled'] is False
    assert metadata['communication_cost'] == 'N/A'
    assert metadata['evaluation_protocol']['seed'] == 20000
    assert metadata['initialization'] == 'torch.nn.Linear default reset_parameters'
    assert len(metadata['initial_parameter_sha256']['model']) == 64
    assert metadata['optimizer_settings']['critic']['betas'] == [.9, .999]
    assert summary['random_baseline']['initialization_sequence_sha256'] == summary['final_evaluation']['initialization_sequence_sha256']
    assert summary['final_evaluation']['episodes'] == 2
    for name in ('resolved_config.yaml', 'env.yaml', 'requirements.freeze.txt', 'git_status.txt',
                 'final.pt', 'best.pt', 'checkpoint_0000500.pt', 'summary.json', 'diagnostics.json', 'random_baseline.json'):
        assert (run_dir / name).is_file()
    diagnostics = json.loads((run_dir / 'diagnostics.json').read_text())
    assert diagnostics['loss_check_status'] == 'PASS'
    assert diagnostics['updates'] == 745
    assert diagnostics['raw_done_count'] == diagnostics['timeout_count'] == 5
    assert all(v > 0 for v in diagnostics['gradient_norm_min'].values())
    assert diagnostics['random_initialization_sequence'] == diagnostics['final_initialization_sequence']
    assert diagnostics['status'] == 'completed' and diagnostics['global_step'] == 1000
    assert diagnostics['numeric'] == dict(nan_count=0, inf_count=0, action_bound_violations=0)
    assert diagnostics['last_diagnostic_step'] == 1000
    for i in range(2):
        agent = diagnostics[f'agent_{i}']
        assert agent['entropy_final'] == -agent['log_pi_mean_final']
        assert agent['actor_grad_norm_max'] >= agent['actor_grad_norm_min'] > 0
        assert agent['action_std_mean'] > 0 and 0 <= agent['action_saturation_fraction'] <= 1
    critic = diagnostics['critic']
    assert critic['td_abs_p50_final'] <= critic['td_abs_p95_final'] <= critic['td_abs_p99_final']
    assert diagnostics['environment']['timeouts'] == 5
    restored, saved_metadata = MASAC.load(run_dir / 'final.pt', load_optimizers=False)
    assert restored.updates == 745
    assert saved_metadata['step'] == 1000
    events = EventAccumulator(str(run_dir / 'tensorboard')).Reload()
    assert 'train/episode_return' in events.Tags()['scalars']
    assert 'eval/success_rate' in events.Tags()['scalars']
    assert 'losses/alpha_0' in events.Tags()['scalars']
    assert {'train/log_pi_mean_0', 'train/entropy_0', 'critic/td_abs_p99',
            'action/std_0', 'action/saturation_1', 'grad/q_norm', 'numeric/nan_count'} <= set(events.Tags()['scalars'])

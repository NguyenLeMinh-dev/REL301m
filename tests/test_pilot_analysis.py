import csv
import importlib.util
import json

import pytest

from rel301m.envs.robosuite_factory import PROJECT_ROOT
from rel301m.evaluation.metrics import EPISODE_FIELDS, episode_record, summarize_episodes


spec = importlib.util.spec_from_file_location('pilot_analysis', PROJECT_ROOT / 'scripts/analyze_pilot.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episodes(count, successes):
    return [episode_record(i, 100. if i < successes else 0., i < successes,
                           100 if i < successes else None, i < successes, 200, 20, f'{i:064x}')
            for i in range(count)]


@pytest.fixture
def runs(tmp_path):
    result = {}
    for seed in (0, 1, 2):
        path = tmp_path / f'seed{seed}'
        path.mkdir()
        result[seed] = path
        config = dict(seed=seed, total_steps=300000, eval_seed=20000, eval_interval=10000,
                      eval_episodes=10, final_eval_episodes=50, algo=analysis.FROZEN_ALGO)
        metadata = dict(seed=seed, config=config, env_config=analysis.FROZEN_ENV, predictor_enabled=False,
                        observation_dims=[66, 66], action_dims=[7, 7], critic_state_dim=119,
                        initialization='torch.nn.Linear default reset_parameters',
                        initial_parameter_sha256={key: f'{seed:064x}' for key in ('model', *analysis.CRITIC_NAMES)},
                        device='cuda',
                        packages={'torch': 'fixture'}, source_sha256={'fixture': 'identical'})
        write_json(path / 'metadata.json', metadata)
        final, random = episodes(50, 26), episodes(50, 0)
        write_rows(path / 'eval_0300000_episodes.csv', final)
        write_rows(path / 'random_baseline_episodes.csv', random)
        final_summary, random_summary = summarize_episodes(final), summarize_episodes(random)
        write_json(path / 'random_baseline.json', random_summary)
        write_json(path / 'summary.json', dict(status='completed', steps=300000, updates=290001,
                                              final_evaluation=final_summary))
        write_json(path / 'diagnostics.json', dict(status='completed', global_step=300000, updates=290001,
              numeric=dict(nan_count=0, inf_count=0, action_bound_violations=0), loss_check_status='PASS',
              final_initialization_sequence=final_summary['initialization_sequence_sha256']))
        # Placeholder exists only inside this synthetic artifact-schema test.
        (path / 'final.pt').write_bytes(b'test fixture')
        periodic = []
        for step in range(10000, 300001, 10000):
            rows = final if step == 300000 else episodes(10, min(5, step // 50000))
            write_rows(path / f'eval_{step:07d}_episodes.csv', rows)
            periodic.append(dict(step=step, **summarize_episodes(rows)))
        write_rows(path / 'evaluation.csv', periodic)
    return result


def test_gate_requires_all_seeds_paired_csv_and_explicit_curve_review(runs):
    report = analysis.analyze(runs)
    assert report['status'] == 'REVIEW_REQUIRED'
    assert report['aggregate']['mean_success_rate'] == .52
    assert report['aggregate']['std_success_rate'] == 0
    assert report['aggregate']['std_return'] == 0
    assert report['aggregate']['std_final_success_rate'] == 0
    assert analysis.analyze(runs, 'pass')['status'] == 'PASS'
    assert analysis.analyze(runs, 'fail')['status'] == 'LEARNING_FAIL'


def test_failed_or_missing_seed_is_incomplete_not_learning_failure(runs):
    path = runs[1]
    (path / 'summary.json').unlink()
    write_json(path / 'failure.json', dict(step=18713, error='controller exception'))
    report = analysis.analyze(runs, 'pass')
    assert report['status'] == 'INCOMPLETE' and report['aggregate'] is None
    assert report['seeds'][1]['failure']['step'] == 18713
    assert analysis.analyze({0: runs[0]})['status'] == 'INCOMPLETE'


@pytest.mark.parametrize('corruption', ['summary', 'paired_hash', 'numeric', 'checkpoint', 'versions', 'horizon', 'hyperparameters'])
def test_corrupt_or_incomparable_artifacts_cannot_pass(runs, corruption):
    path = runs[0]
    if corruption == 'summary':
        data = analysis.read_json(path / 'summary.json')
        data['final_evaluation']['success_rate'] = 1.
        write_json(path / 'summary.json', data)
    elif corruption in ('paired_hash', 'horizon'):
        rows = analysis.read_csv(path / 'eval_0300000_episodes.csv')
        rows[0]['initial_state_sha256' if corruption == 'paired_hash' else 'episode_length'] = 'f' * 64 if corruption == 'paired_hash' else 199
        write_rows(path / 'eval_0300000_episodes.csv', rows)
    elif corruption == 'numeric':
        data = analysis.read_json(path / 'diagnostics.json')
        data['numeric']['nan_count'] = 1
        write_json(path / 'diagnostics.json', data)
    elif corruption == 'checkpoint':
        (path / 'final.pt').unlink()
    else:
        data = analysis.read_json(path / 'metadata.json')
        if corruption == 'versions':
            data['packages']['torch'] = 'different version'
        else:
            data['config']['algo']['gamma'] = .5
        write_json(path / 'metadata.json', data)
    assert analysis.analyze(runs, 'pass')['status'] == 'INVALID'


def test_nonfinite_metric_and_invalid_flag_are_rejected():
    with pytest.raises(ValueError):
        analysis.flag('maybe')
    with pytest.raises(ValueError):
        analysis.number('nan')


def mark_bc(runs):
    for path in runs.values():
        metadata = analysis.read_json(path / 'metadata.json')
        metadata.update(training_variant='bc_initialized_masac',
                        initialization='BC actors; torch.nn.Linear default critics',
                        actor_initialization=dict(method='behavior_cloning', checkpoint_sha256='b' * 64))
        write_json(path / 'metadata.json', metadata)


def test_bc_gate_requires_same_checkpoint_across_seeds(runs):
    mark_bc(runs)
    assert analysis.analyze(runs, 'pass')['status'] == 'PASS'
    metadata = analysis.read_json(runs[1] / 'metadata.json')
    metadata['actor_initialization']['checkpoint_sha256'] = 'c' * 64
    write_json(runs[1] / 'metadata.json', metadata)
    assert analysis.analyze(runs, 'pass')['status'] == 'INVALID'


def test_bc_reference_comparison_rejects_unpaired_or_changed_core(runs):
    from copy import deepcopy
    reference = analysis.analyze(runs)
    mark_bc(runs)
    candidate = analysis.analyze(runs)
    core = ('src/rel301m/algorithms/masac.py', 'src/rel301m/algorithms/networks.py',
            'src/rel301m/algorithms/replay_buffer.py', 'src/rel301m/envs/multi_agent_wrapper.py',
            'src/rel301m/envs/robosuite_factory.py', 'configs/algo/masac.yaml', 'configs/env/two_arm_lift.yaml')
    for report in (candidate, reference):
        for seed in report['seeds']:
            seed['source_sha256'] = {key: 'fixture-identical' for key in core}
    compared = analysis.compare_methods(candidate, reference)
    assert compared['status'] == 'COMPARED' and compared['mean_success_rate_difference'] == 0
    assert compared['seeds_beating_scratch'] == 0 and compared['seeds_tied'] == 3
    assert compared['warm_start_gate']['status'] == 'NOT_MET'
    changed = deepcopy(candidate)
    changed['seeds'][0]['source_sha256'][core[0]] = 'changed'
    assert analysis.compare_methods(changed, reference)['status'] == 'INVALID'
    changed = deepcopy(candidate)
    changed['seeds'][0]['final']['initialization_sequence_sha256'] = 'f' * 64
    assert analysis.compare_methods(changed, reference)['status'] == 'INVALID'
    assert analysis.compare_methods(dict(status='INCOMPLETE'), reference)['status'] == 'INCOMPLETE'


def paired_reports(runs):
    reference = analysis.analyze(runs, 'pass')
    mark_bc(runs)
    candidate = analysis.analyze(runs, 'pass')
    core = ('src/rel301m/algorithms/masac.py', 'src/rel301m/algorithms/networks.py',
            'src/rel301m/algorithms/replay_buffer.py', 'src/rel301m/envs/multi_agent_wrapper.py',
            'src/rel301m/envs/robosuite_factory.py', 'configs/algo/masac.yaml', 'configs/env/two_arm_lift.yaml')
    for report in (candidate, reference):
        for seed in report['seeds']:
            seed['source_sha256'] = {key: 'fixture-identical' for key in core}
    return candidate, reference


@pytest.mark.parametrize('corruption', ['seed', 'device', 'critic', 'missing_critic'])
def test_comparison_requires_paired_execution_and_fresh_critics(runs, corruption):
    candidate, reference = paired_reports(runs)
    source = reference['seeds'][0]
    if corruption == 'seed':
        source['seed'] = 1
    elif corruption == 'device':
        source['device'] = 'cpu'
    else:
        source['initial_critic_sha256']['q1'] = 'changed' if corruption == 'critic' else None
    assert analysis.compare_methods(candidate, reference)['status'] == 'INVALID'


def test_warm_start_gate_keeps_absolute_gate_and_counts_strict_wins(runs):
    candidate, reference = paired_reports(runs)
    for seed in candidate['seeds'][:2]:
        seed['final']['success_rate'] = .7
    comparison = analysis.compare_methods(candidate, reference)
    assert comparison['seeds_beating_scratch'] == 2 and comparison['seeds_tied'] == 1
    assert comparison['warm_start_gate']['status'] == 'PASS'
    candidate['status'] = 'LEARNING_FAIL'
    assert analysis.compare_methods(candidate, reference)['warm_start_gate']['status'] == 'NOT_MET'
    candidate['status'] = 'REVIEW_REQUIRED'
    assert analysis.compare_methods(candidate, reference)['warm_start_gate']['status'] == 'REVIEW_REQUIRED'


def test_milestones_are_observed_crossings_not_interpolated_or_uncensored():
    curve = [dict(step=s, success_rate=r) for s, r in zip((10000, 20000, 30000, 40000), (.1, .5, .2, .6))]
    result = analysis.success_milestones(curve)
    assert result['first_observed_step'] == 20000
    assert result['two_consecutive_confirmation_step'] is None
    curve.append(dict(step=50000, success_rate=.5))
    assert analysis.success_milestones(curve)['two_consecutive_confirmation_step'] == 50000
    never = analysis.success_milestones([dict(step=10000, success_rate=.4)])
    assert never['not_reached'] and never['first_observed_step'] is None


def test_final_fifty_episode_metric_does_not_change_ten_episode_milestone(runs):
    report = analysis.analyze(runs)
    seed = report['seeds'][0]
    assert seed['final']['episodes'] == 50 and seed['final']['success_rate'] == .52
    assert seed['monitoring_curve'][-1]['episodes'] == 10
    assert seed['monitoring_curve'][-1]['success_rate'] == 1.
    assert seed['success_milestones']['first_observed_step'] == 250000
    assert seed['success_milestones']['two_consecutive_confirmation_step'] == 260000


def test_comparison_tables_use_valid_results_and_do_not_fabricate_missing_seeds(runs, tmp_path):
    candidate, reference = paired_reports(runs)
    candidate['reference'] = reference
    analysis.write_comparison_tables(candidate, tmp_path)
    final_rows = analysis.read_csv(tmp_path / 'final_results.csv')
    assert len(final_rows) == 6 and all(row['episodes'] == '50' for row in final_rows)
    milestone_rows = analysis.read_csv(tmp_path / 'success_milestones.csv')
    assert len(milestone_rows) == 6 and all(row['episodes_per_eval'] == '10' for row in milestone_rows)
    empty = tmp_path / 'incomplete'
    empty.mkdir()
    analysis.write_comparison_tables(analysis.analyze({}), empty)
    assert not (empty / 'final_results.csv').exists()


def test_never_reached_milestone_has_no_invented_step_savings(runs):
    candidate, reference = paired_reports(runs)
    reference['seeds'][0]['success_milestones']['first_observed_step'] = None
    result = analysis.compare_methods(candidate, reference)['success_milestone_comparison'][0]
    assert result['scratch_first_observed_step'] is None
    assert result['saved_rl_steps'] is None and result['bc_reached_earlier'] is None

"""Read frozen pilot artifacts, recompute success, and plot available CSV metrics.

Uses only the standard library unless --plots requests the installed matplotlib.
Never trains, evaluates policies, edits run artifacts, or picks a best checkpoint.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


FROZEN_ALGO = dict(hidden_dims=[256, 256], actor_lr=3e-4, critic_lr=3e-4,
                   alpha_lr=3e-4, gamma=.99, tau=.005, batch_size=256,
                   buffer_capacity=300000, warmup_steps=10000, updates_per_step=1,
                   initial_alpha=.2, target_entropy='auto', bootstrap_time_limits=True)
FROZEN_ENV = dict(env_name='TwoArmLift', robots=['Panda', 'Panda'], env_configuration='opposed',
                  controller='BASIC', has_renderer=False, has_offscreen_renderer=False,
                  use_camera_obs=False, use_object_obs=True, reward_shaping=True,
                  reward_scale=1., control_freq=20, horizon=200, ignore_done=False)
MILESTONE_SR = .5
CRITIC_NAMES = ('q1', 'q2', 'target_q1', 'target_q2')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def flag(value):
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    raise ValueError(f'Invalid success flag {value!r}')


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Nonfinite metric {value!r}')
    return result


def episode_metrics(path, expected=50):
    rows = read_csv(path)
    if len(rows) != expected:
        raise ValueError(f'{path.name}: expected {expected} episodes, found {len(rows)}')
    returns, ever, final, first_steps, hashes = [], [], [], [], []
    for episode, row in enumerate(rows):
        if int(row['episode']) != episode or int(row['episode_length']) != 200:
            raise ValueError(f'{path.name}: episode order/horizon mismatch')
        success, final_success = flag(row['ever_success']), flag(row['final_success'])
        first = int(row['first_success_step']) if row['first_success_step'] else None
        if success != (first is not None) or (final_success and not success):
            raise ValueError(f'{path.name}: inconsistent success semantics')
        if first is not None and not 0 <= first <= 200:
            raise ValueError(f'{path.name}: invalid first-success step')
        if first is not None and not math.isclose(number(row['first_success_time_s']), first / 20):
            raise ValueError(f'{path.name}: invalid first-success seconds')
        digest = row['initial_state_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'{path.name}: invalid initial-state hash')
        returns.append(number(row['episode_return']))
        ever.append(success)
        final.append(final_success)
        hashes.append(digest)
        if first is not None:
            first_steps.append(first)
    return dict(episodes=expected, mean_return=statistics.mean(returns),
                success_rate=statistics.mean(ever), final_success_rate=statistics.mean(final),
                mean_first_success_step=statistics.mean(first_steps) if first_steps else None,
                mean_first_success_time_s=statistics.mean(first_steps) / 20 if first_steps else None,
                mean_episode_length=200.,
                initialization_sequence_sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest())


def check_summary(actual, recorded, label):
    for key, value in actual.items():
        observed = recorded.get(key)
        if isinstance(value, (int, float)):
            if not isinstance(observed, (int, float)) or not math.isclose(value, observed, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f'{label}: {key} differs from episode CSV')
        elif value != observed:
            raise ValueError(f'{label}: {key} differs from episode CSV')


def success_milestones(curve):
    """First observed crossings on the fixed 10-episode monitoring sequence."""
    first, confirmed = None, None
    previous = None
    for point in curve:
        if point['success_rate'] >= MILESTONE_SR:
            if first is None:
                first = point['step']
            if previous is not None and previous['success_rate'] >= MILESTONE_SR and confirmed is None:
                confirmed = point['step']
        previous = point
    return dict(threshold=MILESTONE_SR, episodes_per_eval=10,
                first_observed_step=first, two_consecutive_confirmation_step=confirmed,
                not_reached=first is None, monitoring_budget_steps=300000,
                interpretation='10k evaluation grid; observed crossing, not exact convergence or an independent test')


def analyze_run(path, seed):
    result = dict(seed=seed, run_dir=str(path.resolve()) if path is not None else None,
                  complete=False, valid=False, errors=[])
    if path is None or not path.is_dir():
        result['errors'].append('Missing seed directory')
        return result
    if (path / 'failure.json').is_file():
        failure = read_json(path / 'failure.json')
        result['failure'] = failure
        result['errors'].append(f"Run failed at attempted step {failure.get('step')}: {failure.get('error')}")
    if not (path / 'summary.json').is_file():
        result['errors'].append('No completed summary.json; partial run cannot pass the pilot gate')
        return result
    try:
        summary, metadata = read_json(path / 'summary.json'), read_json(path / 'metadata.json')
        result['complete'] = summary.get('status') == 'completed' and summary.get('steps') == 300000
        if not result['complete']:
            raise ValueError('Expected a completed 300k run')
        config = metadata['config']
        expected = dict(seed=seed, total_steps=300000, eval_seed=20000,
                        eval_interval=10000, eval_episodes=10, final_eval_episodes=50)
        if any(config.get(k) != v for k, v in expected.items()) or metadata['seed'] != seed:
            raise ValueError('Frozen training/evaluation protocol mismatch')
        if any(config['algo'].get(k) != v for k, v in FROZEN_ALGO.items()):
            raise ValueError('Frozen MASAC hyperparameter mismatch')
        if any(metadata['env_config'].get(k) != v for k, v in FROZEN_ENV.items()):
            raise ValueError('Frozen environment parameter mismatch')
        if metadata.get('predictor_enabled') is not False or config.get('predictor', {}).get('enabled', False):
            raise ValueError('Predictor must remain disabled')
        variant = metadata.get('training_variant', 'scratch_masac')
        expected_init = 'BC actors; torch.nn.Linear default critics' if variant == 'bc_initialized_masac' else 'torch.nn.Linear default reset_parameters'
        if variant not in ('scratch_masac', 'bc_initialized_masac') or metadata.get('initialization') != expected_init:
            raise ValueError('Initialization scheme missing or inconsistent with training variant')
        actor_init = metadata.get('actor_initialization', {'method': 'random'})
        if variant == 'bc_initialized_masac' and (actor_init.get('method') != 'behavior_cloning' or not actor_init.get('checkpoint_sha256')):
            raise ValueError('Missing BC checkpoint provenance')
        initial_hash = metadata.get('initial_parameter_sha256', {}).get('model', '')
        if len(initial_hash) != 64 or any(c not in '0123456789abcdef' for c in initial_hash):
            raise ValueError('Missing/invalid initial parameter hash')
        if metadata['observation_dims'] != [66, 66] or metadata['action_dims'] != [7, 7] or metadata['critic_state_dim'] != 119:
            raise ValueError('Phase 2 dimension mismatch')
        if not (path / 'final.pt').is_file():
            raise ValueError('Missing final.pt; best.pt does not satisfy the primary gate')
        final = episode_metrics(path / 'eval_0300000_episodes.csv')
        random = episode_metrics(path / 'random_baseline_episodes.csv')
        check_summary(final, summary['final_evaluation'], 'final summary')
        check_summary(random, read_json(path / 'random_baseline.json'), 'random summary')
        if final['initialization_sequence_sha256'] != random['initialization_sequence_sha256']:
            raise ValueError('Random/final evaluation initialization mismatch')
        diagnostics = read_json(path / 'diagnostics.json')
        if diagnostics.get('status') != 'completed' or diagnostics.get('global_step') != 300000:
            raise ValueError('Diagnostics do not cover a completed 300k run')
        if diagnostics.get('updates') != 290001 or summary.get('updates') != 290001:
            raise ValueError('Unexpected optimizer update count')
        if any(diagnostics['numeric'][key] != 0 for key in ('nan_count', 'inf_count', 'action_bound_violations')):
            raise ValueError('Numeric failure or action-bound violation')
        if diagnostics.get('loss_check_status') != 'PASS':
            raise ValueError('Update numeric checks not exercised')
        if diagnostics.get('final_initialization_sequence') != final['initialization_sequence_sha256']:
            raise ValueError('Diagnostics/final initialization mismatch')
        periodic = read_csv(path / 'evaluation.csv')
        monitoring_curve = []
        expected_steps = list(range(10000, 300001, 10000))
        if [int(r['step']) for r in periodic] != expected_steps:
            raise ValueError('Periodic evaluation schedule mismatch')
        for row in periodic:
            count = 50 if int(row['step']) == 300000 else 10
            recomputed = episode_metrics(path / f"eval_{int(row['step']):07d}_episodes.csv", count)
            typed = {k: (None if row[k] == '' else number(row[k])) for k in recomputed if k != 'initialization_sequence_sha256'}
            typed['initialization_sequence_sha256'] = row['initialization_sequence_sha256']
            check_summary(recomputed, typed, f"evaluation step {row['step']}")
            reference = read_csv(path / 'random_baseline_episodes.csv')[:count]
            paired = hashlib.sha256(''.join(r['initial_state_sha256'] for r in reference).encode()).hexdigest()
            if recomputed['initialization_sequence_sha256'] != paired:
                raise ValueError('Periodic evaluation initialization differs from random reference')
            # Final primary eval uses 50 episodes; milestones keep the same first 10.
            monitoring_rows = read_csv(path / f"eval_{int(row['step']):07d}_episodes.csv")[:10]
            monitoring_curve.append(dict(step=int(row['step']), episodes=10,
                success_rate=statistics.mean(flag(r['ever_success']) for r in monitoring_rows),
                final_success_rate=statistics.mean(flag(r['final_success']) for r in monitoring_rows),
                mean_return=statistics.mean(number(r['episode_return']) for r in monitoring_rows)))
        if result['errors']:
            return result
        result.update(valid=True, final=final, random=random, beats_random=final['success_rate'] > random['success_rate'],
                      packages=metadata['packages'], algorithm=config['algo'], source_sha256=metadata['source_sha256'],
                      training_variant=variant, actor_initialization=actor_init,
                      initial_critic_sha256={key: metadata.get('initial_parameter_sha256', {}).get(key) for key in CRITIC_NAMES},
                      device=metadata.get('device'), monitoring_curve=monitoring_curve,
                      success_milestones=success_milestones(monitoring_curve))
    except (ValueError, KeyError, OSError, TypeError) as error:
        result['errors'].append(str(error))
    return result


def analyze(runs, curve_review='pending'):
    results = [analyze_run(runs.get(seed), seed) for seed in (0, 1, 2)]
    report = dict(status='INCOMPLETE', learning_curve_review=curve_review, seeds=results,
                  gate=None, aggregate=None, errors=[])
    if not all(r['complete'] for r in results):
        return report
    if not all(r['valid'] for r in results):
        report['status'] = 'INVALID'
        return report
    for key in ('packages', 'algorithm', 'source_sha256', 'training_variant', 'actor_initialization'):
        if any(r[key] != results[0][key] for r in results[1:]):
            report['errors'].append(f'Across-seed {key} mismatch')
    if len({r['final']['initialization_sequence_sha256'] for r in results}) != 1:
        report['errors'].append('Across-seed initialization mismatch')
    if report['errors']:
        report['status'] = 'INVALID'
        return report
    rates = [r['final']['success_rate'] for r in results]
    report['aggregate'] = dict(mean_success_rate=statistics.mean(rates), std_success_rate=statistics.stdev(rates),
                               mean_final_success_rate=statistics.mean(r['final']['final_success_rate'] for r in results),
                               std_final_success_rate=statistics.stdev(r['final']['final_success_rate'] for r in results),
                               mean_return=statistics.mean(r['final']['mean_return'] for r in results),
                               std_return=statistics.stdev(r['final']['mean_return'] for r in results))
    report['gate'] = dict(mean_sr_at_least_50_percent=statistics.mean(rates) >= .5,
                         at_least_two_seeds_beat_random=sum(r['beats_random'] for r in results) >= 2,
                         no_zero_success_seed=all(rate > 0 for rate in rates))
    numeric_pass = all(report['gate'].values())
    report['status'] = 'LEARNING_FAIL' if not numeric_pass or curve_review == 'fail' else ('PASS' if curve_review == 'pass' else 'REVIEW_REQUIRED')
    return report


def compare_methods(candidate, reference):
    if any(report['status'] == 'INCOMPLETE' for report in (candidate, reference)):
        return dict(status='INCOMPLETE', mean_success_rate_difference=None)
    if any(report['status'] == 'INVALID' for report in (candidate, reference)):
        return dict(status='INVALID', mean_success_rate_difference=None)
    differences = []
    core_files = ('src/rel301m/algorithms/masac.py', 'src/rel301m/algorithms/networks.py',
                  'src/rel301m/algorithms/replay_buffer.py', 'src/rel301m/envs/multi_agent_wrapper.py',
                  'src/rel301m/envs/robosuite_factory.py', 'configs/algo/masac.yaml', 'configs/env/two_arm_lift.yaml')
    for bc, scratch in zip(candidate['seeds'], reference['seeds']):
        if bc['training_variant'] != 'bc_initialized_masac' or scratch['training_variant'] != 'scratch_masac':
            return dict(status='INVALID', reason='Expected BC-init candidate and scratch reference')
        if bc['packages'] != scratch['packages'] or bc['algorithm'] != scratch['algorithm']:
            return dict(status='INVALID', reason='Software/hyperparameter mismatch')
        if bc['seed'] != scratch['seed'] or bc['device'] is None or bc['device'] != scratch['device']:
            return dict(status='INVALID', reason='Training seed or execution device mismatch')
        if any(not isinstance(bc['initial_critic_sha256'][key], str) or
               len(bc['initial_critic_sha256'][key]) != 64 or
               any(c not in '0123456789abcdef' for c in bc['initial_critic_sha256'][key]) or
               bc['initial_critic_sha256'][key] != scratch['initial_critic_sha256'][key] for key in CRITIC_NAMES):
            return dict(status='INVALID', reason='Missing or mismatched fresh critic/target initialization hashes')
        if bc['final']['initialization_sequence_sha256'] != scratch['final']['initialization_sequence_sha256']:
            return dict(status='INVALID', reason='Evaluation initializations mismatch')
        if any(bc['source_sha256'].get(key) is None or bc['source_sha256'].get(key) != scratch['source_sha256'].get(key) for key in core_files):
            return dict(status='INVALID', reason='Core learner/environment source mismatch')
        differences.append(dict(seed=bc['seed'],
             success_rate=bc['final']['success_rate'] - scratch['final']['success_rate'],
             final_success_rate=bc['final']['final_success_rate'] - scratch['final']['final_success_rate'],
             mean_return=bc['final']['mean_return'] - scratch['final']['mean_return']))
    wins = sum(d['success_rate'] > 0 for d in differences)
    mean_difference = statistics.mean(d['success_rate'] for d in differences)
    relative_gate = wins >= 2 and mean_difference > 0
    if not relative_gate or candidate['status'] == 'LEARNING_FAIL':
        warm_start_status = 'NOT_MET'
    else:
        warm_start_status = 'PASS' if candidate['status'] == 'PASS' else 'REVIEW_REQUIRED'
    milestone_pairs = []
    for bc, scratch in zip(candidate['seeds'], reference['seeds']):
        bc_step = bc['success_milestones']['first_observed_step']
        scratch_step = scratch['success_milestones']['first_observed_step']
        milestone_pairs.append(dict(seed=bc['seed'], bc_first_observed_step=bc_step,
            scratch_first_observed_step=scratch_step,
            saved_rl_steps=scratch_step - bc_step if bc_step is not None and scratch_step is not None else None,
            bc_reached_earlier=bc_step < scratch_step if bc_step is not None and scratch_step is not None else None))
    return dict(status='COMPARED', per_seed_differences=differences,
                mean_success_rate_difference=mean_difference,
                std_success_rate_difference=statistics.stdev(d['success_rate'] for d in differences),
                mean_final_success_rate_difference=statistics.mean(d['final_success_rate'] for d in differences),
                mean_return_difference=statistics.mean(d['mean_return'] for d in differences),
                std_return_difference=statistics.stdev(d['mean_return'] for d in differences),
                seeds_beating_scratch=wins, seeds_tied=sum(d['success_rate'] == 0 for d in differences),
                warm_start_gate=dict(status=warm_start_status, at_least_two_seeds_beat_scratch=wins >= 2,
                                     positive_mean_sr_difference=mean_difference > 0,
                                     nominal_learning_status=candidate['status']),
                success_milestone_comparison=milestone_pairs,
                interpretation='Descriptive comparison; includes demonstration/BC compute; not a significance test')


def write_comparison_tables(report, output):
    """Separate per-seed final results from monitoring-only milestone diagnostics."""
    groups = [('candidate', report)]
    if 'reference' in report:
        groups.append(('scratch_reference', report['reference']))
    rows, milestones = [], []
    for label, method in groups:
        for seed in method['seeds']:
            if not seed['valid']:
                continue
            rows.append(dict(method=seed['training_variant'], role=label, seed=seed['seed'], **seed['final']))
            milestones.append(dict(method=seed['training_variant'], role=label, seed=seed['seed'], **seed['success_milestones']))
    for filename, records in (('final_results.csv', rows), ('success_milestones.csv', milestones)):
        if records:
            with (output / filename).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)


def plot_curves(runs, output, reference_runs=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    specs = [
        ('success', 'evaluation', ['success_rate'], 'Ever-success rate', False),
        ('final_success', 'evaluation', ['final_success_rate'], 'Final-success rate', False),
        ('return', 'evaluation', ['mean_return'], 'Evaluation mean return', False),
        ('alpha', 'losses', ['alpha_0', 'alpha_1'], 'Temperature alpha', True),
        ('entropy', 'losses', ['entropy_0', 'entropy_1'], 'Entropy estimate (-mean log pi)', False),
        ('action_std', 'losses', ['action_std_0', 'action_std_1'], 'Mean pre-tanh Gaussian std', False),
        ('critic', 'losses', ['q1_mean', 'q2_mean', 'target_mean'], 'Q1 / Q2 / Bellman target y', False),
        ('td_residual', 'losses', ['td_abs_p50', 'td_abs_p95', 'td_abs_p99'], 'Pooled absolute TD residual', False),
        ('saturation', 'losses', ['action_saturation_0', 'action_saturation_1'], 'Action saturation fraction (abs >= 0.95)', False),
        ('gradient', 'losses', ['actor_0_grad_norm', 'actor_1_grad_norm', 'q_grad_norm'], 'Gradient L2 norm', True),
        ('first_success', 'evaluation', ['mean_first_success_step'], 'Mean first-success step (successful episodes)', False),
    ]
    generated, unavailable = [], []
    for name, source, keys, ylabel, log in specs:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        groups = [('BC-init', runs), ('scratch', reference_runs)] if reference_runs else [('', runs)]
        for method, method_runs in groups:
            seed_values = {}
            for seed, path in sorted(method_runs.items()):
                rows = read_csv(path / f'{source}.csv')
                for key in keys:
                    pairs = [(int(r['step']), number(r[key])) for r in rows if r.get(key) not in (None, '')]
                    if not pairs:
                        continue
                    x, y = zip(*pairs)
                    label = (f'{method} / ' if method else '') + f'seed {seed}'
                    ax.plot(x, y, linewidth=1, marker='.' if source == 'evaluation' else None,
                            label=label + (f' / {key}' if len(keys) > 1 else ''))
                    if len(keys) == 1:
                        seed_values[seed] = dict(pairs)
            if set(seed_values) == {0, 1, 2}:
                common = sorted(set.intersection(*(set(v) for v in seed_values.values())))
                ax.plot(common, [statistics.mean(seed_values[s][x] for s in (0, 1, 2)) for x in common],
                        linewidth=2.5, linestyle='--' if method == 'scratch' else '-',
                        label=f'{method + " / " if method else ""}mean of 3 seeds')
        if not ax.lines:
            unavailable.append(name)
            plt.close(fig)
            continue
        if name == 'entropy':
            ax.axhline(-7, color='black', linestyle=':', label='target entropy -7')
        if name in ('success', 'final_success', 'saturation'):
            ax.set_ylim(-.02, 1.02)
        if log:
            ax.set_yscale('log')
        ax.set(xlabel='Environment steps', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            path = output / f'{name}_vs_steps.{suffix}'
            fig.savefig(path, dpi=180)
            generated.append(str(path))
        plt.close(fig)
    return dict(files=generated, unavailable_metrics=unavailable)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--batch', type=Path, help='Directory containing seed0, seed1, seed2')
    choice.add_argument('--runs', type=Path, nargs='+', help='Explicit standalone run directories; seeds read from metadata')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plots', action='store_true', help='Requires matplotlib in the interpreter running this script')
    parser.add_argument('--reference-batch', type=Path, help='Optional completed scratch pilot batch for BC-init comparison')
    parser.add_argument('--learning-curve-review', choices=('pending', 'pass', 'fail'), default='pending',
                        help='Explicit qualitative review required for learning PASS; numeric gates stay frozen')
    args = parser.parse_args()
    runs = {seed: args.batch / f'seed{seed}' for seed in (0, 1, 2)} if args.batch else {}
    if args.runs:
        for path in args.runs:
            seed = read_json(path / 'metadata.json')['seed']
            if seed not in (0, 1, 2) or seed in runs:
                parser.error('Provide unique training seeds from 0, 1, 2')
            runs[seed] = path
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent = args.batch or args.runs[0].parent
    output = args.output_dir or parent / f'analysis_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    report = analyze(runs, args.learning_curve_review)
    reference_runs = None
    if args.reference_batch:
        reference_runs = {seed: args.reference_batch / f'seed{seed}' for seed in (0, 1, 2)}
        reference = analyze(reference_runs, args.learning_curve_review)
        report['reference'] = reference
        report['comparison'] = compare_methods(report, reference)
    if args.plots:
        report['plots'] = plot_curves(runs, output, reference_runs)
    (output / 'pilot_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_comparison_tables(report, output)
    print(json.dumps(dict(status=report['status'], aggregate=report['aggregate'], gate=report['gate'],
                          comparison=report.get('comparison'), output_dir=str(output.resolve())), indent=2))


if __name__ == '__main__':
    main()

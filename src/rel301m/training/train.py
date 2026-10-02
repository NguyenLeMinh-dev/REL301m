"""Train the Phase 3 cooperative MASAC baseline on the locked Phase 2 wrapper."""

import argparse
from copy import deepcopy
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import sys

from rel301m.utils.console import TrainingProgress, install_panda_warning_filter

# Install before importing robosuite: startup notices otherwise escape the filter.
if __name__ == '__main__' and '--show-all-warnings' not in sys.argv:
    install_panda_warning_filter()

import numpy as np
import torch
import yaml

from rel301m.algorithms.masac import MASAC
from rel301m.algorithms.replay_buffer import ReplayBuffer
from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
from rel301m.envs.robosuite_factory import PROJECT_ROOT, load_env_config, make_two_arm_lift
from rel301m.evaluation.evaluate import evaluate_policy, write_episodes
from rel301m.utils.seed import seed_everything
from rel301m.utils.logger import RunLogger, run_metadata
from rel301m.utils.diagnostics import NonfiniteTrainingError, require_finite, tensorboard_tag, finite_json


LOSS_FIELDS = ['step', 'updates', 'q_loss', 'q1_mean', 'q2_mean', 'target_mean', 'alpha_loss', 'q_grad_norm',
               'actor_0_loss', 'alpha_0', 'actor_0_grad_norm', 'entropy_0',
               'actor_1_loss', 'alpha_1', 'actor_1_grad_norm', 'entropy_1',
               'log_pi_mean_0', 'log_pi_mean_1', 'target_q_mean',
               'td_abs_mean', 'td_abs_p50', 'td_abs_p95', 'td_abs_p99',
               'action_std_0', 'action_std_1', 'action_sample_std_0', 'action_sample_std_1',
               'action_saturation_0', 'action_saturation_1']
EPISODE_FIELDS = ['step', 'episode', 'episode_return', 'ever_success', 'first_success_step', 'final_success',
                  'episode_length', 'first_success_time_s', 'success_rate_100', 'partial']
EVAL_FIELDS = ['step', 'episodes', 'mean_return', 'success_rate', 'final_success_rate', 'mean_first_success_step', 'mean_first_success_time_s', 'mean_episode_length', 'initialization_sequence_sha256']


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    config.setdefault('eval_seed', 20000)
    config.setdefault('final_eval_episodes', config['eval_episodes'])
    algo_path = resolve_path(config['algo_config'])
    config['algo'] = yaml.safe_load(algo_path.read_text(encoding='utf-8'))
    for key in ('total_steps', 'torch_threads', 'eval_interval', 'eval_episodes', 'checkpoint_interval', 'log_interval', 'final_eval_episodes'):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'eval_seed'):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    algo = config['algo']
    for key in ('batch_size', 'buffer_capacity', 'updates_per_step'):
        if not isinstance(algo[key], int) or isinstance(algo[key], bool) or algo[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not isinstance(algo['warmup_steps'], int) or algo['warmup_steps'] < 0:
        raise ValueError('warmup_steps must be a nonnegative integer')
    if algo['buffer_capacity'] < algo['batch_size']:
        raise ValueError('buffer_capacity must be at least batch_size')
    if not 0 <= algo['gamma'] <= 1 or not 0 < algo['tau'] <= 1:
        raise ValueError('Invalid gamma/tau')
    if any(algo[key] <= 0 for key in ('actor_lr', 'critic_lr', 'alpha_lr', 'initial_alpha')):
        raise ValueError('Learning rates and initial_alpha must be positive')
    if algo.get('target_entropy') != 'auto':
        raise ValueError('Baseline derives target entropy from live action dimensions')
    if config.get('predictor', {}).get('enabled', False):
        raise ValueError('Phase 3 baseline does not enable a predictor')
    if config['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, or cuda')
    return config


def select_device(request):
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    return 'cuda' if request == 'auto' and torch.cuda.is_available() else ('cpu' if request == 'auto' else request)


def train(config, run_dir=None):
    seed = config['seed']
    seed_everything(seed)
    torch.set_num_threads(config['torch_threads'])
    device = select_device(config['device'])
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = Path(run_dir) if run_dir else resolve_path(config['output_root']) / f'{config["name"]}_seed{seed}_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f'RUN_DIR={run_dir}', flush=True)
    env_config_path = resolve_path(config['env_config'])
    env_config = load_env_config(env_config_path)
    env = MultiAgentWrapper(make_two_arm_lift(env_config_path, seed=seed))
    eval_env = None
    writer = None
    logger = None
    start = time.perf_counter()
    simulation_seconds, update_seconds, evaluation_seconds = 0.0, 0.0, 0.0
    step = 0
    completed_steps = 0
    numeric = dict(nan_count=0, inf_count=0, action_bound_violations=0)
    gradient_min = {key: None for key in ('q_grad_norm', 'actor_0_grad_norm', 'actor_1_grad_norm')}
    gradient_max = dict(gradient_min)
    last_metrics = None
    diagnostic_metrics = None
    diagnostic_step = None
    action_min, action_max = None, None
    done_count = ever_count = final_count = 0
    baseline = None
    final_evaluation = None
    progress = TrainingProgress(config['total_steps'], seed, enabled=config.get('progress', True))

    def write_diagnostics(status):
        model_ready = last_metrics is not None
        diagnostics = dict(status=status, global_step=completed_steps, attempted_step=step,
                           updates=model.updates, numeric=dict(numeric),
                           loss_check_status='PASS' if model_ready else 'NOT_EXERCISED',
                           gradient_norm_min=gradient_min, gradient_norm_max=gradient_max,
                           alpha_final=model.alpha.detach().cpu().tolist(),
                           target_entropy=model.target_entropy.cpu().tolist(),
                           joint_action_min=action_min.tolist() if action_min is not None else None,
                           joint_action_max=action_max.tolist() if action_max is not None else None,
                           joint_action_low=np.concatenate([low for low, _ in bounds]).tolist(),
                           joint_action_high=np.concatenate([high for _, high in bounds]).tolist(),
                           action_bounds_enforced='live_phase2_wrapper',
                           raw_done_count=done_count, timeout_count=done_count,
                           bootstrap_time_limits=config['algo']['bootstrap_time_limits'],
                           random_initialization_sequence=baseline['initialization_sequence_sha256'] if baseline else None,
                           final_initialization_sequence=final_evaluation['initialization_sequence_sha256'] if final_evaluation else None,
                           communication_cost='N/A', exact_resume_supported=False,
                           environment=dict(training_ever_success_count=ever_count,
                                            training_final_success_count=final_count, timeouts=done_count),
                           evaluation=final_evaluation)
        diagnostics['last_diagnostic_step'] = diagnostic_step
        for i in range(2):
            metrics = diagnostic_metrics or {}
            diagnostics[f'agent_{i}'] = dict(alpha_final=diagnostics['alpha_final'][i],
                entropy_final=metrics.get(f'entropy_{i}'), log_pi_mean_final=metrics.get(f'log_pi_mean_{i}'),
                action_std_mean=metrics.get(f'action_std_{i}'), action_sample_std_mean=metrics.get(f'action_sample_std_{i}'),
                action_saturation_fraction=metrics.get(f'action_saturation_{i}'),
                actor_grad_norm_min=gradient_min[f'actor_{i}_grad_norm'],
                actor_grad_norm_max=gradient_max[f'actor_{i}_grad_norm'])
        diagnostics['critic'] = {f'{key}_final': (diagnostic_metrics or {}).get(key) for key in
                                  ('q1_mean', 'q2_mean', 'target_q_mean', 'td_abs_mean', 'td_abs_p50', 'td_abs_p95', 'td_abs_p99')}
        diagnostics['critic'].update(q_grad_norm_min=gradient_min['q_grad_norm'], q_grad_norm_max=gradient_max['q_grad_norm'])
        (run_dir / 'diagnostics.json').write_text(json.dumps(finite_json(diagnostics), indent=2, allow_nan=False)+'\n', encoding='utf-8')

    try:
        eval_env = MultiAgentWrapper(make_two_arm_lift(env_config_path, seed=config['eval_seed']))
        eval_rng_state = deepcopy(eval_env._env.rng.bit_generator.state)
        observations = env.reset()
        obs_dims = [env.observation_dims[agent]['actor_obs'] for agent in env.agent_ids]
        bounds = [env.action_specs[agent] for agent in env.agent_ids]
        model = MASAC(obs_dims, env.critic_state_dim, bounds, config['algo'], device)
        extra_metadata = dict(training_variant='scratch_masac', actor_initialization=dict(method='random'))
        if config.get('bc_checkpoint'):
            from rel301m.imitation.warm_start import warm_start_from_bc
            actor_init = warm_start_from_bc(model, resolve_path(config['bc_checkpoint']), expected_env_config=env_config)
            extra_metadata = dict(training_variant='bc_initialized_masac', actor_initialization=actor_init,
                                  initialization='BC actors; torch.nn.Linear default critics',
                                  critic_initialization='torch.nn.Linear default reset_parameters')
        replay = ReplayBuffer(config['algo']['buffer_capacity'], obs_dims, env.critic_state_dim, model.action_dims, seed)
        metadata = run_metadata(run_dir, config, env_config, model, replay, device, extra_metadata=extra_metadata)
        logger = RunLogger(run_dir, dict(episodes=EPISODE_FIELDS, losses=LOSS_FIELDS, evaluation=EVAL_FIELDS))
        writer, writers = logger.tensorboard, logger.writers
        rng = np.random.default_rng(seed)
        # Establish an actual random-policy reference before updates.
        tick = time.perf_counter()
        progress.set_phase('random baseline', episodes=config['final_eval_episodes'])
        baseline_rows, baseline = evaluate_policy(eval_env, None, config['final_eval_episodes'], config['eval_seed'], random_policy=True,
                                         initial_rng_state=eval_rng_state, episode_callback=progress.episode_completed)
        write_episodes(run_dir / 'random_baseline_episodes.csv', baseline_rows)
        (run_dir / 'random_baseline.json').write_text(json.dumps(baseline, indent=2)+'\n', encoding='utf-8')
        evaluation_seconds += time.perf_counter() - tick
        progress.write(f'Random baseline: {json.dumps(baseline)}')
        progress.set_phase('warmup' if config['algo']['warmup_steps'] else 'training')
        episode, episode_steps, episode_return = 0, 0, 0.0
        initial = env.check_success()
        ever_success, final_success, first_success = initial, initial, 0 if initial else None
        recent_success = deque(maxlen=100)
        best_score = (-1.0, -float('inf'))
        for step in range(1, config['total_steps'] + 1):
            if step == config['algo']['warmup_steps'] + 1:
                progress.set_phase('training')
            current_obs = [observations[agent]['actor_obs'] for agent in env.agent_ids]
            state = env.critic_state
            if step <= config['algo']['warmup_steps']:
                actions = tuple(rng.uniform(low, high) for low, high in bounds)
            else:
                actions = model.act(current_obs)
            joint_action = np.concatenate(actions)
            require_finite('executed action', joint_action)
            low, high = np.concatenate([b[0] for b in bounds]), np.concatenate([b[1] for b in bounds])
            if joint_action.shape != low.shape:
                raise ValueError('Executed action shape differs from live bounds')
            violations = int(np.count_nonzero((joint_action < low) | (joint_action > high)))
            numeric['action_bound_violations'] += violations
            if violations:
                raise ValueError('Executed actions exceed live robosuite bounds')
            tick = time.perf_counter()
            next_obs, reward, done, _ = env.step(*actions)
            simulation_seconds += time.perf_counter() - tick
            require_finite('reward/next observations', reward, *[next_obs[a]['actor_obs'] for a in env.agent_ids], env.critic_state)
            action_min = joint_action.copy() if action_min is None else np.minimum(action_min, joint_action)
            action_max = joint_action.copy() if action_max is None else np.maximum(action_max, joint_action)
            done_count += int(bool(done))
            episode_steps += 1
            if bool(done) != (episode_steps == env.horizon):
                raise RuntimeError('Termination differs from locked Phase 1 horizon contract')
            # Preserve raw done, plus a time-limit flag to distinguish Bellman terminals.
            replay.add(*current_obs, state, *actions, reward,
                       *[next_obs[a]['actor_obs'] for a in env.agent_ids], env.critic_state, done, timeout=bool(done))
            completed_steps = step
            episode_return += float(reward)
            final_success = env.check_success()
            ever_success |= final_success
            if final_success and first_success is None:
                first_success = episode_steps
            observations = next_obs
            if step >= config['algo']['warmup_steps'] and len(replay) >= config['algo']['batch_size']:
                tick = time.perf_counter()
                for _ in range(config['algo']['updates_per_step']):
                    last_metrics = model.update(replay.sample(config['algo']['batch_size'], device),
                                                collect_diagnostics=step % config['log_interval'] == 0 or step == config['total_steps'])
                    if 'td_abs_p99' in last_metrics:
                        diagnostic_metrics, diagnostic_step = last_metrics, step
                    for key in gradient_min:
                        value = last_metrics[key]
                        gradient_min[key] = value if gradient_min[key] is None else min(gradient_min[key], value)
                        gradient_max[key] = value if gradient_max[key] is None else max(gradient_max[key], value)
                update_seconds += time.perf_counter() - tick
                if step % config['log_interval'] == 0:
                    writers['losses'].writerow(dict(step=step, updates=model.updates, **last_metrics))
                    for key, value in last_metrics.items():
                        writer.add_scalar(f'losses/{key}', value, step)
                        writer.add_scalar(tensorboard_tag(key), value, step)
            if done or step == config['total_steps']:
                partial = not bool(done)
                if not partial:
                    recent_success.append(float(ever_success))
                    ever_count += int(ever_success)
                    final_count += int(final_success)
                row = dict(step=step, episode=episode, episode_return=episode_return,
                           ever_success=bool(ever_success), first_success_step=first_success,
                           final_success=bool(final_success), episode_length=episode_steps,
                           first_success_time_s=first_success / env.control_freq if first_success is not None else None,
                           success_rate_100=float(np.mean(recent_success)) if recent_success else None, partial=partial)
                writers['episodes'].writerow(row)
                for key in ('episode_return', 'ever_success', 'final_success', 'episode_length', 'success_rate_100', 'first_success_step', 'first_success_time_s'):
                    if row[key] is not None:
                        writer.add_scalar(f'train/{key}', row[key], step)
                if done and step < config['total_steps']:
                    tick = time.perf_counter()
                    observations = env.reset()
                    simulation_seconds += time.perf_counter() - tick
                    episode += 1
                    episode_steps, episode_return = 0, 0.0
                    initial = env.check_success()
                    ever_success, final_success, first_success = initial, initial, 0 if initial else None
            progress.update(step, f'updates={model.updates}')
            if step % config['eval_interval'] == 0 or step == config['total_steps']:
                tick = time.perf_counter()
                evaluation_episodes = config['final_eval_episodes'] if step == config['total_steps'] else config['eval_episodes']
                progress.set_phase('evaluation', episodes=evaluation_episodes)
                rows, summary = evaluate_policy(eval_env, model, evaluation_episodes, config['eval_seed'],
                                                initial_rng_state=eval_rng_state, episode_callback=progress.episode_completed)
                expected_hash = hashlib.sha256(''.join(r['initial_state_sha256'] for r in baseline_rows[:evaluation_episodes]).encode()).hexdigest()
                if len(rows) != evaluation_episodes or summary['initialization_sequence_sha256'] != expected_hash:
                    raise RuntimeError('Evaluation initialization sequence differs from paired random reference')
                if step == config['total_steps']:
                    final_evaluation = summary
                evaluation_seconds += time.perf_counter() - tick
                writers['evaluation'].writerow(dict(step=step, **summary))
                write_episodes(run_dir / f'eval_{step:07d}_episodes.csv', rows)
                for key, value in summary.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'eval/{key}', value, step)
                progress.write(f'step={step} updates={model.updates} eval={json.dumps(summary)}')
                selection_rows = rows[:config['eval_episodes']]
                score = (float(np.mean([r['ever_success'] for r in selection_rows])),
                         float(np.mean([r['episode_return'] for r in selection_rows])))
                if score > best_score:
                    best_score = score
                    model.save(run_dir / 'best.pt', dict(metadata, step=step, evaluation=summary,
                                                       selection_episodes=len(selection_rows)))
                progress.set_phase('saving' if step == config['total_steps'] else
                                   ('warmup' if step < config['algo']['warmup_steps'] else 'training'))
            if step % config['checkpoint_interval'] == 0:
                model.save(run_dir / f'checkpoint_{step:07d}.pt', dict(metadata, step=step))
            if step % 1000 == 0:
                logger.flush()
                for key, value in numeric.items():
                    writer.add_scalar(f'numeric/{key}' if key != 'action_bound_violations' else 'action/bound_violations', value, step)
                write_diagnostics('running')
        if config['total_steps'] >= config['algo']['warmup_steps'] and len(replay) >= config['algo']['batch_size'] and not model.updates:
            raise RuntimeError('No optimizer updates after warmup')
        model.save(run_dir / 'final.pt', dict(metadata, step=step))
        elapsed = time.perf_counter() - start
        summary = dict(status='completed', steps=step, updates=model.updates, run_dir=str(run_dir),
                       wall_seconds=elapsed, simulation_seconds=simulation_seconds,
                       update_seconds=update_seconds, evaluation_seconds=evaluation_seconds,
                       steps_per_second=step / elapsed,
                       simulation_steps_per_second=step / simulation_seconds,
                       gradient_updates_per_second=model.updates / update_seconds if update_seconds else None,
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else None,
                       random_baseline=baseline, final_evaluation=summary,
                       last_update_metrics=last_metrics)
        write_diagnostics('completed')
        (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        progress.write(json.dumps(summary, indent=2, allow_nan=False))
        progress.finish('completed')
        return summary
    except BaseException as error:
        progress.update(completed_steps)
        progress.finish('FAILED')
        if isinstance(error, NonfiniteTrainingError):
            numeric['nan_count'] += error.nan_count
            numeric['inf_count'] += error.inf_count
        failure = dict(step=step, completed_env_steps=completed_steps,
                       error=repr(error), traceback=traceback.format_exc())
        (run_dir / 'failure.json').write_text(json.dumps(failure, indent=2)+'\n', encoding='utf-8')
        if 'model' in locals():
            write_diagnostics('failed')
        if 'state' in locals() and 'joint_action' in locals():
            # Evidence for controller failures, not an exact-resume checkpoint.
            np.savez_compressed(run_dir / 'failure_state.npz', critic_state=state, joint_action=joint_action,
                                qpos=np.asarray(env._env.sim.data.qpos), qvel=np.asarray(env._env.sim.data.qvel))
        raise
    finally:
        progress.close()
        if logger:
            logger.close()
        if eval_env:
            eval_env.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--steps', type=int, help='Explicit override for integration/benchmark runs')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--bc-checkpoint', type=Path, help='Actor-only BC initialization; fresh critics/alpha/replay')
    parser.add_argument('--no-progress', action='store_true', help='Disable terminal progress only')
    parser.add_argument('--show-all-warnings', action='store_true', help='Show known optional Panda/BASIC notices too')
    args = parser.parse_args()
    config = load_config(resolve_path(args.config))
    config['progress'] = not args.no_progress
    for argument, key in ((args.steps, 'total_steps'), (args.seed, 'seed'), (args.device, 'device')):
        if argument is not None:
            config[key] = argument
    if args.bc_checkpoint is not None:
        config['bc_checkpoint'] = str(args.bc_checkpoint)
    if config['total_steps'] <= 0:
        parser.error('--steps must be positive')
    train(config, args.run_dir)


if __name__ == '__main__':
    main()

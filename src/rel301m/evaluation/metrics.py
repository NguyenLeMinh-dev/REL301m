"""Task success metrics remain independent of reward and episode return."""

import hashlib
import math

import numpy as np


EPISODE_FIELDS = ['episode', 'episode_return', 'ever_success', 'first_success_step',
                  'first_success_time_s', 'final_success', 'episode_length', 'initial_state_sha256']


def episode_record(episode, episode_return, ever_success, first_success_step,
                   final_success, episode_length, control_freq, initial_state_sha256):
    if episode_length <= 0 or control_freq <= 0 or not math.isfinite(episode_return):
        raise ValueError('Invalid episode length, control frequency, or return')
    if bool(ever_success) != (first_success_step is not None):
        raise ValueError('ever_success must agree with first_success_step')
    if final_success and not ever_success:
        raise ValueError('final_success requires ever_success')
    if first_success_step is not None and not 0 <= first_success_step <= episode_length:
        raise ValueError('first_success_step must lie within the episode')
    return dict(episode=episode, episode_return=float(episode_return), ever_success=bool(ever_success),
                first_success_step=first_success_step,
                first_success_time_s=first_success_step / control_freq if first_success_step is not None else None,
                final_success=bool(final_success), episode_length=episode_length,
                initial_state_sha256=initial_state_sha256)


def summarize_episodes(rows):
    if not rows:
        raise ValueError('Evaluation needs completed episodes')
    steps = [row['first_success_step'] for row in rows if row['ever_success']]
    times = [row['first_success_time_s'] for row in rows if row['ever_success']]
    sequence = hashlib.sha256(''.join(row['initial_state_sha256'] for row in rows).encode()).hexdigest()
    return dict(episodes=len(rows), mean_return=float(np.mean([r['episode_return'] for r in rows])),
                success_rate=float(np.mean([r['ever_success'] for r in rows])),
                final_success_rate=float(np.mean([r['final_success'] for r in rows])),
                mean_first_success_step=float(np.mean(steps)) if steps else None,
                mean_first_success_time_s=float(np.mean(times)) if times else None,
                mean_episode_length=float(np.mean([r['episode_length'] for r in rows])),
                initialization_sequence_sha256=sequence)

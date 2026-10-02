from copy import deepcopy

import numpy as np
import pytest
import torch

from rel301m.envs.multi_agent_wrapper import MultiAgentWrapper
from rel301m.evaluation.evaluate import evaluate_policy
from rel301m.evaluation.metrics import episode_record, summarize_episodes


class ZeroPolicy:
    def act(self, observations, deterministic=False):
        assert deterministic
        assert len(observations) == 2
        assert all(obs.shape == (66,) for obs in observations)
        return np.zeros(7), np.zeros(7)


def test_common_initializations_independent_of_policy_and_previous_evals(env):
    wrapper = MultiAgentWrapper(env)
    initial_rng_state = deepcopy(env.rng.bit_generator.state)
    reference = deepcopy(initial_rng_state)
    torch_rng = torch.get_rng_state().clone()
    random_rows, random_summary = evaluate_policy(wrapper, None, 2, 20000, random_policy=True,
                                                  initial_rng_state=initial_rng_state)
    learned_rows, learned_summary = evaluate_policy(wrapper, ZeroPolicy(), 2, 20000,
                                                    initial_rng_state=initial_rng_state)
    notifications = []
    repeated_rows, repeated_summary = evaluate_policy(wrapper, ZeroPolicy(), 2, 20000,
                                                      initial_rng_state=initial_rng_state,
                                                      episode_callback=lambda count, total: notifications.append((count, total)))
    assert notifications == [(1, 2), (2, 2)]
    assert [r['initial_state_sha256'] for r in random_rows] == [r['initial_state_sha256'] for r in learned_rows]
    assert learned_rows == repeated_rows
    assert learned_summary == repeated_summary
    assert random_summary['initialization_sequence_sha256'] == learned_summary['initialization_sequence_sha256']
    assert initial_rng_state == reference
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert all(r['episode_length'] == 200 for r in learned_rows)


def test_success_time_and_success_not_return_threshold():
    rows = [episode_record(0, 0, True, 96, False, 200, 20, 'a'),
            episode_record(1, 1000, False, None, False, 200, 20, 'b')]
    summary = summarize_episodes(rows)
    assert rows[0]['first_success_time_s'] == 4.8
    assert rows[1]['first_success_time_s'] is None
    assert summary['success_rate'] == .5
    assert summary['final_success_rate'] == 0
    assert summary['mean_first_success_time_s'] == 4.8
    assert summary['mean_return'] == 500


@pytest.mark.parametrize('ever,first,final', [(False, 1, False), (True, None, False), (False, None, True)])
def test_inconsistent_success_metrics_rejected(ever, first, final):
    with pytest.raises(ValueError):
        episode_record(0, 1, ever, first, final, 200, 20, 'a')


def test_same_eval_sequence_across_training_seeds():
    from rel301m.envs.robosuite_factory import make_two_arm_lift
    from rel301m.utils.seed import seed_everything

    sequences = []
    for training_seed in (0, 1):
        seed_everything(training_seed)
        wrapper = MultiAgentWrapper(make_two_arm_lift(seed=20000))
        try:
            state = deepcopy(wrapper._env.rng.bit_generator.state)
            rows, _ = evaluate_policy(wrapper, ZeroPolicy(), 2, 20000, initial_rng_state=state)
            sequences.append([row['initial_state_sha256'] for row in rows])
        finally:
            wrapper.close()
    assert sequences[0] == sequences[1]

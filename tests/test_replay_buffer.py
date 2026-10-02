import numpy as np
import pytest
import torch

from rel301m.algorithms.replay_buffer import ReplayBuffer


def transition(label=1):
    return [np.full(66, label), np.full(66, label + 1), np.full(119, label + 2),
            np.full(7, .1), np.full(7, -.1), label + 3,
            np.full(66, label + 4), np.full(66, label + 5), np.full(119, label + 6), True]


def test_ring_order_sampling_and_owned_copies():
    replay = ReplayBuffer(4, (66, 66), 119, (7, 7), seed=12)
    values = transition(0)
    replay.add(*values, timeout=True)
    values[0][:] = 100
    np.testing.assert_array_equal(replay.arrays['o0'][0], 0)
    for label in range(1, 6):
        replay.add(*transition(label), timeout=True)
    assert len(replay) == 4
    assert replay.position == 2
    assert set(replay.arrays['o0'][:, 0]) == {2, 3, 4, 5}
    batch = replay.sample(4)
    for key, size in dict(o0=66, o1=66, s=119, a0=7, a1=7, r=1, next_o0=66, next_o1=66, next_s=119, done=1, timeout=1).items():
        assert batch[key].shape == (4, size)
        assert batch[key].dtype == torch.float32
    torch.testing.assert_close(batch['o1'][:, 0], batch['o0'][:, 0] + 1)
    torch.testing.assert_close(batch['next_s'][:, 0], batch['o0'][:, 0] + 6)
    batch['o0'][:] = -999
    assert (replay.arrays['o0'] >= 0).all()
    assert replay.allocated_bytes == 4 * 519 * 4


@pytest.mark.parametrize('field,value', [(0, np.zeros(65)), (5, np.nan), (9, 2)])
def test_invalid_transition_is_atomic(field, value):
    replay = ReplayBuffer(4, (66, 66), 119, (7, 7))
    values = transition()
    values[field] = value
    with pytest.raises(ValueError):
        replay.add(*values)
    assert len(replay) == 0
    assert replay.position == 0


def test_not_enough_samples_and_timeout_validation():
    replay = ReplayBuffer(4, (66, 66), 119, (7, 7))
    with pytest.raises(ValueError):
        replay.sample(1)
    values = transition()
    values[-1] = False
    with pytest.raises(ValueError):
        replay.add(*values, timeout=True)

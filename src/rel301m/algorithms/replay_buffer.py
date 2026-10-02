"""Uniform CPU float32 ring buffer; only sampled batches move to the GPU."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dims, state_dim, action_dims, seed=0):
        if not isinstance(capacity, int) or capacity <= 0 or len(obs_dims) != 2 or len(action_dims) != 2:
            raise ValueError("Replay requires a positive capacity and two agents")
        sizes = dict(o0=obs_dims[0], o1=obs_dims[1], s=state_dim,
                     a0=action_dims[0], a1=action_dims[1], r=1,
                     next_o0=obs_dims[0], next_o1=obs_dims[1], next_s=state_dim,
                     done=1, timeout=1)
        if any(not isinstance(size, int) or size <= 0 for size in sizes.values()):
            raise ValueError("Replay dimensions must be positive integers")
        self.arrays = {key: np.empty((capacity, size), dtype=np.float32) for key, size in sizes.items()}
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    @property
    def allocated_bytes(self):
        return sum(array.nbytes for array in self.arrays.values())

    def add(self, o0, o1, s, a0, a1, r, next_o0, next_o1, next_s, done, timeout=False):
        values = dict(o0=o0, o1=o1, s=s, a0=a0, a1=a1, r=r,
                      next_o0=next_o0, next_o1=next_o1, next_s=next_s, done=done, timeout=timeout)
        checked = {}
        for key, value in values.items():
            array = np.asarray(value, dtype=np.float32)
            if key in {'r', 'done', 'timeout'} and array.shape == ():
                array = array.reshape(1)
            if array.shape != self.arrays[key].shape[1:] or not np.isfinite(array).all():
                raise ValueError(f"Invalid replay field {key}: shape or NaN/Inf")
            checked[key] = array
        if checked['done'][0] not in (0, 1) or checked['timeout'][0] not in (0, 1):
            raise ValueError("done and timeout must be binary")
        if checked['timeout'][0] > checked['done'][0]:
            raise ValueError("timeout requires done")
        for key, value in checked.items():
            self.arrays[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cpu'):
        if batch_size <= 0 or self.size < batch_size:
            raise ValueError("Replay has insufficient samples")
        indices = self.rng.integers(self.size, size=batch_size)
        return {key: torch.as_tensor(array[indices], device=device) for key, array in self.arrays.items()}

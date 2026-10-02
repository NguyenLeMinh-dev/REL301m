"""Seed the coordinator; environment/replay retain independent NumPy generators."""

import random

import numpy as np
import torch


def seed_everything(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed also seeds all CUDA devices; no evaluation reseeding.

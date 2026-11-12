"""Reproducible random seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU, MPS, CUDA) RNGs.

    ``deterministic=True`` additionally asks PyTorch to use deterministic
    algorithms where available (may be slower, and some MPS ops warn).
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # also seeds MPS and all CUDA devices
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    """A dedicated CPU generator, independent of global RNG state.

    Batch sampling uses its own generator so that data order is reproducible
    regardless of how many random numbers the model consumes (e.g. dropout).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g

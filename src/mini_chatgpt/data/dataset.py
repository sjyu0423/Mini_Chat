"""Token streams, deterministic splitting, and next-token batches.

A language-modeling dataset here is just a 1-D tensor of token ids. Training
examples are windows of ``context_length + 1`` consecutive tokens: the first
``context_length`` are the inputs ``x`` and the last ``context_length`` are the
targets ``y``, i.e. ``y[t] == x[t + 1]`` — the model predicts the next token at
every position.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


def split_tokens(tokens: torch.Tensor, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic contiguous split: first (1 - f) for train, last f for val.

    A contiguous split (rather than random shuffling of positions) prevents
    overlapping training windows from leaking validation text into training.
    """
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1-D tensor")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n = tokens.numel()
    n_train = int(n * (1.0 - val_fraction))
    if n_train == 0 or n_train == n:
        raise ValueError(
            f"split of {n} tokens with val_fraction={val_fraction} leaves a split empty"
        )
    return tokens[:n_train], tokens[n_train:]


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random windows; return (x, y) of shape (B, T).

    Offsets are drawn on CPU from ``generator`` so the sampled data is
    identical across devices for the same seed.
    """
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    max_start = tokens.numel() - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"token stream of length {tokens.numel()} is too short for context_length "
            f"{context_length} (needs at least {context_length + 1})"
        )
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    offsets = torch.arange(context_length + 1)
    windows = tokens[starts.unsqueeze(1) + offsets]  # (B, T + 1)
    x = windows[:, :-1].contiguous()
    y = windows[:, 1:].contiguous()
    dev = torch.device(device)
    if dev.type != "cpu":
        x, y = x.to(dev), y.to(dev)
    return x, y


def iter_sequential_batches(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    *,
    device: torch.device | str = "cpu",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Non-overlapping windows covering the stream once, for exact evaluation.

    Each target token is predicted exactly once; the tail that does not fill a
    whole window is dropped. The final batch may be smaller than batch_size.
    """
    n_windows = (tokens.numel() - 1) // context_length
    if n_windows == 0:
        raise ValueError("token stream too short for one window")
    dev = torch.device(device)
    for first in range(0, n_windows, batch_size):
        idx = torch.arange(first, min(first + batch_size, n_windows)) * context_length
        offsets = torch.arange(context_length + 1)
        windows = tokens[idx.unsqueeze(1) + offsets]
        yield windows[:, :-1].to(dev), windows[:, 1:].to(dev)


@dataclass(frozen=True)
class TokenSplits:
    train: torch.Tensor
    val: torch.Tensor

    @property
    def num_train(self) -> int:
        return self.train.numel()

    @property
    def num_val(self) -> int:
        return self.val.numel()

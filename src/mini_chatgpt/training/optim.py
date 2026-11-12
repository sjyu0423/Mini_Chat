"""AdamW parameter groups and the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import torch
from torch import nn

from mini_chatgpt.config import TrainingConfig


def parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, list[nn.Parameter] | float]]:
    """Decay matrices (Linear weights, embeddings); never decay biases or LayerNorm.

    ``named_parameters`` de-duplicates shared tensors, so the tied
    embedding / LM-head weight appears (and is decayed) exactly once.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameter_groups(model, cfg.weight_decay),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=1e-8,
    )


def learning_rate(step: int, cfg: TrainingConfig) -> float:
    """LR for optimizer step ``step`` (0-based).

    * Linear warmup:  lr_max * (step + 1) / warmup_steps   for step < warmup_steps
    * Cosine decay:   lr_min + 0.5 (lr_max - lr_min) (1 + cos(pi * p)),
                      p = (step - warmup) / (max_steps - warmup) in [0, 1]
    * Held at lr_min after max_steps.

    The schedule is a pure function of ``step``; restoring ``step`` restores it.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    lr_max, lr_min = cfg.learning_rate, cfg.min_learning_rate
    if step < cfg.warmup_steps:
        return lr_max * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return lr_min
    decay_steps = cfg.max_steps - cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / decay_steps
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

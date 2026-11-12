import dataclasses
from typing import Any

import pytest
import torch

from mini_chatgpt.config import ExperimentConfig, ModelConfig, TrainingConfig

MPS = torch.backends.mps.is_available()
requires_mps = pytest.mark.skipif(not MPS, reason="MPS not available")


def tiny_model_config(**kw) -> ModelConfig:
    base = dict(vocab_size=17, context_length=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)  # type: ignore[arg-type]


def tiny_experiment(model_kw=None, **training_kw) -> ExperimentConfig:
    t: dict[str, Any] = dict(
        seed=0,
        batch_size=4,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        warmup_steps=2,
        max_steps=20,
        eval_interval=10,
        eval_steps=2,
        log_interval=5,
        checkpoint_interval=10,
        device="cpu",
    )
    t.update(training_kw)
    return ExperimentConfig(
        name="tiny", model=tiny_model_config(**(model_kw or {})), training=TrainingConfig(**t)
    )


def with_training(cfg: ExperimentConfig, **kw) -> ExperimentConfig:
    return dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, **kw))


@pytest.fixture
def token_stream() -> torch.Tensor:
    g = torch.Generator().manual_seed(123)
    return torch.randint(0, 17, (2000,), generator=g)

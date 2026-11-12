"""Training and validation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_MAX_EXP = 700.0  # math.exp overflows a float64 around 709.78


def perplexity(mean_loss: float) -> float:
    """exp(cross-entropy in nats). Returns inf for very large losses."""
    return math.inf if mean_loss > _MAX_EXP else math.exp(mean_loss)


def bits_per_token(mean_loss: float) -> float:
    """Cross-entropy in bits. For a char tokenizer this is bits-per-character."""
    return mean_loss / math.log(2)


def uniform_baseline_loss(vocab_size: int) -> float:
    """Loss of a model that predicts uniformly: ln(V). A sanity check for step 0."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return math.log(vocab_size)


@dataclass
class RunningMean:
    """Numerically simple weighted running mean (weights = tokens or batches)."""

    total: float = 0.0
    count: float = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError("weight must be non-negative")
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric value: {value}")
        self.total += value * weight
        self.count += weight

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0.0


@dataclass
class Throughput:
    """Tokens/second over a measured interval."""

    tokens: int = 0
    seconds: float = 0.0

    def update(self, tokens: int, seconds: float) -> None:
        self.tokens += tokens
        self.seconds += seconds

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


@dataclass
class EvalResult:
    split: str
    loss: float
    num_tokens: int
    perplexity: float = field(init=False)
    bits_per_token: float = field(init=False)

    def __post_init__(self) -> None:
        self.perplexity = perplexity(self.loss)
        self.bits_per_token = bits_per_token(self.loss)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "split": self.split,
            "loss": self.loss,
            "num_tokens": self.num_tokens,
            "perplexity": self.perplexity,
            "bits_per_token": self.bits_per_token,
        }

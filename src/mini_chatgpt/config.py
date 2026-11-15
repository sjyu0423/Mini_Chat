"""Typed, validated configuration for mini_chatgpt.

Configurations are plain frozen dataclasses so they are easy to read, compare,
serialize, and log. Every dataclass validates itself in ``__post_init__`` and
raises :class:`ConfigError` with a descriptive message on invalid values.

Configs can be loaded from TOML files (see ``configs/``) and serialized to JSON
for storage alongside training runs.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args

DeviceName = Literal["auto", "cpu", "mps", "cuda"]
DTypeName = Literal["float32", "bfloat16", "float16"]
LRSchedule = Literal["cosine", "constant"]
DataOrder = Literal["random", "single_pass"]
TokenizerName = Literal["char", "bpe"]
AttentionBackend = Literal["sdpa", "explicit"]


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


def _require_positive_int(name: str, value: object) -> None:
    # bool is a subclass of int; reject it explicitly.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value!r}")


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer, got {value!r}")


def _require_number(name: str, value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    return float(value)


def _require_choice(name: str, value: object, literal_type: Any) -> None:
    choices = get_args(literal_type)
    if value not in choices:
        raise ConfigError(f"{name} must be one of {choices}, got {value!r}")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters of the (future, M1) GPT-style decoder-only Transformer.

    Attributes:
        vocab_size: Number of distinct token ids. For the char baseline this is
            set from the tokenizer at data-preparation time.
        context_length: Maximum sequence length (a.k.a. block size).
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads per block.
        n_embd: Residual stream / embedding width. Must be divisible by n_head.
        dropout: Dropout probability in [0, 1).
        bias: Whether Linear and LayerNorm layers carry bias terms.
        tie_embeddings: Share token-embedding and output-projection weights.
        mlp_ratio: Hidden width of the MLP as a multiple of n_embd.
        attention: "sdpa" (torch scaled_dot_product_attention, for training) or
            "explicit" (hand-written reference). Does not change parameters.
    """

    vocab_size: int
    context_length: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    tie_embeddings: bool = True
    mlp_ratio: int = 4
    attention: AttentionBackend = "sdpa"

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "n_layer", "n_head", "n_embd", "mlp_ratio"):
            _require_positive_int(name, getattr(self, name))
        if self.n_embd % self.n_head != 0:
            raise ConfigError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        dropout = _require_number("dropout", self.dropout)
        if not 0.0 <= dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout!r}")
        for name in ("bias", "tie_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a bool, got {getattr(self, name)!r}")
        _require_choice("attention", self.attention, AttentionBackend)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def estimate_parameters(self) -> dict[str, int]:
        """Analytic parameter count for the GPT architecture in ``model/gpt.py``.

        Assumes a GPT-2 style pre-LayerNorm block:
            LN -> causal self-attention (fused QKV + output proj) -> residual
            LN -> MLP (n_embd -> mlp_ratio*n_embd -> n_embd, GELU) -> residual
        plus learned token and absolute position embeddings, a final LayerNorm,
        and an output projection to the vocabulary (tied by default).

        Verified by tests against ``GPT.num_parameters()`` for every shipped config.
        """
        d, hidden, b = self.n_embd, self.mlp_ratio * self.n_embd, int(self.bias)
        ln = d + b * d  # weight (+ bias)
        attn = (d * 3 * d + b * 3 * d) + (d * d + b * d)
        mlp = (d * hidden + b * hidden) + (hidden * d + b * d)
        per_layer = 2 * ln + attn + mlp

        token_embedding = self.vocab_size * d
        position_embedding = self.context_length * d
        blocks = self.n_layer * per_layer
        final_ln = ln
        lm_head = 0 if self.tie_embeddings else self.vocab_size * d  # LM head has no bias

        total = token_embedding + position_embedding + blocks + final_ln + lm_head
        return {
            "token_embedding": token_embedding,
            "position_embedding": position_embedding,
            "per_layer": per_layer,
            "blocks": blocks,
            "final_layernorm": final_ln,
            "lm_head_untied": lm_head,
            "total": total,
            # Non-embedding count is what scaling-law papers usually report.
            "non_embedding": blocks + final_ln,
        }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and bookkeeping hyperparameters."""

    seed: int = 1337
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 5000
    # Cosine-decay horizon of the LR schedule. None (default) means max_steps.
    # Setting it larger than max_steps stops training early on the *same* LR
    # trajectory; smaller compresses the decay and holds lr_min afterwards.
    lr_schedule_steps: int | None = None
    # "cosine": warmup then cosine decay to min_learning_rate at the horizon (default).
    # "constant": warmup (if any) then learning_rate for every step. Used for explicit
    # continuation phases so an old schedule is never silently stretched or restarted.
    lr_schedule: LRSchedule = "cosine"
    # "random" (default, M0-M3): every micro-batch draws random window offsets, i.e.
    # sampling with replacement. "single_pass" (M4.1+): the training stream is cut into
    # non-overlapping windows, visited once in a seeded permutation; training refuses
    # to run past the end instead of silently repeating data.
    data_order: DataOrder = "random"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_steps: int = 50
    log_interval: int = 10
    checkpoint_interval: int = 1000
    # Periodic step_*.pt checkpoints to keep (oldest are deleted); 0 keeps all.
    # best.pt and the final checkpoint are never deleted by retention.
    keep_last_checkpoints: int = 0
    save_best: bool = True  # write checkpoints/best.pt whenever validation improves
    # Also compute the exact full-split validation loss at every eval; when on,
    # best-checkpoint selection uses it instead of the random-batch estimate.
    exact_val_eval: bool = False
    device: DeviceName = "auto"
    dtype: DTypeName = "float32"

    def __post_init__(self) -> None:
        _require_non_negative_int("seed", self.seed)
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "eval_interval",
            "eval_steps",
            "log_interval",
            "checkpoint_interval",
        ):
            _require_positive_int(name, getattr(self, name))
        _require_non_negative_int("warmup_steps", self.warmup_steps)
        _require_non_negative_int("keep_last_checkpoints", self.keep_last_checkpoints)
        for name in ("save_best", "exact_val_eval"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.warmup_steps > self.max_steps:
            raise ConfigError(
                f"warmup_steps ({self.warmup_steps}) cannot exceed max_steps ({self.max_steps})"
            )
        _require_choice("lr_schedule", self.lr_schedule, LRSchedule)
        _require_choice("data_order", self.data_order, DataOrder)
        if self.lr_schedule_steps is not None:
            _require_positive_int("lr_schedule_steps", self.lr_schedule_steps)
            if self.warmup_steps > self.lr_schedule_steps:
                raise ConfigError(
                    f"warmup_steps ({self.warmup_steps}) cannot exceed lr_schedule_steps "
                    f"({self.lr_schedule_steps})"
                )

        lr = _require_number("learning_rate", self.learning_rate)
        min_lr = _require_number("min_learning_rate", self.min_learning_rate)
        if lr <= 0:
            raise ConfigError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if not 0 <= min_lr <= lr:
            raise ConfigError(
                f"min_learning_rate must be in [0, learning_rate], got {self.min_learning_rate!r}"
            )
        if _require_number("weight_decay", self.weight_decay) < 0:
            raise ConfigError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        for name in ("beta1", "beta2"):
            if not 0.0 <= _require_number(name, getattr(self, name)) < 1.0:
                raise ConfigError(f"{name} must be in [0, 1), got {getattr(self, name)!r}")
        if _require_number("grad_clip", self.grad_clip) < 0:
            raise ConfigError(f"grad_clip must be >= 0 (0 disables), got {self.grad_clip!r}")
        _require_choice("device", self.device, DeviceName)
        _require_choice("dtype", self.dtype, DTypeName)

    @property
    def schedule_horizon(self) -> int:
        """Step at which the cosine decay reaches ``min_learning_rate``."""
        return self.max_steps if self.lr_schedule_steps is None else self.lr_schedule_steps

    @property
    def effective_batch_size(self) -> int:
        """Sequences contributing to each optimizer step."""
        return self.batch_size * self.gradient_accumulation_steps


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DataConfig:
    """Dataset and tokenizer selection."""

    dataset: str = "tiny_shakespeare"
    data_dir: str = "data"
    tokenizer: TokenizerName = "char"
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ConfigError(f"dataset must be a non-empty string, got {self.dataset!r}")
        if not isinstance(self.data_dir, str) or not self.data_dir:
            raise ConfigError(f"data_dir must be a non-empty string, got {self.data_dir!r}")
        _require_choice("tokenizer", self.tokenizer, TokenizerName)
        if not 0.0 < _require_number("val_fraction", self.val_fraction) < 1.0:
            raise ConfigError(f"val_fraction must be in (0, 1), got {self.val_fraction!r}")


# --------------------------------------------------------------------------- #
# Experiment (top level)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration bundling model, training, and data settings."""

    name: str
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run_dir: str = "runs"
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("name must be a non-empty string")
        if any(c in self.name for c in "/\\ "):
            raise ConfigError(f"name must not contain spaces or path separators: {self.name!r}")

    # ---- serialization ---------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        raw = dict(raw)
        if "model" not in raw:
            raise ConfigError("experiment config requires a [model] section")
        kwargs: dict[str, Any] = {
            "model": _build(ModelConfig, raw.pop("model")),
            "training": _build(TrainingConfig, raw.pop("training", {})),
            "data": _build(DataConfig, raw.pop("data", {})),
        }
        top_level = {f.name for f in dataclasses.fields(cls)} - set(kwargs)
        unknown = set(raw) - top_level
        if unknown:
            raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")
        kwargs.update(raw)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, text: str) -> ExperimentConfig:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("rb") as f:
            return cls.from_dict(tomllib.load(f))

    def with_vocab_size(self, vocab_size: int) -> ExperimentConfig:
        """Return a copy whose model vocab_size matches a concrete tokenizer."""
        return dataclasses.replace(
            self, model=dataclasses.replace(self.model, vocab_size=vocab_size)
        )


_T = TypeVar("_T")


def _build(cls: type[_T], raw: object) -> _T:
    """Instantiate a config dataclass from a dict, rejecting unknown keys."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{cls.__name__} section must be a table/dict, got {type(raw).__name__}")
    allowed = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    try:
        return cls(**raw)
    except TypeError as exc:  # e.g. missing required field
        raise ConfigError(f"invalid {cls.__name__}: {exc}") from exc


def format_config(config: ExperimentConfig) -> str:
    """Human-readable multi-line summary suitable for logging."""
    lines = [f"experiment: {config.name}"]
    if config.description:
        lines.append(f"description: {config.description}")
    for section in ("model", "training", "data"):
        lines.append(f"[{section}]")
        for key, value in dataclasses.asdict(getattr(config, section)).items():
            lines.append(f"  {key:<28} = {value!r}")
    est = config.model.estimate_parameters()
    lines.append("[parameter count (analytic, verified against model)]")
    lines.append(f"  {'total':<28} = {est['total']:,}")
    lines.append(f"  {'non_embedding':<28} = {est['non_embedding']:,}")
    lines.append(f"  {'effective_batch_size':<28} = {config.training.effective_batch_size}")
    tokens = config.training.effective_batch_size * config.model.context_length
    lines.append(f"  {'tokens_per_optimizer_step':<28} = {tokens:,}")
    return "\n".join(lines)

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from mini_chatgpt.config import (
    ConfigError,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    format_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def model(**kw):
    base: dict[str, Any] = dict(vocab_size=65, context_length=64, n_layer=2, n_head=2, n_embd=64)
    base.update(kw)
    return ModelConfig(**base)


def test_valid_model_config():
    m = model()
    assert m.head_dim == 32


@pytest.mark.parametrize(
    "kw",
    [
        {"n_embd": 65},  # not divisible by n_head
        {"vocab_size": 0},
        {"n_layer": -1},
        {"n_head": 2.0},
        {"context_length": True},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"bias": "yes"},
        {"attention": "flash"},
    ],
)
def test_invalid_model_config(kw):
    with pytest.raises(ConfigError):
        model(**kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"min_learning_rate": 1.0},
        {"warmup_steps": 10, "max_steps": 5},
        {"beta2": 1.0},
        {"device": "tpu"},
        {"dtype": "float64"},
        {"seed": -1},
        {"grad_clip": -1.0},
    ],
)
def test_invalid_training_config(kw):
    with pytest.raises(ConfigError):
        TrainingConfig(**kw)


def test_invalid_data_config():
    with pytest.raises(ConfigError):
        DataConfig(val_fraction=0.0)
    with pytest.raises(ConfigError):
        DataConfig(tokenizer="wordpiece")
    assert DataConfig(tokenizer="bpe").tokenizer == "bpe"  # implemented in M3/M3.1


def test_effective_batch_size():
    assert TrainingConfig(batch_size=4, gradient_accumulation_steps=8).effective_batch_size == 32


def test_parameter_estimate_matches_hand_count():
    # Hand count for d=64, L=2, V=65, T=64, bias, tied embeddings:
    d = 64
    per_layer = (
        2 * (2 * d) + (3 * d * d + 3 * d) + (d * d + d) + (4 * d * d + 4 * d) + (4 * d * d + d)
    )
    expected = 65 * d + 64 * d + 2 * per_layer + 2 * d
    est = model().estimate_parameters()
    assert est["per_layer"] == per_layer
    assert est["total"] == expected


def test_parameter_estimate_untied_and_no_bias():
    tied = model(bias=False).estimate_parameters()
    untied = model(bias=False, tie_embeddings=False).estimate_parameters()
    assert untied["total"] - tied["total"] == 65 * 64
    assert tied["per_layer"] == 12 * 64 * 64 + 2 * 64


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [("gpt_10m", 9e6, 12e6), ("gpt_30m", 27e6, 35e6), ("gpt_100m", 90e6, 110e6)],
)
def test_example_configs_hit_target_sizes(name, low, high):
    cfg = ExperimentConfig.from_toml(CONFIG_DIR / f"{name}.toml")
    assert low <= cfg.model.estimate_parameters()["total"] <= high


def test_all_shipped_configs_load():
    for path in CONFIG_DIR.glob("*.toml"):
        ExperimentConfig.from_toml(path)


def test_json_round_trip():
    cfg = ExperimentConfig.from_toml(CONFIG_DIR / "gpt_10m.toml")
    assert ExperimentConfig.from_json(cfg.to_json()) == cfg


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError, match="unknown"):
        ExperimentConfig.from_dict(
            {"name": "x", "model": {**dataclasses.asdict(model()), "foo": 1}}
        )
    with pytest.raises(ConfigError, match="unknown top-level"):
        ExperimentConfig.from_dict({"name": "x", "model": dataclasses.asdict(model()), "bar": 1})
    with pytest.raises(ConfigError, match="requires"):
        ExperimentConfig.from_dict({"name": "x"})


def test_missing_required_model_field():
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"name": "x", "model": {"vocab_size": 65}})


def test_bad_name():
    with pytest.raises(ConfigError):
        ExperimentConfig(name="a/b", model=model())


def test_with_vocab_size_and_format():
    cfg = ExperimentConfig(name="t", model=model()).with_vocab_size(100)
    assert cfg.model.vocab_size == 100
    text = format_config(cfg)
    assert "vocab_size" in text and "total" in text

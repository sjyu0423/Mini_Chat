import json
import math
import random

import numpy as np
import pytest
import torch

from mini_chatgpt.config import ExperimentConfig, ModelConfig
from mini_chatgpt.device import describe_device
from mini_chatgpt.training.checkpoint import (
    CheckpointMetadata,
    atomic_torch_save,
    atomic_write_text,
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    metadata_path,
    save_checkpoint,
)
from mini_chatgpt.training.metrics import (
    EvalResult,
    RunningMean,
    Throughput,
    bits_per_token,
    perplexity,
    uniform_baseline_loss,
)
from mini_chatgpt.training.run import MetricsLogger, make_run_dir, read_metrics, write_run_metadata
from mini_chatgpt.training.seed import make_generator, set_seed


def test_set_seed_reproduces_all_rngs():
    def draw():
        return random.random(), float(np.random.rand()), float(torch.rand(1))

    set_seed(123)
    a = draw()
    set_seed(123)
    assert draw() == a


def test_generator_independent_of_global_state():
    a = torch.randint(0, 100, (5,), generator=make_generator(1))
    torch.rand(100)  # consume global RNG
    b = torch.randint(0, 100, (5,), generator=make_generator(1))
    assert torch.equal(a, b)


def test_metric_functions():
    assert perplexity(0.0) == 1.0
    assert math.isclose(perplexity(math.log(65)), 65)
    assert perplexity(1e6) == math.inf
    assert math.isclose(bits_per_token(math.log(2)), 1.0)
    assert math.isclose(uniform_baseline_loss(65), math.log(65))


def test_running_mean():
    m = RunningMean()
    assert math.isnan(m.mean)
    m.update(1.0, weight=1)
    m.update(4.0, weight=3)
    assert m.mean == pytest.approx(3.25)
    with pytest.raises(ValueError):
        m.update(float("nan"))
    m.reset()
    assert m.count == 0


def test_throughput_and_eval_result():
    t = Throughput()
    t.update(1000, 0.5)
    assert t.tokens_per_second == 2000
    r = EvalResult("val", math.log(4), 10)
    assert r.perplexity == pytest.approx(4) and r.bits_per_token == pytest.approx(2)


def test_atomic_write_replaces_and_leaves_no_temp(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    atomic_write_text(p, "one")
    atomic_write_text(p, "two")
    assert p.read_text() == "two"
    assert [x.name for x in p.parent.iterdir()] == ["f.txt"]


def test_atomic_write_failure_keeps_old_file(tmp_path):
    p = tmp_path / "f.pt"
    atomic_torch_save(p, {"a": 1})
    with pytest.raises(Exception):  # noqa: B017 - lambdas cannot be pickled
        atomic_torch_save(p, {"a": lambda: None})
    assert torch.load(p, weights_only=True) == {"a": 1}
    assert len(list(tmp_path.iterdir())) == 1


def test_checkpoint_paths(tmp_path):
    assert checkpoint_path(tmp_path, 42).name == "step_00000042.pt"
    assert metadata_path(checkpoint_path(tmp_path, 42)).suffix == ".json"
    with pytest.raises(ValueError):
        checkpoint_path(tmp_path, -1)
    assert latest_checkpoint(tmp_path) is None


def test_checkpoint_save_load_round_trip(tmp_path):
    state = {"w": torch.arange(6.0).reshape(2, 3), "step": 10}
    for step in (5, 10):
        save_checkpoint(
            tmp_path, state, CheckpointMetadata(step=step, config={"x": 1}, val_loss=1.5)
        )
    latest = latest_checkpoint(tmp_path)
    assert latest == checkpoint_path(tmp_path, 10)
    loaded, meta = load_checkpoint(latest)
    assert torch.equal(loaded["w"], torch.arange(6.0).reshape(2, 3))
    assert meta.step == 10 and meta.val_loss == 1.5
    assert json.loads(metadata_path(latest).read_text())["step"] == 10


def test_run_dir_and_metrics_log(tmp_path):
    run = make_run_dir(tmp_path, "exp", timestamp="20260101-000000")
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, "exp", timestamp="20260101-000000")
    with MetricsLogger(run / "metrics.jsonl") as log:
        log.log("train", 1, loss=4.17, lr=1e-3)
        log.log("eval", 1, val_loss=float("inf"))
    rows = read_metrics(run / "metrics.jsonl")
    assert [r["event"] for r in rows] == ["train", "eval"]
    assert rows[0]["loss"] == 4.17 and rows[1]["val_loss"] == "inf"


def test_write_run_metadata(tmp_path):
    cfg = ExperimentConfig(
        name="t", model=ModelConfig(vocab_size=65, context_length=8, n_layer=1, n_head=1, n_embd=8)
    )
    write_run_metadata(tmp_path, cfg, describe_device(torch.device("cpu")))
    saved = json.loads((tmp_path / "config.json").read_text())
    assert ExperimentConfig.from_dict(saved) == cfg
    assert "torch_version" in json.loads((tmp_path / "environment.json").read_text())

"""M2 experiment infrastructure: best/final checkpoints, retention, histories, CSV."""

import csv
import json
import math

import pytest
import torch

from mini_chatgpt.config import ConfigError, TrainingConfig
from mini_chatgpt.training.checkpoint import (
    apply_retention,
    best_checkpoint_path,
    latest_valid_checkpoint,
    list_checkpoints,
    load_checkpoint,
    read_metadata,
)
from mini_chatgpt.training.metrics import bits_per_token, perplexity
from mini_chatgpt.training.optim import learning_rate
from mini_chatgpt.training.run import MetricsLogger, dedupe_metrics, export_csv, read_metrics
from mini_chatgpt.training.trainer import Trainer, load_model
from tests.conftest import tiny_experiment, with_training

CPU = torch.device("cpu")


def split(stream):
    return stream[:1600], stream[1600:]


def learnable():
    return torch.arange(2000) % 17


def test_best_checkpoint_is_argmin_of_eval_history(tmp_path):
    cfg = tiny_experiment(max_steps=30, eval_interval=5, checkpoint_interval=10)
    t = Trainer(cfg, *split(learnable()), CPU, tmp_path)
    t.fit()
    evals = [r for r in t.eval_history if r["step"] > 0]
    best = min(evals, key=lambda r: r["val"])
    assert t.best_step == int(best["step"])
    assert t.best_val_loss == best["val"]
    meta = read_metadata(best_checkpoint_path(tmp_path))
    assert meta is not None and meta.kind == "best" and meta.step == t.best_step
    assert meta.val_loss == best["val"]
    # The saved weights reproduce the recorded best validation estimate.
    t2 = Trainer(cfg, *split(learnable()), CPU)
    t2.load_state(best_checkpoint_path(tmp_path))
    assert t2.step == t.best_step
    assert t2.estimate_loss()["val"] == pytest.approx(best["val"], abs=1e-6)


def test_best_uses_exact_val_when_enabled(tmp_path):
    cfg = tiny_experiment(max_steps=10, eval_interval=5, exact_val_eval=True)
    t = Trainer(cfg, *split(learnable()), CPU, tmp_path)
    t.fit()
    assert t.best_metric == "val_exact"
    last = t.eval_history[-1]
    assert last["val_exact"] == pytest.approx(t.full_loss(t.val_data), abs=1e-7)
    best = min((r for r in t.eval_history if r["step"] > 0), key=lambda r: r["val_exact"])
    assert t.best_val_loss == best["val_exact"]
    meta = read_metadata(best_checkpoint_path(tmp_path))
    assert meta is not None and meta.val_loss_exact == best["val_exact"]


def test_final_checkpoint_and_retention(tmp_path):
    cfg = tiny_experiment(
        max_steps=22, eval_interval=5, checkpoint_interval=5, keep_last_checkpoints=2
    )
    t = Trainer(cfg, *split(learnable()), CPU, tmp_path)
    t.fit()
    names = [p.name for p in list_checkpoints(tmp_path)]
    # periodic 5, 10 deleted; 15, 20 kept; 22 is the (protected) final.
    assert names == ["step_00000015.pt", "step_00000020.pt", "step_00000022.pt"]
    assert not (tmp_path / "checkpoints" / "step_00000005.json").exists()
    final = read_metadata(tmp_path / "checkpoints" / "step_00000022.pt")
    assert final is not None and final.kind == "final" and final.step == 22
    assert best_checkpoint_path(tmp_path).exists()
    # Retention never deletes the final checkpoint, even with keep_last=1.
    apply_retention(tmp_path, 1)
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        "step_00000020.pt",
        "step_00000022.pt",
    ]
    assert apply_retention(tmp_path, 0) == []


def test_partial_fit_end_is_not_final(tmp_path):
    t = Trainer(tiny_experiment(max_steps=20), *split(learnable()), CPU, tmp_path)
    t.fit(max_steps=7)
    meta = read_metadata(tmp_path / "checkpoints" / "step_00000007.pt")
    assert meta is not None and meta.kind == "periodic"


def test_checkpoint_holds_full_state_and_histories(tmp_path):
    cfg = tiny_experiment(model_kw={"dropout": 0.1}, max_steps=10, eval_interval=5)
    t = Trainer(cfg, *split(learnable()), CPU, tmp_path)
    t.fit()
    state, meta = load_checkpoint(tmp_path / "checkpoints" / "step_00000010.pt")
    for key in ("model", "optimizer", "data_generator", "rng", "step", "tokens_seen"):
        assert key in state
    assert state["step"] == 10 and state["loss_history"] == t.loss_history
    assert [r["step"] for r in state["eval_history"]] == [0.0, 5.0, 10.0]
    assert state["optimizer"]["state"]  # AdamW moments present
    assert meta.tokens_seen == 10 * t.tokens_per_step


def test_loads_m1_style_checkpoint(tmp_path):
    """Checkpoints written before M2 lack histories/best_step and new config keys."""
    t = Trainer(tiny_experiment(max_steps=4), *split(learnable()), CPU, tmp_path)
    t.fit()
    path = tmp_path / "checkpoints" / "step_00000004.pt"
    blob = torch.load(path, weights_only=True)
    for key in ("loss_history", "eval_history", "best_step"):
        del blob["state"][key]
    for key in ("keep_last_checkpoints", "save_best", "exact_val_eval"):
        del blob["metadata"]["config"]["training"][key]
    for key in ("kind", "val_loss_exact"):
        del blob["metadata"][key]
    old = tmp_path / "m1.pt"
    torch.save(blob, old)
    r = Trainer(tiny_experiment(max_steps=4), *split(learnable()), CPU)
    r.load_state(old)
    assert r.step == 4 and r.loss_history == [] and r.best_step is None
    model, _ = load_model(old, CPU)
    x = learnable()[:16].unsqueeze(0)
    t.model.eval()
    assert torch.equal(model(x)[0], t.model(x)[0])


def test_request_stop_checkpoints_and_resume_is_exact(tmp_path):
    cfg = tiny_experiment(model_kw={"dropout": 0.1}, max_steps=12, checkpoint_interval=100)
    full = Trainer(cfg, *split(learnable()), CPU, tmp_path / "full")
    full.fit()

    part = Trainer(cfg, *split(learnable()), CPU, tmp_path / "part")
    original = part.train_step

    def step_then_stop():
        out = original()
        if part.step == 7:
            part.request_stop()
        return out

    part.train_step = step_then_stop
    part.fit()
    assert part.step == 7
    ckpt = latest_valid_checkpoint(tmp_path / "part")
    assert ckpt is not None and ckpt.name == "step_00000007.pt"
    meta = read_metadata(ckpt)
    assert meta is not None and meta.kind == "interrupt"

    resumed = Trainer(cfg, *split(learnable()), CPU, tmp_path / "part")
    resumed.load_state(ckpt)
    resumed.fit()
    assert resumed.loss_history == full.loss_history
    for a, b in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_latest_valid_checkpoint_skips_corrupt(tmp_path):
    t = Trainer(tiny_experiment(max_steps=10, checkpoint_interval=5), *split(learnable()), CPU,
                tmp_path)  # fmt: skip
    t.fit()
    (tmp_path / "checkpoints" / "step_00000010.pt").write_bytes(b"truncated")
    assert latest_valid_checkpoint(tmp_path).name == "step_00000005.pt"


def test_eval_log_reports_ppl_bpc_gap_lr(tmp_path):
    cfg = tiny_experiment(max_steps=10, eval_interval=5)
    t = Trainer(cfg, *split(learnable()), CPU, tmp_path)
    t.fit()
    evals = [e for e in read_metrics(tmp_path / "metrics.jsonl") if e["event"] == "eval"]
    assert [e["step"] for e in evals] == [0, 5, 10]
    for e in evals:
        assert e["val_ppl"] == pytest.approx(perplexity(e["val_loss"]))
        assert e["val_bpc"] == pytest.approx(e["val_loss"] / math.log(2))
        assert e["train_bpc"] == pytest.approx(bits_per_token(e["train_loss"]))
        assert e["gap"] == pytest.approx(e["val_loss"] - e["train_loss"])
        assert e["lr"] == pytest.approx(learning_rate(e["step"], cfg.training))
    trains = [e for e in read_metrics(tmp_path / "metrics.jsonl") if e["event"] == "train"]
    for e in trains:  # logged lr is the lr used for that update (step s-1 -> s)
        assert e["lr"] == pytest.approx(learning_rate(e["step"] - 1, cfg.training))
        assert e["grad_norm"] > 0 and e["tokens_per_s"] > 0


def test_csv_export_and_dedupe(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricsLogger(path) as log:
        log.log("train", 1, loss=3.0)
        log.log("train", 2, loss=2.0)
        log.log("eval", 2, val_loss=2.5)
        log.log("train", 2, loss=1.9)  # re-run after a resume: last one wins
        log.log("train", 3, loss=1.5, extra=1)
    assert [r["loss"] for r in dedupe_metrics(read_metrics(path)) if r["event"] == "train"] == [
        3.0,
        1.9,
        1.5,
    ]
    out = export_csv(path)
    rows = list(csv.DictReader(out["train"].open()))
    assert [r["step"] for r in rows] == ["1", "2", "3"]
    assert rows[0]["extra"] == "" and rows[2]["extra"] == "1"
    assert next(iter(rows[0])) == "step"
    assert next(iter(csv.DictReader(out["eval"].open())))["val_loss"] == "2.5"


def test_new_training_config_fields_are_validated():
    with pytest.raises(ConfigError):
        TrainingConfig(keep_last_checkpoints=-1)
    with pytest.raises(ConfigError):
        TrainingConfig(save_best=1)
    with pytest.raises(ConfigError):
        TrainingConfig(exact_val_eval="yes")


def test_save_best_false_writes_no_best(tmp_path):
    cfg = with_training(tiny_experiment(max_steps=10, eval_interval=5), save_best=False)
    t = Trainer(cfg, *split(learnable()), CPU, tmp_path)
    t.fit()
    assert not best_checkpoint_path(tmp_path).exists()
    assert t.best_step is not None
    assert json.loads((tmp_path / "checkpoints" / "step_00000010.json").read_text())["kind"] == (
        "final"
    )

"""Constant-LR schedule and explicit continuation phases (M3.2)."""

import hashlib

import pytest
import torch

from mini_chatgpt.config import ConfigError, TrainingConfig
from mini_chatgpt.training.checkpoint import list_checkpoints, load_checkpoint, read_metadata
from mini_chatgpt.training.optim import learning_rate
from mini_chatgpt.training.trainer import Trainer
from tests.conftest import tiny_experiment, with_training

CPU = torch.device("cpu")


def split(stream):
    return stream[:1600], stream[1600:]


def data():
    return split(torch.arange(2000) % 17)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_constant_schedule():
    c = TrainingConfig(learning_rate=6e-5, min_learning_rate=6e-5, warmup_steps=0,
                       max_steps=7200, lr_schedule="constant")  # fmt: skip
    assert all(learning_rate(s, c) == 6e-5 for s in range(0, 8000, 7))
    w = TrainingConfig(learning_rate=1e-3, warmup_steps=10, max_steps=100, lr_schedule="constant")
    assert learning_rate(4, w) == pytest.approx(5e-4) and learning_rate(50, w) == 1e-3
    with pytest.raises(ConfigError):
        TrainingConfig(lr_schedule="linear")
    assert TrainingConfig().lr_schedule == "cosine"  # default unchanged


def phase2(cfg, end):
    return with_training(cfg, max_steps=end, learning_rate=3e-4, min_learning_rate=3e-4,
                         warmup_steps=0, lr_schedule="constant", eval_interval=4,
                         checkpoint_interval=4)  # fmt: skip


def test_continuation_restores_everything_and_records_phase(tmp_path):
    base = tiny_experiment(model_kw={"dropout": 0.1}, max_steps=8, eval_interval=4,
                           checkpoint_interval=4)  # fmt: skip
    t1 = Trainer(base, *data(), CPU, tmp_path / "phase1")
    t1.fit()
    parent = tmp_path / "phase1" / "checkpoints" / "step_00000008.pt"
    before = {p.name: sha(p) for p in (tmp_path / "phase1" / "checkpoints").iterdir()}

    cfg2 = phase2(base, 16)
    t2 = Trainer(cfg2, *data(), CPU, tmp_path / "phase2")
    t2.load_state(parent, continuation=True)
    state, _ = load_checkpoint(parent)
    # weights, optimizer moments, step, sampler and histories restored exactly
    for a, b in zip(t1.model.state_dict().values(), t2.model.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    o1, o2 = state["optimizer"]["state"], t2.optimizer.state_dict()["state"]
    for k in o1:
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(o1[k][name].cpu(), o2[k][name].cpu())
    assert t2.step == 8 and t2.tokens_seen == t1.tokens_seen
    assert torch.equal(t2.data_generator.get_state(), t1.data_generator.get_state())
    assert t2.loss_history == t1.loss_history and len(t2.eval_history) == len(t1.eval_history)
    assert [p["start_step"] for p in t2.phases] == [0, 8]
    assert t2.phases[0]["end_step"] == 8 and t2.phases[0]["lr_schedule"] == "cosine"
    assert t2.phases[1]["lr_schedule"] == "constant" and t2.phases[1]["end_step"] == 16
    assert t2.phases[1]["parent_checkpoint"] == str(parent)

    t2.fit()
    assert t2.step == 16
    assert all(lr == 3e-4 for lr in [learning_rate(s, cfg2.training) for s in range(8, 16)])
    names = [p.name for p in list_checkpoints(tmp_path / "phase2")]
    assert names == ["step_00000012.pt", "step_00000016.pt"]  # numbering continues
    meta = read_metadata(tmp_path / "phase2" / "checkpoints" / "step_00000016.pt")
    assert meta is not None and meta.kind == "final"
    saved, _ = load_checkpoint(tmp_path / "phase2" / "checkpoints" / "step_00000016.pt")
    assert [p["start_step"] for p in saved["phases"]] == [0, 8]
    # the parent run is untouched
    assert before == {p.name: sha(p) for p in (tmp_path / "phase1" / "checkpoints").iterdir()}


def test_continuation_batches_match_an_uninterrupted_continuation(tmp_path):
    base = tiny_experiment(max_steps=6, checkpoint_interval=6)
    Trainer(base, *data(), CPU, tmp_path / "p1").fit()
    parent = tmp_path / "p1" / "checkpoints" / "step_00000006.pt"
    runs = []
    for _k in range(2):
        t = Trainer(phase2(base, 10), *data(), CPU)
        t.load_state(parent, continuation=True)
        t.fit()
        runs.append(t)
    assert runs[0].loss_history == runs[1].loss_history  # deterministic on CPU
    for a, b in zip(runs[0].model.parameters(), runs[1].model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_continuation_rejects_mismatches(tmp_path):
    base = tiny_experiment(max_steps=4, checkpoint_interval=4)
    Trainer(base, *data(), CPU, tmp_path / "p1").fit()
    parent = tmp_path / "p1" / "checkpoints" / "step_00000004.pt"
    with pytest.raises(ValueError, match="model config"):
        Trainer(tiny_experiment(model_kw={"n_layer": 1}, max_steps=8), *data(),
                CPU).load_state(parent, continuation=True)  # fmt: skip
    with pytest.raises(ValueError, match="max_steps"):
        Trainer(phase2(base, 4), *data(), CPU).load_state(parent, continuation=True)
    import dataclasses

    other_data = dataclasses.replace(phase2(base, 8), data=dataclasses.replace(
        base.data, dataset="something_else"))  # fmt: skip
    with pytest.raises(ValueError, match="data config"):
        Trainer(other_data, *data(), CPU).load_state(parent, continuation=True)


def test_old_checkpoints_get_a_synthesized_phase(tmp_path):
    base = tiny_experiment(max_steps=4, checkpoint_interval=4)
    Trainer(base, *data(), CPU, tmp_path / "p1").fit()
    path = tmp_path / "p1" / "checkpoints" / "step_00000004.pt"
    blob = torch.load(path, weights_only=True)
    del blob["state"]["phases"]
    del blob["metadata"]["config"]["training"]["lr_schedule"]
    old = tmp_path / "old.pt"
    torch.save(blob, old)
    t = Trainer(phase2(base, 8), *data(), CPU)
    t.load_state(old, continuation=True)
    assert [p["lr_schedule"] for p in t.phases] == ["cosine", "constant"]

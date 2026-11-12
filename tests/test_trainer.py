import math

import pytest
import torch

from mini_chatgpt.model import GPT
from mini_chatgpt.training.checkpoint import latest_checkpoint, list_checkpoints
from mini_chatgpt.training.run import read_metrics
from mini_chatgpt.training.trainer import NonFiniteError, Trainer, load_model
from tests.conftest import requires_mps, tiny_experiment, with_training

CPU = torch.device("cpu")


def split(stream):
    return stream[:1600], stream[1600:]


def test_overfit_single_batch():
    """The model must be able to memorize one fixed batch."""
    torch.manual_seed(0)
    from tests.conftest import tiny_model_config

    m = GPT(tiny_model_config(n_embd=64, n_head=4))
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 17, (4, 16), generator=g)
    y = torch.randint(0, 17, (4, 16), generator=g)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    losses = []
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 2.5
    assert losses[-1] < 0.1, losses[-1]


def test_training_reduces_loss_and_logs(tmp_path, token_stream):
    # A learnable stream: next token = (prev + 1) mod 17, with noise-free structure.
    stream = torch.arange(2000) % 17
    cfg = tiny_experiment(max_steps=40, eval_interval=20, checkpoint_interval=20)
    t = Trainer(cfg, *split(stream), CPU, tmp_path)
    before = t.estimate_loss()
    after = t.fit()
    assert after["val"] < before["val"] - 1.0
    events = read_metrics(tmp_path / "metrics.jsonl")
    assert {e["event"] for e in events} == {"train", "eval"}
    assert [p.name for p in list_checkpoints(tmp_path)] == ["step_00000020.pt", "step_00000040.pt"]
    assert t.tokens_seen == 40 * t.tokens_per_step


def test_gradient_accumulation_matches_large_batch(token_stream):
    """accum=2 x batch=4 must equal batch=8 when the 8 sequences are identical."""
    big = Trainer(tiny_experiment(batch_size=8), *split(token_stream), CPU)
    acc = Trainer(
        tiny_experiment(batch_size=4, gradient_accumulation_steps=2), *split(token_stream), CPU
    )
    acc.model.load_state_dict(big.model.state_dict())
    x, y = big._batch(big.train_data, torch.Generator().manual_seed(9))
    for trainer, chunks in ((big, [(x, y)]), (acc, [(x[:4], y[:4]), (x[4:], y[4:])])):
        trainer.optimizer.zero_grad()
        for cx, cy in chunks:
            _, loss = trainer.model(cx, cy)
            (loss / len(chunks)).backward()
    for (n, a), (_, b) in zip(
        big.model.named_parameters(), acc.model.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5, msg=n)


def test_nan_detection_leaves_weights_unchanged(token_stream):
    t = Trainer(tiny_experiment(), *split(token_stream), CPU)
    with torch.no_grad():
        t.model.wte.weight[0, 0] = float("nan")
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    with pytest.raises(NonFiniteError):
        t.train_step()
    for k, v in t.model.state_dict().items():
        assert torch.equal(v, before[k]) or torch.isnan(v).any(), k
    assert t.step == 0


def test_rejects_unverified_dtype(token_stream):
    with pytest.raises(ValueError, match="not verified"):
        Trainer(tiny_experiment(dtype="bfloat16"), *split(token_stream), CPU)


def test_checkpoint_preserves_predictions(tmp_path, token_stream):
    t = Trainer(tiny_experiment(), *split(token_stream), CPU, tmp_path)
    t.fit(max_steps=5)
    path = t.save()
    model, cfg = load_model(path, CPU)
    x = token_stream[:16].unsqueeze(0)
    t.model.eval()
    assert torch.equal(model(x)[0], t.model(x)[0])
    assert cfg == t.config
    assert model.lm_head.weight is model.wte.weight


def _resume_matches(device, tmp_path, token_stream, exact):
    cfg = tiny_experiment(model_kw={"dropout": 0.1}, max_steps=12, checkpoint_interval=6)
    cfg = with_training(cfg, device=device.type)
    full = Trainer(cfg, *split(token_stream), device, tmp_path / "full")
    full.fit()

    first = Trainer(cfg, *split(token_stream), device, tmp_path / "part")
    first.fit(max_steps=6)
    ckpt = latest_checkpoint(tmp_path / "part")
    torch.manual_seed(999)  # scramble global RNG to prove it is restored
    resumed = Trainer(cfg, *split(token_stream), device, tmp_path / "part")
    resumed.load_state(ckpt)
    assert resumed.step == 6
    resumed.fit()
    assert resumed.step == 12
    assert resumed.loss_history == full.loss_history[6:] or not exact
    for (n, a), (_, b) in zip(
        full.model.state_dict().items(), resumed.model.state_dict().items(), strict=True
    ):
        if exact:
            assert torch.equal(a, b), n
        else:
            torch.testing.assert_close(a.cpu(), b.cpu(), atol=1e-5, rtol=1e-4, msg=n)
    return full, resumed


def test_resume_is_bit_exact_on_cpu(tmp_path, token_stream):
    _resume_matches(CPU, tmp_path, token_stream, exact=True)


@requires_mps
def test_resume_on_mps(tmp_path, token_stream):
    full, resumed = _resume_matches(torch.device("mps"), tmp_path, token_stream, exact=False)
    for a, b in zip(full.loss_history[6:], resumed.loss_history, strict=True):
        assert math.isclose(a, b, rel_tol=1e-4)


def test_resume_rejects_different_model(tmp_path, token_stream):
    t = Trainer(tiny_experiment(), *split(token_stream), CPU, tmp_path)
    path = t.save()
    other = Trainer(tiny_experiment(model_kw={"n_layer": 1}), *split(token_stream), CPU)
    with pytest.raises(ValueError, match="model config differs"):
        other.load_state(path)


@requires_mps
def test_mps_training_smoke(token_stream):
    t = Trainer(
        with_training(tiny_experiment(), device="mps"), *split(token_stream), torch.device("mps")
    )
    t.fit(max_steps=3)
    assert t.step == 3 and all(math.isfinite(x) for x in t.loss_history)
    assert t.stats.peak_memory["mps_driver_allocated@after_step"] > 0
    assert next(t.model.parameters()).device.type == "mps"

"""M6 local chat demo: engine, API validation, stopping, history and stability."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from mini_chatgpt.chat import format as cf
from mini_chatgpt.config import ModelConfig
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.serve.app import ChatRequest, create_app, validate_request
from mini_chatgpt.serve.engine import ChatEngine, Limits
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer

ROOT = Path(__file__).resolve().parent.parent
M5_CKPT = ROOT / "runs/m5_sft_chat-20261114-175334/checkpoints/best.pt"
SPECIALS = ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>")
CORPUS = ["Hello there. What is Python? Python is a programming language.\n"] * 40


@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(CORPUS, 300, special_tokens=SPECIALS)


def tiny_model(vocab: int, ctx: int = 64) -> GPT:
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, context_length=ctx, n_layer=1, n_head=2, n_embd=16))


def rigged(vocab: int, favourite: int, ctx: int = 64) -> GPT:
    """A model whose greedy (and near-certain sampled) next token is always ``favourite``."""
    model = tiny_model(vocab, ctx)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.ln_f.weight.zero_()
        model.ln_f.bias.zero_()
        model.ln_f.bias[0] = 1.0
        model.lm_head.weight[favourite, 0] = 50.0
    return model


def client_for(engine: ChatEngine) -> tuple[TestClient, Any]:
    app = create_app(lambda: engine)
    client = TestClient(app)
    client.__enter__()
    app.state.holder.wait(10)
    return client, app


def body(message: str = "Hi", history: list[dict[str, str]] | None = None, **settings: Any):
    out: dict[str, Any] = {"message": message, "history": history or []}
    if settings:
        out["settings"] = settings
    return out


# --------------------------------------------------------------------------- #
# request validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (body(""), "empty"),
        (body("x" * 2001), "longer than 2000"),
        (body(history=[{"role": "user", "content": "a"}]), "complete user/assistant pairs"),
        (
            body(history=[{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]),
            "must have role 'user'",
        ),
        (
            body(history=[{"role": "user", "content": "a"}, {"role": "assistant", "content": " "}]),
            "is empty",
        ),
        (body(temperature=3.0), "temperature"),
        (body(max_new_tokens=1000), "max_new_tokens"),
        (body(max_new_tokens=0), "max_new_tokens"),
    ],
)
def test_validation_rejects_malformed_requests(payload, reason):
    req = ChatRequest(**payload)
    with pytest.raises(Exception) as err:
        validate_request(req, Limits())
    assert reason in str(err.value.detail)


def test_validation_limits_history_and_total_size():
    pairs = [{"role": r, "content": "x"} for _ in range(21) for r in ("user", "assistant")]
    with pytest.raises(Exception, match="history longer"):
        validate_request(ChatRequest(**body(history=pairs)), Limits())
    big = [{"role": r, "content": "y" * 1999} for _ in range(5) for r in ("user", "assistant")]
    with pytest.raises(Exception, match="conversation longer"):
        validate_request(ChatRequest(**body(history=big)), Limits())


def test_validation_builds_message_list_in_order():
    hist = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    msgs = validate_request(ChatRequest(**body("c", hist)), Limits())
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [m["content"] for m in msgs] == ["a", "b", "c"]


def test_api_rejects_unknown_fields_and_roles(tok):
    client, _ = client_for(ChatEngine(tiny_model(tok.vocab_size), tok))
    assert client.post("/api/chat", json={**body(), "extra": 1}).status_code == 422
    bad = body(history=[{"role": "system", "content": "x"}, {"role": "assistant", "content": "y"}])
    assert client.post("/api/chat", json=bad).status_code == 422
    assert client.post("/api/chat", json={"history": []}).status_code == 422


# --------------------------------------------------------------------------- #
# generation through the API
# --------------------------------------------------------------------------- #
def test_health_reports_ready_with_model_info(tok):
    client, _ = client_for(ChatEngine(tiny_model(tok.vocab_size), tok, info={"name": "tiny"}))
    h = client.get("/api/health").json()
    assert h["status"] == "ready" and h["model"]["name"] == "tiny"
    assert h["defaults"] == {
        "temperature": 0.7,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "max_new_tokens": 200,
    }


def test_health_reports_loading_then_error():
    gate = threading.Event()

    def slow_fail() -> ChatEngine:
        gate.wait(5)
        raise RuntimeError("bad checkpoint")

    app = create_app(slow_fail)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "loading"
        assert client.post("/api/chat", json=body()).status_code == 503
        gate.set()
        app.state.holder.wait(5)
        h = client.get("/api/health").json()
        assert h["status"] == "error" and "bad checkpoint" in h["error"]


def test_end_of_turn_stop(tok):
    end = cf.ChatTokens.from_tokenizer(tok).end
    client, _ = client_for(ChatEngine(rigged(tok.vocab_size, end), tok))
    r = client.post("/api/chat", json=body(seed=1)).json()
    assert r["stop_reason"] == "end_of_turn" and r["new_tokens"] == 0 and r["reply"] == ""


def test_max_token_cutoff_is_reported_as_truncation(tok):
    word = tok.encode("Hello")[0]
    client, _ = client_for(ChatEngine(rigged(tok.vocab_size, word), tok))
    r = client.post("/api/chat", json=body(max_new_tokens=5, seed=1)).json()
    assert r["stop_reason"] == "max_tokens" and r["new_tokens"] == 5
    assert r["settings"]["max_new_tokens"] == 5 and r["settings"]["repetition_penalty"] == 1.2
    assert r["duration_ms"] >= 0 and r["seed"] == 1


def test_structural_token_stop(tok):
    user = cf.ChatTokens.from_tokenizer(tok).user
    client, _ = client_for(ChatEngine(rigged(tok.vocab_size, user), tok))
    assert client.post("/api/chat", json=body(seed=1)).json()["stop_reason"] == "structural_token"


def test_multi_turn_history_is_serialized_like_training(tok):
    engine = ChatEngine(tiny_model(tok.vocab_size, ctx=512), tok)
    client, _ = client_for(engine)
    hist = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    r = client.post("/api/chat", json=body("What is Python?", hist, max_new_tokens=3)).json()
    expected, _ = cf.build_prompt(tok, [*hist, {"role": "user", "content": "What is Python?"}])
    assert r["prompt_tokens"] == len(expected) and r["dropped_history_messages"] == 0


def test_history_truncated_to_fit_context(tok):
    client, _ = client_for(ChatEngine(tiny_model(tok.vocab_size, ctx=64), tok))
    hist = [
        {"role": "user", "content": "Python " * 10},
        {"role": "assistant", "content": "ok " * 10},
    ]
    r = client.post("/api/chat", json=body("Hi", hist * 2, max_new_tokens=8)).json()
    assert r["dropped_history_messages"] >= 2 and r["prompt_tokens"] <= 64 - 8


def test_prompt_too_long_for_context_is_413(tok):
    client, _ = client_for(ChatEngine(tiny_model(tok.vocab_size, ctx=64), tok))
    r = client.post("/api/chat", json=body("Python is a language. " * 20))
    assert r.status_code == 413 and "shorten" in r.json()["detail"]


def test_generation_error_returns_error_stop_reason(tok):
    engine = ChatEngine(tiny_model(tok.vocab_size), tok)

    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("MPS exploded")

    engine.model.forward = boom
    client, _ = client_for(engine)
    r = client.post("/api/chat", json=body())
    assert r.status_code == 500 and r.json()["stop_reason"] == "error"
    assert "MPS exploded" in r.json()["detail"]
    # the lock was released: a later request is not stuck behind the failure
    engine.model.forward = GPT.forward.__get__(engine.model)
    assert client.post("/api/chat", json=body(max_new_tokens=2)).status_code == 200


def test_busy_engine_returns_503(tok):
    engine = ChatEngine(tiny_model(tok.vocab_size), tok, limits=Limits(queue_timeout_s=0.05))
    client, _ = client_for(engine)
    engine._lock.acquire()
    try:
        assert client.post("/api/chat", json=body()).status_code == 503
    finally:
        engine._lock.release()


def test_repeated_requests_are_stable_and_seeded(tok):
    engine = ChatEngine(tiny_model(tok.vocab_size), tok)
    client, _ = client_for(engine)
    replies = [
        client.post("/api/chat", json=body(max_new_tokens=6, seed=7)).json() for _ in range(30)
    ]
    assert len({r["reply"] for r in replies}) == 1  # same seed, same reply
    assert engine.requests_served == 30
    assert all(r["new_tokens"] <= 6 for r in replies)


def test_concurrent_requests_are_serialized(tok):
    engine = ChatEngine(tiny_model(tok.vocab_size), tok)
    active, peak = 0, 0
    real = engine.model.forward
    guard = threading.Lock()

    def counting(*a: Any, **k: Any) -> Any:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            return real(*a, **k)
        finally:
            with guard:
                active -= 1

    engine.model.forward = counting
    client, _ = client_for(engine)
    threads = [
        threading.Thread(target=lambda: client.post("/api/chat", json=body(max_new_tokens=4)))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert engine.requests_served == 6 and peak == 1


def test_static_frontend_served_when_built(tok, tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>demo</title>")
    app = create_app(lambda: ChatEngine(tiny_model(tok.vocab_size), tok), static_dir=tmp_path)
    with TestClient(app) as client:
        assert "demo" in client.get("/").text
        app.state.holder.wait(5)
        assert client.get("/api/health").json()["status"] == "ready"


# --------------------------------------------------------------------------- #
# the real M5 checkpoint (skipped if the git-ignored run is absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not M5_CKPT.exists(), reason="M5 checkpoint not present")
def test_m5_checkpoint_and_tokenizer_compatibility(tmp_path):
    engine = ChatEngine.from_checkpoint(M5_CKPT, device="cpu")
    assert engine.info["parameters"] == 13_989_888
    assert engine.info["context_length"] == 512
    assert engine.info["tokenizer_fingerprint"].startswith("e6be77cc")
    assert not engine.model.training
    ct = cf.ChatTokens.from_tokenizer(engine.tokenizer)
    assert (ct.user, ct.assistant, ct.system, ct.end) == (8188, 8189, 8190, 8191)
    with pytest.raises(RuntimeError, match="fingerprint"):
        ChatEngine.from_checkpoint(M5_CKPT, device="cpu", expected_fingerprint="0" * 64)

"""The M4 general-language evaluation suite and its scorer."""

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from mini_chatgpt.analysis import general_eval as ge
from mini_chatgpt.config import ModelConfig
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "evals" / "general_v1.json"
CORPUS = ["The cat sat on the mat. The dog ran in the park.", "def f(x):\n    return x + 1\n"] * 30
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(CORPUS, 290, special_tokens=("<|endoftext|>",))


@pytest.fixture(scope="module")
def chat_tok() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(CORPUS, 290, special_tokens=("<|endoftext|>", *ge.CHAT_TOKENS))


def tiny_model(vocab: int) -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=vocab, context_length=32, n_layer=1, n_head=2, n_embd=16)
    return GPT(cfg).eval()


def test_shipped_suite_is_valid_and_complete():
    suite, sha = ge.load_suite(SUITE)
    assert len(sha) == 64
    s = suite["sections"]
    assert len(s["facts_mc"]["items"]) == 40 and len(s["qa_mc"]["items"]) == 30
    assert len(s["grammar_pairs"]["items"]) == 30 and len(s["code_mc"]["items"]) == 20
    assert s["chat"]["track"] == "instruction"
    assert all(sec["track"] == "next_token" for k, sec in s.items() if k != "chat")
    texts = ge.all_eval_texts(suite)
    assert texts["fact01"].endswith("The capital of France is Paris")


def test_validate_suite_rejects_bad_items():
    suite = json.loads(SUITE.read_text())
    bad = copy.deepcopy(suite)
    bad["sections"]["facts_mc"]["items"][0]["answer"] = 9
    with pytest.raises(ValueError, match="answer index"):
        ge.validate_suite(bad)
    bad = copy.deepcopy(suite)
    bad["sections"]["qa_mc"]["items"][1]["id"] = "qa01"
    with pytest.raises(ValueError, match="duplicate"):
        ge.validate_suite(bad)
    bad = copy.deepcopy(suite)
    bad["sections"]["chat"]["items"][0]["turns"].append({"role": "assistant", "content": "x"})
    with pytest.raises(ValueError, match="end with a user turn"):
        ge.validate_suite(bad)


def test_continuation_logprob_matches_manual_sum(tok):
    model = tiny_model(tok.vocab_size)
    ctx, cont = tok.encode("The cat"), tok.encode(" sat on")
    ids = ctx + cont
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids[:-1]]))
    logp = torch.log_softmax(logits[0], -1)
    manual = sum(float(logp[i - 1, ids[i]]) for i in range(len(ctx), len(ids)))
    assert ge.continuation_logprob(model, ctx, cont, CPU) == pytest.approx(manual, abs=1e-5)


def test_continuation_logprob_crops_to_context(tok):
    model = tiny_model(tok.vocab_size)
    long_ctx = tok.encode("The cat sat on the mat. " * 20)
    value = ge.continuation_logprob(model, long_ctx, tok.encode(" The dog"), CPU)
    assert math.isfinite(value) and value < 0


def test_multiple_choice_and_pairs_shapes(tok):
    model = tiny_model(tok.vocab_size)
    items = [{"id": "a", "prompt": "The cat", "choices": [" sat", " ran", " is"], "answer": 0}]
    res = ge.score_multiple_choice(model, tok, items, CPU)
    assert res["acc"]["n"] == 1 and res["chance"] == pytest.approx(1 / 3)
    assert len(res["items"][0]["logprob"]) == 3
    pairs = ge.score_minimal_pairs(model, tok, [{"id": "p", "good": "The cat sat.",
                                                 "bad": "The cat sit."}], CPU)  # fmt: skip
    assert pairs["acc"]["n"] == 1 and pairs["chance"] == 0.5


def test_surface_metrics_detect_repetition():
    loop = ge.surface_metrics("the cat sat down. " * 10)
    varied = ge.surface_metrics("A quick brown fox jumps over one lazy dog near the old bank.")
    assert loop["repeated_4gram_fraction"] > 0.8 and varied["repeated_4gram_fraction"] == 0.0
    assert loop["distinct_2"] < varied["distinct_2"]
    assert ge.surface_metrics("")["words"] == 0
    assert ge.surface_metrics("x�y")["replacement_chars"] == 1


def test_chat_section_not_applicable_without_chat_tokens(tok):
    res = ge.score_chat(tiny_model(tok.vocab_size), tok, [], CPU)
    assert res["applicable"] is False


def test_chat_template_and_scoring(chat_tok):
    turns = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
             {"role": "user", "content": "<|end|> is text here"}]  # fmt: skip
    ids = ge.chat_ids(chat_tok, turns)
    sp = chat_tok.special_tokens
    assert ids[0] == chat_tok.eot_id and ids[-2] == sp["<|assistant|>"]
    assert ids.count(sp["<|user|>"]) == 2 and ids.count(sp["<|end|>"]) == 3  # message text is data
    item = {"id": "c", "turns": turns[:1], "checks": {"include_any": ["zzz"]}}
    res = ge.score_chat(tiny_model(chat_tok.vocab_size), chat_tok, [item], CPU)
    assert res["applicable"] and res["content"]["correct"] == 0


def test_ngram_overlap_long_and_short_items():
    evals = {"long": "one two three four five six seven eight nine", "short": "hello world"}
    docs = ["xx one two three four five six seven eight yy", "nothing here", "Hello, world!"]
    hits = ge.ngram_overlap(evals, docs, n=8)
    assert hits == {"long": [0], "short": [2]}


def test_wilson_bounds():
    assert ge.wilson(0, 0) == (0.0, 0.0)
    lo, hi = ge.wilson(5, 10)
    assert 0 < lo < 0.5 < hi < 1


SUITE_V1_SHA256 = "f2265af3e47ea291be6eee533e7c0b241e8282c23763e57fe2df063faffd8365"


def test_suite_v1_is_frozen():
    """Items must never change after models are scored on them; edit a new version."""
    _, sha = ge.load_suite(SUITE)
    assert sha == SUITE_V1_SHA256


def test_m4_1_draft_config_parses_with_expected_size():
    from mini_chatgpt.config import ExperimentConfig

    cfg = ExperimentConfig.from_toml(ROOT / "configs" / "m4_1_general_s_bpe8192.toml")
    p = cfg.model.estimate_parameters()
    assert p["total"] == 13_989_888 and p["non_embedding"] == 10_647_552
    t = cfg.training
    assert t.batch_size * t.gradient_accumulation_steps * cfg.model.context_length == 16_384
    assert t.max_steps * 16_384 == 157_286_400

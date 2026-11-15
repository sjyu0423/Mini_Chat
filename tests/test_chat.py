"""M5 conversational post-training: format, masking, data, SFT and chat generation."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from mini_chatgpt.analysis import general_eval as ge
from mini_chatgpt.chat import data as cd
from mini_chatgpt.chat import format as cf
from mini_chatgpt.chat.generate import apply_repetition_penalty, chat_reply
from mini_chatgpt.chat.sft import (
    ChatSplit,
    SFTConfig,
    SFTTrainer,
    batch_plan,
    evaluate,
    make_batch,
    masked_loss,
)
from mini_chatgpt.config import ExperimentConfig, ModelConfig
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import CheckpointMetadata, save_checkpoint

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ["Hello there. What is Python? Python is a programming language.\n"] * 40
SPECIALS = ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>")
CPU = torch.device("cpu")
CHAT_SUITE_SHA = "c5b264d3456e6923c8c9f9d4e9b3ee1c9977288d653f22d823b9a56b49c34b30"


@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(CORPUS, 300, special_tokens=SPECIALS)


def conv(*texts: str) -> list[cf.Message]:
    roles = ("user", "assistant")
    return [cast(cf.Message, {"role": roles[i % 2], "content": t}) for i, t in enumerate(texts)]


# --------------------------------------------------------------------------- #
# serialization and masking
# --------------------------------------------------------------------------- #
def test_serialization_layout_and_mask(tok):
    ct = cf.ChatTokens.from_tokenizer(tok)
    ids, mask = cf.encode_conversation(tok, conv("What is Python?", "A language."))
    nl = list(ct.newline)
    q, a = tok.encode("What is Python?"), tok.encode("A language.")
    expected = [ct.eot, ct.user, *nl, *q, ct.end, *nl, ct.assistant, *nl, *a, ct.end, *nl]
    assert ids == expected
    # targets: exactly the assistant content and the <|end|> that closes it
    targets = [i for i, m in zip(ids, mask, strict=True) if m]
    assert targets == [*a, ct.end]
    assert len(mask) == len(ids)


def test_multi_turn_mask_covers_every_assistant_turn_only(tok):
    ct = cf.ChatTokens.from_tokenizer(tok)
    msgs = conv("Hi", "Hello!", "What is Python?", "A language.")
    ids, mask = cf.encode_conversation(tok, msgs)
    targets = [i for i, m in zip(ids, mask, strict=True) if m]
    assert targets == [*tok.encode("Hello!"), ct.end, *tok.encode("A language."), ct.end]
    assert sum(1 for i, m in zip(ids, mask, strict=True) if m and i == ct.end) == 2
    for header in (ct.user, ct.assistant, ct.eot):
        assert all(not m for i, m in zip(ids, mask, strict=True) if i == header)


def test_labels_are_next_token_aligned(tok):
    ids, mask = cf.encode_conversation(tok, conv("Hi", "Hello!"))
    x, y = cf.labels_from_mask(ids, mask)
    assert x == ids[:-1] and len(y) == len(x)
    for t in range(len(x)):
        assert y[t] == (ids[t + 1] if mask[t + 1] else cf.IGNORE)
    # the position holding "\n" after <|assistant|> predicts the first reply token
    ct = cf.ChatTokens.from_tokenizer(tok)
    p = ids.index(ct.assistant) + len(ct.newline)
    assert y[p] == tok.encode("Hello!")[0]
    assert y[ids.index(ct.assistant)] == cf.IGNORE  # header newline is not a target


def test_special_token_text_in_messages_is_data(tok):
    ct = cf.ChatTokens.from_tokenizer(tok)
    ids, _ = cf.encode_conversation(tok, conv("say <|end|> please", "ok <|user|>"))
    assert ids.count(ct.end) == 2 and ids.count(ct.user) == 1


def test_invalid_conversations_rejected(tok):
    with pytest.raises(ValueError):
        cf.encode_conversation(tok, conv("only a user turn"))
    with pytest.raises(ValueError):
        cf.encode_conversation(
            tok, [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
        )
    with pytest.raises(ValueError):
        cf.encode_conversation(tok, conv("q", "   "))


def test_prompt_matches_frozen_suite_template(tok):
    turns: list[cf.Message] = [
        {"role": "user", "content": "My name is Lena."},
        {"role": "assistant", "content": "Nice to meet you!"},
        {"role": "user", "content": "What is my name?"},
    ]
    ids, dropped = cf.build_prompt(tok, turns)
    assert ids == ge.chat_ids(tok, [dict(t) for t in turns]) and dropped == 0
    # prompt == conversation prefix up to the open assistant header
    full, _ = cf.encode_conversation(tok, [*turns, {"role": "assistant", "content": "Lena."}])
    assert full[: len(ids)] == ids


def test_prompt_drops_oldest_exchanges_to_fit(tok):
    turns = conv("first question " * 20, "first answer " * 20, "second?")
    full, _ = cf.build_prompt(tok, turns)
    short, dropped = cf.build_prompt(tok, turns, max_prompt_tokens=len(full) - 1)
    assert dropped == 2 and short == cf.build_prompt(tok, conv("second?"))[0]
    with pytest.raises(ValueError):
        cf.build_prompt(tok, conv("x " * 100), max_prompt_tokens=10)


def test_fit_conversation_policy(tok):
    msgs = conv("q1", "a1 " * 5, "q2", "a2 " * 200)
    ids_short, _ = cf.encode_conversation(tok, msgs[:2])
    kept, status = cf.fit_conversation(tok, msgs, len(ids_short) + 5)
    assert status == "truncated" and kept == msgs[:2]
    assert cf.fit_conversation(tok, msgs, 10_000) == (msgs, "full")
    assert cf.fit_conversation(tok, msgs, 3) == ([], "dropped")
    # boundary: len(ids) == max_tokens + 1 still fits (x has max_tokens positions)
    assert cf.fit_conversation(tok, msgs[:2], len(ids_short) - 1)[1] == "full"


def test_display_text_removes_structural_tokens(tok):
    ct = cf.ChatTokens.from_tokenizer(tok)
    assert cf.display_text(tok, [ct.user, *tok.encode(" Hi "), ct.end]) == "Hi"


# --------------------------------------------------------------------------- #
# data pipeline
# --------------------------------------------------------------------------- #
def _msg(mid, role, text, rank=None, replies=(), **kw):
    return {
        "message_id": mid,
        "role": role,
        "text": text,
        "lang": kw.get("lang", "en"),
        "rank": rank,
        "deleted": kw.get("deleted", False),
        "synthetic": False,
        "review_result": kw.get("review", True),
        "replies": list(replies),
    }


def test_oasst_best_path_and_filters(tmp_path):
    tree = _msg(
        "r",
        "prompter",
        "Question?",
        replies=[
            _msg("a2", "assistant", "second best", rank=1),
            _msg(
                "a1",
                "assistant",
                "best",
                rank=0,
                replies=[
                    _msg(
                        "p",
                        "prompter",
                        "follow-up",
                        replies=[
                            _msg("x", "assistant", "deleted", rank=0, deleted=True),
                            _msg("y", "assistant", "kept", rank=1),
                        ],
                    )
                ],
            ),
            _msg("a0", "assistant", "spam", rank=0, review=False),
        ],
    )
    assert [m["text"] for m in cd.best_path(tree)] == ["Question?", "best", "follow-up", "kept"]
    path = tmp_path / "t.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write(
            json.dumps({"message_tree_id": "T", "tree_state": "ready_for_export", "prompt": tree})
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "message_tree_id": "U",
                    "tree_state": "ready_for_export",
                    "prompt": _msg("z", "prompter", "Hola", lang="es"),
                }
            )
            + "\n"
        )
    stats: dict[str, int] = {}
    out = list(cd.oasst2_conversations(path, stats))
    assert len(out) == 1 and stats["non_english_root"] == 1
    assert [m["role"] for m in out[0].messages] == ["user", "assistant", "user", "assistant"]


def test_dolly_formatting_and_cleaning(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps(
            {
                "instruction": "Summarize.\r\n",
                "context": "Text [12] here.",
                "response": "Short [3].",
                "category": "summarization",
            }
        )
        + "\n"
    )
    (c,) = cd.dolly_conversations(path, {})
    assert c.messages[0]["content"] == "Summarize.\n\nText  here."
    assert c.messages[1]["content"] == "Short ."
    assert cd.malformed_reason(c.messages) is None


def test_malformed_dedup_and_split():
    assert cd.malformed_reason(conv("q")) == "no_complete_exchange"
    assert cd.malformed_reason(conv("q", "")) == "empty_message"
    assert cd.malformed_reason(conv("Привет как дела", "Хорошо спасибо")) == "not_english_script"
    assert cd.dedup_key(conv("What  is X?", "a")) == cd.dedup_key(conv("what is x?", "b"))
    assert cd.split_of_group("g1") == cd.split_of_group("g1")
    shares = [cd.split_of_group(f"g{i}") for i in range(4000)]
    assert 0.03 < shares.count("val") / 4000 < 0.07


def test_pinned_chat_sources():
    names = {s.name: s for s in cd.CHAT_SOURCES}
    assert names["oasst2"].revision.startswith("179dd21")
    assert names["dolly15k"].revision.startswith("bdd27f4")
    assert all(len(s.sha256) == 64 for s in cd.CHAT_SOURCES)


def test_chat_suite_is_frozen():
    raw = (ROOT / "evals" / "chat_v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CHAT_SUITE_SHA
    items = json.loads(raw)["items"]
    assert len({i["id"] for i in items}) == len(items) == 40
    for i in items:
        cf.validate_messages(i["turns"], require_final_assistant=False)
        assert i["turns"][-1]["role"] == "user"


# --------------------------------------------------------------------------- #
# SFT batches, loss and trainer
# --------------------------------------------------------------------------- #
def make_split(tok, convs) -> ChatSplit:
    ids_all, mask_all, offsets = [], [], [0]
    for c in convs:
        ids, mask = cf.encode_conversation(tok, c)
        ids_all += ids
        mask_all += mask
        offsets.append(len(ids_all))
    return ChatSplit(
        np.asarray(ids_all, np.uint16),
        np.asarray(mask_all, np.uint8),
        np.asarray(offsets, np.int64),
    )


def tiny_model(vocab: int, ctx: int = 64) -> GPT:
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=vocab, context_length=ctx, n_layer=1, n_head=2, n_embd=16))


def test_make_batch_padding_and_labels(tok):
    convs = [conv("Hi", "Hello!"), conv("What is Python?", "A programming language.")]
    split = make_split(tok, convs)
    x, y = make_batch(split, [0, 1])
    for b, c in enumerate(convs):
        ids, mask = cf.encode_conversation(tok, c)
        rx, ry = cf.labels_from_mask(ids, mask)
        n = len(rx)
        assert x[b, :n].tolist() == rx and y[b, :n].tolist() == ry
        assert (y[b, n:] == cf.IGNORE).all()


def test_masked_loss_counts_only_assistant_targets(tok):
    split = make_split(tok, [conv("Hi", "Hello!"), conv("What is Python?", "A language.")])
    x, y = make_batch(split, [0, 1])
    model = tiny_model(tok.vocab_size)
    logits, _ = model(x)
    s, n = masked_loss(logits, y)
    keep = y != cf.IGNORE
    manual = -F.log_softmax(logits, -1).gather(-1, y.clamp(min=0)[..., None])[..., 0][keep].sum()
    assert n == int(keep.sum()) and torch.allclose(s, manual)
    # user/prompt logits do not influence the loss
    logits2 = logits.detach().clone()
    logits2[~keep] += torch.randn_like(logits2[~keep]) * 5
    assert torch.allclose(masked_loss(logits2, y)[0], s.detach())


def test_batch_plan_covers_each_conversation_once():
    lengths = np.arange(1, 101)
    plan = batch_plan(lengths, 8, seed=1, epoch=0, bucket_factor=3)
    flat = [i for b in plan for i in b]
    assert sorted(flat) == list(range(100))
    assert plan == batch_plan(lengths, 8, seed=1, epoch=0, bucket_factor=3)
    assert plan != batch_plan(lengths, 8, seed=1, epoch=1, bucket_factor=3)


def _base_checkpoint(tmp_path: Path, vocab: int) -> Path:
    cfg = ExperimentConfig(
        name="tiny",
        model=ModelConfig(vocab_size=vocab, context_length=64, n_layer=1, n_head=2, n_embd=16),
    )
    torch.manual_seed(0)
    model = GPT(cfg.model)
    return save_checkpoint(
        tmp_path,
        {"model": model.state_dict()},
        CheckpointMetadata(step=0, config=cfg.to_dict()),
        path=tmp_path / "base.pt",
    )


def test_sft_trainer_learns_and_state_round_trips(tok, tmp_path):
    convs = [conv("What is Python?", "A programming language."), conv("Hi", "Hello there.")] * 4
    split = make_split(tok, convs)
    base = _base_checkpoint(tmp_path, tok.vocab_size)
    end = cf.ChatTokens.from_tokenizer(tok).end
    cfg = SFTConfig(
        name="t",
        base_checkpoint=str(base),
        batch_size=4,
        epochs=150,
        learning_rate=3e-3,
        min_learning_rate=3e-3,
        warmup_steps=1,
        device="cpu",
    )
    tr = SFTTrainer(cfg, run_dir=None, device=CPU, train=split, val=split, end_id=end)
    before = evaluate(tr.model, split, end, CPU)
    for _ in range(200):
        tr.train_step()
    after = evaluate(tr.model, split, end, CPU)
    assert after["loss"] < 0.2 * before["loss"] and after["end_recall"] == 1.0
    # restoring model + fresh-trainer optimizer state reproduces the next step exactly
    state = {
        k: (v if not hasattr(v, "items") else json.loads(json.dumps(v, default=str)))
        for k, v in tr.state_dict().items()
        if k not in ("model", "optimizer")
    }
    snap = {
        "model": {k: v.clone() for k, v in tr.model.state_dict().items()},
        "optimizer": torch.load(_dump(tr.optimizer.state_dict(), tmp_path)),
        **state,
    }
    m1 = tr.train_step()
    tr2 = SFTTrainer(cfg, run_dir=None, device=CPU, train=split, val=split, end_id=end)
    tr2.load_state_dict(snap)
    m2 = tr2.train_step()
    assert m1["loss"] == pytest.approx(m2["loss"], abs=1e-6)
    for (k, a), b in zip(
        tr.model.state_dict().items(), tr2.model.state_dict().values(), strict=True
    ):
        assert torch.allclose(a, b, atol=1e-6), k


def _dump(obj, tmp_path: Path) -> Path:
    p = tmp_path / "opt.pt"
    torch.save(obj, p)
    return p


def test_sft_starts_from_base_weights_with_fresh_optimizer(tok, tmp_path):
    split = make_split(tok, [conv("Hi", "Hello!")] * 4)
    base = _base_checkpoint(tmp_path, tok.vocab_size)
    cfg = SFTConfig(
        name="t", base_checkpoint=str(base), batch_size=2, epochs=1, warmup_steps=1, device="cpu"
    )
    tr = SFTTrainer(cfg, run_dir=None, device=CPU, train=split, val=split, end_id=0)
    ref = torch.load(base, weights_only=True)["state"]["model"]
    assert all(torch.equal(ref[k], v) for k, v in tr.model.state_dict().items())
    assert len(tr.optimizer.state) == 0


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def test_repetition_penalty_rule():
    logits = torch.tensor([2.0, -2.0, 1.0, 3.0])
    out = apply_repetition_penalty(logits, [0, 1, 3], 2.0, frozenset({3}))
    assert out.tolist() == [1.0, -4.0, 1.0, 3.0]  # 3 is exempt (structural)
    assert apply_repetition_penalty(logits, [0], 1.0, frozenset()) is logits


def _rigged(tok, favourite: int) -> GPT:
    model = tiny_model(tok.vocab_size)
    with torch.no_grad():
        model.lm_head.weight.zero_()  # tied: also zeroes token embeddings (fine here)
        model.ln_f.bias.zero_()
        model.ln_f.weight.zero_()
        model.ln_f.bias[0] = 1.0
        model.lm_head.weight[favourite, 0] = 10.0
    return model


def test_chat_reply_stop_reasons(tok):
    ct = cf.ChatTokens.from_tokenizer(tok)
    hist = conv("Hi")
    r = chat_reply(_rigged(tok, ct.end), tok, hist, max_new_tokens=5, greedy=True)
    assert r.stop_reason == "end_of_turn" and r.ended_naturally and r.ids == []
    r = chat_reply(_rigged(tok, ct.user), tok, hist, max_new_tokens=5, greedy=True)
    assert r.stop_reason == "structural_token" and not r.ended_naturally
    word = tok.encode("Hello")[0]
    r = chat_reply(_rigged(tok, word), tok, hist, max_new_tokens=5, greedy=True)
    assert r.stop_reason == "max_tokens" and r.truncated and r.ids == [word] * 5
    # the penalty cannot stop a fully rigged model from repeating, but must not crash
    r = chat_reply(
        _rigged(tok, word), tok, hist, max_new_tokens=3, greedy=True, repetition_penalty=1.3
    )
    assert len(r.ids) == 3


def test_chat_reply_sampling_is_seeded(tok):
    model = tiny_model(tok.vocab_size)
    a = chat_reply(model, tok, conv("Hi"), max_new_tokens=8, seed=3)
    b = chat_reply(model, tok, conv("Hi"), max_new_tokens=8, seed=3)
    assert a.ids == b.ids and a.text == b.text


# --------------------------------------------------------------------------- #
# conversational scoring
# --------------------------------------------------------------------------- #
def test_check_content_rules():
    from mini_chatgpt.chat.evaluate import check_content

    assert check_content("It is Jupiter.", {"include_any": ["jupiter"]}) == {"include_any": True}
    assert check_content("- a\n- b\n3. c", {"min_list_items": 3})["min_list_items"]
    assert not check_content("a, b, c", {"min_list_items": 3})["min_list_items"]
    assert check_content("Yes.", {"max_words": 3})["max_words"]
    assert not check_content("Yes it is very hot.", {"max_words": 3})["max_words"]
    assert check_content("One. Two.", {"max_sentences": 2})["max_sentences"]
    assert check_content("fruit and vegetable", {"include_all": ["Fruit", "vegetable"]})[
        "include_all"
    ]


def test_score_reply_separates_truncation_from_success():
    from mini_chatgpt.chat.evaluate import aggregate, score_reply
    from mini_chatgpt.chat.generate import ChatReply

    item = {"id": "q", "category": "qa", "checks": {"include_any": ["Paris"]}}
    ok = score_reply(item, ChatReply("Paris.", [1], "end_of_turn", 5, settings={"seed": 1}))
    cut = score_reply(item, ChatReply("Paris " * 50, [1] * 200, "max_tokens", 5))
    assert ok["format_ok"] and ok["both_ok"]
    assert not cut["format_ok"] and cut["content_ok"] and not cut["both_ok"] and cut["repetitive"]
    agg = aggregate([ok, cut])
    assert agg["natural_end_of_turn"]["correct"] == 1 and agg["stop_reasons"]["max_tokens"] == 1


def test_padding_to_multiple_keeps_loss_identical(tok):
    split = make_split(tok, [conv("Hi", "Hello!"), conv("What is Python?", "A language.")])
    x1, y1 = make_batch(split, [0, 1])
    x2, y2 = make_batch(split, [0, 1], pad_multiple=16, max_len=64)
    assert x2.shape[1] % 16 == 0 and x2.shape[1] >= x1.shape[1]
    assert torch.equal(x2[:, : x1.shape[1]], x1) and (y2[:, x1.shape[1] :] == cf.IGNORE).all()
    model = tiny_model(tok.vocab_size)
    l1, n1 = masked_loss(model(x1)[0], y1)
    l2, n2 = masked_loss(model(x2)[0], y2)
    assert n1 == n2 and torch.allclose(l1, l2, atol=1e-5)  # causal: padding is invisible
    x3, _ = make_batch(split, [0, 1], pad_multiple=1000, max_len=x1.shape[1])
    assert x3.shape == x1.shape  # never padded beyond the context window


def test_shipped_sft_config_loads():
    cfg = SFTConfig.from_toml(ROOT / "configs" / "sft" / "m5_sft_chat.toml")
    assert cfg.base_checkpoint.endswith("step_00009600.pt")
    assert cfg.learning_rate == 1e-4 and cfg.epochs == 3 and cfg.batch_size == 16
    assert cfg.pad_multiple == 64 and cfg.empty_cache_interval == 25
    assert cfg.expected_fingerprint.startswith("e6be77cc")


DIAG_SUITE_SHA = "c5e376486626a6dd97753b7e7b84b4bd07bda4b0333f2c670a5e67a9b6b19e2f"


def test_diagnostic_suite_is_frozen():
    raw = (ROOT / "evals" / "chat_diag_v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DIAG_SUITE_SHA
    items = json.loads(raw)["items"]
    assert len({i["id"] for i in items}) == len(items) == 50
    for i in items:
        cf.validate_messages(i["turns"], require_final_assistant=False)
        assert i["turns"][-1]["role"] == "user"


def test_top_p_keeps_smallest_nucleus():
    from mini_chatgpt.chat.generate import apply_top_p

    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    out = apply_top_p(logits, 0.7, 1.0)
    assert torch.isfinite(out[:2]).all() and torch.isinf(out[2:]).all()
    out = apply_top_p(logits, 0.8, 1.0)  # 0.5 + 0.3 = 0.8 reaches p: third token dropped
    assert torch.isfinite(out[:2]).all() and torch.isinf(out[2:]).all()
    assert torch.isfinite(apply_top_p(logits, 0.01, 1.0)[0])  # always keeps the argmax
    assert apply_top_p(logits, None, 1.0) is logits
    shuffled = logits[[2, 0, 3, 1]]
    kept = torch.isfinite(apply_top_p(shuffled, 0.7, 1.0))
    assert kept.tolist() == [False, True, False, True]
    with pytest.raises(ValueError):
        apply_top_p(logits, 0.0, 1.0)


def test_chat_reply_with_top_p_is_seeded(tok):
    model = tiny_model(tok.vocab_size)
    a = chat_reply(model, tok, conv("Hi"), max_new_tokens=8, seed=3, top_k=None, top_p=0.9)
    b = chat_reply(model, tok, conv("Hi"), max_new_tokens=8, seed=3, top_k=None, top_p=0.9)
    assert a.ids == b.ids and a.settings["top_p"] == 0.9

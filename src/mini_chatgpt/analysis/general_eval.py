"""Scoring for the fixed general-language suite (``evals/general_v1.json``).

Two tracks are kept apart:

* **next_token**: what a base language model can be measured on. Multiple-choice
  items are scored by log-likelihood (no generation), minimal pairs by sentence
  log-probability, code completion by greedy exact prefix, and free generations
  by automatic surface metrics only.
* **instruction**: chat-formatted conversations. They need the chat special tokens
  (``<|user|>``, ``<|assistant|>``, ``<|end|>``); for a tokenizer without them the
  section is reported as not applicable rather than scored.

Every prompt is preceded by ``<|endoftext|>`` when the tokenizer has it, so items
start like a fresh document, as during training.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from mini_chatgpt.inference.generate import generate
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer

CHAT_TOKENS = ("<|user|>", "<|assistant|>", "<|end|>")
GEN_SEEDS = (11, 22, 33)
GEN_TEMPERATURE, GEN_TOP_K, GEN_MAX_NEW = 0.7, 40, 200
CODE_MAX_NEW = 16
CHAT_MAX_NEW = 128
MC_SECTIONS = ("facts_mc", "qa_mc", "code_mc")
WORD = re.compile(r"\w+|[^\w\s]")  # words and punctuation (surface metrics)
ALNUM = re.compile(r"\w+")  # words only (contamination n-grams ignore punctuation)
SENT_END = re.compile(r"[.!?](?:\s|$)")


# --------------------------------------------------------------------------- #
# suite loading
# --------------------------------------------------------------------------- #
def load_suite(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and validate a suite; returns it with the SHA-256 of the file bytes."""
    raw = Path(path).read_bytes()
    suite = json.loads(raw)
    validate_suite(suite)
    return suite, hashlib.sha256(raw).hexdigest()


def validate_suite(suite: dict[str, Any]) -> None:
    sections = suite.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("suite has no 'sections'")
    ids: list[str] = []
    for name, sec in sections.items():
        if sec.get("track") not in ("next_token", "instruction"):
            raise ValueError(f"section {name}: track must be next_token or instruction")
        for item in sec.get("items", []):
            ids.append(item["id"])
            if name in MC_SECTIONS:
                ch = item["choices"]
                if len(ch) < 2 or len(set(ch)) != len(ch):
                    raise ValueError(f"{item['id']}: need >= 2 distinct choices")
                if not 0 <= item["answer"] < len(ch):
                    raise ValueError(f"{item['id']}: answer index out of range")
            if name == "grammar_pairs" and item["good"] == item["bad"]:
                raise ValueError(f"{item['id']}: good and bad are identical")
            if name == "code_generation" and not item["accept"]:
                raise ValueError(f"{item['id']}: empty accept list")
            if name == "chat" and item["turns"][-1]["role"] != "user":
                raise ValueError(f"{item['id']}: conversation must end with a user turn")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item ids")


def all_eval_texts(suite: dict[str, Any]) -> dict[str, str]:
    """Every text the suite shows or expects, keyed by item id (for contamination checks)."""
    out: dict[str, str] = {}
    for sec in suite["sections"].values():
        for item in sec.get("items", []):
            parts: list[str] = []
            if "prompt" in item:
                parts.append(item["prompt"])
            if "choices" in item:
                parts.append(item["prompt"] + item["choices"][item["answer"]])
            for key in ("good", "bad"):
                if key in item:
                    parts.append(item[key])
            for acc in item.get("accept", []):
                parts.append(item["prompt"] + acc)
            for turn in item.get("turns", []):
                parts.append(turn["content"])
            out[item["id"]] = "\n".join(parts)
    return out


# --------------------------------------------------------------------------- #
# log-likelihood scoring
# --------------------------------------------------------------------------- #
def _prefix(tok: ByteBPETokenizer) -> list[int]:
    return [tok.eot_id] if tok.eot_id is not None else []


@torch.no_grad()
def continuation_logprob(
    model: GPT, context: list[int], continuation: list[int], device: torch.device
) -> float:
    """Sum of log p(continuation | context), cropped to the model's context window."""
    if not continuation:
        raise ValueError("empty continuation")
    ids = (context + continuation)[-(model.config.context_length + 1) :]
    n_cont = min(len(continuation), len(ids) - 1)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    picked = logp[torch.arange(len(ids) - 1, device=device), y]
    return float(picked[-n_cont:].sum())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"correct": k, "n": n, "rate": k / n if n else 0.0, "ci95": [lo, hi]}


def score_multiple_choice(
    model: GPT, tok: ByteBPETokenizer, items: list[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    rows = []
    for item in items:
        ctx = _prefix(tok) + tok.encode(item["prompt"])
        sums, norms = [], []
        for choice in item["choices"]:
            lp = continuation_logprob(model, ctx, tok.encode(choice), device)
            sums.append(lp)
            norms.append(lp / len(choice.encode("utf-8")))
        pick = max(range(len(sums)), key=sums.__getitem__)
        pick_norm = max(range(len(norms)), key=norms.__getitem__)
        rows.append({"id": item["id"], "logprob": sums, "logprob_per_byte": norms,
                     "pred": pick, "pred_norm": pick_norm, "answer": item["answer"]})  # fmt: skip
    n = len(rows)
    return {
        "acc": _rate(sum(r["pred"] == r["answer"] for r in rows), n),
        "acc_norm": _rate(sum(r["pred_norm"] == r["answer"] for r in rows), n),
        "chance": sum(1 / len(i["choices"]) for i in items) / n if n else 0.0,
        "items": rows,
    }


def score_minimal_pairs(
    model: GPT, tok: ByteBPETokenizer, items: list[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    rows = []
    for item in items:
        good = continuation_logprob(model, _prefix(tok), tok.encode(item["good"]), device)
        bad = continuation_logprob(model, _prefix(tok), tok.encode(item["bad"]), device)
        rows.append({"id": item["id"], "good": good, "bad": bad, "correct": good > bad})
    return {"acc": _rate(sum(r["correct"] for r in rows), len(rows)), "chance": 0.5,
            "items": rows}  # fmt: skip


def score_code_generation(
    model: GPT, tok: ByteBPETokenizer, items: list[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    rows = []
    for item in items:
        ids = _prefix(tok) + tok.encode(item["prompt"])
        out = generate(model, torch.tensor([ids], device=device), CODE_MAX_NEW, greedy=True)
        text = tok.decode(out[0, len(ids) :].tolist())
        ok = any(text.startswith(a) for a in item["accept"])
        rows.append({"id": item["id"], "completion": text, "correct": ok})
    return {"acc": _rate(sum(r["correct"] for r in rows), len(rows)), "items": rows}


# --------------------------------------------------------------------------- #
# generation and surface metrics
# --------------------------------------------------------------------------- #
def surface_metrics(text: str) -> dict[str, Any]:
    """Automatic, model-free statistics of a generated text."""
    toks = WORD.findall(text.lower())
    n = len(toks)

    def distinct(k: int) -> float | None:
        grams = [tuple(toks[i : i + k]) for i in range(n - k + 1)]
        return len(set(grams)) / len(grams) if grams else None

    four = [tuple(toks[i : i + 4]) for i in range(n - 3)]
    counts = Counter(four)
    repeated = sum(c for c in counts.values() if c > 1)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_counts = Counter(lines)
    return {
        "words": n,
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "repeated_4gram_fraction": repeated / len(four) if four else 0.0,
        "max_line_repeats": max(line_counts.values()) if line_counts else 0,
        "replacement_chars": text.count("�"),
    }


def sample_text(
    model: GPT,
    tok: ByteBPETokenizer,
    prompt: str,
    seed: int,
    device: torch.device,
    *,
    special: bool = False,
    max_new: int = GEN_MAX_NEW,
) -> dict[str, Any]:
    ids = _prefix(tok) + tok.encode(prompt, allowed_special="all" if special else "none")
    out = generate(model, torch.tensor([ids], device=device), max_new, seed=seed,
                   temperature=GEN_TEMPERATURE, top_k=GEN_TOP_K)  # fmt: skip
    new = out[0, len(ids) :].tolist()
    eot = tok.eot_id
    ended = eot is not None and eot in new
    if ended:
        new = new[: new.index(eot)]
    text = tok.decode(new)
    return {"seed": seed, "text": text, "ended_with_eot": ended, "new_tokens": len(new),
            **surface_metrics(text)}  # fmt: skip


def score_generations(
    model: GPT, tok: ByteBPETokenizer, items: list[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    rows = []
    for item in items:
        for seed in GEN_SEEDS:
            s = sample_text(model, tok, item["prompt"], seed, device,
                            special=bool(item.get("special")))  # fmt: skip
            rows.append({"id": item["id"], **s})

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": len(rows),
        "mean_words": mean("words"),
        "mean_distinct_2": mean("distinct_2"),
        "mean_repeated_4gram_fraction": mean("repeated_4gram_fraction"),
        "share_repeated_4gram_over_0.3": (
            sum(r["repeated_4gram_fraction"] > 0.3 for r in rows) / len(rows) if rows else 0.0
        ),
        "share_ended_with_eot": sum(r["ended_with_eot"] for r in rows) / len(rows) if rows else 0,
        "share_empty": sum(r["words"] == 0 for r in rows) / len(rows) if rows else 0.0,
        "replacement_chars": sum(r["replacement_chars"] for r in rows),
        "items": rows,
    }


# --------------------------------------------------------------------------- #
# instruction track
# --------------------------------------------------------------------------- #
def has_chat_tokens(tok: ByteBPETokenizer) -> bool:
    return all(t in tok.special_tokens for t in CHAT_TOKENS)


def chat_ids(tok: ByteBPETokenizer, turns: list[dict[str, str]]) -> list[int]:
    """Template: ``<|user|>\\n{text}<|end|>\\n<|assistant|>\\n{text}<|end|>\\n``, then an
    open assistant header. Message text is encoded as data (no special tokens)."""
    sp = tok.special_tokens
    ids = _prefix(tok)
    for turn in turns:
        ids += [sp[f"<|{turn['role']}|>"], *tok.encode("\n"), *tok.encode(turn["content"]),
                sp["<|end|>"], *tok.encode("\n")]  # fmt: skip
    return [*ids, sp["<|assistant|>"], *tok.encode("\n")]


def score_chat(
    model: GPT, tok: ByteBPETokenizer, items: list[dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    if not has_chat_tokens(tok):
        return {"applicable": False,
                "reason": f"tokenizer lacks chat special tokens {list(CHAT_TOKENS)}"}  # fmt: skip
    sp = tok.special_tokens
    role_ids = {sp["<|user|>"], sp["<|assistant|>"]}
    rows = []
    for item in items:
        ids = chat_ids(tok, item["turns"])
        out = generate(model, torch.tensor([ids], device=device), CHAT_MAX_NEW, greedy=True)
        new = out[0, len(ids) :].tolist()
        end = new.index(sp["<|end|>"]) if sp["<|end|>"] in new else None
        reply_ids = new[:end] if end is not None else new
        role_break = any(i in role_ids for i in reply_ids)
        reply = tok.decode([i for i in reply_ids if i not in tok.special_ids])
        checks = item.get("checks", {})
        content_ok = True
        if "include_any" in checks:
            content_ok &= any(w.lower() in reply.lower() for w in checks["include_any"])
        if "max_sentences" in checks:
            content_ok &= len(SENT_END.findall(reply)) <= checks["max_sentences"]
        fmt_ok = end is not None and not role_break
        rows.append({"id": item["id"], "reply": reply, "format_ok": fmt_ok,
                     "content_ok": content_ok, "both_ok": fmt_ok and content_ok})  # fmt: skip
    n = len(rows)
    return {
        "applicable": True,
        "format": _rate(sum(r["format_ok"] for r in rows), n),
        "content": _rate(sum(r["content_ok"] for r in rows), n),
        "both": _rate(sum(r["both_ok"] for r in rows), n),
        "items": rows,
    }


# --------------------------------------------------------------------------- #
def run_suite(
    model: GPT, tok: ByteBPETokenizer, suite: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Run every itemized section. ``lm_heldout`` needs data and is run separately."""
    s = suite["sections"]
    model.eval()
    return {
        "next_token": {
            "facts_mc": score_multiple_choice(model, tok, s["facts_mc"]["items"], device),
            "qa_mc": score_multiple_choice(model, tok, s["qa_mc"]["items"], device),
            "code_mc": score_multiple_choice(model, tok, s["code_mc"]["items"], device),
            "grammar_pairs": score_minimal_pairs(model, tok, s["grammar_pairs"]["items"], device),
            "code_generation": score_code_generation(
                model, tok, s["code_generation"]["items"], device
            ),
            "continuation": score_generations(model, tok, s["continuation"]["items"], device),
            "explanations": score_generations(model, tok, s["explanations"]["items"], device),
            "repetition_stress": score_generations(
                model, tok, s["repetition_stress"]["items"], device
            ),
        },
        "instruction": {"chat": score_chat(model, tok, s["chat"]["items"], device)},
    }


def ngram_overlap(
    eval_texts: dict[str, str], documents: list[str], n: int = 8
) -> dict[str, list[int]]:
    """For each eval item, the indices of documents sharing a word n-gram with it
    (lower-cased word tokens; items shorter than n words are matched whole)."""
    index: dict[tuple[str, ...], set[str]] = {}
    for key, text in eval_texts.items():
        toks = ALNUM.findall(text.lower())
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)] or [tuple(toks)]
        for g in grams:
            if g:
                index.setdefault(g, set()).add(key)
    lengths = {len(g) for g in index}
    hits: dict[str, list[int]] = {k: [] for k in eval_texts}
    for d, doc in enumerate(documents):
        toks = ALNUM.findall(doc.lower())
        found: set[str] = set()
        for k in lengths:
            for i in range(len(toks) - k + 1):
                found |= index.get(tuple(toks[i : i + k]), set())
        for key in found:
            hits[key].append(d)
    return hits

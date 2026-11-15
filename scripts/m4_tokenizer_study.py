"""M4 tokenizer study: byte-level BPE at 4,096 vs 8,192 on general-domain samples.

Protocol
--------
* Every sample is de-duplicated (exact) and split by ``sha256(text)[:8] % 100 < 10``
  into validation (10%) and training (90%) **before** any fitting. The same split is
  used by ``evaluate_general.py`` for held-out bits per byte.
* Candidate tokenizers are trained only on the training documents of the proposed
  pretraining mixture (FineWeb-Edu + Cosmopedia v2 samples), with five special
  tokens reserved for later chat formatting.
* The historical TinyStories tokenizer (M3.1-M3.3) is evaluated as a baseline; it
  is read, never modified.
* Evaluation domains (validation documents only): FineWeb-Edu, Cosmopedia v2,
  Wikipedia (en, simple), TinyStories, chat text and fenced code from SmolTalk.
* A fitting-size sensitivity run trains the 8,192 tokenizer on 25%, 50% and 100%
  of the training documents.

Tokenizer files go to ``--save-dir`` (git-ignored ``runs/``); the report directory
gets ``tokenizer_study.json``.

Usage: python scripts/m4_tokenizer_study.py --out docs/reports/m4 --save-dir runs/m4_tokenizers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mini_chatgpt.config import ModelConfig
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json

SAMPLES = Path("data/raw/samples")
SPECIALS = ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>")
TRAIN_MIX = ("fineweb_edu_10bt", "cosmopedia_v2")
TEXT_DOMAINS = {
    "fineweb_edu": "fineweb_edu_10bt",
    "cosmopedia_v2": "cosmopedia_v2",
    "wikipedia_en": "wikipedia_en",
    "wikipedia_simple": "wikipedia_simple",
    "tinystories": "tinystories",
}
HISTORICAL = Path("data/processed/tinystories_v2_bpe/tokenizer.json")
ARCH = {"S": (6, 6, 384), "M": (8, 8, 512), "X": (10, 10, 640), "L": (12, 12, 768)}
FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)


def is_val(text: str) -> bool:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 100 < 10


def split_docs(docs: list[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    train: list[str] = []
    val: list[str] = []
    for d in docs:
        if not d or d in seen:
            continue
        seen.add(d)
        (val if is_val(d) else train).append(d)
    return train, val


def read_rows(name: str) -> list[dict[str, Any]]:
    with (SAMPLES / name / "sample.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chat_domains() -> tuple[list[str], list[str]]:
    """Validation chat transcripts (role-tagged plain text) and fenced code blocks."""
    rows = read_rows("smol_smoltalk")
    chats = ["\n".join(f"{m['role']}: {m['content']}" for m in r["messages"]) for r in rows]
    _, val = split_docs(chats)
    code = [c for t in val for c in FENCE.findall(t) if c.strip()]
    return val, code


def zipf_stats(usage: Counter[int]) -> dict[str, Any]:
    counts = sorted(usage.values(), reverse=True)
    total = sum(counts)
    out: dict[str, Any] = {}
    for q in (0.5, 0.9, 0.99):
        acc, k = 0, 0
        for c in counts:
            acc += c
            k += 1
            if acc >= q * total:
                break
        out[f"types_covering_{int(q * 100)}pct"] = k
    # least-squares slope of log(freq) vs log(rank) over ranks 10..1000
    pts = [(math.log(r + 1), math.log(c)) for r, c in enumerate(counts) if 10 <= r < 1000]
    if len(pts) > 2:
        mx = statistics.fmean(x for x, _ in pts)
        my = statistics.fmean(y for _, y in pts)
        num = sum((x - mx) * (y - my) for x, y in pts)
        den = sum((x - mx) ** 2 for x, _ in pts)
        out["zipf_slope_rank10_1000"] = num / den
    return out


def evaluate(tok: ByteBPETokenizer, docs: list[str]) -> tuple[dict[str, Any], Counter[int]]:
    fresh = ByteBPETokenizer.from_dict(tok.to_dict())  # cold cache for honest throughput
    t0 = time.perf_counter()
    enc = [fresh.encode(d) for d in docs]
    dt = time.perf_counter() - t0
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    lens = [len(e) for e in enc]
    usage: Counter[int] = Counter()
    for e in enc:
        usage.update(e)
    n_tok = sum(lens)
    res = {
        "documents": len(docs),
        "bytes": n_bytes,
        "tokens": n_tok,
        "bytes_per_token": n_bytes / n_tok,
        "tokens_per_doc_median": statistics.median(lens),
        "tokens_per_doc_p95": sorted(lens)[int(0.95 * (len(lens) - 1))],
        "share_docs_over_512_tokens": sum(n > 512 for n in lens) / len(lens),
        "share_docs_over_1024_tokens": sum(n > 1024 for n in lens) / len(lens),
        "encode_mb_per_s_cold": n_bytes / dt / 1e6,
        "round_trip_exact": all(fresh.decode(e) == d for e, d in zip(enc, docs, strict=True)),
        "non_ascii_byte_token_share": sum(c for i, c in usage.items() if 128 <= i < 256) / n_tok,
    }
    return res, usage


def vocab_overhead(v: int) -> dict[str, Any]:
    out = {}
    for name, (layer, head, embd) in ARCH.items():
        p = ModelConfig(vocab_size=v, context_length=512, n_layer=layer, n_head=head,
                        n_embd=embd).estimate_parameters()  # fmt: skip
        out[name] = {"total": int(p["total"]), "token_embedding": int(p["token_embedding"]),
                     "embedding_share": p["token_embedding"] / p["total"]}  # fmt: skip
    return out


def study(tok: ByteBPETokenizer, train_docs: list[str], domains: dict[str, list[str]],
          train_seconds: float | None) -> dict[str, Any]:  # fmt: skip
    _, train_usage = evaluate(tok, train_docs)
    n_ordinary = 256 + len(tok.merges)
    merged = range(256, n_ordinary)
    per_domain: dict[str, Any] = {}
    val_usage: Counter[int] = Counter()
    for name, docs in domains.items():
        res, usage = evaluate(tok, docs)
        per_domain[name] = res
        if name in ("fineweb_edu", "cosmopedia_v2"):
            val_usage.update(usage)
    return {
        "vocab_size": tok.vocab_size,
        "fingerprint": tok.fingerprint(),
        "special_tokens": tok.special_tokens,
        "train_seconds": train_seconds,
        "domains": per_domain,
        "frequency_on_mix_val": zipf_stats(val_usage),
        "rare_tokens": {
            "merged_tokens": len(merged),
            "unused_in_fit_corpus": sum(train_usage[i] == 0 for i in merged),
            "used_lt_10_in_fit_corpus": sum(train_usage[i] < 10 for i in merged),
            "used_lt_100_in_fit_corpus": sum(train_usage[i] < 100 for i in merged),
            "unused_in_mix_val": sum(val_usage[i] == 0 for i in merged),
            "median_fit_count_of_last_500_merges": statistics.median(
                train_usage[i] for i in range(max(256, n_ordinary - 500), n_ordinary)
            ),
        },
        "vocab_overhead": vocab_overhead(tok.vocab_size),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--vocab", type=int, nargs="+", default=[4096, 8192])
    args = p.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    splits = {k: split_docs([r["text"] for r in read_rows(v)]) for k, v in TEXT_DOMAINS.items()}
    chat_val, code_val = chat_domains()
    domains = {k: v[1] for k, v in splits.items()} | {"chat": chat_val, "code": code_val}
    train_docs = [d for k, v in TEXT_DOMAINS.items() if v in TRAIN_MIX for d in splits[k][0]]
    fit_digest = hashlib.sha256(b"".join(hashlib.sha256(d.encode()).digest()
                                         for d in train_docs)).hexdigest()  # fmt: skip
    result: dict[str, Any] = {
        "protocol": {
            "fit_corpus": list(TRAIN_MIX),
            "fit_documents": len(train_docs),
            "fit_bytes": sum(len(d.encode("utf-8")) for d in train_docs),
            "fit_documents_sha256": fit_digest,
            "special_tokens": list(SPECIALS),
            "val_documents": {k: len(v) for k, v in domains.items()},
        },
        "candidates": {},
        "historical_tinystories_4096": None,
        "fit_size_sensitivity_8192": {},
    }
    print(json.dumps(result["protocol"]), flush=True)

    for v in args.vocab:
        t0 = time.perf_counter()
        tok = ByteBPETokenizer.train(train_docs, v, special_tokens=SPECIALS)
        dt = time.perf_counter() - t0
        tok.metadata = {"fit_documents_sha256": fit_digest, "study": "m4"}
        tok.save(args.save_dir / f"m4_mix_bpe{v}.json")
        result["candidates"][str(v)] = study(tok, train_docs, domains, dt)
        d = result["candidates"][str(v)]["domains"]
        print(v, f"{dt:.1f}s", {k: round(x["bytes_per_token"], 3) for k, x in d.items()},
              flush=True)  # fmt: skip

    hist = ByteBPETokenizer.load(HISTORICAL)
    result["historical_tinystories_4096"] = study(hist, train_docs, domains, None)
    d = result["historical_tinystories_4096"]["domains"]
    print("historical", {k: round(x["bytes_per_token"], 3) for k, x in d.items()}, flush=True)

    ordered = sorted(train_docs, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    for frac in (0.25, 0.5, 1.0):
        subset = ordered[: int(len(ordered) * frac)]
        tok = ByteBPETokenizer.train(subset, 8192, special_tokens=SPECIALS)
        mix_val = domains["fineweb_edu"] + domains["cosmopedia_v2"]
        res, _ = evaluate(tok, mix_val)
        result["fit_size_sensitivity_8192"][str(frac)] = {
            "fit_bytes": sum(len(s.encode("utf-8")) for s in subset),
            "mix_val_bytes_per_token": res["bytes_per_token"],
        }
        print("fit", frac, result["fit_size_sensitivity_8192"][str(frac)], flush=True)

    atomic_write_json(args.out / "tokenizer_study.json", result)


if __name__ == "__main__":
    main()

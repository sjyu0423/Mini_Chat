"""Content statistics and duplication checks for bounded dataset samples.

For each sample: document-length distribution, character composition, word
vocabulary, metadata summaries (FineWeb-Edu scores, crawl dumps, domains), exact
duplicates, and near-duplicates found with MinHash over word 5-gram shingles
(128 permutations, LSH with 32 bands x 4 rows; candidate pairs are confirmed with
an estimated Jaccard >= 0.8). Near-duplicates that straddle the hash-based
train/validation split (the one ``benchmark_tokenizers.py`` uses) are counted
separately as potential leakage.

Usage: python scripts/sample_stats.py --corpus tinystories=data/raw/samples/tinystories \
    --corpus fineweb_edu=data/raw/samples/fineweb_edu_10bt --out docs/reports/m3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from mini_chatgpt.training.checkpoint import atomic_write_json

WORD = re.compile(r"[A-Za-z']+")
P = (1 << 31) - 1  # Mersenne prime; a*x + b stays below 2**63 in uint64
N_PERM, BANDS = 128, 32
ROWS = N_PERM // BANDS


def shingles(text: str, n: int = 5) -> np.ndarray:
    ws = text.lower().split()
    grams = {" ".join(ws[i : i + n]) for i in range(max(1, len(ws) - n + 1))}
    return np.array(
        [int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") % P
         for g in grams], dtype=np.uint64,
    )  # fmt: skip


def minhash(sh: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash signature: min over shingles of (a*x + b) mod P, for each permutation."""
    sig: np.ndarray = ((a[:, None] * sh[None, :] + b[:, None]) % P).min(axis=1)
    return sig


def is_val(text: str, val_percent: int = 10) -> bool:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 100 < val_percent


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [r["text"] for r in rows]
    chars = [len(t) for t in texts]
    words_per = [len(t.split()) for t in texts]
    allchars = Counter("".join(texts))
    total = sum(allchars.values())
    non_ascii = sum(c for ch, c in allchars.items() if ord(ch) > 127)
    vocab = Counter(w.lower() for t in texts for w in WORD.findall(t))
    out: dict[str, Any] = {
        "documents": len(texts),
        "chars_mean": statistics.fmean(chars),
        "chars_median": statistics.median(chars),
        "chars_p95": sorted(chars)[int(0.95 * (len(chars) - 1))],
        "chars_max": max(chars),
        "words_mean": statistics.fmean(words_per),
        "utf8_bytes_total": sum(len(t.encode("utf-8")) for t in texts),
        "non_ascii_char_fraction": non_ascii / total,
        "distinct_chars": len(allchars),
        "distinct_words": len(vocab),
        "words_seen_once_fraction": sum(1 for c in vocab.values() if c == 1) / len(vocab),
        "top_words": [w for w, _ in vocab.most_common(15)],
    }
    if "int_score" in rows[0]:
        out["int_score_counts"] = dict(sorted(Counter(r["int_score"] for r in rows).items()))
        out["score_mean"] = statistics.fmean(r["score"] for r in rows)
        out["language_score_min"] = min(r["language_score"] for r in rows)
        out["token_count_gpt2_mean"] = statistics.fmean(r["token_count"] for r in rows)
        out["dumps_distinct"] = len({r["dump"] for r in rows})
        out["dump_years"] = dict(sorted(Counter(r["dump"][8:12] for r in rows).items()))
        domains = Counter(urlparse(r["url"]).netloc.removeprefix("www.") for r in rows)
        out["domains_distinct"] = len(domains)
        out["top_domains"] = domains.most_common(10)
    # exact and near duplicates
    hashes = Counter(hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts)
    out["exact_duplicate_documents"] = sum(c - 1 for c in hashes.values() if c > 1)
    rng = np.random.default_rng(0)
    a = rng.integers(1, P - 1, N_PERM).astype(np.uint64)
    b = rng.integers(0, P - 1, N_PERM).astype(np.uint64)
    sigs = [minhash(shingles(t), a, b) for t in texts]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for i, s in enumerate(sigs):
        for band in range(BANDS):
            key = (band, tuple(int(v) for v in s[band * ROWS : (band + 1) * ROWS]))
            buckets.setdefault(key, []).append(i)
    pairs: set[tuple[int, int]] = set()
    for idx in buckets.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                pairs.add((idx[x], idx[y]))
    near = [(i, j) for i, j in sorted(pairs)
            if np.mean(sigs[i] == sigs[j]) >= 0.8 and texts[i] != texts[j]]  # fmt: skip
    split = [is_val(t) for t in texts]
    out["near_duplicate_pairs_jaccard_ge_0.8"] = len(near)
    out["near_duplicate_docs"] = len({k for p in near for k in p})
    out["near_duplicate_pairs_crossing_train_val"] = sum(split[i] != split[j] for i, j in near)
    out["near_duplicate_examples"] = [[texts[i][:120], texts[j][:120]] for i, j in near[:3]]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = {}
    for name, d in (c.split("=", 1) for c in args.corpus):
        with (Path(d) / "sample.jsonl").open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        result[name] = {"manifest": json.loads((Path(d) / "manifest.json").read_text()),
                        "stats": analyze(rows)}  # fmt: skip
        result[name]["manifest"].pop("requests", None)
        print(name, json.dumps({k: v for k, v in result[name]["stats"].items()
                                if k != "near_duplicate_examples"}, default=str))  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out / "sample_stats.json", result)


if __name__ == "__main__":
    main()

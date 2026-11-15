"""Train byte-level BPE tokenizers at several vocabulary sizes and compare them.

Procedure (all deterministic):
1. Load each bounded sample (``sample.jsonl`` from ``fetch_samples.py``), drop
   exact-duplicate documents (SHA-256 of the UTF-8 text), and split documents
   into train/validation by hash: a document is validation iff
   ``int(sha256[:8], 16) % 100 < val_percent``. The split is fixed before any
   tokenizer is fitted, and tokenizers only ever see training documents.
2. For every training set (``mixed`` = all corpora's train documents, plus each
   corpus alone) and every vocabulary size, train a tokenizer and record the time.
3. Evaluate on every corpus's validation split: compression (UTF-8 bytes per
   token), characters per token, tokens per document, encode throughput (cold
   cache), exact round-trip, and rare-token statistics.

Usage:
    python scripts/benchmark_tokenizers.py --corpus tinystories=data/raw/samples/tinystories \
        --corpus fineweb_edu=data/raw/samples/fineweb_edu_10bt --vocab 2048 4096 8192 \
        --out docs/reports/m3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from mini_chatgpt.analysis.svgplot import Chart, Series, render
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json
from mini_chatgpt.training.run import make_run_dir, setup_logging

log = setup_logging()


def load_split(path: Path, val_percent: int) -> dict[str, Any]:
    docs: list[str] = []
    seen: set[str] = set()
    dupes = 0
    with (path / "sample.jsonl").open(encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen:
                dupes += 1
                continue
            seen.add(h)
            docs.append(text)
    train: list[str] = []
    val: list[str] = []
    for d in docs:
        h = hashlib.sha256(d.encode("utf-8")).hexdigest()
        (val if int(h[:8], 16) % 100 < val_percent else train).append(d)

    def digest(xs: list[str]) -> str:
        s = hashlib.sha256()
        for x in xs:
            s.update(hashlib.sha256(x.encode("utf-8")).digest())
        return s.hexdigest()

    manifest = json.loads((path / "manifest.json").read_text())
    train_bytes: int = sum(len(x.encode("utf-8")) for x in train)
    val_bytes: int = sum(len(x.encode("utf-8")) for x in val)
    info: dict[str, Any] = {
        "source": f"{manifest['dataset']}:{manifest['config']}:{manifest['split']}",
        "sample_sha256": manifest["file_sha256"],
        "documents": len(docs),
        "exact_duplicates_dropped": dupes,
        "train_documents": len(train),
        "val_documents": len(val),
        "train_bytes": train_bytes,
        "val_bytes": val_bytes,
        "train_sha256": digest(train),
        "val_sha256": digest(val),
    }
    return {"train": train, "val": val, "info": info}


def evaluate(tok: ByteBPETokenizer, docs: list[str], train_usage: Counter[int]) -> dict[str, Any]:
    fresh = ByteBPETokenizer.from_dict(tok.to_dict())  # cold cache for honest throughput
    t0 = time.perf_counter()
    encoded = [fresh.encode(d) for d in docs]
    dt = time.perf_counter() - t0
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    n_chars = sum(len(d) for d in docs)
    lens = [len(e) for e in encoded]
    n_tok = sum(lens)
    usage: Counter[int] = Counter()
    for e in encoded:
        usage.update(e)
    n_ordinary = 256 + len(tok.merges)
    single_byte = sum(c for i, c in usage.items() if i < 256)
    non_ascii_single = sum(c for i, c in usage.items() if 128 <= i < 256)
    round_trip = all(fresh.decode(e) == d for e, d in zip(encoded, docs, strict=True))
    merged_ids = range(256, n_ordinary)
    return {
        "documents": len(docs),
        "bytes": n_bytes,
        "tokens": n_tok,
        "bytes_per_token": n_bytes / n_tok,
        "chars_per_token": n_chars / n_tok,
        "tokens_per_doc_mean": statistics.fmean(lens),
        "tokens_per_doc_median": statistics.median(lens),
        "tokens_per_doc_p95": sorted(lens)[int(0.95 * (len(lens) - 1))],
        "encode_mb_per_s_cold": n_bytes / dt / 1e6,
        "encode_tokens_per_s_cold": n_tok / dt,
        "round_trip_exact": round_trip,
        "single_byte_token_share": single_byte / n_tok,
        "non_ascii_byte_token_share": non_ascii_single / n_tok,
        "vocab_used_fraction": sum(1 for i in range(n_ordinary) if usage[i] > 0) / n_ordinary,
        "merged_tokens_unused_in_eval": sum(1 for i in merged_ids if usage[i] == 0),
        "merged_tokens_unused_in_train": sum(1 for i in merged_ids if train_usage[i] == 0),
        "merged_tokens_rare_in_train_lt10": sum(1 for i in merged_ids if train_usage[i] < 10),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--vocab", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--val-percent", type=int, default=10)
    p.add_argument("--out", type=Path, required=True, help="report directory (json + svg)")
    p.add_argument("--no-single", action="store_true", help="only train on the mixed set")
    args = p.parse_args()
    corpora = {n: load_split(Path(d), args.val_percent)
               for n, d in (c.split("=", 1) for c in args.corpus)}  # fmt: skip
    run_dir = make_run_dir("runs", "m3_tokenizers")
    train_sets = {"mixed": [d for c in corpora.values() for d in c["train"]]}
    if not args.no_single:
        train_sets.update({n: c["train"] for n, c in corpora.items()})

    results: dict[str, Any] = {
        "corpora": {n: c["info"] for n, c in corpora.items()},
        "val_percent": args.val_percent,
        "run_dir": str(run_dir),
        "tokenizers": [],
    }
    for set_name, docs in train_sets.items():
        for v in args.vocab:
            t0 = time.perf_counter()
            tok = ByteBPETokenizer.train(docs, vocab_size=v)
            train_s = time.perf_counter() - t0
            path = run_dir / f"bpe_{set_name}_{v}.json"
            tok.save(path)
            usage: Counter[int] = Counter()
            for d in docs:
                usage.update(tok.encode(d))
            longest = sorted(range(256, 256 + len(tok.merges)),
                             key=lambda i: (-len(tok.token_bytes(i)), i))[:8]  # fmt: skip
            entry = {
                "train_set": set_name,
                "vocab_size": v,
                "train_seconds": train_s,
                "train_documents": len(docs),
                "train_bytes": tok.metadata["trained_on_bytes"],
                "unique_pretokens": tok.metadata["unique_pretokens"],
                "fingerprint": tok.fingerprint(),
                "file": str(path),
                "first_merges": [
                    tok.token_bytes(256 + i).decode("utf-8", "replace") for i in range(10)
                ],
                "longest_tokens": [tok.token_bytes(i).decode("utf-8", "replace") for i in longest],
                "eval": {n: evaluate(tok, c["val"], usage) for n, c in corpora.items()},
            }
            results["tokenizers"].append(entry)
            log.info(
                "%s v=%d: %.1fs | %s", set_name, v, train_s,
                " | ".join(f"{n} {e['bytes_per_token']:.2f} B/tok"
                           for n, e in entry["eval"].items()),
            )  # fmt: skip

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "tokenizer_benchmark.json", results)
    atomic_write_json(run_dir / "tokenizer_benchmark.json", results)
    series = []
    for slot, name in enumerate(corpora):
        rows = [t for t in results["tokenizers"] if t["train_set"] == "mixed"]
        series.append(Series(f"{name} validation", [t["vocab_size"] for t in rows],
                             [t["eval"][name]["bytes_per_token"] for t in rows], slot=slot,
                             markers=True))  # fmt: skip
    chart = Chart("BPE compression vs. vocabulary size (tokenizer trained on mixed sample)",
                  "vocabulary size", "UTF-8 bytes per token (higher = more compression)",
                  series, y_min=0.0)  # fmt: skip
    (out / "tokenizer_bytes_per_token.svg").write_text(render(chart), encoding="utf-8")
    log.info("wrote %s and %s", out, run_dir)


if __name__ == "__main__":
    main()

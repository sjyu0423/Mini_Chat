"""Prepare the M4.1 general pretraining data: FineWeb-Edu + Cosmopedia v2, BPE 8192.

Pipeline (deterministic; every count and hash goes to ``manifest.json``):

1. Verify both pinned Parquet files (size + Hub LFS SHA-256). Download is a separate,
   explicit step (``--download``); nothing is fetched otherwise.
2. Read only the ``text`` column. Cosmopedia v2 is reduced to a 25% hash-sampled pool.
3. Clean (``general.clean_text``) and drop obvious garbage (``general.quality_reject``).
4. Exact + MinHash near-duplicate removal over both sources together, **before** splitting.
5. Document-hash split: 0.5% validation, 0.5% holdout, 99% train.
6. Decontamination: drop training documents sharing a word 8-gram with any item of
   ``evals/general_v1.json`` that has at least 8 words; held-out hits are counted.
7. Fit byte-level BPE (8,192, five special tokens) on the first 70 MB of FineWeb-Edu
   and 30 MB of Cosmopedia v2 **training** documents in SHA-256 order.
8. Encode training documents per source in SHA-256 order until its token quota
   (70% / 30% of ``--train-tokens``) is met. A source that runs out is an error:
   the run is single-pass, so nothing is ever repeated to fill a quota.
9. Write uint16 ``train.bin`` / ``val.bin`` / ``holdout.bin`` (each document followed by
   ``<|endoftext|>``; documents ordered by SHA-256, which mixes the sources), the
   tokenizer, ``meta.json`` (read by ``load_prepared``) and held-out text as JSON Lines
   for per-source bits-per-byte evaluation.

Usage:
    python scripts/prepare_general.py --download   # once: fetch the two pinned files
    python scripts/prepare_general.py              # prepare
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mini_chatgpt.analysis.general_eval import all_eval_texts, load_suite
from mini_chatgpt.data import general as g
from mini_chatgpt.data.tinystories import MinHashConfig, list_digest, text_sha256
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)
from mini_chatgpt.training.run import setup_logging

log = setup_logging()
RAW = Path("data/raw/general_m4_1")
OUT = Path("data/processed/general_m4_1_bpe")
SUITE = Path("evals/general_v1.json")
TS_VALID = Path("data/raw/tinystories_v2/TinyStoriesV2-GPT4-valid.txt")


def download(src: g.Source) -> None:
    path = RAW / src.local_name
    if path.exists():
        log.info("%s exists; verifying", path)
    else:
        RAW.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        req = urllib.request.Request(src.url, headers={"User-Agent": "mini-chatgpt/0.1"})
        with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as f:
            while block := resp.read(1 << 22):
                f.write(block)
        tmp.rename(path)
    g.verify_file(path, src)
    log.info("%s verified (%d bytes, sha256 %s)", path.name, src.size, src.sha256[:12])


def load_source(src: g.Source) -> tuple[list[g.Doc], dict[str, Any]]:
    path = RAW / src.local_name
    t0 = time.perf_counter()
    digest = g.verify_file(path, src)
    rows = pooled = 0
    rejects: Counter[str] = Counter()
    raw_bytes = clean_bytes = 0
    docs: list[g.Doc] = []
    for text in g.read_texts(path, src.text_column):
        rows += 1
        sha = text_sha256(text)
        if int(sha[:8], 16) % src.pool_modulus != 0:
            continue
        pooled += 1
        raw_bytes += len(text.encode("utf-8"))
        cleaned = g.clean_text(text)
        reason = g.quality_reject(cleaned)
        if reason:
            rejects[reason] += 1
            continue
        clean_bytes += len(cleaned.encode("utf-8"))
        docs.append(g.Doc(src.name, cleaned))
    info = {
        "repo": src.repo, "revision": src.revision, "path": src.path, "url": src.url,
        "bytes": src.size, "sha256": digest, "sha256_matches_pinned": True,
        "license": src.license, "text_column": src.text_column,
        "rows_in_file": rows, "pool_modulus": src.pool_modulus, "rows_in_pool": pooled,
        "pool_text_bytes": raw_bytes, "rejected": dict(rejects),
        "kept_after_cleaning": len(docs), "clean_text_bytes": clean_bytes,
        "changed_by_cleaning": None, "seconds": time.perf_counter() - t0,
    }  # fmt: skip
    log.info("%s: %d rows, pool %d, kept %d, rejected %s (%.0f s)", src.name, rows, pooled,
             len(docs), dict(rejects), info["seconds"])  # fmt: skip
    return docs, info


def encode_docs(tok: ByteBPETokenizer, docs: list[g.Doc]) -> list[np.ndarray]:
    eot = tok.eot_id
    assert eot is not None
    return [np.asarray([*tok.encode_ordinary(d.text), eot], dtype="<u2") for d in docs]


def stream(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.empty(0, dtype="<u2")


def write_jsonl(path: Path, docs: list[g.Doc]) -> None:
    body = "".join(json.dumps({"text": d.text, "sha256": d.sha}, ensure_ascii=False) + "\n"
                   for d in docs)  # fmt: skip
    atomic_write_text(path / "sample.jsonl", body)


def tokenizer_checks(tok: ByteBPETokenizer, val_docs: list[g.Doc]) -> dict[str, Any]:
    reloaded = ByteBPETokenizer.load(OUT / "tokenizer.json")
    sp = tok.special_tokens
    probe = "Hello <|user|> world <|endoftext|>"
    return {
        "round_trip_exact_val_holdout_docs": all(
            tok.decode(tok.encode_ordinary(d.text)) == d.text for d in val_docs
        ),
        "round_trip_docs_checked": len(val_docs),
        "reload_fingerprint_matches": reloaded.fingerprint() == tok.fingerprint(),
        "reload_merges_identical": reloaded.merges == tok.merges,
        "special_ids": sp,
        "special_text_is_data_by_default": not any(i in tok.special_ids for i in tok.encode(probe)),
        "special_allowed_gives_single_ids": tok.encode(probe, allowed_special="all").count(
            sp["<|user|>"]
        )
        == 1,
        "special_decode": tok.decode([sp["<|assistant|>"]]) == "<|assistant|>",
    }


def frequency_stats(train: np.ndarray, tok: ByteBPETokenizer) -> dict[str, Any]:
    counts = np.bincount(train, minlength=tok.vocab_size)
    order = np.sort(counts)[::-1]
    cum = np.cumsum(order) / order.sum()
    merged = counts[256 : 256 + len(tok.merges)]
    top = np.argsort(counts)[::-1][:20]
    return {
        "types_covering_50pct": int(np.searchsorted(cum, 0.5) + 1),
        "types_covering_90pct": int(np.searchsorted(cum, 0.9) + 1),
        "types_covering_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "ordinary_tokens_never_used": int((counts[: 256 + len(tok.merges)] == 0).sum()),
        "byte_tokens_never_used": int((counts[:256] == 0).sum()),
        "merged_tokens_never_used": int((merged == 0).sum()),
        "merged_tokens_used_lt_100": int((merged < 100).sum()),
        "merged_tokens_used_lt_1000": int((merged < 1000).sum()),
        "special_token_counts": {s: int(counts[i]) for s, i in tok.special_tokens.items()},
        "top20": [[tok.decode([int(i)]), int(counts[i])] for i in top],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true", help="fetch + verify the pinned files")
    p.add_argument("--train-tokens", type=int, default=157_286_400)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--fit-mb", type=float, default=100.0)
    args = p.parse_args()
    if args.download:
        for src in g.SOURCES:
            download(src)
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} is not empty; refusing to overwrite prepared data")
    t_start = time.perf_counter()

    # 1-3 read, clean, filter
    docs: list[g.Doc] = []
    sources: dict[str, Any] = {}
    for src in g.SOURCES:
        src_docs, info = load_source(src)
        docs.extend(src_docs)
        sources[src.name] = info

    # 4 dedup across sources, before splitting
    t0 = time.perf_counter()
    mh = MinHashConfig()
    docs, dstats = g.dedup_documents(docs, mh)
    dedup_s = time.perf_counter() - t0
    log.info("dedup: %s (%.0f s)", {k: v for k, v in g.stats_dict(dstats).items()
                                    if k != "examples"}, dedup_s)  # fmt: skip

    # 5 split
    splits: dict[str, dict[str, list[g.Doc]]] = {
        s: {src.name: [] for src in g.SOURCES} for s in ("train", "val", "holdout")
    }
    for d in docs:  # already in SHA-256 order
        splits[g.split_of(d.sha)][d.source].append(d)

    # 6 decontamination
    suite, suite_sha = load_suite(SUITE)
    index = g.eval_ngrams(all_eval_texts(suite))
    decon: dict[str, Any] = {"suite_sha256": suite_sha, "ngram": 8,
                             "items_indexed": len({k for v in index.values() for k in v}),
                             "removed_train": {}, "item_hits_train": Counter(),
                             "heldout_docs_with_hits": {}}  # fmt: skip
    for name in splits["train"]:
        kept = []
        removed = 0
        for d in splits["train"][name]:
            hits = g.contaminating_items(d.text, index)
            if hits:
                removed += 1
                decon["item_hits_train"].update(hits)
            else:
                kept.append(d)
        splits["train"][name] = kept
        decon["removed_train"][name] = removed
    for s in ("val", "holdout"):
        for name, ds in splits[s].items():
            decon["heldout_docs_with_hits"][f"{s}/{name}"] = sum(
                bool(g.contaminating_items(d.text, index)) for d in ds
            )
    decon["item_hits_train"] = dict(decon["item_hits_train"])
    log.info("decontamination: %s", decon)

    # 7 tokenizer fit on training documents only
    fit_docs: list[str] = []
    fit_info: dict[str, Any] = {}
    for src in g.SOURCES:
        budget = args.fit_mb * 1e6 * src.token_share
        used = n = 0
        for d in splits["train"][src.name]:
            if used >= budget:
                break
            fit_docs.append(d.text)
            used += len(d.text.encode("utf-8"))
            n += 1
        fit_info[src.name] = {"documents": n, "bytes": used}
    t0 = time.perf_counter()
    tok = ByteBPETokenizer.train(fit_docs, args.vocab, special_tokens=g.SPECIAL_TOKENS)
    fit_s = time.perf_counter() - t0
    tok.metadata = {"dataset": "general_m4_1", "fit_documents_sha256": list_digest(fit_docs),
                    "fit": fit_info}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    tok.save(OUT / "tokenizer.json")
    log.info("tokenizer: V=%d fingerprint %s (%.1f s on %d docs)", tok.vocab_size,
             tok.fingerprint(), fit_s, len(fit_docs))  # fmt: skip

    # 8 encode training docs up to each source's quota (no repetition)
    t0 = time.perf_counter()
    selected: list[tuple[g.Doc, np.ndarray]] = []
    quota_info: dict[str, Any] = {}
    for src in g.SOURCES:
        quota = round(args.train_tokens * src.token_share)
        got = 0
        n = 0
        pool = splits["train"][src.name]
        for d in pool:
            if got >= quota:
                break
            arr = encode_docs(tok, [d])[0]
            selected.append((d, arr))
            got += arr.size
            n += 1
        if got < quota:
            raise SystemExit(
                f"{src.name}: only {got} training tokens available for a quota of {quota}; "
                "refusing to repeat data (single-pass run)"
            )
        sel_bytes = sum(len(d.text.encode("utf-8")) for d, _ in selected[-n:])
        unused_bytes = sum(len(d.text.encode("utf-8")) for d in pool[n:])
        quota_info[src.name] = {
            "quota_tokens": quota, "selected_tokens": got, "selected_documents": n,
            "train_documents_available": len(pool),
            "unused_train_documents": len(pool) - n,
            "selected_bytes_per_token": sel_bytes / got,
            "unused_train_tokens_estimate": round(unused_bytes / (sel_bytes / got)),
        }  # fmt: skip
        log.info("%s: %s", src.name, quota_info[src.name])
    selected.sort(key=lambda t: t[0].sha)  # mixes sources deterministically
    train = stream([a for _, a in selected])
    encode_s = time.perf_counter() - t0
    windows = (train.size - 1) // args.context_length

    # 9 held-out streams and files
    streams: dict[str, Any] = {}
    held: dict[str, list[g.Doc]] = {}
    for s in ("val", "holdout"):
        hdocs = sorted((d for ds in splits[s].values() for d in ds), key=lambda d: d.sha)
        held[s] = hdocs
        arr = stream(encode_docs(tok, hdocs))
        n_bytes = sum(len(d.text.encode("utf-8")) for d in hdocs)
        atomic_write_bytes(OUT / f"{s}.bin", arr.tobytes())
        streams[s] = {"documents": len(hdocs), "tokens": int(arr.size), "text_bytes": n_bytes,
                      "bytes_per_token_incl_eot": n_bytes / arr.size,
                      "per_source_documents": {k: len(v) for k, v in splits[s].items()},
                      "sha256": g.file_sha256(OUT / f"{s}.bin")}  # fmt: skip
        for name, ds in splits[s].items():
            write_jsonl(OUT / "heldout_text" / f"{name}_{s}", ds)
    atomic_write_bytes(OUT / "train.bin", train.tobytes())
    tr_bytes = sum(len(d.text.encode("utf-8")) for d, _ in selected)
    streams["train"] = {"documents": len(selected), "tokens": int(train.size),
                        "text_bytes": tr_bytes, "bytes_per_token_incl_eot": tr_bytes / train.size,
                        "sha256": g.file_sha256(OUT / "train.bin"),
                        "single_pass_windows": windows,
                        "windows_needed_for_plan": args.train_tokens // args.context_length,
                        "per_source_tokens": {k: v["selected_tokens"]
                                              for k, v in quota_info.items()}}  # fmt: skip
    if windows < args.train_tokens // args.context_length:
        raise SystemExit(f"train stream gives {windows} windows; plan needs more")

    # TinyStories V2 validation text for the regression bits-per-byte check
    ts_docs = [s.strip() for s in TS_VALID.read_text(encoding="utf-8").split("<|endoftext|>")]
    write_jsonl(OUT / "heldout_text" / "tinystories_v2_val",
                [g.Doc("tinystories", t) for t in ts_docs if t])  # fmt: skip

    checks = tokenizer_checks(tok, held["val"] + held["holdout"])
    freq = frequency_stats(train, tok)
    per_source_bpt = {}
    for name in ("fineweb_edu", "cosmopedia_v2"):
        ds = splits["val"][name]
        n_tok = sum(len(tok.encode_ordinary(d.text)) for d in ds)
        per_source_bpt[name] = sum(len(d.text.encode("utf-8")) for d in ds) / n_tok

    meta = {
        "dataset": "general_m4_1",
        "tokenizer": "bpe",
        "vocab_size": tok.vocab_size,
        "tokenizer_fingerprint": tok.fingerprint(),
        "eot_id": tok.eot_id,
        "special_tokens": tok.special_tokens,
        "token_dtype": "uint16 little-endian",
        "num_train_tokens": streams["train"]["tokens"],
        "num_val_tokens": streams["val"]["tokens"],
        "num_holdout_tokens": streams["holdout"]["tokens"],
        "train_sha256": streams["train"]["sha256"],
        "val_sha256": streams["val"]["sha256"],
        "holdout_sha256": streams["holdout"]["sha256"],
        "val_bytes_per_token": streams["val"]["bytes_per_token_incl_eot"],
        "bytes_per_token_note": "UTF-8 document bytes / tokens (incl. one EOT per document)",
        "license": {s.name: s.license for s in g.SOURCES},
        "source": {s.name: f"{s.repo}@{s.revision}/{s.path}" for s in g.SOURCES},
    }
    atomic_write_json(OUT / "meta.json", meta)
    manifest = {
        "sources": sources,
        "deduplication": {**g.stats_dict(dstats), "minhash": g.stats_dict(mh), "seconds": dedup_s},
        "split_rule": "int(sha256(clean_text)[:8], 16) % 1000: 0-4 val, 5-9 holdout, else train",
        "split_documents": {s: {k: len(v) for k, v in d.items()} for s, d in splits.items()},
        "split_fingerprints": {
            f"{s}/{k}": list_digest([d.text for d in v])
            for s, dd in splits.items()
            for k, v in dd.items()
        },
        "split_overlap_check": {
            "train_val": len({d.sha for d, _ in selected} & {d.sha for d in held["val"]}),
            "train_holdout": len({d.sha for d, _ in selected} & {d.sha for d in held["holdout"]}),
            "val_holdout": len({d.sha for d in held["val"]} & {d.sha for d in held["holdout"]}),
        },
        "decontamination": decon,
        "tokenizer": {
            "vocab_size": tok.vocab_size,
            "fingerprint": tok.fingerprint(),
            "special_tokens": tok.special_tokens,
            "fit": fit_info,
            "fit_documents_sha256": list_digest(fit_docs),
            "fit_seconds": fit_s,
            "checks": checks,
            "val_bytes_per_token_excl_eot": per_source_bpt,
            "train_frequency": freq,
        },
        "train_selection": quota_info,
        "streams": streams,
        "encode_seconds": encode_s,
        "doc_tokens_median": {
            s: statistics.median(len(tok.encode_ordinary(d.text)) + 1 for d in held[s][:500])
            for s in held
        },
        "total_seconds": time.perf_counter() - t_start,
    }
    atomic_write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"streams": streams, "selection": quota_info, "dedup": {
        k: v for k, v in g.stats_dict(dstats).items() if k != "examples"},
        "decontamination": decon["removed_train"], "tokenizer": tok.fingerprint(),
        "checks": checks}, indent=1, default=str))  # fmt: skip


if __name__ == "__main__":
    main()

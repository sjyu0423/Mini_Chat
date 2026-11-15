"""Fetch, clean, deduplicate, tokenize, and save TinyStories V2 (GPT-4-only).

Pipeline (every step is deterministic and recorded in ``manifest.json``):
1. Download the official validation file whole and verify its SHA-256 against the
   LFS oid published by Hugging Face.
2. Plan ``--ranges`` non-overlapping byte ranges of ``--range-mb`` MB in the 2.23 GB
   training file (one per equal stratum, seeded offsets) and fetch each with a
   verified HTTP Range request (status 206, exact Content-Range, file size).
3. Recover complete stories (drop the partial story at each range edge).
4. Remove exact and near-duplicate training stories, and any training story that
   exactly or nearly matches a validation story.
5. Hold out ``--holdout-percent`` of training stories by content hash.
6. Train byte-level BPE (``--vocab``) on the training stories only.
7. Encode every story followed by ``<|endoftext|>`` into uint16 token streams.

Raw downloads stay under ``data/raw/tinystories_v2/`` and outputs under
``data/processed/tinystories_v2_bpe/``; both are git-ignored.

Usage:
    python scripts/prepare_tinystories.py --ranges 8 --range-mb 50 --vocab 4096 \
        --compare-tokenizer runs/m3_tokenizers-*/bpe_tinystories_4096.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from mini_chatgpt.data import tinystories as ts
from mini_chatgpt.data.prepare import file_sha256
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_bytes, atomic_write_json
from mini_chatgpt.training.run import setup_logging

log = setup_logging()


def fetch_validation(path: Path) -> dict[str, Any]:
    if not path.exists():
        req = urllib.request.Request(ts.file_url(ts.VALID_FILE),
                                     headers={"User-Agent": ts.USER_AGENT})  # fmt: skip
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read()
        atomic_write_bytes(path, body)
    digest = file_sha256(path)
    size = path.stat().st_size
    if digest != ts.VALID_SHA256 or size != ts.VALID_SIZE:
        raise SystemExit(f"validation file mismatch: sha256 {digest}, size {size}")
    return {"file": ts.VALID_FILE, "url": ts.file_url(ts.VALID_FILE), "bytes": size,
            "sha256": digest, "sha256_matches_upstream_lfs_oid": True}  # fmt: skip


def fetch_ranges(
    raw: Path, plan: list[tuple[int, int]]
) -> tuple[list[bytes], list[dict[str, Any]]]:
    blobs: list[bytes] = []
    infos: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(plan):
        path = raw / f"range_{i:02d}_{start}_{end}.bin"
        if path.exists() and path.stat().st_size == end - start + 1:
            body = path.read_bytes()
            info: dict[str, Any] = {
                "start": start,
                "end": end,
                "bytes": len(body),
                "reused_local_copy": True,
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        else:
            t0 = time.perf_counter()
            body, info = ts.fetch_range(ts.file_url(ts.TRAIN_FILE), start, end, ts.TRAIN_SIZE)
            info["seconds"] = time.perf_counter() - t0
            atomic_write_bytes(path, body)
        log.info("range %d: %d-%d (%.1f MB) sha256 %s", i, start, end, len(body) / 1e6,
                 info["sha256"][:12])  # fmt: skip
        blobs.append(body)
        infos.append(info)
    return blobs, infos


def encode_stream(tok: ByteBPETokenizer, stories: list[str]) -> tuple[np.ndarray, float, int]:
    eot = tok.eot_id
    assert eot is not None
    t0 = time.perf_counter()
    ids: list[int] = []
    for s in stories:
        ids.extend(tok.encode_ordinary(s))
        ids.append(eot)
    dt = time.perf_counter() - t0
    n_bytes = sum(len(s.encode("utf-8")) for s in stories)
    return np.asarray(ids, dtype=np.uint16), dt, n_bytes


def usage_stats(counts: np.ndarray, n_ordinary: int) -> dict[str, Any]:
    merged = counts[256:n_ordinary]
    total = int(counts.sum())
    order = np.sort(counts[:n_ordinary])[::-1]
    return {
        "tokens": total,
        "vocab_used": int((counts > 0).sum()),
        "vocab_used_fraction": float((counts > 0).mean()),
        "merged_tokens_unused": int((merged == 0).sum()),
        "merged_tokens_lt10": int((merged < 10).sum()),
        "merged_tokens_lt100": int((merged < 100).sum()),
        "merged_tokens_lt_1e-6_share": int((merged < 1e-6 * total).sum()),
        "byte_tokens_used": int((counts[:256] > 0).sum()),
        "top10_share": float(order[:10].sum() / total),
        "top100_share": float(order[:100].sum() / total),
        "top1000_share": float(order[:1000].sum() / total),
        "median_count_merged": float(np.median(merged)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranges", type=int, default=8)
    p.add_argument("--range-mb", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--holdout-percent", type=float, default=1.0)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--compare-tokenizer", type=Path, help="M3 tokenizer for rare-token comparison")
    args = p.parse_args()
    raw = args.data_dir / "raw" / "tinystories_v2"
    out = args.data_dir / "processed" / "tinystories_v2_bpe"
    raw.mkdir(parents=True, exist_ok=True)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite a prepared dataset")
    t_start = time.perf_counter()

    val_info = fetch_validation(raw / ts.VALID_FILE)
    range_bytes = int(args.range_mb * 1_000_000)
    plan = ts.plan_ranges(ts.TRAIN_SIZE, args.ranges, range_bytes, args.seed)
    blobs, range_infos = fetch_ranges(raw, plan)

    pstats = ts.ParseStats()
    train_raw: list[str] = []
    for blob, info in zip(blobs, range_infos, strict=True):
        before = pstats.stories
        train_raw.extend(ts.split_stories(blob, partial_edges=True, stats=pstats))
        info["complete_stories"] = pstats.stories - before
    vstats = ts.ParseStats()
    val = ts.split_stories((raw / ts.VALID_FILE).read_bytes(), partial_edges=False, stats=vstats)
    log.info("parsed %d train stories, %d validation stories", len(train_raw), len(val))

    t0 = time.perf_counter()
    mh = ts.MinHashConfig()
    deduped, drep = ts.deduplicate(train_raw, val, mh)
    dedup_s = time.perf_counter() - t0
    train, holdout = ts.holdout_split(deduped, args.holdout_percent)
    log.info("dedup: %s (%.0f s); train %d, holdout %d", ts.report_dict(drep), dedup_s,
             len(train), len(holdout))  # fmt: skip

    t0 = time.perf_counter()
    tok = ByteBPETokenizer.train(train, vocab_size=args.vocab)
    tok_train_s = time.perf_counter() - t0
    log.info("tokenizer trained in %.1f s (fingerprint %s)", tok_train_s, tok.fingerprint()[:12])

    streams: dict[str, dict[str, Any]] = {}
    for name, stories in (("train", train), ("val", val), ("holdout", holdout)):
        arr, dt, n_bytes = encode_stream(tok, stories)
        atomic_write_bytes(out / f"{name}.bin", arr.astype("<u2").tobytes())
        lens = np.diff(np.flatnonzero(arr == tok.eot_id), prepend=-1)
        streams[name] = {"stories": len(stories), "tokens": int(arr.size),
                         "text_bytes": n_bytes, "bytes_per_token": n_bytes / arr.size,
                         "encode_seconds": dt, "encode_mb_per_s": n_bytes / dt / 1e6,
                         "tokens_per_story_mean": float(lens.mean()),
                         "tokens_per_story_median": float(np.median(lens)),
                         "tokens_per_story_p95": float(np.percentile(lens, 95)),
                         "tokens_per_story_max": int(lens.max()),
                         "stories_longer_than_512_tokens": int((lens > 512).sum()),
                         "sha256": file_sha256(out / f"{name}.bin"),
                         "stories_digest": ts.list_digest(stories)}  # fmt: skip
        log.info("%s: %d stories, %d tokens, %.2f bytes/token, %.1f MB/s", name, len(stories),
                 arr.size, n_bytes / arr.size, n_bytes / dt / 1e6)  # fmt: skip
    # tokens before deduplication (all parsed training stories, same tokenizer)
    pre_tokens = sum(len(tok.encode_ordinary(s)) + 1 for s in train_raw)
    tok.save(out / "tokenizer.json")

    n_ord = 256 + len(tok.merges)
    train_arr = np.fromfile(out / "train.bin", dtype="<u2")
    val_arr = np.fromfile(out / "val.bin", dtype="<u2")
    counts = np.bincount(train_arr, minlength=tok.vocab_size)
    vcounts = np.bincount(val_arr, minlength=tok.vocab_size)

    def piece(i: int) -> str:
        return tok.token_bytes(i).decode("utf-8", "replace")

    longest = sorted(range(256, n_ord), key=lambda i: (-len(tok.token_bytes(i)), i))
    tok_stats: dict[str, Any] = {
        "vocab_size": tok.vocab_size,
        "merges": len(tok.merges),
        "special_tokens": tok.special_tokens,
        "fingerprint": tok.fingerprint(),
        "train_seconds": tok_train_s,
        "unique_pretokens": tok.metadata["unique_pretokens"],
        "train_usage": usage_stats(counts, n_ord),
        "val_usage": usage_stats(vcounts, n_ord),
        "longest_tokens": [piece(i) for i in longest[:12]],
        "most_frequent_tokens": [[piece(int(i)), int(counts[i])] for i in np.argsort(-counts)[:20]],
    }
    if args.compare_tokenizer:
        old = ByteBPETokenizer.load(args.compare_tokenizer)
        old_ids = np.asarray([i for s in train for i in old.encode_ordinary(s)], dtype=np.int64)
        ocounts = np.bincount(old_ids, minlength=old.vocab_size)
        overlap = len(set(old.merges) & set(tok.merges))
        tok_stats["m3_sample_tokenizer_on_this_corpus"] = {
            "file": str(args.compare_tokenizer),
            "fingerprint": old.fingerprint(),
            "trained_on_bytes": old.metadata.get("trained_on_bytes"),
            "usage": usage_stats(ocounts, 256 + len(old.merges)),
            "bytes_per_token_excluding_eot": streams["train"]["text_bytes"] / int(old_ids.size),
            "shared_merges_with_new": overlap,
        }
    val_bpt = streams["val"]["text_bytes"] / streams["val"]["tokens"]
    meta = {
        "dataset": "tinystories_v2",
        "tokenizer": "bpe",
        "vocab_size": tok.vocab_size,
        "tokenizer_fingerprint": tok.fingerprint(),
        "eot_id": tok.eot_id,
        "num_train_tokens": streams["train"]["tokens"],
        "num_val_tokens": streams["val"]["tokens"],
        "num_holdout_tokens": streams["holdout"]["tokens"],
        "train_sha256": streams["train"]["sha256"],
        "val_sha256": streams["val"]["sha256"],
        "holdout_sha256": streams["holdout"]["sha256"],
        "val_bytes_per_token": val_bpt,
        "bytes_per_token_note": "UTF-8 story-text bytes / tokens (incl. one EOT per story)",
        "token_dtype": "uint16 little-endian",
        "license": ts.LICENSE,
        "source": f"https://huggingface.co/datasets/{ts.REPO}@{ts.REVISION}",
    }  # fmt: skip
    atomic_write_json(out / "meta.json", meta)
    manifest = {
        "source": {
            "repo": ts.REPO, "revision": ts.REVISION, "license": ts.LICENSE,
            "train_file": ts.TRAIN_FILE, "train_size": ts.TRAIN_SIZE,
            "train_upstream_sha256": ts.TRAIN_SHA256, "train_url": ts.file_url(ts.TRAIN_FILE),
            "validation": val_info,
        },
        "ranges": {"seed": args.seed, "n_ranges": args.ranges, "range_bytes": range_bytes,
                   "method": "one range per equal stratum, seeded offset; verified HTTP 206",
                   "fetched_bytes": sum(i["bytes"] for i in range_infos),
                   "fraction_of_file": sum(i["bytes"] for i in range_infos) / ts.TRAIN_SIZE,
                   "items": range_infos},
        "parsing": {"train": ts.report_dict(pstats), "validation": ts.report_dict(vstats),
                    "decisions": ["split raw bytes on b'<|endoftext|>' before decoding",
                                  "drop the bytes before the first and after the last "
                                  "delimiter of each range (partial stories)",
                                  "strict UTF-8 decode; undecodable stories dropped",
                                  "strip surrounding whitespace; empty stories dropped",
                                  "no Unicode normalization; text kept as published"]},
        "deduplication": {**ts.report_dict(drep), "minhash": ts.report_dict(mh),
                          "seconds": dedup_s,
                          "train_tokens_before_dedup": pre_tokens,
                          "train_plus_holdout_tokens_after_dedup":
                              streams["train"]["tokens"] + streams["holdout"]["tokens"]},
        "holdout": {"percent": args.holdout_percent,
                    "rule": "int(sha256(story)[:8], 16) % 10000 < percent * 100"},
        "streams": streams,
        "tokenizer": tok_stats,
        "total_seconds": time.perf_counter() - t_start,
    }  # fmt: skip
    atomic_write_json(out / "manifest.json", manifest)
    brief = {k: {kk: vv for kk, vv in v.items() if "sha" not in kk and "digest" not in kk}
             for k, v in streams.items()}  # fmt: skip
    summary = {"streams": brief, "dedup": ts.report_dict(drep),
               "tokenizer_train_usage": tok_stats["train_usage"]}  # fmt: skip
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

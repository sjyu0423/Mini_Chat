"""Fetch a bounded, reproducible sample of a Hugging Face dataset via the dataset-viewer API.

No full download: the script asks ``datasets-server.huggingface.co/rows`` for
``--requests`` windows of ``--rows`` consecutive rows (at most 100 per request) at
offsets drawn from a seeded RNG. The result is written as JSON Lines plus a
``manifest.json`` that records every request, the dataset revision metadata, the
number of rows and bytes, and SHA-256 hashes. Rows whose cells the API truncated
are dropped and counted.

Examples:
    python scripts/fetch_samples.py roneneldan/TinyStories --config default --split train \
        --requests 60 --out data/raw/samples/tinystories
    python scripts/fetch_samples.py HuggingFaceFW/fineweb-edu --config sample-10BT \
        --split train --requests 40 --out data/raw/samples/fineweb_edu_10bt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mini_chatgpt.training.checkpoint import atomic_write_json, atomic_write_text

API = "https://datasets-server.huggingface.co"
USER_AGENT = "mini-chatgpt-research/0.1 (bounded sample fetch)"
KEEP_FIELDS = (
    "text", "id", "url", "dump", "language_score", "score", "int_score", "token_count",
    # M4 additions (conversation and synthetic corpora); absent fields are simply skipped
    "messages", "source", "title", "format", "audience", "seed_data", "full_topic",
)  # fmt: skip


def row_text_bytes(row: dict[str, Any]) -> int:
    """UTF-8 bytes of a row's text, or of its message contents for chat datasets."""
    if "text" in row:
        return len(str(row["text"]).encode("utf-8"))
    return sum(len(str(m.get("content", "")).encode("utf-8")) for m in row.get("messages", []))


def dataset_revision(dataset: str) -> str | None:
    """Commit SHA of the dataset's main branch at fetch time (the rows API serves main)."""
    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{dataset}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            sha = json.loads(resp.read().decode("utf-8")).get("sha")
            return str(sha) if sha else None
    except (urllib.error.URLError, TimeoutError):
        return None


def get_json(endpoint: str, params: dict[str, Any], retries: int = 6) -> Any:
    """GET with polite retries: honours Retry-After on 429, exponential backoff otherwise."""
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if attempt == retries - 1 or exc.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 0.0
            time.sleep(max(wait, 10.0 * 2**attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(10.0 * 2**attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    p.add_argument("--config", default="default")
    p.add_argument("--split", default="train")
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--rows", type=int, default=100, help="rows per request (API max 100)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--sleep", type=float, default=3.0, help="seconds between requests")
    p.add_argument("--max-bytes", type=int, default=40_000_000, help="hard cap on text bytes")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.rows <= 100:
        raise SystemExit("--rows must be in 1..100")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; refusing to overwrite a sample")
    args.out.mkdir(parents=True, exist_ok=True)

    revision = dataset_revision(args.dataset)
    size = get_json("size", {"dataset": args.dataset, "config": args.config})
    split_info = next(s for s in size["size"]["splits"] if s["split"] == args.split)
    n_rows = int(split_info["num_rows"])
    rng = random.Random(args.seed)
    offsets = sorted(rng.sample(range(0, n_rows - args.rows), args.requests))

    lines: list[str] = []
    requests: list[dict[str, Any]] = []
    seen: set[int] = set()
    truncated = 0
    text_bytes = 0
    for off in offsets:
        if text_bytes >= args.max_bytes:
            break
        params = {"dataset": args.dataset, "config": args.config, "split": args.split,
                  "offset": off, "length": args.rows}  # fmt: skip
        data = get_json("rows", params)
        kept = 0
        for r in data["rows"]:
            idx = int(r["row_idx"])
            if idx in seen:
                continue
            seen.add(idx)
            if r.get("truncated_cells"):
                truncated += 1
                continue
            row = {k: v for k, v in r["row"].items() if k in KEEP_FIELDS}
            row["row_idx"] = idx
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            lines.append(line)
            text_bytes += row_text_bytes(row)
            kept += 1
        requests.append({"offset": off, "length": args.rows, "returned": len(data["rows"]),
                         "kept": kept})  # fmt: skip
        time.sleep(args.sleep)

    body = "\n".join(lines) + "\n"
    path = atomic_write_text(args.out / "sample.jsonl", body)
    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "source_api": f"{API}/rows",
        "dataset_revision_at_fetch": revision,
        "split_num_rows": n_rows,
        "split_num_bytes_parquet": split_info.get("num_bytes_parquet_files"),
        "seed": args.seed,
        "rows_per_request": args.rows,
        "requests": requests,
        "rows_kept": len(lines),
        "rows_dropped_truncated": truncated,
        "text_bytes_utf8": text_bytes,
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "file_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "fetched_at": datetime.now(UTC).isoformat(),
        "note": "Bounded sample for tokenizer research; not a full download.",
    }
    atomic_write_json(args.out / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()

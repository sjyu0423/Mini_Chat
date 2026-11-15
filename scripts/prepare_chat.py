"""Prepare the M5 conversational SFT dataset (OASST2 + Dolly-15k).

Steps (all deterministic):

1. Verify the pinned raw files (size + SHA-256). ``--download`` fetches them first.
2. Build conversations: OASST2 best-ranked English path per tree; Dolly one turn each.
3. Normalize text; drop malformed conversations (reasons counted per source).
4. Decontaminate against both frozen evaluation suites: drop a conversation if it
   shares a word 8-gram with any eval item, or if its first user message equals an eval
   user message (case/whitespace-insensitive).
5. Exact deduplication on the first user message (survivor: smallest hash of
   ``group``), then split by group hash: 5% validation, 95% train.
6. Context policy (``chat.format.fit_conversation``): keep whole conversations that fit
   512 input tokens; otherwise drop trailing exchanges; drop if the first exchange alone
   does not fit. No message is ever cut.
7. Tokenize with the unchanged M4.1 tokenizer; write per split ``tokens.bin`` (uint16),
   ``mask.bin`` (uint8 target mask), ``offsets.npy`` (int64 conversation starts),
   ``conversations.jsonl`` (normalized text) and one ``manifest.json``.

Usage: python scripts/prepare_chat.py [--download]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mini_chatgpt.analysis.general_eval import all_eval_texts, load_suite
from mini_chatgpt.chat import data as cd
from mini_chatgpt.chat.format import encode_conversation, fit_conversation
from mini_chatgpt.data.general import contaminating_items, eval_ngrams, file_sha256
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json

RAW = Path("data/raw/chat_m5")
OUT = Path("data/processed/chat_m5")
TOKENIZER = Path("data/processed/general_m4_1_bpe/tokenizer.json")
FINGERPRINT = "e6be77ccff4929187ed84ee1b71d15b7654824abd9e23f3ad889c57232ce91c8"
CONTEXT = 512
_WS = " ".join


def verify(src: cd.ChatSource) -> None:
    path = RAW / src.local_name
    if not path.exists():
        raise SystemExit(f"{path} missing; run with --download")
    if path.stat().st_size != src.size or file_sha256(path) != src.sha256:
        raise SystemExit(f"{path}: size or sha256 differs from the pinned value")


def norm_q(text: str) -> str:
    return _WS(text.lower().split())


def chat_eval_texts(path: Path) -> dict[str, str]:
    suite = json.loads(path.read_text())
    return {i["id"]: "\n".join(t["content"] for t in i["turns"]) for i in suite["items"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    if args.download:
        for src in cd.CHAT_SOURCES:
            dest = RAW / src.local_name
            if not dest.exists():
                subprocess.run(["curl", "-sSL", "-o", str(dest), src.url], check=True)
    for src in cd.CHAT_SOURCES:
        verify(src)
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} is not empty; refusing to overwrite")
    OUT.mkdir(parents=True, exist_ok=True)

    tok = ByteBPETokenizer.load(TOKENIZER)
    if tok.fingerprint() != FINGERPRINT:
        raise SystemExit("tokenizer fingerprint differs from M4.1")

    # 2-3. build, normalize, reject
    source_stats: dict[str, dict[str, int]] = {"oasst2": {}, "dolly15k": {}}
    convs = [
        *cd.oasst2_conversations(RAW / cd.CHAT_SOURCES[0].local_name, source_stats["oasst2"]),
        *cd.dolly_conversations(RAW / cd.CHAT_SOURCES[1].local_name, source_stats["dolly15k"]),
    ]
    rejected: Counter[str] = Counter()
    kept: list[cd.Conversation] = []
    for c in convs:
        reason = cd.malformed_reason(c.messages)
        if reason:
            rejected[f"{c.source}:{reason}"] += 1
        else:
            kept.append(c)

    # 4. decontaminate against both frozen suites
    general, general_sha = load_suite("evals/general_v1.json")
    chat_raw = Path("evals/chat_v1.json").read_bytes()
    eval_texts = {**all_eval_texts(general), **chat_eval_texts(Path("evals/chat_v1.json"))}
    index = eval_ngrams(eval_texts)
    eval_questions = {norm_q(t.split("\n")[-1]) for t in eval_texts.values()}
    eval_questions |= {norm_q(t) for t in eval_texts.values()}
    contaminated: list[dict[str, Any]] = []
    clean: list[cd.Conversation] = []
    for c in kept:
        text = "\n".join(m["content"] for m in c.messages)
        hits = contaminating_items(text, index)
        same_q = norm_q(c.messages[0]["content"]) in eval_questions
        if hits or same_q:
            contaminated.append(
                {
                    "group": c.group,
                    "items": sorted(hits),
                    "exact_question": same_q,
                    "first_user": c.messages[0]["content"][:120],
                }
            )
        else:
            clean.append(c)

    # 5. exact dedup, split
    def gh(c: cd.Conversation) -> str:
        return hashlib.sha256(c.group.encode()).hexdigest()

    seen: set[str] = set()
    unique: list[cd.Conversation] = []
    dup_removed: Counter[str] = Counter()
    for c in sorted(clean, key=lambda c: (cd.dedup_key(c.messages), gh(c))):
        k = cd.dedup_key(c.messages)
        if k in seen:
            dup_removed[c.source] += 1
            continue
        seen.add(k)
        unique.append(c)
    unique.sort(key=gh)  # canonical order, mixes sources

    # 6-7. length policy and tokenization
    splits: dict[str, dict[str, Any]] = {
        s: {"ids": [], "mask": [], "offsets": [0], "rows": []} for s in ("train", "val")
    }
    length_status: Counter[str] = Counter()
    for c in unique:
        msgs, status = fit_conversation(tok, c.messages, CONTEXT)
        length_status[f"{c.source}:{status}"] += 1
        if status == "dropped":
            continue
        ids, mask = encode_conversation(tok, msgs)
        s = splits[cd.split_of_group(c.group)]
        s["ids"] += ids
        s["mask"] += mask
        s["offsets"].append(len(s["ids"]))
        s["rows"].append(
            {
                "source": c.source,
                "group": c.group,
                "length_status": status,
                "tokens": len(ids),
                "targets": int(sum(mask)),
                "turns": len(msgs),
                **c.meta,
                "messages": msgs,
            }
        )

    split_stats: dict[str, Any] = {}
    for name, s in splits.items():
        arr_ids = np.asarray(s["ids"], dtype=np.uint16)
        arr_mask = np.asarray(s["mask"], dtype=np.uint8)
        offsets = np.asarray(s["offsets"], dtype=np.int64)
        assert int(arr_ids.max()) < tok.vocab_size
        assert arr_ids.size == arr_mask.size == offsets[-1]
        arr_ids.tofile(OUT / f"{name}_tokens.bin")
        arr_mask.tofile(OUT / f"{name}_mask.bin")
        np.save(OUT / f"{name}_offsets.npy", offsets)
        with (OUT / f"{name}_conversations.jsonl").open("w", encoding="utf-8") as f:
            for r in s["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows = s["rows"]
        lens = np.asarray([r["tokens"] for r in rows])
        split_stats[name] = {
            "conversations": len(rows),
            "by_source": dict(Counter(r["source"] for r in rows)),
            "multi_turn_conversations": sum(r["turns"] > 2 for r in rows),
            "assistant_turns": sum(r["turns"] // 2 for r in rows),
            "turns_histogram": dict(sorted(Counter(r["turns"] for r in rows).items())),
            "dolly_categories": dict(Counter(r["category"] for r in rows if "category" in r)),
            "tokens": int(arr_ids.size),
            "target_tokens": int(arr_mask.sum()),
            "target_tokens_by_source": {
                src: int(sum(r["targets"] for r in rows if r["source"] == src))
                for src in ("oasst2", "dolly15k")
            },
            "tokens_per_conversation": {
                "mean": float(lens.mean()),
                "median": float(np.median(lens)),
                "p95": float(np.percentile(lens, 95)),
                "max": int(lens.max()),
            },
            "tokens_sha256": file_sha256(OUT / f"{name}_tokens.bin"),
            "mask_sha256": file_sha256(OUT / f"{name}_mask.bin"),
            "offsets_sha256": file_sha256(OUT / f"{name}_offsets.npy"),
        }
    train_groups = {r["group"] for r in splits["train"]["rows"]}
    val_groups = {r["group"] for r in splits["val"]["rows"]}
    train_keys = {cd.dedup_key(r["messages"]) for r in splits["train"]["rows"]}
    val_keys = {cd.dedup_key(r["messages"]) for r in splits["val"]["rows"]}
    shutil.copyfile(TOKENIZER, OUT / "tokenizer.json")
    manifest = {
        "sources": [{**src.__dict__, "url": src.url} for src in cd.CHAT_SOURCES],
        "source_stats": source_stats,
        "conversations_built": dict(Counter(c.source for c in convs)),
        "malformed_rejected": dict(sorted(rejected.items())),
        "decontamination": {
            "general_suite_sha256": general_sha,
            "chat_suite_sha256": hashlib.sha256(chat_raw).hexdigest(),
            "eval_items_indexed": len(eval_texts),
            "removed": len(contaminated),
            "removed_by_source": dict(Counter(c["group"].split(":")[0] for c in contaminated)),
            "examples": contaminated[:25],
        },
        "exact_duplicates_removed": dict(dup_removed),
        "after_dedup": dict(Counter(c.source for c in unique)),
        "length_policy": {"context": CONTEXT, "status": dict(sorted(length_status.items()))},
        "splits": split_stats,
        "leakage_check": {
            "shared_groups": len(train_groups & val_groups),
            "shared_first_user_messages": len(train_keys & val_keys),
        },
        "tokenizer_fingerprint": tok.fingerprint(),
        "format": "<|endoftext|> then per message: <|role|> \\n content <|end|> \\n",
        "seconds": round(time.time() - t0, 1),
    }
    atomic_write_json(OUT / "manifest.json", manifest)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("sources",)},
            indent=1,
            ensure_ascii=False,
        )[:6000]
    )


if __name__ == "__main__":
    main()

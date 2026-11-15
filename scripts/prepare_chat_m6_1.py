"""M6.1 SFT data: the M5 conversations with ONE added rule, assistant replies <= 100 words.

Input: ``data/processed/chat_m5/{train,val}_conversations.jsonl`` (already cleaned,
deduplicated, decontaminated against general_v1/chat_v1 and split by group in M5).
Output: ``data/processed/chat_m6_1/`` in the same format (token stream, target mask,
offsets, conversations, manifest), so ``chat.sft`` trains on it unchanged.

Rules, applied in this order:
1. Decontaminate against the M6.1 diagnostic suite ``evals/chat_diag_v1.json`` (word
   8-grams of every item, and exact first-user-message match), as M5 did for its suites.
2. **Length cap (the experimental variable):** walk each conversation; at the first
   assistant reply with more than ``MAX_WORDS`` words, cut the conversation just before
   that user/assistant exchange. If that is the first exchange, drop the conversation.
   No message is shortened; replies are kept or removed whole.
3. The M5 context policy is unchanged (``fit_conversation``, 512 inputs); after the cap
   every kept conversation already fits.

Audit motivation (docs/reports/m6_1/sft_audit.json): replies over 100 words are 24.9% of
M5 training replies but carry 60.8% of the assistant-target tokens; the M5 model rambles
(19% of replies over 80 words) and ignores short-answer instructions.

Usage: python scripts/prepare_chat_m6_1.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mini_chatgpt.chat import data as cd
from mini_chatgpt.chat.format import Message, encode_conversation, fit_conversation
from mini_chatgpt.data.general import contaminating_items, eval_ngrams, file_sha256
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json

SRC = Path("data/processed/chat_m5")
OUT = Path("data/processed/chat_m6_1")
DIAG = Path("evals/chat_diag_v1.json")
MAX_WORDS = 100
WORD = re.compile(r"[A-Za-z0-9']+")


def n_words(text: str) -> int:
    return len(WORD.findall(text))


def cap_conversation(messages: list[Message], max_words: int) -> list[Message]:
    """Longest prefix of whole exchanges whose assistant replies are all <= max_words."""
    kept: list[Message] = []
    for i in range(0, len(messages) - 1, 2):
        user, asst = messages[i], messages[i + 1]
        if n_words(asst["content"]) > max_words:
            break
        kept += [user, asst]
    return kept


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} is not empty; refusing to overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    tok = ByteBPETokenizer.load(SRC / "tokenizer.json")
    diag_items = json.loads(DIAG.read_text())["items"]
    diag_texts = {i["id"]: "\n".join(t["content"] for t in i["turns"]) for i in diag_items}
    index = eval_ngrams(diag_texts)
    diag_q = {" ".join(t["content"].lower().split()) for i in diag_items for t in i["turns"]}
    manifest: dict[str, Any] = {
        "source": str(SRC),
        "source_manifest_sha256": file_sha256(SRC / "manifest.json"),
        "rule": f"assistant replies <= {MAX_WORDS} words (conversation cut before the first "
        "longer reply; dropped if it is the first)",
        "diag_suite_sha256": hashlib.sha256(DIAG.read_bytes()).hexdigest(),
        "splits": {},
    }
    for split in ("train", "val"):
        rows = [json.loads(line) for line in (SRC / f"{split}_conversations.jsonl").open()]
        st: Counter[str] = Counter()
        ids_all: list[int] = []
        mask_all: list[bool] = []
        offsets = [0]
        out_rows = []
        for r in rows:
            msgs: list[Message] = r["messages"]
            text = "\n".join(m["content"] for m in msgs)
            if (
                contaminating_items(text, index)
                or " ".join(msgs[0]["content"].lower().split()) in diag_q
            ):
                st[f"{r['source']}:diag_decontaminated"] += 1
                continue
            capped = cap_conversation(msgs, MAX_WORDS)
            if not capped:
                st[f"{r['source']}:dropped_first_reply_too_long"] += 1
                continue
            if len(capped) < len(msgs):
                st[f"{r['source']}:cut_to_earlier_turns"] += 1
            else:
                st[f"{r['source']}:kept_whole"] += 1
            fitted, status = fit_conversation(tok, capped, 512)
            assert status == "full", status
            ids, mask = encode_conversation(tok, fitted)
            ids_all += ids
            mask_all += mask
            offsets.append(len(ids_all))
            out_rows.append({**{k: v for k, v in r.items() if k != "messages"},
                             "turns": len(fitted), "tokens": len(ids),
                             "targets": int(sum(mask)), "messages": fitted})  # fmt: skip
        arr_ids = np.asarray(ids_all, dtype=np.uint16)
        arr_mask = np.asarray(mask_all, dtype=np.uint8)
        arr_off = np.asarray(offsets, dtype=np.int64)
        arr_ids.tofile(OUT / f"{split}_tokens.bin")
        arr_mask.tofile(OUT / f"{split}_mask.bin")
        np.save(OUT / f"{split}_offsets.npy", arr_off)
        with (OUT / f"{split}_conversations.jsonl").open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        lens = np.diff(arr_off)
        manifest["splits"][split] = {
            "input_conversations": len(rows),
            "conversations": len(out_rows),
            "by_source": dict(Counter(r["source"] for r in out_rows)),
            "multi_turn_conversations": sum(r["turns"] > 2 for r in out_rows),
            "assistant_turns": sum(r["turns"] // 2 for r in out_rows),
            "rule_counts": dict(sorted(st.items())),
            "tokens": int(arr_ids.size),
            "target_tokens": int(arr_mask.sum()),
            "tokens_per_conversation_median": float(np.median(lens)),
            "tokens_sha256": file_sha256(OUT / f"{split}_tokens.bin"),
            "mask_sha256": file_sha256(OUT / f"{split}_mask.bin"),
            "offsets_sha256": file_sha256(OUT / f"{split}_offsets.npy"),
        }
    tr = [json.loads(x) for x in (OUT / "train_conversations.jsonl").open()]
    va = [json.loads(x) for x in (OUT / "val_conversations.jsonl").open()]
    manifest["leakage_check"] = {
        "shared_groups": len({r["group"] for r in tr} & {r["group"] for r in va}),
        "shared_first_user_messages": len(
            {cd.dedup_key(r["messages"]) for r in tr} & {cd.dedup_key(r["messages"]) for r in va}
        ),
    }
    manifest["tokenizer_fingerprint"] = tok.fingerprint()
    shutil.copyfile(SRC / "tokenizer.json", OUT / "tokenizer.json")
    atomic_write_json(OUT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

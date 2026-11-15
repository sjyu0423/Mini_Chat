"""M4 dataset analysis on bounded samples (no downloads; reads data/raw/samples/*).

For every text corpus:
* heuristic quality filters in the spirit of Gopher/C4 rules (share of documents
  each rule would flag), English-language heuristics (stop-word rate, Latin
  letters), and markup density;
* cross-corpus verbatim overlap: the share of documents in corpus A that share at
  least one word 13-gram with any document in corpus B (e.g. Cosmopedia text that
  copies its FineWeb seed page, or Wikipedia text inside FineWeb-Edu);
* eval-suite contamination: documents that contain a word 8-gram of any item in
  ``evals/general_v1.json`` (items shorter than 8 words are matched whole).

For conversation corpora: turns, roles, lengths, sources and code share.

Usage: python scripts/m4_data_analysis.py --out docs/reports/m4
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

from mini_chatgpt.analysis.general_eval import all_eval_texts, load_suite, ngram_overlap
from mini_chatgpt.training.checkpoint import atomic_write_json

SAMPLES = Path("data/raw/samples")
TEXT = {
    "fineweb_edu": "fineweb_edu_10bt",
    "cosmopedia_v2": "cosmopedia_v2",
    "wikipedia_en": "wikipedia_en",
    "wikipedia_simple": "wikipedia_simple",
    "tinystories": "tinystories",
}
CHAT = {"smol_smoltalk": "smol_smoltalk", "everyday_conversations": "everyday_conversations"}
WORDS = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
STOP = {"the", "be", "to", "of", "and", "that", "have", "with", "is", "a", "in", "it", "for"}
END_PUNCT = (".", "!", "?", '"', "'", ")", ":", "”")
MARKUP = set("|{}=<>[]#*")


def load(name: str) -> list[dict[str, Any]]:
    with (SAMPLES / name / "sample.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def quality_flags(text: str) -> dict[str, bool]:
    words = text.split()
    alpha_words = WORDS.findall(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n_words = max(len(words), 1)
    mean_len = statistics.fmean(len(w) for w in words) if words else 0.0
    line_counts = Counter(lines)
    dup_lines = sum(c for c in line_counts.values() if c > 1)
    lower = [w.lower() for w in alpha_words]
    stop_rate = sum(w in STOP for w in lower) / max(len(lower), 1)
    letters = [ch for ch in text if ch.isalpha()]
    latin = sum(ch.isascii() for ch in letters) / max(len(letters), 1)
    return {
        "too_short_lt_50_words": len(words) < 50,
        "mean_word_length_outside_3_10": not 3 <= mean_len <= 10,
        "alphabetic_word_share_lt_0.8": len(alpha_words) / n_words < 0.8,
        "lines_ending_in_punct_lt_0.5": (
            sum(ln.endswith(END_PUNCT) for ln in lines) / max(len(lines), 1) < 0.5
        ),
        "duplicate_line_share_gt_0.3": dup_lines / max(len(lines), 1) > 0.3,
        "markup_char_share_gt_0.05": sum(ch in MARKUP for ch in text) / max(len(text), 1) > 0.05,
        "stopword_rate_lt_0.15_non_english_or_list": stop_rate < 0.15,
        "latin_letter_share_lt_0.95": latin < 0.95,
        "boilerplate_terms": bool(
            re.search(r"cookie|javascript|all rights reserved|click here|subscribe", text, re.I)
        ),
    }


def shingle_hashes(text: str, n: int = 13) -> set[int]:
    toks = [w.lower() for w in WORDS.findall(text)]
    return {
        int.from_bytes(hashlib.blake2b(" ".join(toks[i : i + n]).encode(), digest_size=8).digest())
        for i in range(len(toks) - n + 1)
    }


def text_corpus(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags = [quality_flags(r["text"]) for r in rows]
    n = len(flags)
    rates = {k: sum(f[k] for f in flags) / n for k in flags[0]}
    any_rule = sum(any(v for k, v in f.items() if k != "boilerplate_terms") for f in flags) / n
    return {"documents": n, "flag_rates": rates, "any_quality_rule_flagged": any_rule}


def chat_corpus(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [len(r["messages"]) for r in rows]
    roles = Counter(m["role"] for r in rows for m in r["messages"])
    user_len = [len(m["content"]) for r in rows for m in r["messages"] if m["role"] == "user"]
    asst = [m["content"] for r in rows for m in r["messages"] if m["role"] == "assistant"]
    asst_len = [len(c) for c in asst]
    code = sum("```" in m["content"] for r in rows for m in r["messages"]) / len(rows)
    latin = []
    for r in rows:
        letters = [ch for m in r["messages"] for ch in m["content"] if ch.isalpha()]
        latin.append(sum(ch.isascii() for ch in letters) / max(len(letters), 1))
    return {
        "conversations": len(rows),
        "messages_per_conversation_mean": statistics.fmean(turns),
        "messages_per_conversation_max": max(turns),
        "multi_turn_share_gt_2_messages": sum(t > 2 for t in turns) / len(turns),
        "roles": dict(roles),
        "user_chars_median": statistics.median(user_len),
        "assistant_chars_median": statistics.median(asst_len),
        "assistant_chars_p95": sorted(asst_len)[int(0.95 * (len(asst_len) - 1))],
        "share_with_code_fence": code,
        "share_latin_letters_lt_0.95": sum(x < 0.95 for x in latin) / len(latin),
        "sources": dict(Counter(r.get("source", "?") for r in rows).most_common()),
        "assistant_text_bytes": sum(len(c.encode()) for c in asst),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, default=Path("evals/general_v1.json"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    corpora = {k: load(v) for k, v in TEXT.items()}
    out: dict[str, Any] = {"text": {}, "chat": {}, "cross_overlap_13gram": {},
                           "eval_contamination_8gram": {}}  # fmt: skip
    for name, rows in corpora.items():
        out["text"][name] = text_corpus(rows)
        print(name, json.dumps(out["text"][name]["flag_rates"]), flush=True)
    for name, d in CHAT.items():
        out["chat"][name] = chat_corpus(load(d))
        print(name, json.dumps({k: v for k, v in out["chat"][name].items() if k != "sources"}))

    shingles = {k: [shingle_hashes(r["text"]) for r in rows] for k, rows in corpora.items()}
    union = {k: set().union(*s) for k, s in shingles.items()}
    for a in corpora:
        out["cross_overlap_13gram"][a] = {
            b: sum(bool(s & union[b]) for s in shingles[a]) / len(shingles[a])
            for b in corpora
            if b != a
        }
    print("cross overlap", json.dumps(out["cross_overlap_13gram"]))

    suite, sha = load_suite(args.suite)
    evals = all_eval_texts(suite)
    out["eval_suite_sha256"] = sha
    for name, rows in corpora.items():
        hits = ngram_overlap(evals, [r["text"] for r in rows])
        flagged = {k: len(v) for k, v in hits.items() if v}
        out["eval_contamination_8gram"][name] = {
            "items_with_any_hit": len(flagged),
            "items_total": len(evals),
            "hits_per_item": dict(sorted(flagged.items())),
        }
    for name, d in CHAT.items():
        docs = [" ".join(m["content"] for m in r["messages"]) for r in load(d)]
        hits = ngram_overlap(evals, docs)
        flagged = {k: len(v) for k, v in hits.items() if v}
        out["eval_contamination_8gram"][name] = {
            "items_with_any_hit": len(flagged), "items_total": len(evals),
            "hits_per_item": dict(sorted(flagged.items())),
        }  # fmt: skip
    counts = {k: v["items_with_any_hit"] for k, v in out["eval_contamination_8gram"].items()}
    print("contamination", json.dumps(counts))
    atomic_write_json(args.out / "data_analysis.json", out)


if __name__ == "__main__":
    main()

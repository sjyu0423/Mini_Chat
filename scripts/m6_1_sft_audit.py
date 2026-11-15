"""M6.1 audit of the M5 SFT training data (read-only; nothing is modified).

Reads ``data/processed/chat_m5/{train,val}_conversations.jsonl`` and the token streams and
reports, per source and Dolly category:

* assistant reply length (tokens, words) and the share of long replies;
* exact and near-duplicate first user messages (MinHash/LSH, word 3-shingles, J >= 0.8);
* boilerplate / persona / refusal phrases in assistant replies;
* "copying": the share of an assistant reply's words that already occur in the user turn
  (high for context-based Dolly tasks: closed QA, information extraction, summarization);
* replies that restate the user's question (>= 60% of its words, question >= 4 words);
* very short replies, URLs, code blocks;
* truncation effects of the M5 context policy (from the M5 manifest);
* special-token balance: ``<|end|>`` targets vs all assistant targets;
* a fixed-seed random sample of 40 conversations for manual reading.

Usage: python scripts/m6_1_sft_audit.py --out docs/reports/m6_1/sft_audit.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mini_chatgpt.chat.sft import load_split
from mini_chatgpt.data.tinystories import MinHashConfig, minhash_signatures, near_duplicate_pairs
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer

D = Path("data/processed/chat_m5")
WORD = re.compile(r"[a-z0-9']+")
PATTERNS = {
    "as_an_ai": re.compile(r"\bas an? (ai|artificial intelligence|language model)", re.I),
    "language_model_self": re.compile(
        r"\b(i am|i'm) (an? )?(ai|language model|chatbot|assistant)\b", re.I
    ),
    "open_assistant_name": re.compile(r"open ?assistant", re.I),
    "refusal": re.compile(
        r"\b(i cannot|i can't|i am not able to|i'm not able to|i am unable|i'm unable)\b", re.I
    ),
    "sorry": re.compile(r"\b(i'm sorry|i am sorry|sorry, but)\b", re.I),
    "url": re.compile(r"https?://|www\."),
    "code_block": re.compile(r"```"),
}


def words(t: str) -> list[str]:
    return WORD.findall(t.lower())


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/reports/m6_1/sft_audit.json"))
    args = p.parse_args()
    tok = ByteBPETokenizer.load(D / "tokenizer.json")
    rows = [json.loads(line) for line in (D / "train_conversations.jsonl").open()]
    manifest = json.loads((D / "manifest.json").read_text())

    def group(r: dict[str, Any]) -> str:
        return r["source"] if r["source"] == "oasst2" else f"dolly:{r['category']}"

    stats: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    examples: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        g = group(r)
        msgs = r["messages"]
        for i in range(1, len(msgs), 2):
            user, asst = msgs[i - 1]["content"], msgs[i]["content"]
            aw, uw = words(asst), set(words(user))
            s = stats[g]
            s["reply_tokens"].append(len(tok.encode(asst)) + 1)
            s["reply_words"].append(len(aw))
            s["copy_frac"].append(sum(w in uw for w in aw) / len(aw) if aw else 0.0)
            q = set(words(user.split("\n")[0]))
            s["restates"].append(len(q) >= 4 and len(q & set(aw)) / len(q) >= 0.6)
            s["short"].append(len(aw) <= 3)
            for name, pat in PATTERNS.items():
                hit = bool(pat.search(asst))
                s[name].append(hit)
                if hit and len(examples[name]) < 4:
                    examples[name].append(asst[:160])

    def summarize(s: dict[str, list[Any]]) -> dict[str, Any]:
        n = len(s["reply_tokens"])
        out: dict[str, Any] = {
            "assistant_replies": n,
            "reply_tokens_median": st.median(s["reply_tokens"]),
            "reply_tokens_p90": pct(s["reply_tokens"], 90),
            "reply_words_median": st.median(s["reply_words"]),
            "share_reply_over_100_words": float(np.mean([w > 100 for w in s["reply_words"]])),
            "share_reply_over_200_words": float(np.mean([w > 200 for w in s["reply_words"]])),
            "mean_copy_fraction": float(np.mean(s["copy_frac"])),
            "share_copy_over_0.8": float(np.mean([c > 0.8 for c in s["copy_frac"]])),
            "share_restates_question": float(np.mean(s["restates"])),
            "share_reply_3_words_or_fewer": float(np.mean(s["short"])),
        }
        for name in PATTERNS:
            out[f"share_{name}"] = float(np.mean(s[name]))
        return out

    all_stats: dict[str, list[Any]] = defaultdict(list)
    for s in stats.values():
        for k, v in s.items():
            all_stats[k] += v
    by_group = {g: summarize(s) for g, s in sorted(stats.items())}

    # near-duplicate first user messages
    firsts = [r["messages"][0]["content"] for r in rows]
    sigs = minhash_signatures(firsts, MinHashConfig())
    pairs = near_duplicate_pairs(sigs, MinHashConfig())
    dup_examples = [[firsts[i][:100], firsts[j][:100]] for i, j, _ in pairs[:6]]

    # special-token balance from the token streams
    train = load_split(D, "train")
    end_id = tok.special_tokens["<|end|>"]
    targets = int(train.mask.sum())
    end_targets = int(((train.tokens == end_id) & (train.mask == 1)).sum())
    rng = random.Random(0)
    sample = [
        {"group": group(rows[i]), "messages": rows[i]["messages"]}
        for i in rng.sample(range(len(rows)), 40)
    ]
    report = {
        "train_conversations": len(rows),
        "assistant_replies": len(all_stats["reply_tokens"]),
        "overall": summarize(all_stats),
        "by_group": by_group,
        "near_duplicate_first_user_pairs": len(pairs),
        "near_duplicate_examples": dup_examples,
        "pattern_examples": examples,
        "special_tokens": {
            "assistant_target_tokens": targets,
            "end_targets": end_targets,
            "end_share_of_targets": end_targets / targets,
            "mean_targets_per_assistant_turn": targets / end_targets,
        },
        "m5_length_policy": manifest["length_policy"],
        "manual_sample_seed0": sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "manual_sample_seed0"}, indent=1,
                     ensure_ascii=False)[:9000])  # fmt: skip


if __name__ == "__main__":
    main()

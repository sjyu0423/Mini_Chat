"""M6.1 decoding experiments on a fixed checkpoint (the checkpoint is never modified).

Prompts: the frozen ``evals/chat_v1.json`` (40) and ``evals/chat_diag_v1.json`` (50).
Seeds 11, 22, 33 for every sampled setting; greedy is deterministic (one pass).

Settings (all with max 200 new tokens):

| name            | decoding                                   |
|-----------------|--------------------------------------------|
| greedy          | argmax                                     |
| greedy_rp1.2    | argmax + repetition penalty 1.2            |
| t0.7_k40        | temperature 0.7, top-k 40                   |
| t0.7_k40_rp1.1  | + penalty 1.1                              |
| t0.7_k40_rp1.2  | + penalty 1.2 (M5/M6 default)              |
| t0.7_k40_rp1.3  | + penalty 1.3                              |
| t0.7_p0.9_rp1.2 | temperature 0.7, top-p 0.9 (no top-k), 1.2 |
| t0.5_k40_rp1.2  | temperature 0.5, top-k 40, penalty 1.2     |

Selection rule (fixed before any M6.1 output was generated):
1. Eligible settings have natural end-of-turn >= 95% and repetitive replies (>30%
   repeated 4-grams) <= 5% over all 270 sampled replies (or 90 greedy replies).
2. Among eligible settings, rank by ``both_ok`` (terminated AND passed its content
   checks) over all checked items of both suites.
3. Change the default only if the best setting's ``both_ok`` exceeds the current default's
   (t0.7_k40_rp1.2) and their 95% Wilson intervals do not overlap; otherwise keep the
   current default and report the differences as not significant.

Usage: python scripts/m6_1_decoding.py <checkpoint> --label m5 --out docs/reports/m6_1
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path
from typing import Any

from mini_chatgpt.analysis.general_eval import wilson
from mini_chatgpt.chat.evaluate import aggregate, score_reply
from mini_chatgpt.chat.generate import chat_reply
from mini_chatgpt.device import select_device
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json
from mini_chatgpt.training.trainer import load_model

SUITES = {"chat_v1": Path("evals/chat_v1.json"), "diag_v1": Path("evals/chat_diag_v1.json")}
SEEDS = (11, 22, 33)
DEFAULT = "t0.7_k40_rp1.2"
SETTINGS: dict[str, dict[str, Any]] = {
    "greedy": {"greedy": True, "repetition_penalty": 1.0},
    "greedy_rp1.2": {"greedy": True, "repetition_penalty": 1.2},
    "t0.7_k40": {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.0},
    "t0.7_k40_rp1.1": {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.1},
    "t0.7_k40_rp1.2": {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.2},
    "t0.7_k40_rp1.3": {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.3},
    "t0.7_p0.9_rp1.2": {"temperature": 0.7, "top_k": None, "top_p": 0.9, "repetition_penalty": 1.2},
    "t0.5_k40_rp1.2": {"temperature": 0.5, "top_k": 40, "repetition_penalty": 1.2},
}


def restates_question(question: str, reply: str) -> bool:
    """Reply contains >= 60% of the question's words (question of >= 4 words)."""
    q = set(question.lower().rstrip("?.!").split())
    return len(q) >= 4 and len(q & set(reply.lower().split())) / len(q) >= 0.6


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = aggregate(rows)
    words = [r["words"] for r in rows]
    agg.update(
        {
            "median_words": st.median(words),
            "empty_replies": sum(not r["reply"].strip() for r in rows),
            "structural_stops": sum(r["stop_reason"] == "structural_token" for r in rows),
            "restates_question": sum(r["restates"] for r in rows),
            "mean_latency_ms": round(st.mean(r["latency_ms"] for r in rows), 1),
        }
    )
    return agg


def run(checkpoint: Path, label: str, out: Path, settings: list[str]) -> dict[str, Any]:
    device = select_device("auto")
    model, _ = load_model(checkpoint, device)
    tok = ByteBPETokenizer.load(checkpoint.resolve().parent.parent / "tokenizer.json")
    suites = {k: json.loads(p.read_text())["items"] for k, p in SUITES.items()}
    results: dict[str, Any] = {"checkpoint": str(checkpoint), "label": label, "settings": {}}
    for name in settings:
        cfg = dict(SETTINGS[name])
        seeds = (None,) if cfg.get("greedy") else SEEDS
        rows: list[dict[str, Any]] = []
        for suite, items in suites.items():
            for seed in seeds:
                for item in items:
                    t0 = time.perf_counter()
                    r = chat_reply(model, tok, item["turns"], max_new_tokens=200, seed=seed,
                                   temperature=cfg.get("temperature", 0.7),
                                   top_k=cfg.get("top_k", 40), top_p=cfg.get("top_p"),
                                   greedy=cfg.get("greedy", False),
                                   repetition_penalty=cfg["repetition_penalty"])  # fmt: skip
                    row = score_reply(item, r)
                    row["latency_ms"] = (time.perf_counter() - t0) * 1000
                    row["suite"] = suite
                    row["restates"] = restates_question(item["turns"][-1]["content"], r.text)
                    rows.append(row)
        summary = {"all": summarize(rows)}
        for suite in suites:
            summary[suite] = summarize([r for r in rows if r["suite"] == suite])
        results["settings"][name] = {"config": cfg, "summary": summary, "rows": rows}
        a = summary["all"]
        print(f"{label} {name}: eot {a['natural_end_of_turn']['rate']:.3f} "
              f"rep {a['repetitive_over_0.3']['rate']:.3f} both {a['both_ok']['correct']}/"
              f"{a['both_ok']['n']} words~{a['median_words']} restate {a['restates_question']}",
              flush=True)  # fmt: skip
    results["selection"] = select(results["settings"])
    atomic_write_json(out / f"decoding_{label}.json", results)
    return results


def select(settings: dict[str, Any]) -> dict[str, Any]:
    table = {}
    for name, s in settings.items():
        a = s["summary"]["all"]
        k, n = a["both_ok"]["correct"], a["both_ok"]["n"]
        table[name] = {
            "eligible": a["natural_end_of_turn"]["rate"] >= 0.95
            and a["repetitive_over_0.3"]["rate"] <= 0.05,
            "both_ok": k,
            "n": n,
            "rate": k / n,
            "ci95": list(wilson(k, n)),
        }
    eligible = [m for m, v in table.items() if v["eligible"]]
    best = max(eligible, key=lambda m: table[m]["rate"]) if eligible else None
    chosen, reason = DEFAULT, "no eligible setting"
    if best is not None and DEFAULT in table:
        b, d = table[best], table[DEFAULT]
        if best != DEFAULT and b["rate"] > d["rate"] and b["ci95"][0] > d["ci95"][1]:
            chosen, reason = best, "significantly better than the default"
        elif best == DEFAULT:
            reason = "the default is the best eligible setting"
        else:
            reason = (
                f"best eligible setting {best} ({b['rate']:.3f}) is not significantly better than "
                f"the default ({d['rate']:.3f}); intervals overlap"
            )
    return {"table": table, "best_eligible": best, "chosen": chosen, "reason": reason}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--out", type=Path, default=Path("docs/reports/m6_1"))
    p.add_argument("--settings", nargs="*", default=list(SETTINGS))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res = run(args.checkpoint, args.label, args.out, args.settings)
    print(json.dumps(res["selection"], indent=1))


if __name__ == "__main__":
    main()

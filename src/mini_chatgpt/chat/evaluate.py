"""Scoring for the fixed conversational suite ``evals/chat_v1.json`` (M5).

Each reply gets:

* **format_ok**: stopped with ``<|end|>`` (``stop_reason == "end_of_turn"``) and the
  displayed reply is non-empty. Truncation at the token limit is *not* success.
* **content_ok** (items with checks only): every rule in ``checks`` holds, using
  case-insensitive substring tests and simple counts (documented in the suite file).
* surface metrics (repetition) from ``general_eval.surface_metrics``.

Content is checked on the displayed text whether or not the reply terminated, and
reported both alone and combined with format (``both_ok``).
"""

from __future__ import annotations

import re
from typing import Any

from mini_chatgpt.analysis.general_eval import SENT_END, WORD, _rate, surface_metrics
from mini_chatgpt.chat.generate import ChatReply

LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)
REPETITIVE = 0.3  # repeated-4-gram fraction above which a reply counts as repetitive


def check_content(text: str, checks: dict[str, Any]) -> dict[str, bool]:
    low = text.lower()
    out: dict[str, bool] = {}
    if "include_any" in checks:
        out["include_any"] = any(s.lower() in low for s in checks["include_any"])
    if "include_all" in checks:
        out["include_all"] = all(s.lower() in low for s in checks["include_all"])
    if "exclude_any" in checks:
        out["exclude_any"] = not any(s.lower() in low for s in checks["exclude_any"])
    if "max_words" in checks:
        out["max_words"] = (
            len([w for w in WORD.findall(text) if w[0].isalnum()]) <= checks["max_words"]
        )
    if "max_sentences" in checks:
        out["max_sentences"] = len(SENT_END.findall(text)) <= checks["max_sentences"]
    if "min_list_items" in checks:
        out["min_list_items"] = len(LIST_ITEM.findall(text)) >= checks["min_list_items"]
    return out


def score_reply(item: dict[str, Any], reply: ChatReply) -> dict[str, Any]:
    checks = item.get("checks", {})
    detail = check_content(reply.text, checks)
    fmt_ok = reply.ended_naturally and bool(reply.text)
    content_ok = all(detail.values()) if checks else None
    surf = surface_metrics(reply.text)
    return {
        "id": item["id"],
        "category": item["category"],
        "seed": reply.settings.get("seed"),
        "reply": reply.text,
        "stop_reason": reply.stop_reason,
        "new_tokens": len(reply.ids),
        "format_ok": fmt_ok,
        "content_ok": content_ok,
        "both_ok": (fmt_ok and content_ok) if content_ok is not None else None,
        "checks": detail,
        "repetitive": surf["repeated_4gram_fraction"] > REPETITIVE,
        **surf,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rates with 95% Wilson intervals, overall and per category."""
    n = len(rows)

    def rate(key: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        vals = [r[key] for r in subset if r[key] is not None]
        return _rate(sum(bool(v) for v in vals), len(vals))

    stops: dict[str, int] = {}
    for r in rows:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    cats = sorted({r["category"] for r in rows})
    return {
        "replies": n,
        "natural_end_of_turn": _rate(stops.get("end_of_turn", 0), n),
        "stop_reasons": stops,
        "format_ok": rate("format_ok", rows),
        "repetitive_over_0.3": rate("repetitive", rows),
        "mean_repeated_4gram_fraction": sum(r["repeated_4gram_fraction"] for r in rows) / n,
        "mean_words": sum(r["words"] for r in rows) / n,
        "mean_new_tokens": sum(r["new_tokens"] for r in rows) / n,
        "content_ok": rate("content_ok", rows),
        "both_ok": rate("both_ok", rows),
        "by_category": {
            c: {
                "content_ok": rate("content_ok", [r for r in rows if r["category"] == c]),
                "format_ok": rate("format_ok", [r for r in rows if r["category"] == c]),
                "both_ok": rate("both_ok", [r for r in rows if r["category"] == c]),
            }
            for c in cats
        },
    }

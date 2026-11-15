"""Write M5 chat transcripts verbatim from the evaluation JSON (no manual editing).

Selections are fixed by rule and were written before the SFT outputs existed:

* ``FIXED``: five transcripts for the report and completion summary
  (qa01, ins01, mt01, mt04, op01), greedy decoding, base vs SFT.
* Behaviour examples: the first SFT reply (settings greedy, greedy_rp, sampled,
  sampled_rp in that order; items in suite order) that matches each rule: successful
  answer, wrong answer, repeated text, failed instruction, successful stop, failed stop.
  If no reply matches a rule, that is reported instead of an example.
* Repetition penalty: the same five fixed items, sampled seed 11, penalty off vs on.
* All 40 greedy SFT replies, and all 8 multi-turn items, in ``transcripts_all.md``.

Usage: python scripts/m5_transcripts.py docs/reports/m5
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

SUITE = Path("evals/chat_v1.json")
FIXED = ("qa01", "ins01", "mt01", "mt04", "op01")
Row = dict[str, Any]
RULES: list[tuple[str, Callable[[Row], bool]]] = [
    ("Successful answer", lambda r: r["category"] == "qa" and bool(r["both_ok"])),
    (
        "Wrong answer (well-formed but incorrect)",
        lambda r: r["category"] == "qa" and r["format_ok"] and r["content_ok"] is False,
    ),
    ("Repeated text", lambda r: r["repetitive"]),
    (
        "Failed instruction following",
        lambda r: r["category"] == "instruction" and r["content_ok"] is False,
    ),
    (
        "Successful stopping (open question)",
        lambda r: r["category"] == "open" and r["stop_reason"] == "end_of_turn",
    ),
    ("Failed stopping", lambda r: r["stop_reason"] != "end_of_turn"),
]


def fence(text: str) -> str:
    return "````text\n" + (text if text else "(empty)") + "\n````\n"


def transcript(item: Row, row: Row, label: str = "") -> str:
    lines = [
        f"**{item['id']}** ({item['category']}){' — ' + label if label else ''}"
        f" · stop: `{row['stop_reason']}` · {row['new_tokens']} tokens"
        f" · content check: {row['content_ok']}\n"
    ]
    for t in item["turns"]:
        lines.append(f"*{t['role']}*:\n\n" + fence(t["content"]))
    lines.append("*assistant (model)*:\n\n" + fence(row["reply"]))
    return "\n".join(lines)


def main() -> None:
    out = Path(sys.argv[1])
    items = {i["id"]: i for i in json.loads(SUITE.read_text())["items"]}
    base = json.loads((out / "chat_eval_base_m4_1.json").read_text())
    sft = json.loads((out / "chat_eval_sft_best.json").read_text())

    def rows(res: dict[str, Any], setting: str, seed: int | None = None) -> dict[str, Row]:
        return {r["id"]: r for r in res[setting]["rows"] if seed is None or r["seed"] == seed}

    sg, bg = rows(sft, "greedy"), rows(base, "greedy")
    md = ["\n#### Fixed transcripts: base model vs SFT (greedy)\n"]
    for i in FIXED:
        md.append(transcript(items[i], bg[i], "M4.1 base"))
        md.append(transcript(items[i], sg[i], "M5 SFT"))
    md.append("\n#### Behaviour examples (SFT; first match in suite order)\n")
    md.append(
        "Settings are searched in the fixed order greedy, greedy_rp, sampled, sampled_rp "
        "(seed order 11, 22, 33); the first reply matching each rule is shown.\n"
    )
    for label, rule in RULES:
        hit, where = None, ""
        for setting in ("greedy", "greedy_rp", "sampled", "sampled_rp"):
            hit = next((r for r in sft[setting]["rows"] if rule(r)), None)
            if hit:
                where = f"{label} [{setting}" + (f", seed {hit['seed']}]" if hit["seed"] else "]")
                break
        md.append(
            transcript(items[hit["id"]], hit, where)
            if hit
            else f"**{label}**: no SFT reply matched this rule.\n"
        )
    md.append("\n#### Repetition penalty off vs on (SFT, sampled T=0.7, top-k 40, seed 11)\n")
    off, on = rows(sft, "sampled", 11), rows(sft, "sampled_rp", 11)
    for i in FIXED:
        md.append(transcript(items[i], off[i], "penalty 1.0 (off)"))
        md.append(transcript(items[i], on[i], "penalty 1.2"))
    (out / "transcripts_excerpt.md").write_text("\n".join(md), encoding="utf-8")

    full = [
        "# M5 transcripts (verbatim)\n",
        "All 40 chat_v1 items, greedy decoding, M5 SFT best checkpoint, then the base "
        "model. Copied programmatically from `chat_eval_*.json`.\n",
    ]
    for name, res in (("M5 SFT", sg), ("M4.1 base", bg)):
        full.append(f"\n## {name}\n")
        full += [transcript(items[i], res[i]) for i in items]
    (out / "transcripts_all.md").write_text("\n".join(full), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Verbatim M6.2 samples (rule-based selection, no editing).

* ``samples_excerpt.md``: for each free-generation section, the M6.2 seed-11 sample with the
  lowest, median and highest repeated-4-gram fraction (the M4.1 report's rule), and the
  first five continuation prompts side by side for M4.1 and M6.2 (seed 11).
* ``samples.md``: every seed-11 generation of both models and all greedy code completions.

Usage: python scripts/m6_2_samples.py docs/reports/m6_2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SUITE = Path("evals/general_v1.json")
SECTIONS = ("continuation", "explanations", "repetition_stress")


def fence(text: str) -> str:
    return "````text\n" + text + "\n````\n"


def head(item: dict[str, Any]) -> str:
    return (
        f"**{item['id']}** (seed {item['seed']}, repeated 4-gram fraction "
        f"{item['repeated_4gram_fraction']:.2f}, "
        f"ended with end-of-text: {item['ended_with_eot']})\n\n"
    )


def main() -> None:
    out = Path(sys.argv[1])
    res = {
        m: json.loads((out / f"general_eval_{m}.json").read_text())
        for m in ("m4_1_final", "m6_2_final")
    }
    suite = json.loads(SUITE.read_text())["sections"]
    excerpt: list[str] = []
    full = ["# M6.2 vs M4.1 generated samples (verbatim, seed 11)\n"]
    for sec in SECTIONS:
        prompts = {i["id"]: i["prompt"] for i in suite[sec]["items"]}
        new = [i for i in res["m6_2_final"]["next_token"][sec]["items"] if i["seed"] == 11]
        old = {i["id"]: i for i in res["m4_1_final"]["next_token"][sec]["items"] if i["seed"] == 11}
        ranked = sorted(new, key=lambda i: (i["repeated_4gram_fraction"], i["id"]))
        picks = {
            "least repetitive": ranked[0],
            "median": ranked[len(ranked) // 2],
            "most repetitive": ranked[-1],
        }
        excerpt.append(f"\n#### M6.2, {sec}\n")
        for label, i in picks.items():
            excerpt.append(
                f"*{label}*: "
                + head(i)
                + "Prompt:\n\n"
                + fence(prompts[i["id"]])
                + "\nContinuation:\n\n"
                + fence(i["text"])
            )
        full.append(f"\n## {sec}\n")
        for i in new:
            full.append(
                f"### {i['id']}\n\nPrompt:\n\n"
                + fence(prompts[i["id"]])
                + "\nM4.1 (14.0M): "
                + head(old[i["id"]])
                + fence(old[i["id"]]["text"])
                + "\nM6.2 (29.7M): "
                + head(i)
                + fence(i["text"])
            )
    excerpt.append("\n#### Side by side: first five continuation prompts (seed 11)\n")
    prompts = {i["id"]: i["prompt"] for i in suite["continuation"]["items"]}
    rows = {
        m: {i["id"]: i for i in res[m]["next_token"]["continuation"]["items"] if i["seed"] == 11}
        for m in res
    }
    for cid in list(prompts)[:5]:
        excerpt.append(
            f"**{cid}**: prompt `{prompts[cid]}`\n\n*M4.1:*\n\n"
            + fence(rows["m4_1_final"][cid]["text"])
            + "\n*M6.2:*\n\n"
            + fence(rows["m6_2_final"][cid]["text"])
        )
    code = {i["id"]: i["prompt"] for i in suite["code_generation"]["items"]}
    full.append("\n## code_generation (greedy, 16 tokens)\n")
    for m in res:
        full.append(f"\n### {m}\n")
        for it in res[m]["next_token"]["code_generation"]["items"]:
            full.append(
                f"**{it['id']}** (correct: {it['correct']})\n\n"
                + fence(code[it["id"]] + it["completion"])
            )
    (out / "samples_excerpt.md").write_text("\n".join(excerpt), encoding="utf-8")
    (out / "samples.md").write_text("\n".join(full), encoding="utf-8")


if __name__ == "__main__":
    main()

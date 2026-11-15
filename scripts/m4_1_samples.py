"""Write the M4.1 generated samples verbatim to Markdown (no manual selection or editing).

Selection is rule-based so nothing is cherry-picked:

* ``samples.md`` holds every seed-11 generation of the final checkpoint for the
  continuation, explanations and repetition-stress sections, every chat reply, and every
  greedy code completion, exactly as stored in ``general_eval_m4_1_final.json``.
* ``samples_excerpt.md`` (embedded in the report) holds, per free-generation section,
  the seed-11 sample with the lowest, median and highest repeated-4-gram fraction, so the
  report shows the best, typical and worst case rather than a chosen favourite.

Usage: python scripts/m4_1_samples.py docs/reports/m4_1
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


def block(item: dict[str, Any], prompt: str) -> str:
    head = (
        f"**{item['id']}** (seed {item['seed']}, repeated 4-gram fraction "
        f"{item['repeated_4gram_fraction']:.2f}, "
        f"ended with end-of-text: {item['ended_with_eot']})\n\n"
    )
    return head + "Prompt:\n\n" + fence(prompt) + "\nModel continuation:\n\n" + fence(item["text"])


def main() -> None:
    out = Path(sys.argv[1])
    res = json.loads((out / "general_eval_m4_1_final.json").read_text())
    suite = json.loads(SUITE.read_text())["sections"]
    nt = res["next_token"]
    full = ["# M4.1 generated samples (verbatim)\n",
            f"Checkpoint: `{res['checkpoint']}` (sha256 `{res['checkpoint_sha256'][:16]}…`). "
            "Sampling: temperature 0.7, top-k 40, 200 new tokens, seed 11. "
            "Text is copied programmatically from `general_eval_m4_1_final.json`.\n"]  # fmt: skip
    excerpt = []
    for sec in SECTIONS:
        prompts = {i["id"]: i["prompt"] for i in suite[sec]["items"]}
        items = [i for i in nt[sec]["items"] if i["seed"] == 11]
        full.append(f"\n## {sec}\n")
        full += [block(i, prompts[i["id"]]) for i in items]
        ranked = sorted(items, key=lambda i: (i["repeated_4gram_fraction"], i["id"]))
        picks = {"least repetitive": ranked[0], "median": ranked[len(ranked) // 2],
                 "most repetitive": ranked[-1]}  # fmt: skip
        excerpt.append(f"\n#### {sec}\n")
        for label, i in picks.items():
            excerpt.append(f"*{label}*: " + block(i, prompts[i["id"]]))
    chat_prompts = {i["id"]: i["turns"][-1]["content"] for i in suite["chat"]["items"]}
    full.append("\n## chat (base model, chat tokens never seen in training)\n")
    for it in res["instruction"]["chat"]["items"]:
        full.append(f"**{it['id']}** (format ok: {it['format_ok']}, content ok: "
                    f"{it['content_ok']})\n\nUser:\n\n" + fence(chat_prompts[it["id"]])
                    + "\nReply:\n\n" + fence(it["reply"]))  # fmt: skip
    code_prompts = {i["id"]: i["prompt"] for i in suite["code_generation"]["items"]}
    full.append("\n## code_generation (greedy, 16 tokens)\n")
    for it in nt["code_generation"]["items"]:
        full.append(f"**{it['id']}** (correct: {it['correct']})\n\n"
                    + fence(code_prompts[it["id"]] + it["completion"]))  # fmt: skip
    (out / "samples.md").write_text("\n".join(full), encoding="utf-8")
    (out / "samples_excerpt.md").write_text("\n".join(excerpt), encoding="utf-8")


if __name__ == "__main__":
    main()

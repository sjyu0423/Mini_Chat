"""Build the web app's showcase data from committed evidence (no model is run here).

Writes ``apps/chat-web/src/showcase/data.json`` containing:

* ``recorded``: real, unedited replies from the M6.3 matched evaluation
  (``docs/reports/m6_3/decoding_{m5,m6_3}.json``: production decoding T 0.7, top-k 40,
  penalty 1.2, seed 11) for a fixed list of prompts chosen to show both successes and
  failures. Used as the fallback when no live model server is reachable, and by static builds.
* ``facts``: the numbers shown on the project page, copied from ``docs/showcase/facts.json``
  so every value on screen is traceable to a report.

Usage: python scripts/build_showcase_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path("apps/chat-web/src/showcase/data.json")
SETTING = "t0.7_k40_rp1.2"
# Fixed selection: two correct-ish, the rest typical failures, across categories.
ITEMS = [
    ("diag_v1", "df01"),  # sky colour (both models correct)
    ("chat_v1", "qa01"),  # largest planet
    ("diag_v1", "df03"),  # capital of France
    ("diag_v1", "di01"),  # yes/no instruction
    ("diag_v1", "de02"),  # cat's name (entity)
    ("diag_v1", "dm01"),  # topic switch in multi-turn
    ("diag_v1", "do04"),  # open: how to make tea
    ("chat_v1", "op01"),  # open: why exercise
]


def main() -> None:
    suites = {
        "chat_v1": {
            i["id"]: i for i in json.loads(Path("evals/chat_v1.json").read_text())["items"]
        },
        "diag_v1": {
            i["id"]: i for i in json.loads(Path("evals/chat_diag_v1.json").read_text())["items"]
        },
    }
    rows = {
        m: json.loads(Path(f"docs/reports/m6_3/decoding_{m}.json").read_text())["settings"][
            SETTING
        ]["rows"]
        for m in ("m5", "m6_3")
    }
    recorded: list[dict[str, Any]] = []
    for suite, iid in ITEMS:
        replies = {}
        for m in ("m5", "m6_3"):
            r = next(
                x for x in rows[m] if x["suite"] == suite and x["id"] == iid and x["seed"] == 11
            )
            replies[m] = {
                "reply": r["reply"],
                "stop_reason": r["stop_reason"],
                "new_tokens": r["new_tokens"],
                "content_check": r["content_ok"],
            }
        recorded.append(
            {
                "id": f"{suite}:{iid}",
                "category": suites[suite][iid]["category"],
                "turns": suites[suite][iid]["turns"],
                "replies": replies,
            }
        )
    facts = {f["id"]: f for f in json.loads(Path("docs/showcase/facts.json").read_text())["facts"]}
    data = {
        "source": {
            "recorded": "docs/reports/m6_3/decoding_{m5,m6_3}.json (seed 11, T 0.7, top-k 40, "
            "repetition penalty 1.2), copied verbatim",
            "facts": "docs/showcase/facts.json",
        },
        "recorded": recorded,
        "facts": {k: {"value": v["value"], "label": v["label"]} for k, v in facts.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(recorded)} recorded conversations, {len(facts)} facts")


if __name__ == "__main__":
    main()

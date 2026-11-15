"""Before/after M6.1 transcripts, copied verbatim from the decoding JSON files.

Fixed selection (independent of results): the five M5 transcript items (qa01, ins01,
mt01, mt04, op01) and the first item of every diagnostic category (df01, di01, de01, dm01,
do01); seed 11 of the default decoding (t0.7_k40_rp1.2) and greedy, for M5 and for the
capped-reply SFT model.

Usage: python scripts/m6_1_transcripts.py docs/reports/m6_1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

IDS = ("qa01", "ins01", "mt01", "mt04", "op01", "df01", "di01", "de01", "dm01", "do01")
SETTINGS = (("t0.7_k40_rp1.2", 11), ("greedy", None))


def fence(text: str) -> str:
    return "````text\n" + (text or "(empty)") + "\n````\n"


def main() -> None:
    out = Path(sys.argv[1])
    items = {
        i["id"]: i
        for p in ("evals/chat_v1.json", "evals/chat_diag_v1.json")
        for i in json.loads(Path(p).read_text())["items"]
    }
    res = {m: json.loads((out / f"decoding_{m}.json").read_text()) for m in ("m5", "cap100")}

    def row(model: str, setting: str, seed: int | None, item: str) -> dict[str, Any]:
        rows = res[model]["settings"][setting]["rows"]
        return next(r for r in rows if r["id"] == item and r["seed"] == seed)

    md = []
    for setting, seed in SETTINGS:
        md.append(f"\n#### Decoding `{setting}`" + (f", seed {seed}" if seed else "") + "\n")
        for i in IDS:
            it = items[i]
            md.append(f"**{i}** ({it['category']})\n")
            for t in it["turns"]:
                md.append(f"*{t['role']}*:\n\n" + fence(t["content"]))
            for model, label in (("m5", "M5"), ("cap100", "M6.1 cap100")):
                r = row(model, setting, seed, i)
                md.append(
                    f"*{label}* · stop `{r['stop_reason']}` · {r['new_tokens']} tokens · "
                    f"content check {r['content_ok']}:\n\n" + fence(r["reply"])
                )
    (out / "transcripts.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

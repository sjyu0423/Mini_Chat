"""Unblinded M6.3 transcripts: the 20 pre-registered blind-review items, M5 vs M6.3, verbatim.

Reads ``blind_key.json``, ``blind_grades.json`` and both decoding files (seed 11, production
decoding) and writes ``transcripts.md``. Run only after grading (``evaluate_m6_3.py unblind``).

Usage: python scripts/m6_3_transcripts.py docs/reports/m6_3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SETTING = "t0.7_k40_rp1.2"


def main() -> None:
    out = Path(sys.argv[1])
    key = json.loads((out / "blind_key.json").read_text())
    grades = json.loads((out / "blind_grades.json").read_text())
    rows = {
        m: json.loads((out / f"decoding_{m}.json").read_text())["settings"][SETTING]["rows"]
        for m in ("m5", "m6_3")
    }
    suites = {
        "chat_v1": json.loads(Path("evals/chat_v1.json").read_text())["items"],
        "diag_v1": json.loads(Path("evals/chat_diag_v1.json").read_text())["items"],
    }
    turns = {(s, i["id"]): i["turns"] for s, items in suites.items() for i in items}
    names = {"m5": "M5 (14.0M)", "m6_3": "M6.3 (29.7M)"}
    md = [
        "# M6.3 matched transcripts (unblinded, verbatim; seed 11, T 0.7, top-k 40, penalty 1.2)\n"
    ]
    for n, k in key.items():
        md.append(f"\n## {n}. {k['suite']} / {k['id']}\n")
        for t in turns[(k["suite"], k["id"])]:
            md.append(f"*{t['role']}:* {t['content']}\n")
        label_of = {k["A"]: "A", k["B"]: "B"}
        for m in ("m5", "m6_3"):
            r = next(
                x
                for x in rows[m]
                if x["suite"] == k["suite"] and x["id"] == k["id"] and x["seed"] == 11
            )
            md.append(
                f"\n**{names[m]}** · stop `{r['stop_reason']}` · {r['new_tokens']} tokens · "
                f"blind grade `{grades[n][label_of[m]]}`\n\n"
                f"````text\n{r['reply'] or '(empty)'}\n````\n"
            )
        pref = grades[n]["preference"]
        md.append(
            f"\nBlind preference: **{'tie' if pref == 'tie' else names[k[pref]]}**. "
            f"Note: {grades[n].get('note', '')}\n"
        )
    (out / "transcripts.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

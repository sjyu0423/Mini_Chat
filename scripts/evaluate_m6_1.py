"""M6.1 comparison: M5 vs the capped-reply SFT model, both with the selected decoding.

Run after ``scripts/m6_1_decoding.py`` has produced ``decoding_m5.json`` and
``decoding_cap100.json``. This script adds:

1. assistant-token validation loss of BOTH checkpoints on BOTH validation sets (M5's full
   validation split and the capped M6.1 split) with ``chat.sft.evaluate``;
2. the frozen general suite + held-out bits/byte for the new checkpoint
   (``scripts/evaluate_general.py``, same documents and flags as M5's evaluation, whose
   result ``docs/reports/m5/general_eval_sft_best.json`` is reused for M5);
3. a side-by-side summary for each decoding setting, per suite and category, with 95%
   Wilson intervals, written to ``comparison.json``.

Usage: python scripts/evaluate_m6_1.py runs/<m6_1 run> --out docs/reports/m6_1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mini_chatgpt.analysis.general_eval import wilson
from mini_chatgpt.chat.format import ChatTokens
from mini_chatgpt.chat.sft import evaluate, load_split
from mini_chatgpt.device import select_device
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json
from mini_chatgpt.training.trainer import load_model

M5 = "runs/m5_sft_chat-20261114-175334/checkpoints/best.pt"
PROC = Path("data/processed/general_m4_1_bpe/heldout_text")
HELDOUT = {
    "fineweb_edu_val": PROC / "fineweb_edu_val",
    "cosmopedia_v2_val": PROC / "cosmopedia_v2_val",
    "wikipedia_en_ood": Path("data/raw/samples/wikipedia_en"),
    "tinystories_v2_val": PROC / "tinystories_v2_val",
}
VAL_SETS = {"m5_val": "data/processed/chat_m5", "m6_1_val": "data/processed/chat_m6_1"}


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "ci95": list(wilson(k, n))}


def compare_decoding(out: Path) -> dict[str, Any]:
    res = {m: json.loads((out / f"decoding_{m}.json").read_text()) for m in ("m5", "cap100")}
    table: dict[str, Any] = {}
    for setting in res["m5"]["settings"]:
        row: dict[str, Any] = {}
        for model, r in res.items():
            if setting not in r["settings"]:
                continue
            s = r["settings"][setting]["summary"]
            a = s["all"]
            cats: dict[str, Any] = {}
            for suite in ("chat_v1", "diag_v1"):
                for cat, v in s[suite]["by_category"].items():
                    c = v["content_ok"]
                    if c["n"]:
                        cats[f"{suite}:{cat}"] = rate(c["correct"], c["n"])
            row[model] = {
                "natural_end_of_turn": rate(
                    a["natural_end_of_turn"]["correct"], a["natural_end_of_turn"]["n"]
                ),
                "repetitive": rate(a["repetitive_over_0.3"]["correct"], a["replies"]),
                "both_ok": rate(a["both_ok"]["correct"], a["both_ok"]["n"]),
                "content_ok": rate(a["content_ok"]["correct"], a["content_ok"]["n"]),
                "median_words": a["median_words"],
                "mean_latency_ms": a["mean_latency_ms"],
                "restates_question": a["restates_question"],
                "empty_replies": a["empty_replies"],
                "structural_stops": a["structural_stops"],
                "content_by_category": cats,
            }
        table[setting] = row
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=Path("docs/reports/m6_1"))
    args = p.parse_args()
    new = str(args.run / "checkpoints" / "best.pt")
    device = select_device("auto")
    tok = ByteBPETokenizer.load(args.run / "tokenizer.json")
    end = ChatTokens.from_tokenizer(tok).end

    cross: dict[str, Any] = {}
    for label, ckpt in (("m5", M5), ("cap100", new)):
        model, _ = load_model(ckpt, device)
        cross[label] = {
            name: evaluate(model, load_split(d, "val"), end, device) for name, d in VAL_SETS.items()
        }
        del model

    gen_out = args.out / "general_eval_cap100.json"
    cmd = [sys.executable, "scripts/evaluate_general.py", new, "--out", str(gen_out),
           "--heldout-docs", "400",
           "--tokenizer", "data/processed/general_m4_1_bpe/tokenizer.json"]  # fmt: skip
    for name, d in HELDOUT.items():
        cmd += ["--heldout", f"{name}={d}"]
    subprocess.run(cmd, check=True)
    general = {}
    for label, path in (
        ("base_m4_1", Path("docs/reports/m5/general_eval_base_m4_1.json")),
        ("m5", Path("docs/reports/m5/general_eval_sft_best.json")),
        ("cap100", gen_out),
    ):
        g = json.loads(path.read_text())
        nt = g["next_token"]
        general[label] = {
            "heldout_bpb": {k: v["bits_per_byte"] for k, v in g["heldout"].items()},
            **{k: nt[k]["acc_norm"] for k in ("facts_mc", "qa_mc", "code_mc")},
            "grammar_pairs": nt["grammar_pairs"]["acc"],
            "code_generation": nt["code_generation"]["acc"],
            "chat_section": {k: g["instruction"]["chat"][k] for k in ("format", "content", "both")},
        }
    result = {
        "checkpoints": {"m5": M5, "cap100": new},
        "validation_loss": cross,
        "general": general,
        "decoding": compare_decoding(args.out),
    }
    atomic_write_json(args.out / "comparison.json", result)
    print(json.dumps({"validation_loss": cross, "general": general}, indent=1, default=str))


if __name__ == "__main__":
    main()

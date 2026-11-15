"""M6.2 base-model evaluation: M4.1 (14.0M) vs M6.2 (29.7M), identical data and scoring.

1. ``scripts/evaluate_general.py`` (frozen ``evals/general_v1.json`` + held-out bits per byte
   on the same 400-document samples M4.1 was evaluated on) for the M4.1 final checkpoint and
   the M6.2 final and best checkpoints. M4.1 is re-run here so both models are scored by the
   same code in the same session.
2. Reproducibility: the M6.2 final checkpoint is scored twice; deterministic sections
   (multiple choice, minimal pairs, greedy code) must match exactly.
3. Memorization: every generated sample of both models is re-encoded and checked for
   verbatim k-gram overlap (k = 8, 16, 32, 48 tokens) with ``train.bin``.
4. Paired comparison of per-item multiple-choice / minimal-pair / code results (McNemar).

Usage: python scripts/evaluate_m6_2.py runs/<m6_2 run> --out docs/reports/m6_2
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mini_chatgpt.analysis.story_eval import KGramIndex
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json

PROC = Path("data/processed/general_m4_1_bpe")
HELD = PROC / "heldout_text"
HELDOUT = {
    "fineweb_edu_val": HELD / "fineweb_edu_val",
    "cosmopedia_v2_val": HELD / "cosmopedia_v2_val",
    "fineweb_edu_holdout": HELD / "fineweb_edu_holdout",
    "cosmopedia_v2_holdout": HELD / "cosmopedia_v2_holdout",
    "wikipedia_en_ood": Path("data/raw/samples/wikipedia_en"),
    "tinystories_v2_val": HELD / "tinystories_v2_val",
}
M41 = "runs/m4_1_general_s_bpe8192-20261114-190514/checkpoints/step_00009600.pt"
DETERMINISTIC = ("facts_mc", "qa_mc", "code_mc", "grammar_pairs", "code_generation")
GEN_SECTIONS = ("continuation", "explanations", "repetition_stress")


def run_suite(ckpt: str, out: Path) -> dict[str, Any]:
    cmd = [sys.executable, "scripts/evaluate_general.py", ckpt, "--out", str(out),
           "--heldout-docs", "400", "--tokenizer", str(PROC / "tokenizer.json")]  # fmt: skip
    for name, d in HELDOUT.items():
        cmd += ["--heldout", f"{name}={d}"]
    subprocess.run(cmd, check=True)
    result: dict[str, Any] = json.loads(out.read_text())
    return result


def item_correct(r: dict[str, Any], section: str) -> list[bool]:
    items = r["next_token"][section]["items"]
    if section in ("facts_mc", "qa_mc", "code_mc"):
        return [i["pred_norm"] == i["answer"] for i in items]
    return [bool(i["correct"]) for i in items]


def mcnemar(a: list[bool], b: list[bool]) -> dict[str, Any]:
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_m4_1": only_a, "only_m6_2": only_b, "p_two_sided": p}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/reports/m6_2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ck = args.run / "checkpoints"
    final = sorted(ck.glob("step_*.pt"))[-1]
    models = {"m4_1_final": M41, "m6_2_final": str(final), "m6_2_best": str(ck / "best.pt")}
    res = {
        name: run_suite(path, args.out / f"general_eval_{name}.json")
        for name, path in models.items()
    }
    again = run_suite(str(final), args.out / "general_eval_m6_2_final_repeat.json")
    repro = all(item_correct(again, s) == item_correct(res["m6_2_final"], s) for s in DETERMINISTIC)

    paired = {
        s: mcnemar(item_correct(res["m4_1_final"], s), item_correct(res["m6_2_final"], s))
        for s in DETERMINISTIC
    }

    tok = ByteBPETokenizer.load(PROC / "tokenizer.json")
    index = KGramIndex(
        np.asarray(np.memmap(PROC / "train.bin", dtype="<u2", mode="r")), [8, 16, 32, 48]
    )
    mem: dict[str, Any] = {}
    for name in ("m4_1_final", "m6_2_final"):
        rows = []
        for section in GEN_SECTIONS:
            for item in res[name]["next_token"][section]["items"]:
                ids = np.asarray(tok.encode_ordinary(item["text"]), dtype=np.int64)
                rows.append(index.overlap(ids))
        fr = [r["frac_16gram_in_train"] for r in rows if r["frac_16gram_in_train"] is not None]
        mem[name] = {
            "samples": len(rows),
            "mean_frac_16gram_in_train": float(np.mean(fr)),
            "samples_with_32gram_match": sum(r["longest_match_at_least"] >= 32 for r in rows),
            "samples_with_48gram_match": sum(r["longest_match_at_least"] >= 48 for r in rows),
        }
    summary = {
        "checkpoints": models,
        "reproducible_deterministic_sections": repro,
        "paired_mcnemar_m4_1_vs_m6_2_final": paired,
        "memorization": mem,
    }
    atomic_write_json(args.out / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

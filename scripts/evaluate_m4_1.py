"""M4.1 base-model evaluation (evaluation only; nothing is trained).

1. Runs ``scripts/evaluate_general.py`` (the frozen ``evals/general_v1.json`` suite plus
   held-out bits per byte) on the M4.1 final and best checkpoints and, as baselines on
   the *same* held-out documents, the M3.1 and M3.3 TinyStories checkpoints. Bits per
   byte is tokenizer-independent, so these numbers are directly comparable.
2. Reproducibility: the M4.1 final checkpoint is scored twice; multiple-choice,
   minimal-pair and greedy code results must be identical.
3. Memorization: every M4.1 generation (continuation, explanations, stress prompts) is
   re-encoded and checked for verbatim k-gram overlap (k = 8, 16, 32, 48 tokens) with
   the training stream that the model actually saw (``train.bin``).

Usage: python scripts/evaluate_m4_1.py runs/<m4_1 run> --out docs/reports/m4_1
"""

from __future__ import annotations

import argparse
import json
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
BASELINES = {
    "m3_1_s12m_tinystories": "runs/ts_s_bpe4096-20261113-172905/checkpoints/step_00004800.pt",
    "m3_3_m28m_tinystories": "runs/ts_m_bpe4096-20261114-042121/checkpoints/step_00004800.pt",
}
DETERMINISTIC = ("facts_mc", "qa_mc", "code_mc", "grammar_pairs", "code_generation")


def run_suite(ckpt: str, out: Path, docs: int) -> dict[str, Any]:
    cmd = [sys.executable, "scripts/evaluate_general.py", ckpt, "--out", str(out),
           "--heldout-docs", str(docs)]  # fmt: skip
    for name, d in HELDOUT.items():
        cmd += ["--heldout", f"{name}={d}"]
    subprocess.run(cmd, check=True)
    result: dict[str, Any] = json.loads(out.read_text())
    return result


def deterministic_view(r: dict[str, Any]) -> dict[str, Any]:
    nt = r["next_token"]
    return {k: [(i.get("pred_norm"), i.get("correct"), i.get("completion")) for i in nt[k]["items"]]
            for k in DETERMINISTIC}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--heldout-docs", type=int, default=400)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ck = args.run / "checkpoints"
    final = sorted(ck.glob("step_*.pt"))[-1]
    models = {"m4_1_final": str(final), "m4_1_best": str(ck / "best.pt"), **BASELINES}
    results = {}
    for name, path in models.items():
        results[name] = run_suite(path, args.out / f"general_eval_{name}.json", args.heldout_docs)

    again = run_suite(str(final), args.out / "general_eval_m4_1_final_repeat.json",
                      args.heldout_docs)  # fmt: skip
    repro = deterministic_view(again) == deterministic_view(results["m4_1_final"])

    tok = ByteBPETokenizer.load(PROC / "tokenizer.json")
    train = np.memmap(PROC / "train.bin", dtype="<u2", mode="r")
    index = KGramIndex(np.asarray(train), [8, 16, 32, 48])
    mem: dict[str, Any] = {}
    for section in ("continuation", "explanations", "repetition_stress"):
        rows = []
        for item in results["m4_1_final"]["next_token"][section]["items"]:
            ids = np.asarray(tok.encode_ordinary(item["text"]), dtype=np.int64)
            rows.append({"id": item["id"], "seed": item["seed"], **index.overlap(ids)})
        mem[section] = {
            "samples": len(rows),
            "mean_frac_16gram_in_train": float(np.mean(
                [r["frac_16gram_in_train"] for r in rows if r["frac_16gram_in_train"] is not None]
            )),
            "samples_with_32gram_match": sum(r["longest_match_at_least"] >= 32 for r in rows),
            "samples_with_48gram_match": sum(r["longest_match_at_least"] >= 48 for r in rows),
            "rows": rows,
        }  # fmt: skip
    atomic_write_json(args.out / "evaluation_summary.json", {
        "checkpoints": models, "reproducible_deterministic_sections": repro,
        "memorization": mem,
    })  # fmt: skip
    print("reproducible:", repro)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      for k, v in mem.items()}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()

"""M5 evaluation: M4.1 base model vs the SFT conversational model (no training).

1. Frozen general suite ``evals/general_v1.json`` + held-out bits/byte on the same
   documents as M4.1 (``scripts/evaluate_general.py``) for every model: measures
   catastrophic forgetting and runs the suite's own 12-item chat section.
2. Frozen conversational suite ``evals/chat_v1.json`` (40 items) under four fixed
   decoding settings, 200 new tokens max:
   ``greedy`` / ``greedy_rp`` (repetition penalty 1.2) and ``sampled`` / ``sampled_rp``
   (temperature 0.7, top-k 40, seeds 11, 22, 33). Every reply records its stop reason,
   so natural ``<|end|>`` termination is never confused with hitting the token limit.
3. Memorization: SFT replies are checked for verbatim 32/48-token spans of the SFT
   training conversations.

Usage: python scripts/evaluate_m5.py runs/<m5 run> --out docs/reports/m5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_chatgpt.analysis.story_eval import KGramIndex
from mini_chatgpt.chat.evaluate import aggregate, score_reply
from mini_chatgpt.chat.generate import chat_reply
from mini_chatgpt.chat.sft import load_split
from mini_chatgpt.device import select_device
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json
from mini_chatgpt.training.trainer import load_model

BASE = "runs/m4_1_general_s_bpe8192-20261114-190514/checkpoints/step_00009600.pt"
PROC = Path("data/processed/general_m4_1_bpe/heldout_text")
HELDOUT = {
    "fineweb_edu_val": PROC / "fineweb_edu_val",
    "cosmopedia_v2_val": PROC / "cosmopedia_v2_val",
    "wikipedia_en_ood": Path("data/raw/samples/wikipedia_en"),
    "tinystories_v2_val": PROC / "tinystories_v2_val",
}
SUITE = Path("evals/chat_v1.json")
SEEDS = (11, 22, 33)
PENALTY = 1.2
MAX_NEW = 200
SETTINGS: dict[str, dict[str, Any]] = {
    "greedy": {"greedy": True, "repetition_penalty": 1.0, "seeds": (None,)},
    "greedy_rp": {"greedy": True, "repetition_penalty": PENALTY, "seeds": (None,)},
    "sampled": {"greedy": False, "repetition_penalty": 1.0, "seeds": SEEDS},
    "sampled_rp": {"greedy": False, "repetition_penalty": PENALTY, "seeds": SEEDS},
}


def general(ckpt: str, out: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/evaluate_general.py",
        ckpt,
        "--out",
        str(out),
        "--heldout-docs",
        "400",
        "--tokenizer",
        "data/processed/general_m4_1_bpe/tokenizer.json",
    ]
    for name, d in HELDOUT.items():
        cmd += ["--heldout", f"{name}={d}"]
    subprocess.run(cmd, check=True)
    result: dict[str, Any] = json.loads(out.read_text())
    return result


def chat_suite(ckpt: str, tok: ByteBPETokenizer, device: torch.device) -> dict[str, Any]:
    model, _ = load_model(ckpt, device)
    items = json.loads(SUITE.read_text())["items"]
    out: dict[str, Any] = {}
    for name, s in SETTINGS.items():
        t0 = time.time()
        rows = []
        for seed in s["seeds"]:
            for item in items:
                reply = chat_reply(
                    model,
                    tok,
                    item["turns"],
                    max_new_tokens=MAX_NEW,
                    temperature=0.7,
                    top_k=40,
                    greedy=s["greedy"],
                    repetition_penalty=s["repetition_penalty"],
                    seed=seed,
                )
                rows.append(score_reply(item, reply))
        out[name] = {
            "settings": {k: v for k, v in s.items() if k != "seeds"}
            | {
                "seeds": list(s["seeds"]),
                "temperature": 0.7,
                "top_k": 40,
                "max_new_tokens": MAX_NEW,
            },
            "summary": aggregate(rows),
            "rows": rows,
            "seconds": time.time() - t0,
        }
        print(
            name,
            json.dumps(
                {
                    k: v
                    for k, v in out[name]["summary"].items()
                    if k in ("natural_end_of_turn", "repetitive_over_0.3", "content_ok", "both_ok")
                }
            ),
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = select_device("auto")
    tok = ByteBPETokenizer.load(args.run / "tokenizer.json")
    ck = args.run / "checkpoints"
    models = {"base_m4_1": BASE, "sft_best": str(ck / "best.pt")}
    best_sd = torch.load(ck / "best.pt", map_location="cpu", weights_only=True)["state"]["model"]
    final_sd = torch.load(ck / "final.pt", map_location="cpu", weights_only=True)["state"]["model"]
    same = all(torch.equal(best_sd[k], final_sd[k]) for k in best_sd)
    if not same:
        models["sft_final"] = str(ck / "final.pt")
    summary: dict[str, Any] = {
        "checkpoints": models,
        "best_equals_final": same,
        "chat_suite_sha256": hashlib.sha256(SUITE.read_bytes()).hexdigest(),
        "general": {},
        "chat": {},
    }
    for name, path in models.items():
        g = general(path, args.out / f"general_eval_{name}.json")
        nt = g["next_token"]
        summary["general"][name] = {
            "heldout_bpb": {k: v["bits_per_byte"] for k, v in g["heldout"].items()},
            **{k: nt[k]["acc_norm"] for k in ("facts_mc", "qa_mc", "code_mc")},
            "grammar_pairs": nt["grammar_pairs"]["acc"],
            "code_generation": nt["code_generation"]["acc"],
            "general_chat_section": {
                k: g["instruction"]["chat"][k] for k in ("format", "content", "both")
            },
            "continuation_rep4": nt["continuation"]["mean_repeated_4gram_fraction"],
            "explanations_rep4": nt["explanations"]["mean_repeated_4gram_fraction"],
        }
        res = chat_suite(path, tok, device)
        atomic_write_json(args.out / f"chat_eval_{name}.json", res)
        summary["chat"][name] = {k: v["summary"] for k, v in res.items()}

    # memorization of SFT training replies
    train = load_split("data/processed/chat_m5", "train")
    index = KGramIndex(train.tokens.astype(np.int64), [8, 16, 32, 48])
    res = json.loads((args.out / "chat_eval_sft_best.json").read_text())
    mem = []
    for row in res["sampled"]["rows"]:
        ids = np.asarray(tok.encode(row["reply"]), dtype=np.int64)
        mem.append(index.overlap(ids))
    summary["memorization_sft_sampled"] = {
        "replies": len(mem),
        "with_32gram_match": sum(m["longest_match_at_least"] >= 32 for m in mem),
        "with_48gram_match": sum(m["longest_match_at_least"] >= 48 for m in mem),
        "mean_frac_16gram_in_train": float(
            np.mean([m["frac_16gram_in_train"] or 0.0 for m in mem])
        ),
    }
    atomic_write_json(args.out / "evaluation_summary.json", summary)
    print(json.dumps(summary["memorization_sft_sampled"], indent=1))


if __name__ == "__main__":
    main()

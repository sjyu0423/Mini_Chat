"""Extraction-style memorization probe.

Prompt the model with the first ``--prompt-chars`` characters of passages taken
from the *training* split and decode greedily. Count how many characters of the
true continuation it reproduces exactly before the first mistake. The same probe
on *validation* passages (never trained on) is the control. If training
continuations come out much longer than validation ones, the model is
reproducing memorized training text rather than just predicting plausible text.

Usage: python scripts/memorization_probe.py runs/<run> --out docs/reports/m2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.inference.generate import generate
from mini_chatgpt.training.checkpoint import (
    atomic_write_json,
    best_checkpoint_path,
    list_checkpoints,
)
from mini_chatgpt.training.run import setup_logging
from mini_chatgpt.training.trainer import load_model


def matched_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def probe(
    model: torch.nn.Module,
    ids: torch.Tensor,
    starts: list[int],
    prompt_len: int,
    cont_len: int,
    device: torch.device,
) -> dict[str, Any]:
    prompts = torch.stack([ids[s : s + prompt_len] for s in starts]).to(device)
    out = generate(model, prompts, cont_len, greedy=True)  # type: ignore[arg-type]
    gen = out[:, prompt_len:].cpu()
    truth = [ids[s + prompt_len : s + prompt_len + cont_len].tolist() for s in starts]
    prefix = [matched_prefix(g.tolist(), t) for g, t in zip(gen, truth, strict=True)]
    char_acc = [
        sum(x == y for x, y in zip(g.tolist(), t, strict=True)) / cont_len
        for g, t in zip(gen, truth, strict=True)
    ]
    return {
        "n": len(starts),
        "matched_prefix_mean": statistics.fmean(prefix),
        "matched_prefix_median": statistics.median(prefix),
        "matched_prefix_max": max(prefix),
        "full_continuation_reproduced": sum(p == cont_len for p in prefix),
        "reproduced_at_least_64": sum(p >= 64 for p in prefix),
        "position_char_accuracy_mean": statistics.fmean(char_acc),
        "matched_prefix_all": prefix,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="mps")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--prompt-chars", type=int, default=128)
    p.add_argument("--continuation-chars", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    log = setup_logging()
    run: Path = args.run_dir
    out: Path = args.out or run / "analysis"
    cfg = ExperimentConfig.from_dict(json.loads((run / "config.json").read_text()))
    data = load_prepared(processed_dir(cfg.data))
    device = select_device(args.device)
    span = args.prompt_chars + args.continuation_chars
    if span > cfg.model.context_length:
        raise SystemExit("prompt + continuation must fit in the context window")

    g = torch.Generator().manual_seed(args.seed)
    splits = {"train": data.splits.train, "val": data.splits.val}
    starts = {
        k: torch.randint(0, len(v) - span, (args.n,), generator=g).tolist()
        for k, v in splits.items()
    }
    ckpts = {"best": best_checkpoint_path(run), "final": list_checkpoints(run)[-1]}
    by_step = {int(c.stem.split("_")[1]): c for c in list_checkpoints(run)}
    for s in (500, 1000, 2500):
        if s in by_step:
            ckpts[f"step_{s}"] = by_step[s]

    result: dict[str, Any] = {
        "run_dir": str(run),
        "prompt_chars": args.prompt_chars,
        "continuation_chars": args.continuation_chars,
        "decoding": "greedy",
        "seed": args.seed,
        "results": {},
    }
    for name, path in ckpts.items():
        model, _ = load_model(path, device)
        r = {
            split: probe(model, ids, starts[split], args.prompt_chars,
                         args.continuation_chars, device)
            for split, ids in splits.items()
        }  # fmt: skip
        result["results"][name] = {"checkpoint": str(path), **r}
        log.info(
            "%s: train prefix mean %.1f (max %d, >=64: %d) | val prefix mean %.1f (max %d)",
            name, r["train"]["matched_prefix_mean"], r["train"]["matched_prefix_max"],
            r["train"]["reproduced_at_least_64"], r["val"]["matched_prefix_mean"],
            r["val"]["matched_prefix_max"],
        )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "memorization_probe.json", result)


if __name__ == "__main__":
    main()

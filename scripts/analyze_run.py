# ruff: noqa: RUF001  (chart labels intentionally use typographic symbols)
"""Evaluate and analyze a finished training run (checkpoints, samples, plots).

Produces, in ``--out`` (default ``<run>/analysis``):

* ``*.svg``: train loss, validation loss, learning rate, throughput, generalization gap
* ``analysis.json``: exact losses for the best and final checkpoints, emergence
  statistics for every checkpoint, memorization checks
* ``samples.md``: all generated text, with prompts, seeds and decoding settings

All sampling uses fixed prompts and seeds, so reruns produce the same text up to
MPS kernel nondeterminism.

Usage: python scripts/analyze_run.py runs/<run> --out docs/reports/m2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.analysis.svgplot import Chart, Series, ema, render
from mini_chatgpt.analysis.text_stats import (
    longest_match,
    ngram_overlap,
    speaker_stats,
    speakers,
    vocabulary,
    word_stats,
)
from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.inference.generate import generate
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.char import CharTokenizer
from mini_chatgpt.training.checkpoint import (
    atomic_write_json,
    atomic_write_text,
    best_checkpoint_path,
    list_checkpoints,
    read_metadata,
)
from mini_chatgpt.training.metrics import bits_per_token, perplexity
from mini_chatgpt.training.run import dedupe_metrics, read_metrics, setup_logging
from mini_chatgpt.training.seed import set_seed
from mini_chatgpt.training.trainer import Trainer

PROMPTS = ["ROMEO:\n", "JULIET:\n", "First Citizen:\n", "KING RICHARD III:\n", "To be, or not"]
DECODINGS: list[dict[str, Any]] = [
    {"name": "T0.7_k40", "temperature": 0.7, "top_k": 40, "greedy": False},
    {"name": "T0.9_k40", "temperature": 0.9, "top_k": 40, "greedy": False},
    {"name": "greedy", "temperature": 1.0, "top_k": None, "greedy": True},
]
SEED = 1337
log = setup_logging()


def sample(
    model: GPT,
    tok: CharTokenizer,
    prompt: str,
    n: int,
    dec: dict[str, Any],
    rows: int = 1,
    seed: int = SEED,
) -> list[str]:
    idx = torch.tensor([tok.encode(prompt)] * rows, dtype=torch.long)
    idx = idx.to(next(model.parameters()).device)
    out = generate(
        model,
        idx,
        n,
        temperature=dec["temperature"],
        top_k=dec["top_k"],
        greedy=dec["greedy"],
        seed=seed,
    )
    return [tok.decode(r) for r in out[:, len(prompt) :].tolist()]


def init_model(cfg: ExperimentConfig, device: torch.device) -> GPT:
    """Step-0 weights: the trainer seeds then builds the model on the CPU RNG."""
    set_seed(cfg.training.seed)
    return GPT(cfg.model).to(device).eval()


def plots(records: list[dict[str, Any]], out: Path, cfg: ExperimentConfig) -> list[str]:
    train = [r for r in records if r["event"] == "train"]
    evals = [r for r in records if r["event"] == "eval"]
    steps = [r["step"] for r in train]
    loss = [r["loss"] for r in train]
    es = [r["step"] for r in evals]
    written = []

    def save(name: str, chart: Chart) -> None:
        (out / name).write_text(render(chart), encoding="utf-8")
        written.append(name)

    save(
        "train_loss.svg",
        Chart(
            "Training loss",
            "optimizer step",
            "cross-entropy (nats/char)",
            [
                Series("per-step batch loss", steps, loss, opacity=0.25, width=1),
                Series("EMA (α = 0.02)", steps, ema(loss)),
                Series(
                    "fixed-batch train estimate",
                    es,
                    [r["train_loss"] for r in evals],
                    slot=2,
                    markers=True,
                    width=0,
                ),
            ],
            subtitle="gpt_10m · Tiny Shakespeare (char) · batch 64 × 256 · MPS float32",
            y_min=0.4,
            y_max=4.4,
        ),
    )
    late = [r for r in evals if r["step"] > 0]
    save(
        "val_loss.svg",
        Chart(
            "Validation loss vs. training loss",
            "optimizer step",
            "cross-entropy (nats/char)",
            [
                Series(
                    "validation (exact, full split)",
                    [r["step"] for r in late],
                    [r["val_loss_exact"] for r in late],
                    slot=1,
                    markers=True,
                ),
                Series(
                    "train (fixed-batch estimate)",
                    [r["step"] for r in late],
                    [r["train_loss"] for r in late],
                    slot=0,
                    markers=True,
                ),
            ],
            subtitle=f"from step 250 (step 0: val {evals[0]['val_loss_exact']:.3f}, "
            f"uniform ln 65 = {math.log(65):.3f})",
            hlines=[(min(r["val_loss_exact"] for r in late), "best validation")],
        ),
    )
    save(
        "learning_rate.svg",
        Chart(
            "Learning rate",
            "optimizer step",
            "learning rate",
            [Series("lr", steps, [r["lr"] for r in train])],
            subtitle=f"linear warmup {cfg.training.warmup_steps} steps to "
            f"{cfg.training.learning_rate:g}, cosine to {cfg.training.min_learning_rate:g}",
            y_min=0.0,
            y_format="{:.1e}",
        ),
    )
    tps = [r["tokens_per_s"] for r in train]
    save(
        "throughput.svg",
        Chart(
            "Training throughput",
            "optimizer step",
            "tokens / second",
            [
                Series("per step", steps, tps, opacity=0.25, width=1),
                Series("EMA (α = 0.02)", steps, ema(tps)),
            ],
            subtitle="16,384 tokens per optimizer step; excludes evaluation and checkpoint time",
            y_min=0.0,
            y_format="{:,.0f}",
        ),
    )
    save(
        "generalization_gap.svg",
        Chart(
            "Generalization gap (validation − train)",
            "optimizer step",
            "Δ cross-entropy (nats/char)",
            [
                Series(
                    "exact val − train estimate",
                    es,
                    [r["val_loss_exact"] - r["train_loss"] for r in evals],
                    slot=1,
                    markers=True,
                ),
                Series(
                    "val estimate − train estimate (same batch count)",
                    es,
                    [r["gap"] for r in evals],
                    slot=1,
                    dash="5 4",
                ),
            ],
            hlines=[(0.0, "no gap")],
            y_format="{:+.2f}",
        ),
    )
    return written


def exact_losses(
    cfg: ExperimentConfig, path: Path, train: torch.Tensor, val: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    t = Trainer(cfg, train, val, device)
    t.load_state(path)  # full trainer restore: also a checkpoint-compatibility check
    tl, vl = t.full_loss(train), t.full_loss(val)
    return {
        "checkpoint": str(path),
        "step": t.step,
        "tokens_seen": t.tokens_seen,
        "train_loss": tl,
        "val_loss": vl,
        "gap": vl - tl,
        "train_ppl": perplexity(tl),
        "val_ppl": perplexity(vl),
        "train_bpc": bits_per_token(tl),
        "val_bpc": bits_per_token(vl),
    }


def memorization(text: str, train_text: str) -> dict[str, Any]:
    m = longest_match(text, train_text)
    return {
        "chars": len(text),
        "longest_verbatim_match": m.length,
        "longest_match_text": m.text,
        "overlap_32": ngram_overlap(text, train_text, 32),
        "overlap_64": ngram_overlap(text, train_text, 64),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="mps")
    p.add_argument("--sample-tokens", type=int, default=500)
    p.add_argument("--emergence-tokens", type=int, default=1000)
    p.add_argument("--emergence-rows", type=int, default=4)
    args = p.parse_args()
    run: Path = args.run_dir
    out: Path = args.out or run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    cfg = ExperimentConfig.from_dict(json.loads((run / "config.json").read_text()))
    tok = CharTokenizer.load(run / "tokenizer.json")
    data = load_prepared(processed_dir(cfg.data))
    train_ids, val_ids = data.splits.train, data.splits.val
    train_text, val_text = tok.decode(train_ids.tolist()), tok.decode(val_ids.tolist())
    vocab, known_speakers = vocabulary(train_text), speakers(train_text)
    device = select_device(args.device)
    records = dedupe_metrics(read_metrics(run / "metrics.jsonl"))

    result: dict[str, Any] = {"run_dir": str(run), "plots": plots(records, out, cfg)}
    log.info("plots written: %s", result["plots"])

    # ---- exact metrics, best and final -------------------------------------------------- #
    ckpts = list_checkpoints(run)
    final = ckpts[-1]
    best = best_checkpoint_path(run)
    best_meta, final_meta = read_metadata(best), read_metadata(final)
    assert best_meta is not None and final_meta is not None
    result["final_checkpoint_kind"] = final_meta.kind
    result["exact"] = {
        "best": exact_losses(cfg, best, train_ids, val_ids, device),
        "final": exact_losses(cfg, final, train_ids, val_ids, device),
    }
    init = init_model(cfg, device)
    t0 = Trainer(cfg, train_ids, val_ids, device)
    t0.model = init
    result["exact"]["init"] = {"train_loss": t0.full_loss(train_ids),
                               "val_loss": t0.full_loss(val_ids)}  # fmt: skip
    log.info("exact: %s", json.dumps(result["exact"], indent=1))

    # ---- samples from best model ------------------------------------------------------- #
    def load(path: Path) -> GPT:
        t = Trainer(cfg, train_ids, val_ids, device)
        t.load_state(path)
        return t.model.eval()

    models = {"best": load(best), "final": load(final)}
    md = [f"# Samples — `{run.name}`\n",
          f"Seed {SEED} for every sample; {args.sample_tokens} new characters; prompt shown in "
          "**bold**. Text is verbatim model output.\n"]  # fmt: skip
    result["samples"] = {}
    for which, model in models.items():
        meta = best_meta if which == "best" else final_meta
        md.append(f"\n## {which} checkpoint (step {meta.step})\n")
        for dec in DECODINGS:
            md.append(f"\n### {dec['name']}\n")
            for prompt in PROMPTS:
                text = sample(model, tok, prompt, args.sample_tokens, dec)[0]
                mem = memorization(prompt + text, train_text)
                result["samples"].setdefault(which, []).append(
                    {"prompt": prompt, "decoding": dec["name"], "text": text, **mem,
                     **asdict(word_stats(text, vocab)),
                     **{f"speaker_{k}": v for k, v in asdict(speaker_stats(text, known_speakers))
                        .items()}}
                )  # fmt: skip
                md.append(
                    f"\n**{prompt!r}** · longest verbatim train match "
                    f"{mem['longest_verbatim_match']} chars · 32-char overlap "
                    f"{mem['overlap_32']:.0%}\n\n```text\n{prompt}{text}\n```\n"
                )

    # ---- progression: beginning / middle / end ---------------------------------------- #
    by_step = {int(p.stem.split("_")[1]): p for p in ckpts}
    prog_steps: list[int] = [s for s in (0, 250, 500, 1000, 2500, 5000) if s == 0 or s in by_step]
    md.append("\n## Progression over training (prompt `ROMEO:\\n`)\n")
    result["progression"] = []
    for step in prog_steps:
        model = init if step == 0 else load(by_step[step])
        md.append(f"\n### step {step}\n")
        for dec in DECODINGS:
            text = sample(model, tok, "ROMEO:\n", 300, dec)[0]
            result["progression"].append({"step": step, "decoding": dec["name"], "text": text})
            md.append(f"\n{dec['name']}:\n\n```text\nROMEO:\n{text}\n```\n")

    # ---- emergence statistics over all checkpoints ------------------------------------ #
    evals = {r["step"]: r for r in records if r["event"] == "eval"}
    dec_t1: dict[str, Any] = {"name": "T1.0", "temperature": 1.0, "top_k": None, "greedy": False}
    result["emergence"] = []
    for step in [0, *sorted(by_step)]:
        model = init if step == 0 else load(by_step[step])
        row: dict[str, Any] = {"step": step, "val_loss_exact": evals[step]["val_loss_exact"]}
        for dec in (DECODINGS[0], dec_t1):
            texts = sample(model, tok, "\n", args.emergence_tokens, dec, rows=args.emergence_rows)
            joined = "\n".join(texts)
            ws, ss = word_stats(joined, vocab), speaker_stats(joined, known_speakers)
            row[dec["name"]] = {
                "valid_word_fraction": ws.valid_fraction,
                "valid_word_fraction_len4": ws.valid_fraction_len4,
                "n_words": ws.n_words,
                "speaker_labels": ss.n_labels,
                "speaker_known_fraction": ss.known_fraction,
                "speaker_distinct": ss.distinct,
                "longest_verbatim_match": max(longest_match(t, train_text).length for t in texts),
                "overlap_32": sum(ngram_overlap(t, train_text, 32) for t in texts) / len(texts),
            }
        result["emergence"].append(row)
        log.info("emergence step %d: %s", step, json.dumps(row["T0.7_k40"]))

    # ---- reference: how much held-out text overlaps training text naturally ----------- #
    g = torch.Generator().manual_seed(SEED)
    n = args.emergence_tokens
    starts = torch.randint(0, len(val_text) - n, (args.emergence_rows * 2,), generator=g).tolist()
    chunks = [val_text[s : s + n] for s in starts]
    result["val_reference"] = {
        "chars_each": n,
        "chunks": len(chunks),
        "longest_verbatim_match_max": max(longest_match(c, train_text).length for c in chunks),
        "longest_verbatim_match_mean": sum(longest_match(c, train_text).length for c in chunks)
        / len(chunks),
        "overlap_32_mean": sum(ngram_overlap(c, train_text, 32) for c in chunks) / len(chunks),
        "valid_word_fraction": word_stats("\n".join(chunks), vocab).valid_fraction,
        "valid_word_fraction_len4": word_stats("\n".join(chunks), vocab).valid_fraction_len4,
        "speaker_known_fraction": speaker_stats("\n".join(chunks), known_speakers).known_fraction,
    }
    result["analysis_wall_s"] = time.perf_counter() - t_start
    atomic_write_json(out / "analysis.json", result)
    atomic_write_text(out / "samples.md", "".join(md))
    log.info("wrote %s (%.0f s)", out, result["analysis_wall_s"])


if __name__ == "__main__":
    main()

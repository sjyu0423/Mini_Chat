# ruff: noqa: RUF001  (chart labels intentionally use typographic symbols)
"""Evaluate a TinyStories BPE run: exact losses, curves, stories, memorization.

Produces in ``--out``:
* ``eval_tinystories.json``: exact losses (val, train-distribution holdout, fixed
  train slice) for the randomly initialized model and the intermediate, best and
  final checkpoints, in nats/token, bits/byte and perplexity; per-sample
  measured story metrics; k-gram overlap with the full training stream; an
  extraction probe (train vs. validation story prefixes).
* ``stories.md``: every generated story verbatim, with its settings.
* ``*.svg``: training/validation curves, learning rate, gradient norm, throughput.

Usage: python scripts/evaluate_tinystories.py runs/<run> --out docs/reports/m3_1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_chatgpt.analysis.story_eval import (
    LADDER,
    MAX_NEW,
    TEMPERATURE,
    TOP_K,
    WORD,
    KGramIndex,
    exact_loss,
    sample,
    story_metrics,
)
from mini_chatgpt.analysis.svgplot import Chart, Series, ema, render
from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.inference.generate import generate
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import (
    atomic_write_json,
    atomic_write_text,
    best_checkpoint_path,
    checkpoint_path,
    list_checkpoints,
)
from mini_chatgpt.training.run import dedupe_metrics, read_metrics, setup_logging
from mini_chatgpt.training.seed import set_seed
from mini_chatgpt.training.trainer import Trainer

log = setup_logging()
PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "One day, a boy found a strange box.",
    "A small dragon lived in a forest.",
    "Tom wanted to help his friend.",
    "There was a dog who was afraid of water.",
]


def plots(run: Path, out: Path) -> list[str]:
    recs = dedupe_metrics(read_metrics(run / "metrics.jsonl"))
    tr = [r for r in recs if r["event"] == "train"]
    ev = [r for r in recs if r["event"] == "eval"]
    steps = [r["step"] for r in tr]
    written = []

    def save(name: str, chart: Chart) -> None:
        (out / name).write_text(render(chart), encoding="utf-8")
        written.append(name)

    ev_steps = [r["step"] for r in ev]
    train_loss = [r["loss"] for r in tr]
    gnorm = [r["grad_norm"] for r in tr]
    tps = [r["tokens_per_s"] for r in tr]
    save(
        "loss.svg",
        Chart(
            "Training and validation loss (TinyStories V2, BPE 4,096)",
            "optimizer step",
            "cross-entropy (nats / BPE token)",
            [
                Series("train: per-step batch loss", steps, train_loss, opacity=0.25, width=1),
                Series("train: EMA (α = 0.02)", steps, ema(train_loss)),
                Series(
                    "validation: exact, full official V2 set",
                    ev_steps,
                    [r["val_loss_exact"] for r in ev],
                    slot=1,
                    markers=True,
                ),
            ],
            subtitle="6L/6H/384, ctx 512, 16,384 tokens/step; step 0 = random init (8.42)",
            y_min=1.0,
            y_max=8.6,
        ),
    )
    save(
        "val_bpb.svg",
        Chart(
            "Validation bits per byte (exact)",
            "optimizer step",
            "bits / UTF-8 byte",
            [
                Series(
                    "validation bpb",
                    ev_steps[1:],
                    [r["val_bpb_exact"] for r in ev[1:]],
                    slot=1,
                    markers=True,
                )
            ],
            subtitle="nats/token ÷ ln 2 ÷ bytes/token; comparable across tokenizers",
            y_format="{:.3f}",
        ),
    )
    save(
        "learning_rate.svg",
        Chart(
            "Learning rate",
            "optimizer step",
            "learning rate",
            [Series("lr", steps, [r["lr"] for r in tr])],
            subtitle="linear warmup 200 steps to 6e-4, cosine to 6e-5 at step 4,800",
            y_min=0.0,
            y_format="{:.1e}",
        ),
    )
    save(
        "grad_norm.svg",
        Chart(
            "Gradient norm (before clipping at 1.0)",
            "optimizer step",
            "global L2 norm",
            [
                Series("per step", steps, gnorm, opacity=0.3, width=1),
                Series("EMA (α = 0.02)", steps, ema(gnorm)),
            ],
            y_min=0.0,
            y_max=2.0,
            hlines=[(1.0, "clip threshold")],
        ),
    )
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
            subtitle="excludes evaluation and checkpoint time",
            y_min=0.0,
            y_format="{:,.0f}",
        ),
    )
    save(
        "generalization_gap.svg",
        Chart(
            "Validation − train (random-batch estimates, same batch count)",
            "optimizer step",
            "Δ cross-entropy (nats / token)",
            [Series("val − train", ev_steps, [r["gap"] for r in ev], slot=1, markers=True)],
            hlines=[(0.0, "no gap")],
            y_format="{:+.3f}",
        ),
    )
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--train-slice", type=int, default=5_000_000)
    p.add_argument("--intermediate-step", type=int, default=2400)
    p.add_argument("--probe-n", type=int, default=64)
    args = p.parse_args()
    run: Path = args.run_dir
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    cfg = ExperimentConfig.from_dict(json.loads((run / "config.json").read_text()))
    device = select_device(args.device)
    data = load_prepared(processed_dir(cfg.data))
    tok = ByteBPETokenizer.load(run / "tokenizer.json")
    if tok.fingerprint() != data.meta["tokenizer_fingerprint"]:
        raise SystemExit("run tokenizer differs from the dataset tokenizer")
    pdir = processed_dir(cfg.data)
    manifest = json.loads((pdir / "manifest.json").read_text())
    holdout = torch.from_numpy(np.memmap(pdir / "holdout.bin", dtype="<u2", mode="c"))
    train_np = np.asarray(data.splits.train)
    val_np = np.asarray(data.splits.val)
    bpt = {
        "val": data.meta["val_bytes_per_token"],
        "holdout": manifest["streams"]["holdout"]["bytes_per_token"],
        "train_slice": manifest["streams"]["train"]["bytes_per_token"],
    }
    result: dict[str, Any] = {"run_dir": str(run), "plots": plots(run, out), "bytes_per_token": bpt}

    # ---- models ---------------------------------------------------------------------- #
    def load(path: Path) -> GPT:
        t = Trainer(cfg, data.splits.train, data.splits.val, device)
        t.load_state(path)
        return t.model.eval()

    set_seed(cfg.training.seed)
    models: dict[str, tuple[int, GPT]] = {"init": (0, GPT(cfg.model).to(device).eval())}
    inter = checkpoint_path(run, args.intermediate_step)
    if inter.exists():
        models["intermediate"] = (args.intermediate_step, load(inter))
    best = best_checkpoint_path(run)
    best_model = load(best)
    best_step = int(torch.load(best, map_location="cpu", weights_only=True)["state"]["step"])
    models["best"] = (best_step, best_model)
    final = list_checkpoints(run)[-1]
    models["final"] = (int(final.stem.split("_")[1]), load(final))

    # ---- exact losses ------------------------------------------------------------------ #
    train_slice = data.splits.train[: args.train_slice]
    result["exact"] = {}
    for name, (step, model) in models.items():
        row: dict[str, Any] = {"step": step}
        for split, stream in (
            ("val", data.splits.val),
            ("holdout", holdout),
            ("train_slice", train_slice),
        ):
            loss = exact_loss(model, stream, cfg, device)
            row[split] = {
                "loss_nats_per_token": loss,
                "ppl_per_token": math.exp(loss),
                "bits_per_byte": loss / math.log(2) / bpt[split],
            }
        row["gap_val_minus_train_slice"] = (
            row["val"]["loss_nats_per_token"] - row["train_slice"]["loss_nats_per_token"]
        )
        result["exact"][name] = row
        log.info(
            "%s (step %d): val %.4f | holdout %.4f | train-slice %.4f",
            name,
            step,
            row["val"]["loss_nats_per_token"],
            row["holdout"]["loss_nats_per_token"],
            row["train_slice"]["loss_nats_per_token"],
        )

    # ---- training-text vocabulary and k-gram index ---------------------------------- #
    vocab = {w.lower() for w in WORD.findall(tok.decode(train_np[:20_000_000].tolist()))}
    index = KGramIndex(train_np, LADDER)

    # ---- stories ----------------------------------------------------------------------- #
    md = [
        "# M3.1 generated stories (verbatim model output)\n\n",
        f"Decoding: temperature {TEMPERATURE}, top-k {TOP_K}, up to "
        f"{MAX_NEW} new BPE tokens, stopped at `<|endoftext|>`. The prompt is shown in bold; "
        "the same seed is used for every checkpoint.\n",
    ]
    plan = [(name, s) for name in models for s in ([1, 2, 3] if name == "best" else [1])]
    samples: list[dict[str, Any]] = []
    for name, seed in plan:
        step, model = models[name]
        for prompt in PROMPTS:
            s = sample(model, tok, prompt, seed, device)
            s.update(
                checkpoint=name,
                step=step,
                **story_metrics(s["text"], vocab),
                **index.overlap(np.asarray(tok.encode(prompt) + s["new_token_ids"])),
            )
            samples.append(s)
    for prompt in PROMPTS:  # greedy decoding from the best checkpoint
        s = sample(best_model, tok, prompt, 0, device, greedy=True)
        s.update(
            checkpoint="best",
            step=best_step,
            **story_metrics(s["text"], vocab),
            **index.overlap(np.asarray(tok.encode(prompt) + s["new_token_ids"])),
        )
        samples.append(s)
    for s in samples:
        s.pop("new_token_ids")
    for name in [*models, "best-greedy"]:
        rows = [
            s
            for s in samples
            if (s["checkpoint"] == name and not s["greedy"])
            or (name == "best-greedy" and s["greedy"])
        ]
        if not rows:
            continue
        md.append(f"\n## {name} (step {rows[0]['step']}){' greedy' if 'greedy' in name else ''}\n")
        for s in rows:
            ending = "ended with <|endoftext|>" if s["completed_with_eot"] else "hit length limit"
            md.append(
                f"\n**{s['prompt']}** · seed {s['seed']} · {s['new_tokens']} tokens · "
                f"{ending}"
                f" · word-trigram repetition {s['word_trigram_repetition']:.0%} · longest "
                f"verbatim train match ≥ {s['longest_match_at_least']} tokens\n\n"
                f"```text\n{s['prompt']}{s['text']}\n```\n"
            )
    result["samples"] = samples
    agg = {}
    for name in [*models, "best-greedy"]:
        rows = [
            s
            for s in samples
            if (s["checkpoint"] == name and not s["greedy"])
            or (name == "best-greedy" and s["greedy"])
        ]
        if rows:
            agg[name] = {
                k: statistics.fmean(float(s[k]) for s in rows)
                for k in (
                    "new_tokens",
                    "completed_with_eot",
                    "valid_word_fraction",
                    "word_trigram_repetition",
                    "distinct_2",
                    "frac_8gram_in_train",
                    "frac_16gram_in_train",
                    "frac_32gram_in_train",
                    "longest_match_at_least",
                )
            }
            agg[name]["n"] = len(rows)
            agg[name]["max_longest_match_at_least"] = max(s["longest_match_at_least"] for s in rows)
    # reference: real validation stories measured the same way (natural overlap)
    eot = tok.eot_id
    ends = np.flatnonzero(val_np == eot)
    starts = np.concatenate([[0], ends[:-1] + 1])
    rng = np.random.default_rng(7)
    picks = rng.choice(len(starts), 40, replace=False)
    ref = []
    for i in picks:
        story = val_np[starts[i] : ends[i]]
        text = tok.decode(story.tolist())
        ref.append({**story_metrics(text, vocab), **index.overlap(story), "new_tokens": story.size})
    agg["validation_reference"] = {
        k: statistics.fmean(float(r[k]) for r in ref)
        for k in (
            "new_tokens",
            "valid_word_fraction",
            "word_trigram_repetition",
            "distinct_2",
            "frac_8gram_in_train",
            "frac_16gram_in_train",
            "frac_32gram_in_train",
            "longest_match_at_least",
        )
    }
    agg["validation_reference"]["n"] = len(ref)
    agg["validation_reference"]["max_longest_match_at_least"] = max(
        r["longest_match_at_least"] for r in ref
    )
    result["sample_summary"] = agg

    # ---- extraction probe -------------------------------------------------------------- #
    def story_starts(arr: np.ndarray, min_len: int) -> list[int]:
        e = np.flatnonzero(arr == eot)
        s = np.concatenate([[0], e[:-1] + 1])
        return [int(a) for a, b in zip(s, e, strict=True) if b - a >= min_len]

    probe: dict[str, Any] = {"prompt_tokens": 32, "continuation_tokens": 64, "decoding": "greedy"}
    for split, arr in (("train", train_np), ("val", val_np)):
        cands = story_starts(arr[:20_000_000] if split == "train" else arr, 96)
        chosen = np.random.default_rng(11).choice(cands, args.probe_n, replace=False)
        prompts = torch.tensor(np.stack([arr[c : c + 32] for c in chosen]).astype(np.int64))
        for name in ("best", "final"):
            gen = generate(models[name][1], prompts.to(device), 64, greedy=True)[:, 32:].cpu()
            truth = np.stack([arr[c + 32 : c + 96] for c in chosen]).astype(np.int64)
            pref = []
            for g, t in zip(gen.numpy(), truth, strict=True):
                n = 0
                while n < 64 and g[n] == t[n]:
                    n += 1
                pref.append(n)
            probe[f"{name}_{split}"] = {
                "matched_prefix_mean": statistics.fmean(pref),
                "matched_prefix_median": statistics.median(pref),
                "matched_prefix_max": max(pref),
                "full_64_reproduced": sum(x == 64 for x in pref),
            }
    result["extraction_probe"] = probe
    result["wall_s"] = time.perf_counter() - t_start
    atomic_write_json(out / "eval_tinystories.json", result)
    atomic_write_text(out / "stories.md", "".join(md))
    log.info("wrote %s (%.0f s)", out, result["wall_s"])


if __name__ == "__main__":
    main()

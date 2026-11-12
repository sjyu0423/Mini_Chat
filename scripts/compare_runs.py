# ruff: noqa: RUF001  (chart labels intentionally use typographic symbols)
"""Compare several training runs: metrics table, overlay plots, matched samples.

Usage:
    python scripts/compare_runs.py --run M2=runs/gpt_10m_m2-... --run A=runs/... \
        --run B=runs/... --out docs/reports/m2_1 [--max-step 2000]

Outputs in ``--out``: ``comparison.json``, ``samples.md`` and five SVG plots
(train loss, exact validation loss, learning rate, generalization gap, validation
loss vs. tokens). Exact losses are recomputed from the best and final checkpoints
of every run with the same code path. The first ``--run`` is treated as the
reference for the reproduction/noise analysis.
"""

from __future__ import annotations

import argparse
import json
import statistics
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
    words,
)
from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.inference.generate import generate
from mini_chatgpt.tokenizer.char import CharTokenizer
from mini_chatgpt.training.checkpoint import (
    atomic_write_json,
    atomic_write_text,
    best_checkpoint_path,
    checkpoint_path,
    list_checkpoints,
)
from mini_chatgpt.training.metrics import bits_per_token, perplexity
from mini_chatgpt.training.run import dedupe_metrics, read_metrics, setup_logging
from mini_chatgpt.training.trainer import Trainer

log = setup_logging()
SLOTS = {0: 2, 1: 0, 2: 1}  # run index -> categorical slot (reference aqua, A blue, B orange)
SAMPLE_PROMPT = "ROMEO:\n"
SAMPLE_SEEDS = [1, 2, 3, 4, 5]
SAMPLE_DEC: dict[str, Any] = {"temperature": 0.7, "top_k": 40}
MATCHED_STEPS = [500, 1000, 1500, 1750, 2000]


class Run:
    def __init__(self, label: str, path: Path) -> None:
        self.label, self.path = label, path
        self.cfg = ExperimentConfig.from_dict(json.loads((path / "config.json").read_text()))
        recs = dedupe_metrics(read_metrics(path / "metrics.jsonl"))
        self.train = [r for r in recs if r["event"] == "train"]
        self.evals = [r for r in recs if r["event"] == "eval"]
        self.memory = [r for r in recs if r["event"] == "memory"]
        self.eval_at = {int(r["step"]): r for r in self.evals}
        summaries = sorted(path.glob("summary_step*.json"))
        self.summary = json.loads(summaries[-1].read_text()) if summaries else {}

    def best(self) -> dict[str, Any]:
        e = min((r for r in self.evals if r["step"] > 0), key=lambda r: r["val_loss_exact"])
        return e


def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of word n-grams that repeat an earlier n-gram in the same text."""
    ws = words(text)
    grams = [tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)]
    if not grams:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    rep = 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams)


def exact(cfg: ExperimentConfig, path: Path, data: Any, device: torch.device) -> dict[str, Any]:
    t = Trainer(cfg, data.splits.train, data.splits.val, device)
    t.load_state(path)
    tl, vl = t.full_loss(data.splits.train), t.full_loss(data.splits.val)
    return {
        "checkpoint": str(path),
        "step": t.step,
        "tokens_seen": t.tokens_seen,
        "train_loss": tl,
        "val_loss": vl,
        "gap": vl - tl,
        "val_ppl": perplexity(vl),
        "val_bpc": bits_per_token(vl),
        "train_bpc": bits_per_token(tl),
    }


def timing(run: Run, upto: int | None = None) -> dict[str, Any]:
    tr = [r for r in run.train if upto is None or r["step"] <= upto]
    ev = [r for r in run.evals if upto is None or r["step"] <= upto]
    step_s = sum(r["step_time_s"] for r in tr)
    tokens = int(tr[-1]["tokens_seen"])
    wall = ev[-1]["time"] - ev[0]["time"]
    return {
        "steps": int(tr[-1]["step"]),
        "tokens": tokens,
        "optimizer_step_time_min": step_s / 60,
        "wall_min_eval0_to_last_eval": wall / 60,
        "evals": len(ev),
        "tokens_per_s_mean_over_steps": tokens / step_s,
        "tokens_per_s_median_step": statistics.median(r["tokens_per_s"] for r in tr[10:]),
        "steps_per_s": len(tr) / step_s,
        "tokens_per_s_wall": tokens / wall,
    }


def plots(runs: list[Run], out: Path, max_step: int) -> list[str]:
    written: list[str] = []

    def save(name: str, chart: Chart) -> None:
        (out / name).write_text(render(chart), encoding="utf-8")
        written.append(name)

    def series(
        fn: Any, *, markers: bool = False, start: int = 0, x_tokens: bool = False
    ) -> list[Series]:
        out_s = []
        for i, r in enumerate(runs):
            xs, ys = fn(r, start)
            if x_tokens:
                xs = [x * r.summary.get("tokens_per_step", 16384) / 1e6 for x in xs]
            out_s.append(
                Series(r.label, xs, ys, slot=SLOTS.get(i, i), markers=markers,
                       dash="6 4" if i == 0 else None, width=2)
            )  # fmt: skip
        return out_s

    def train_ema(r: Run, _s: int) -> tuple[list[float], list[float]]:
        tr = [x for x in r.train if x["step"] <= max_step]
        return [x["step"] for x in tr], ema([x["loss"] for x in tr])

    def val_exact(r: Run, s: int) -> tuple[list[float], list[float]]:
        ev = [x for x in r.evals if s <= x["step"] <= max_step]
        return [x["step"] for x in ev], [x["val_loss_exact"] for x in ev]

    def lr(r: Run, _s: int) -> tuple[list[float], list[float]]:
        tr = [x for x in r.train if x["step"] <= max_step]
        return [x["step"] for x in tr], [x["lr"] for x in tr]

    def gap(r: Run, _s: int) -> tuple[list[float], list[float]]:
        ev = [x for x in r.evals if x["step"] <= max_step]
        return [x["step"] for x in ev], [x["val_loss_exact"] - x["train_loss"] for x in ev]

    sub = f"steps 0–{max_step:,}; {runs[0].label} is dashed"
    save("train_loss.svg", Chart("Training loss (EMA of per-step batch loss, α = 0.02)",
         "optimizer step", "cross-entropy (nats/char)", series(train_ema), subtitle=sub,
         y_min=0.8, y_max=4.4))  # fmt: skip
    save("val_loss.svg", Chart("Validation loss (exact, full split)", "optimizer step",
         "cross-entropy (nats/char)", series(val_exact, markers=True, start=200),
         subtitle=sub + "; from step 200 (all runs start at 4.125)"))  # fmt: skip
    save("learning_rate.svg", Chart("Learning rate", "optimizer step", "learning rate",
         series(lr), subtitle=sub + "; A follows the M2 curve exactly", y_min=0.0,
         y_format="{:.1e}"))  # fmt: skip
    save("generalization_gap.svg", Chart("Generalization gap (exact val − fixed-batch train)",
         "optimizer step", "Δ cross-entropy (nats/char)", series(gap, markers=True),
         subtitle=sub, hlines=[(0.0, "no gap")], y_format="{:+.2f}"))  # fmt: skip
    save("val_loss_vs_tokens.svg", Chart("Validation loss vs. training tokens",
         "training tokens (millions)", "cross-entropy (nats/char)",
         series(val_exact, markers=True, start=200, x_tokens=True),
         subtitle="16,384 tokens per step; 1M tokens ≈ 1 epoch of Tiny Shakespeare",
         x_format="{:,.0f}"))  # fmt: skip

    def val_zoom(r: Run, _s: int) -> tuple[list[float], list[float]]:
        return val_exact(r, 1000)

    def train_zoom(r: Run, _s: int) -> tuple[list[float], list[float]]:
        tr = [x for x in r.train if x["step"] <= max_step]
        sm = ema([x["loss"] for x in tr])
        keep = [(x["step"], y) for x, y in zip(tr, sm, strict=True) if x["step"] >= 1000]
        return [k for k, _ in keep], [v for _, v in keep]

    save("val_loss_zoom.svg", Chart("Validation loss (exact), steps 1,000–2,000",
         "optimizer step", "cross-entropy (nats/char)", series(val_zoom, markers=True),
         subtitle=f"zoomed; {runs[0].label} is dashed and evaluated every 250 steps",
         y_format="{:.3f}"))  # fmt: skip
    save("train_loss_zoom.svg", Chart("Training loss (EMA), steps 1,000–2,000",
         "optimizer step", "cross-entropy (nats/char)", series(train_zoom),
         subtitle=f"zoomed; dropout-on batch loss; {runs[0].label} is dashed",
         y_format="{:.2f}"))  # fmt: skip
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--max-step", type=int, default=2000)
    p.add_argument("--sample-tokens", type=int, default=500)
    args = p.parse_args()
    runs = [Run(lbl, Path(d)) for lbl, d in (s.split("=", 1) for s in args.run)]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    cfg0 = runs[0].cfg
    data = load_prepared(processed_dir(cfg0.data))
    tok = CharTokenizer.load(runs[0].path / "tokenizer.json")
    train_text = tok.decode(data.splits.train.tolist())
    vocab, known = vocabulary(train_text), speakers(train_text)

    result: dict[str, Any] = {"runs": {}, "plots": plots(runs, out, args.max_step)}
    for r in runs:
        b = r.best()
        final_ckpt = list_checkpoints(r.path)[-1]
        entry: dict[str, Any] = {
            "path": str(r.path),
            "config": {
                "max_steps": r.cfg.training.max_steps,
                "schedule_horizon": r.cfg.training.schedule_horizon,
                "eval_interval": r.cfg.training.eval_interval,
            },
            "best_step": int(b["step"]),
            "best_val_loss_exact": b["val_loss_exact"],
            "val_exact_at": {
                s: (r.eval_at[s]["val_loss_exact"] if s in r.eval_at else None)
                for s in MATCHED_STEPS
            },
            "train_estimate_at": {
                s: (r.eval_at[s]["train_loss"] if s in r.eval_at else None) for s in MATCHED_STEPS
            },
            "batch_loss_mean_50_ending_at": {
                s: statistics.fmean(x["loss"] for x in r.train if s - 50 < x["step"] <= s)
                for s in MATCHED_STEPS
                if any(x["step"] == s for x in r.train)
            },
            "final_step": int(r.evals[-1]["step"]),
            "final_val_loss_exact": r.evals[-1]["val_loss_exact"],
            "exact_best": exact(r.cfg, best_checkpoint_path(r.path), data, device),
            "exact_final": exact(r.cfg, final_ckpt, data, device),
            "exact_at_max_step": (
                exact(r.cfg, checkpoint_path(r.path, args.max_step), data, device)
                if checkpoint_path(r.path, args.max_step).exists()
                else None
            ),
            "timing_full_run": timing(r),
            "timing_to_max_step": timing(r, args.max_step),
            "grad_norm": {
                "max_after_warmup": max(
                    x["grad_norm"]
                    for x in r.train
                    if r.cfg.training.warmup_steps < x["step"] <= args.max_step
                ),
                "median_to_max_step": statistics.median(
                    x["grad_norm"] for x in r.train if x["step"] <= args.max_step
                ),
                "clipped_fraction_to_max_step": sum(
                    x["grad_norm"] > r.cfg.training.grad_clip
                    for x in r.train
                    if x["step"] <= args.max_step
                )
                / min(len(r.train), args.max_step),
            },
            "memory_max_bytes": r.summary.get("memory_max_bytes"),
        }
        result["runs"][r.label] = entry
        log.info("%s: best %.4f @ %d | final %.4f", r.label, entry["best_val_loss_exact"],
                 entry["best_step"], entry["final_val_loss_exact"])  # fmt: skip

    # ---- reproduction / noise: every run vs the reference over shared steps ----------- #
    ref = runs[0]
    for r in runs[1:]:
        ref_loss = {int(x["step"]): x["loss"] for x in ref.train}
        shared = [x for x in r.train if int(x["step"]) in ref_loss and x["step"] <= args.max_step]
        d_train = [abs(x["loss"] - ref_loss[int(x["step"])]) for x in shared]
        shared_eval = sorted(set(ref.eval_at) & set(r.eval_at))
        d_val = {s: r.eval_at[s]["val_loss_exact"] - ref.eval_at[s]["val_loss_exact"]
                 for s in shared_eval if s <= args.max_step}  # fmt: skip
        d_lr = max(abs(x["lr"] - ref.train[int(x["step"]) - 1]["lr"]) for x in shared)
        param = None
        ca, cb = checkpoint_path(ref.path, args.max_step), checkpoint_path(r.path, args.max_step)
        if ca.exists() and cb.exists():
            sa = torch.load(ca, map_location="cpu", weights_only=True)["state"]["model"]
            sb = torch.load(cb, map_location="cpu", weights_only=True)["state"]["model"]
            diffs = [float((sa[k].float() - sb[k].float()).abs().max()) for k in sa]
            rel = [float((sa[k].float() - sb[k].float()).norm() / sa[k].float().norm())
                   for k in sa if sa[k].float().norm() > 0]  # fmt: skip
            param = {"max_abs": max(diffs), "max_relative_l2": max(rel),
                     "median_relative_l2": statistics.median(rel)}  # fmt: skip
        result["runs"][r.label]["vs_reference"] = {
            "reference": ref.label,
            "train_batch_loss_abs_diff": {
                "mean": statistics.fmean(d_train),
                "max": max(d_train),
                "first_step_diff": d_train[0],
                "mean_steps_1_100": statistics.fmean(d_train[:100]),
                "mean_last_100": statistics.fmean(d_train[-100:]),
            },
            "val_exact_diff_at_shared_evals": d_val,
            "lr_max_abs_diff": d_lr,
            f"param_diff_at_step_{args.max_step}": param,
        }

    # ---- matched samples from best checkpoints ---------------------------------------- #
    md = ["# M2.1 samples: best checkpoint of each run\n\n",
          f"Prompt `{SAMPLE_PROMPT!r}`, temperature {SAMPLE_DEC['temperature']}, top-k "
          f"{SAMPLE_DEC['top_k']}, {args.sample_tokens} new characters, seeds {SAMPLE_SEEDS} "
          "(the same five seeds for every model). Verbatim model output.\n"]  # fmt: skip
    result["samples"] = {}
    for r in runs:
        t = Trainer(r.cfg, data.splits.train, data.splits.val, device)
        t.load_state(best_checkpoint_path(r.path))
        model = t.model.eval()
        md.append(f"\n## {r.label} (best checkpoint, step {t.step})\n")
        rows: list[dict[str, Any]] = []
        for seed in SAMPLE_SEEDS:
            idx = torch.tensor([tok.encode(SAMPLE_PROMPT)], dtype=torch.long, device=device)
            ids = generate(model, idx, args.sample_tokens, seed=seed, **SAMPLE_DEC)
            text = tok.decode(ids[0, len(SAMPLE_PROMPT) :].tolist())
            m = longest_match(SAMPLE_PROMPT + text, train_text)
            ws, ss = word_stats(text, vocab), speaker_stats(text, known)
            row: dict[str, Any] = {
                "seed": seed,
                "text": text,
                "longest_verbatim_match": m.length,
                "longest_match_text": m.text,
                "overlap_32": ngram_overlap(SAMPLE_PROMPT + text, train_text, 32),
                "valid_word_fraction": ws.valid_fraction,
                "valid_word_fraction_len4": ws.valid_fraction_len4,
                "word_trigram_repetition": repetition_rate(text),
                "speaker_labels": ss.n_labels,
                "speaker_known_fraction": ss.known_fraction,
            }
            rows.append(row)
            md.append(
                f"\n### seed {seed} · longest train match {m.length} chars · word-trigram "
                f"repetition {row['word_trigram_repetition']:.0%} · valid ≥4-letter words "
                f"{ws.valid_fraction_len4:.0%}\n\n```text\n{SAMPLE_PROMPT}{text}\n```\n"
            )
        keys = (
            "longest_verbatim_match",
            "overlap_32",
            "valid_word_fraction",
            "valid_word_fraction_len4",
            "word_trigram_repetition",
            "speaker_known_fraction",
        )
        agg: dict[str, float] = {k: statistics.fmean(x[k] for x in rows) for k in keys}
        agg["longest_verbatim_match_max"] = max(x["longest_verbatim_match"] for x in rows)
        result["samples"][r.label] = {"step": t.step, "mean": agg, "rows": rows}
    atomic_write_json(out / "comparison.json", result)
    atomic_write_text(out / "samples.md", "".join(md))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()

# ruff: noqa: RUF001  (chart labels intentionally use typographic symbols)
"""Evaluate an extended-training phase against its parent (M3.2 vs M3.1).

Outputs in ``--out``:
* ``eval_m3_2.json``: exact val / holdout / train-slice losses for parent and child
  checkpoints, story metrics, entity-consistency checks, memorization overlap,
  extraction probe, throughput and memory.
* ``stories.md``: every story, labelled by model (for reading after blind scoring).
* ``blind_packet.md`` and ``blind_key.json``: the same stories shuffled under
  neutral ids (S01, S02, ...) for blind rating; the key maps ids to models.
* ``*.svg``: combined M3.1 + M3.2 trajectories with the schedule change marked.

Blind scores (``blind_scores.json``: ``{"S01": {"grammar": 2, ...}, ...}``) can be
unblinded later with ``--unblind``.

Usage:
    python scripts/evaluate_m3_2.py --parent runs/<m3_1> --run runs/<m3_2> --out docs/reports/m3_2
    python scripts/evaluate_m3_2.py --unblind docs/reports/m3_2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_chatgpt.analysis.story_eval import (
    WORD,
    KGramIndex,
    entity_check,
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
from mini_chatgpt.training.trainer import Trainer

log = setup_logging()
STORY_PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "One day, a boy found a strange box.",
    "A small dragon lived in a forest.",
    "Tom wanted to help his friend.",
    "There was a dog who was afraid of water.",
]
ENTITY_PROMPTS: list[dict[str, Any]] = [
    {
        "prompt": "Tom and Lily went into the forest. Tom carried a basket. "
        "Lily carried a lantern.",
        "names": ["Tom", "Lily"],
        "facts": [("Tom", "basket"), ("Lily", "lantern")],
    },
    {
        "prompt": "Mia had a red ball and Ben had a blue kite. They went to the park.",
        "names": ["Mia", "Ben"],
        "facts": [("Mia", "ball"), ("Ben", "kite")],
    },
    {
        "prompt": "Sam was a cat and Max was a dog. Sam liked to sleep in the sun. "
        "Max liked to dig in the garden.",
        "names": ["Sam", "Max"],
        "facts": [("Sam", "cat"), ("Max", "dog")],
    },
    {
        "prompt": "Anna, Leo and Zoe built a sandcastle. Anna found a shell. "
        "Leo found a stick. Zoe found a flag.",
        "names": ["Anna", "Leo", "Zoe"],
        "facts": [("Anna", "shell"), ("Leo", "stick"), ("Zoe", "flag")],
    },
    {
        "prompt": "Grandma gave Lucy a green hat and gave Jack a yellow scarf.",
        "names": ["Lucy", "Jack"],
        "facts": [("Lucy", "hat"), ("Jack", "scarf")],
    },
    {
        "prompt": "Ella lost her doll. Her brother Tim wanted to help her find it.",
        "names": ["Ella", "Tim"],
        "facts": [("Ella", "doll")],
    },
]
SEEDS = [1, 2]
RUBRIC = [
    "grammar",
    "character_consistency",
    "entity_tracking",
    "cause_effect",
    "story_progression",
    "ending_quality",
    "repetition",
]


def records(run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recs = dedupe_metrics(read_metrics(run / "metrics.jsonl"))
    return [r for r in recs if r["event"] == "train"], [r for r in recs if r["event"] == "eval"]


def plots(parent: Path, run: Path, out: Path, exact: dict[str, Any], switch: int) -> list[str]:
    ptr, pev = records(parent)
    ctr, cev = records(run)
    tr = [r for r in ptr if r["step"] <= switch] + ctr
    ev = [r for r in pev if r["step"] <= switch] + cev
    steps = [r["step"] for r in tr]
    vline = [(float(switch), "constant LR 6e-5 →")]
    written = []

    def save(name: str, chart: Chart) -> None:
        (out / name).write_text(render(chart), encoding="utf-8")
        written.append(name)

    sm = ema([r["loss"] for r in tr])
    late = [i for i, r in enumerate(tr) if r["step"] >= 2400]
    save(
        "combined_loss.svg",
        Chart(
            "M3.1 → M3.2: training and validation loss",
            "optimizer step",
            "cross-entropy (nats / BPE token)",
            [
                Series(
                    "train: EMA of batch loss (α = 0.02)",
                    [steps[i] for i in late],
                    [sm[i] for i in late],
                ),
                Series(
                    "validation: exact, full official V2 set",
                    [r["step"] for r in ev if r["step"] >= 2400],
                    [r["val_loss_exact"] for r in ev if r["step"] >= 2400],
                    slot=1,
                    markers=True,
                ),
            ],
            subtitle="steps 2,400–7,200; M3.1 cosine schedule until step 4,800, then a new "
            "constant-LR phase",
            vlines=vline,
            y_format="{:.2f}",
        ),
    )
    save(
        "combined_lr.svg",
        Chart(
            "Learning rate across both phases",
            "optimizer step",
            "learning rate",
            [Series("lr", steps, [r["lr"] for r in tr])],
            subtitle="M3.1: warmup 200 → 6e-4, cosine to 6e-5 at 4,800; M3.2: constant 6e-5",
            y_min=0.0,
            y_format="{:.1e}",
            vlines=vline,
        ),
    )
    save(
        "combined_val_bpb.svg",
        Chart(
            "Validation bits per byte (exact)",
            "optimizer step",
            "bits / UTF-8 byte",
            [
                Series(
                    "validation bpb",
                    [r["step"] for r in ev if r["step"] >= 1200],
                    [r["val_bpb_exact"] for r in ev if r["step"] >= 1200],
                    slot=1,
                    markers=True,
                )
            ],
            vlines=vline,
            y_format="{:.3f}",
        ),
    )
    rows = sorted(exact["checkpoints"].values(), key=lambda r: r["step"])
    xs = [r["step"] for r in rows]
    save(
        "generalization.svg",
        Chart(
            "Exact losses per checkpoint: unseen vs. seen text",
            "optimizer step",
            "cross-entropy (nats / BPE token)",
            [
                Series(
                    "validation (official V2)",
                    xs,
                    [r["val"]["loss"] for r in rows],
                    slot=1,
                    markers=True,
                ),
                Series(
                    "holdout (1% of train distribution, never trained on)",
                    xs,
                    [r["holdout"]["loss"] for r in rows],
                    slot=2,
                    markers=True,
                    dash="5 4",
                ),
                Series(
                    "train slice (first 5M training tokens, seen)",
                    xs,
                    [r["train_slice"]["loss"] for r in rows],
                    slot=0,
                    markers=True,
                ),
            ],
            vlines=vline,
            y_format="{:.3f}",
        ),
    )
    save(
        "combined_throughput.svg",
        Chart(
            "Training throughput across both phases",
            "optimizer step",
            "tokens / second",
            [
                Series("per step", steps, [r["tokens_per_s"] for r in tr], opacity=0.25, width=1),
                Series("EMA (α = 0.02)", steps, ema([r["tokens_per_s"] for r in tr])),
            ],
            y_min=0.0,
            y_format="{:,.0f}",
            vlines=vline,
        ),
    )
    return written


def unblind(out: Path) -> None:
    key = json.loads((out / "blind_key.json").read_text())
    scores = json.loads((out / "blind_scores.json").read_text())
    by_model: dict[str, list[dict[str, int]]] = {}
    for sid, s in scores.items():
        by_model.setdefault(key[sid]["model"], []).append(s)
    summary: dict[str, Any] = {}
    for model, rows in by_model.items():
        summary[model] = {c: statistics.fmean(r[c] for r in rows) for c in RUBRIC}
        summary[model]["total_percent"] = 100 * statistics.fmean(
            sum(r[c] for c in RUBRIC) / (2 * len(RUBRIC)) for r in rows
        )
        summary[model]["n"] = len(rows)
        for kind in ("story", "entity"):
            sub = [scores[s] for s in scores if key[s]["model"] == model and key[s]["kind"] == kind]
            summary[model][f"total_percent_{kind}"] = 100 * statistics.fmean(
                sum(r[c] for c in RUBRIC) / (2 * len(RUBRIC)) for r in sub
            )
    # paired comparison: same prompt and seed, different model
    pairs: dict[tuple[str, int], dict[str, float]] = {}
    for sid, s in scores.items():
        k = key[sid]
        pairs.setdefault((k["prompt"], k["seed"]), {})[k["model"]] = sum(s[c] for c in RUBRIC)
    models = sorted(by_model)
    if len(models) == 2:
        a, b = models
        diffs = [p[b] - p[a] for p in pairs.values() if a in p and b in p]
        summary["paired"] = {
            "models": [a, b],
            "n_pairs": len(diffs),
            "mean_total_diff_b_minus_a": statistics.fmean(diffs),
            "b_better": sum(d > 0 for d in diffs),
            "a_better": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs),
        }
    atomic_write_json(out / "blind_results.json", summary)
    print(json.dumps(summary, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path)
    p.add_argument("--run", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--unblind", type=Path, metavar="OUT_DIR")
    p.add_argument("--device", default="mps")
    p.add_argument("--train-slice", type=int, default=5_000_000)
    p.add_argument("--probe-n", type=int, default=64)
    args = p.parse_args()
    if args.unblind:
        unblind(args.unblind)
        return
    parent, run, out = args.parent, args.run, args.out
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    cfg = ExperimentConfig.from_dict(json.loads((run / "config.json").read_text()))
    device = select_device(args.device)
    data = load_prepared(processed_dir(cfg.data))
    tok = ByteBPETokenizer.load(run / "tokenizer.json")
    assert tok.fingerprint() == data.meta["tokenizer_fingerprint"]
    assert ByteBPETokenizer.load(parent / "tokenizer.json").fingerprint() == tok.fingerprint()
    pdir = processed_dir(cfg.data)
    manifest = json.loads((pdir / "manifest.json").read_text())
    holdout = torch.from_numpy(np.memmap(pdir / "holdout.bin", dtype="<u2", mode="c"))
    train_np, val_np = np.asarray(data.splits.train), np.asarray(data.splits.val)
    bpt = {
        "val": data.meta["val_bytes_per_token"],
        "holdout": manifest["streams"]["holdout"]["bytes_per_token"],
        "train_slice": manifest["streams"]["train"]["bytes_per_token"],
    }
    switch = int(json.loads((run / "continuation.json").read_text())["parent_step"])

    def load(path: Path) -> GPT:
        t = Trainer(cfg, data.splits.train, data.splits.val, device)
        t.load_state(path, continuation=True) if path.is_relative_to(parent) else t.load_state(path)
        return t.model.eval()

    # ---- exact losses per checkpoint ------------------------------------------------- #
    ckpts: list[tuple[str, Path]] = [
        (f"m3_1_step_{s}", checkpoint_path(parent, s)) for s in (3600, 4200, switch)
    ]
    ckpts += [(f"m3_2_step_{int(c.stem.split('_')[1])}", c) for c in list_checkpoints(run)]
    _, pev = records(parent)
    _, cev = records(run)
    logged = {int(r["step"]): r["val_loss_exact"] for r in pev + cev}
    exact: dict[str, Any] = {"checkpoints": {}}
    for name, path in ckpts:
        model = load(path)
        step = int(name.rsplit("_", 1)[1])
        row: dict[str, Any] = {"step": step, "path": str(path)}
        for split, stream in (
            ("val", data.splits.val),
            ("holdout", holdout),
            ("train_slice", data.splits.train[: args.train_slice]),
        ):
            if split == "val" and step not in (switch, cfg.training.max_steps):
                loss = logged[step]  # exact full-val loss logged during training
            else:
                loss = exact_loss(model, stream, cfg, device)
            row[split] = {
                "loss": loss,
                "bpb": loss / math.log(2) / bpt[split],
                "ppl": math.exp(loss),
            }
        row["val_recomputed_matches_log"] = (
            abs(row["val"]["loss"] - logged[step]) < 1e-6 if step in logged else None
        )
        row["gap_val_minus_train_slice"] = row["val"]["loss"] - row["train_slice"]["loss"]
        row["gap_holdout_minus_train_slice"] = row["holdout"]["loss"] - row["train_slice"]["loss"]
        exact["checkpoints"][name] = row
        log.info(
            "%s: val %.4f | holdout %.4f | train-slice %.4f",
            name,
            row["val"]["loss"],
            row["holdout"]["loss"],
            row["train_slice"]["loss"],
        )
    best_child = best_checkpoint_path(run)
    exact["m3_2_best_step"] = (
        int(torch.load(best_child, map_location="cpu", weights_only=True)["state"]["step"])
        if best_child.exists()
        else None
    )
    result: dict[str, Any] = {"exact": exact, "bytes_per_token": bpt, "switch_step": switch}
    result["plots"] = plots(parent, run, out, exact, switch)

    # ---- stories ----------------------------------------------------------------------- #
    final_step = cfg.training.max_steps
    models = {
        "M3.1@4800": load(checkpoint_path(parent, switch)),
        f"M3.2@{final_step}": load(checkpoint_path(run, final_step)),
    }
    vocab = {w.lower() for w in WORD.findall(tok.decode(train_np[:20_000_000].tolist()))}
    index = KGramIndex(train_np, [8, 16, 32, 48])
    items: list[dict[str, Any]] = []
    for model_name, model in models.items():
        for seed in SEEDS:
            for prompt in STORY_PROMPTS:
                items.append(
                    {
                        "model": model_name,
                        "kind": "story",
                        "seed": seed,
                        **sample(model, tok, prompt, seed, device),
                    }
                )
            for ent in ENTITY_PROMPTS:
                s = sample(model, tok, ent["prompt"], seed, device)
                s.update(entity_check(s["text"], ent["names"], ent["facts"]))
                items.append({"model": model_name, "kind": "entity", "seed": seed, **s})
    for it in items:
        ids = np.asarray(tok.encode(it["prompt"]) + it.pop("new_token_ids"))
        it.update(story_metrics(it["text"], vocab), **index.overlap(ids))
    result["stories"] = items

    def mean(rows: list[dict[str, Any]], k: str) -> float:
        return statistics.fmean(float(r[k]) for r in rows)

    summ: dict[str, Any] = {}
    for m in models:
        for kind in ("story", "entity", "all"):
            rows = [r for r in items if r["model"] == m and (kind == "all" or r["kind"] == kind)]
            d = {
                k: mean(rows, k)
                for k in (
                    "new_tokens",
                    "completed_with_eot",
                    "word_trigram_repetition",
                    "distinct_2",
                    "valid_word_fraction",
                    "frac_8gram_in_train",
                    "frac_16gram_in_train",
                    "frac_32gram_in_train",
                )
            }
            d["max_longest_match_at_least"] = max(r["longest_match_at_least"] for r in rows)
            if kind == "entity":
                d["names_kept_fraction"] = mean(rows, "names_kept_fraction")
                d["fact_supports"] = sum(r["fact_supports"] for r in rows)
                d["fact_conflicts"] = sum(r["fact_conflicts"] for r in rows)
                d["stories_with_any_conflict"] = sum(r["fact_conflicts"] > 0 for r in rows)
                d["stories_introducing_new_names"] = sum(
                    bool(r["new_capitalized_names"]) for r in rows
                )
            d["n"] = len(rows)
            summ[f"{m}/{kind}"] = d
    result["story_summary"] = summ

    # ---- labelled stories + blind packet ------------------------------------------- #
    md = [
        "# M3.2 stories (labelled; verbatim model output)\n\n",
        "Temperature 0.7, top-k 40, up to 300 new tokens, stopped at `<|endoftext|>`; "
        f"seeds {SEEDS}; the same prompt+seed for both models.\n",
    ]
    for it in items:
        extra = ""
        if it["kind"] == "entity":
            extra = (
                f" · names kept {it['names_kept_fraction']:.0%} · fact supports "
                f"{it['fact_supports']} / conflicts {it['fact_conflicts']}"
            )
        md.append(
            f"\n### {it['model']} · {it['kind']} · seed {it['seed']}{extra}\n\n"
            f"```text\n{it['prompt']}{it['text']}\n```\n"
        )
    atomic_write_text(out / "stories.md", "".join(md))
    order = list(range(len(items)))
    random.Random(20261113).shuffle(order)
    key, packet = (
        {},
        [
            "# Blind story packet\n\nModel labels are hidden. Rate each story on "
            "the rubric in the M3.2 report (0/1/2 per criterion).\n"
        ],
    )
    for n, i in enumerate(order, 1):
        sid = f"S{n:02d}"
        it = items[i]
        key[sid] = {
            "model": it["model"],
            "kind": it["kind"],
            "seed": it["seed"],
            "prompt": it["prompt"],
        }
        packet.append(f"\n## {sid}\n\n```text\n{it['prompt']}{it['text']}\n```\n")
    atomic_write_text(out / "blind_packet.md", "".join(packet))
    atomic_write_json(out / "blind_key.json", key)

    # ---- extraction probe (same story choices as M3.1) ----------------------------- #
    eot = tok.eot_id

    def starts(arr: np.ndarray) -> list[int]:
        e = np.flatnonzero(arr == eot)
        s = np.concatenate([[0], e[:-1] + 1])
        return [int(a) for a, b in zip(s, e, strict=True) if b - a >= 96]

    probe: dict[str, Any] = {}
    for split, arr in (("train", train_np), ("val", val_np)):
        cands = starts(arr[:20_000_000] if split == "train" else arr)
        chosen = np.random.default_rng(11).choice(cands, args.probe_n, replace=False)
        prompts = torch.tensor(np.stack([arr[c : c + 32] for c in chosen]).astype(np.int64))
        truth = np.stack([arr[c + 32 : c + 96] for c in chosen]).astype(np.int64)
        for name, model in models.items():
            gen = generate(model, prompts.to(device), 64, greedy=True)[:, 32:].cpu().numpy()
            pref = [
                int(np.argmin(np.append(g == t, False))) for g, t in zip(gen, truth, strict=True)
            ]
            probe[f"{name}/{split}"] = {
                "mean": statistics.fmean(pref),
                "median": statistics.median(pref),
                "max": max(pref),
                "full_64": sum(x == 64 for x in pref),
            }
    result["extraction_probe"] = probe

    # ---- throughput and memory ----------------------------------------------------- #
    def summary(r: Path) -> dict[str, Any]:
        s = json.loads(sorted(r.glob("summary_step*.json"))[-1].read_text())
        return {
            "timing": s["timing_this_process"],
            "memory_max_bytes": s["memory_max_bytes"],
            "steps": s["steps"],
            "tokens_seen": s["tokens_seen"],
        }

    ctr, _ = records(run)
    result["compute"] = {
        "m3_1": summary(parent),
        "m3_2": summary(run),
        "m3_2_wall_min": (cev[-1]["time"] - ctr[0]["time"]) / 60,
        "m3_2_step_time_min": sum(r["step_time_s"] for r in ctr) / 60,
    }
    result["wall_s"] = time.perf_counter() - t_start
    atomic_write_json(out / "eval_m3_2.json", result)
    log.info("wrote %s (%.0f s)", out, result["wall_s"])


if __name__ == "__main__":
    main()

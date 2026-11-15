# ruff: noqa: RUF001  (chart labels intentionally use typographic symbols)
"""M3.3 controlled evaluation: 12.4M (M3.1, M3.2) vs 27.6M (M3.3) on TinyStories.

Models: A = M3.1 step 4800 (12.4M), B = M3.2 step 7200 (12.4M, +50% tokens),
C = M3.3 step 4800 (27.6M, same tokens as A).

Outputs in ``--out``:
* ``eval_m3_3.json``: exact val / holdout / train-slice losses, bits per byte,
  perplexity; estimated training FLOPs and equal-compute brackets; throughput and
  memory; measured story metrics; memorization (k-gram overlap, extraction probe).
* ``blind_packet.md`` / ``blind_key.json``: every generated story under an anonymous
  id, in seeded random order; the key maps ids to models.
* ``stories.md``: the same stories labelled by model (read only after scoring).
* ``*.svg``: validation loss vs tokens and vs estimated training FLOPs; train loss.

``--unblind OUT`` reads ``blind_scores.json`` (the rubric in ``rubric.md``) and writes
``blind_results.json`` with means, bootstrap 95% intervals, paired comparisons,
sign tests and contradiction rates.

FLOP estimate per training token (Kaplan et al. 2020 style; stated assumption):
6 * N_total + 6 * n_layer * n_ctx * d_model, where N_total includes the tied
embedding (which carries the LM-head matmul). Evaluation FLOPs are excluded.
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

from mini_chatgpt.analysis.story_eval import WORD, KGramIndex, exact_loss, sample, story_metrics
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
)
from mini_chatgpt.training.run import dedupe_metrics, read_metrics, setup_logging
from mini_chatgpt.training.trainer import Trainer

log = setup_logging()
GRADED = ["G", "E", "C", "N", "R"]  # grammar, entity, coherence, ending, repetition (0-2)
LABELS = {"A": "M3.1 12.4M @4800", "B": "M3.2 12.4M @7200", "C": "M3.3 27.6M @4800"}


def flops_per_token(cfg: ExperimentConfig) -> float:
    m = cfg.model
    n = m.estimate_parameters()["total"]
    return 6.0 * n + 6.0 * m.n_layer * m.context_length * m.n_embd


def records(run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recs = dedupe_metrics(read_metrics(run / "metrics.jsonl"))
    return [r for r in recs if r["event"] == "train"], [r for r in recs if r["event"] == "eval"]


def bootstrap_ci(xs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, dtype=float)
    means = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p: float = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------- #
def unblind(out: Path) -> None:
    key = json.loads((out / "blind_key.json").read_text())
    scores = json.loads((out / "blind_scores.json").read_text())
    missing = set(key) - set(scores)
    if missing:
        raise SystemExit(f"{len(missing)} stories are not scored yet")
    res: dict[str, Any] = {
        "n_scored": len(scores),
        "per_model": {},
        "paired": {},
        "per_category": {},
    }
    by: dict[str, list[dict[str, Any]]] = {}
    for sid, s in scores.items():
        by.setdefault(key[sid]["model"], []).append({**s, **key[sid]})
    for m, rows in sorted(by.items()):
        d: dict[str, Any] = {"n": len(rows), "label": LABELS[m]}
        for c in GRADED:
            v = [r[c] for r in rows]
            d[c] = {"mean": statistics.fmean(v), "ci95": bootstrap_ci(v)}
        tot = [sum(r[c] for c in GRADED) for r in rows]
        d["total_0_10"] = {"mean": statistics.fmean(tot), "ci95": bootstrap_ci(tot)}
        k = sum(r["X"] for r in rows)
        d["contradiction_rate"] = {
            "k": k,
            "n": len(rows),
            "rate": k / len(rows),
            "wilson95": wilson(k, len(rows)),
        }
        res["per_model"][m] = d
    item: dict[tuple[str, int], dict[str, dict[str, Any]]] = {
        (r["prompt_id"], r["seed"]): {} for rows in by.values() for r in rows
    }
    for m, rows in by.items():
        for r in rows:
            item[(r["prompt_id"], r["seed"])][m] = r
    for a, b in (("C", "A"), ("C", "B"), ("B", "A")):
        pairs = [v for v in item.values() if a in v and b in v]
        diff = [sum(v[a][c] for c in GRADED) - sum(v[b][c] for c in GRADED) for v in pairs]
        wins, losses = sum(x > 0 for x in diff), sum(x < 0 for x in diff)
        x_only_a = sum(v[a]["X"] == 1 and v[b]["X"] == 0 for v in pairs)
        x_only_b = sum(v[a]["X"] == 0 and v[b]["X"] == 1 for v in pairs)
        crit = {c: statistics.fmean(v[a][c] - v[b][c] for v in pairs) for c in GRADED}
        res["paired"][f"{a}-{b}"] = {
            "n_pairs": len(pairs),
            "mean_total_diff": statistics.fmean(diff),
            "ci95": bootstrap_ci(diff),
            "wins": wins,
            "losses": losses,
            "ties": len(diff) - wins - losses,
            "sign_test_p": sign_test(wins, losses),
            "mean_diff_by_criterion": crit,
            "contradiction_discordant": {
                f"only_{a}": x_only_a,
                f"only_{b}": x_only_b,
                "exact_binomial_p": sign_test(x_only_a, x_only_b),
            },
        }
    cats = sorted({r["category"] for rows in by.values() for r in rows})
    for cat in cats:
        res["per_category"][cat] = {
            m: {
                "total_mean": statistics.fmean(
                    sum(r[c] for c in GRADED) for r in rows if r["category"] == cat
                ),
                "E_mean": statistics.fmean(r["E"] for r in rows if r["category"] == cat),
                "X_rate": statistics.fmean(r["X"] for r in rows if r["category"] == cat),
            }
            for m, rows in sorted(by.items())
        }
    atomic_write_json(out / "blind_results.json", res)
    print(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m31", type=Path)
    p.add_argument("--m32", type=Path)
    p.add_argument("--m33", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--unblind", type=Path, metavar="OUT_DIR")
    p.add_argument("--device", default="mps")
    p.add_argument("--train-slice", type=int, default=5_000_000)
    args = p.parse_args()
    if args.unblind:
        unblind(args.unblind)
        return
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    runs = {"A": args.m31, "B": args.m32, "C": args.m33}
    cfgs = {
        k: ExperimentConfig.from_dict(json.loads((r / "config.json").read_text()))
        for k, r in runs.items()
    }
    device = select_device(args.device)
    data = load_prepared(processed_dir(cfgs["C"].data))
    tok = ByteBPETokenizer.load(runs["C"] / "tokenizer.json")
    fps = {tok.fingerprint()} | {
        ByteBPETokenizer.load(r / "tokenizer.json").fingerprint() for r in runs.values()
    }
    assert fps == {data.meta["tokenizer_fingerprint"]}, "tokenizer mismatch across runs"
    pdir = processed_dir(cfgs["C"].data)
    manifest = json.loads((pdir / "manifest.json").read_text())
    holdout = torch.from_numpy(np.memmap(pdir / "holdout.bin", dtype="<u2", mode="c"))
    train_np, val_np = np.asarray(data.splits.train), np.asarray(data.splits.val)
    bpt = {
        "val": data.meta["val_bytes_per_token"],
        "holdout": manifest["streams"]["holdout"]["bytes_per_token"],
        "train_slice": manifest["streams"]["train"]["bytes_per_token"],
    }

    def load(key: str, path: Path) -> GPT:
        t = Trainer(cfgs[key], data.splits.train, data.splits.val, device)
        t.load_state(path)
        return t.model.eval()

    steps = {"A": 4800, "B": 7200, "C": cfgs["C"].training.max_steps}
    models = {k: load(k, checkpoint_path(runs[k], steps[k])) for k in runs}
    c_best = int(
        torch.load(best_checkpoint_path(runs["C"]), map_location="cpu", weights_only=True)["state"][
            "step"
        ]
    )

    # ---- exact losses ---------------------------------------------------------------- #
    result: dict[str, Any] = {
        "labels": LABELS,
        "bytes_per_token": bpt,
        "m3_3_best_step": c_best,
        "exact": {},
    }
    for k, model in models.items():
        row: dict[str, Any] = {"step": steps[k], "params": cfgs[k].model.estimate_parameters()}
        for split, stream in (
            ("val", data.splits.val),
            ("holdout", holdout),
            ("train_slice", data.splits.train[: args.train_slice]),
        ):
            loss = exact_loss(model, stream, cfgs[k], device)
            row[split] = {
                "loss": loss,
                "bpb": loss / math.log(2) / bpt[split],
                "ppl": math.exp(loss),
            }
        row["gap_holdout_minus_train_slice"] = row["holdout"]["loss"] - row["train_slice"]["loss"]
        result["exact"][k] = row
        log.info(
            "%s: val %.4f holdout %.4f train %.4f",
            k,
            row["val"]["loss"],
            row["holdout"]["loss"],
            row["train_slice"]["loss"],
        )

    # ---- compute accounting ------------------------------------------------------------ #
    tr = {k: records(r) for k, r in runs.items()}
    fpt = {k: flops_per_token(cfgs[k]) for k in runs}
    tps = cfgs["A"].training.batch_size * cfgs["A"].training.gradient_accumulation_steps * 512
    comp: dict[str, Any] = {"flops_per_token": fpt, "tokens_per_step": tps}
    a_flops = 4800 * tps * fpt["A"]
    b_flops = 7200 * tps * fpt["B"]
    c_flops = steps["C"] * tps * fpt["C"]
    comp["train_flops"] = {"A": a_flops, "B": b_flops, "C": c_flops}
    comp["c_step_matching_A_flops"] = a_flops / (tps * fpt["C"])
    comp["c_step_matching_B_flops"] = b_flops / (tps * fpt["C"])

    def wall_and_steps(trs: list[dict[str, Any]], evs: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "wall_min": (evs[-1]["time"] - evs[0]["time"]) / 60,
            "step_min": sum(r["step_time_s"] for r in trs) / 60,
            "median_tok_s": statistics.median(r["tokens_per_s"] for r in trs[10:]),
        }

    comp["measured"] = {k: wall_and_steps(*tr[k]) for k in runs}
    comp["measured"]["A+B (M3.1 then M3.2)"] = {
        "wall_min": comp["measured"]["A"]["wall_min"] + comp["measured"]["B"]["wall_min"]
    }
    c_evals = {int(r["step"]): r for r in tr["C"][1]}
    a_evals = {int(r["step"]): r for r in tr["A"][1]}
    b_evals = {int(r["step"]): r for r in tr["B"][1]}
    comp["c_val_at_checkpoints"] = {s: c_evals[s]["val_loss_exact"] for s in sorted(c_evals)}
    for name, target in (
        ("A", comp["c_step_matching_A_flops"]),
        ("B", comp["c_step_matching_B_flops"]),
    ):
        lo = max(s for s in c_evals if s <= target)
        hi = min(s for s in c_evals if s >= target)
        comp[f"equal_flops_bracket_{name}"] = {
            "target_step": target,
            "lower": [lo, c_evals[lo]["val_loss_exact"], lo / target],
            "upper": [hi, c_evals[hi]["val_loss_exact"], hi / target],
            "reference_val": (a_evals[4800] if name == "A" else b_evals[7200])["val_loss_exact"],
        }
    # equal wall-clock: M3.3 checkpoint closest to A's and A+B's training time
    c_start = tr["C"][1][0]["time"]
    c_time = {int(r["step"]): (r["time"] - c_start) / 60 for r in tr["C"][1]}
    comp["c_minutes_at_eval"] = c_time
    for name, mins in (
        ("A", comp["measured"]["A"]["wall_min"]),
        ("A+B", comp["measured"]["A+B (M3.1 then M3.2)"]["wall_min"]),
    ):
        lo_step: int = max((s for s in c_time if c_time[s] <= mins), default=0)
        hi_step: int | None = min((s for s in c_time if c_time[s] >= mins), default=None)
        comp[f"equal_wallclock_bracket_{name}"] = {
            "minutes": mins,
            "lower_step": lo_step,
            "upper_step": hi_step,
        }
    comp["memory_max_bytes"] = {
        k: json.loads(sorted(r.glob("summary_step*.json"))[-1].read_text())["memory_max_bytes"]
        for k, r in runs.items()
    }
    result["compute"] = comp

    # ---- plots --------------------------------------------------------------------------- #
    ab_ev = [r for r in tr["A"][1]] + [r for r in tr["B"][1]]
    c_ev = tr["C"][1]

    def series(
        evs: list[dict[str, Any]],
        x: str,
        label: str,
        slot: int,
        fpt_: float,
        dash: str | None = None,
    ) -> Series:
        rows = [r for r in evs if r["step"] >= 600]
        xs = [
            r["step"] * tps / 1e6 if x == "tokens" else r["step"] * tps * fpt_ / 1e15 for r in rows
        ]
        return Series(
            label, xs, [r["val_loss_exact"] for r in rows], slot=slot, markers=True, dash=dash
        )

    vl = [(4800 * tps / 1e6, "M3.1 end / M3.2 start")]
    (out / "val_vs_tokens.svg").write_text(
        render(
            Chart(
                "Validation loss vs. training tokens (equal-token comparison)",
                "training tokens (millions)",
                "exact validation loss (nats / token)",
                [
                    series(
                        ab_ev, "tokens", "12.4M: M3.1 (cosine) → M3.2 (constant 6e-5)", 0, fpt["A"]
                    ),
                    series(c_ev, "tokens", "27.6M: M3.3 (cosine)", 1, fpt["C"]),
                ],
                subtitle="from step 600; identical data order, tokenizer and schedule shape",
                vlines=vl,
                y_format="{:.2f}",
            )
        ),
        encoding="utf-8",
    )
    (out / "val_vs_flops.svg").write_text(
        render(
            Chart(
                "Validation loss vs. estimated training compute (equal-compute view)",
                "estimated training FLOPs (×10^15)",
                "exact validation loss (nats / token)",
                [
                    series(ab_ev, "flops", "12.4M: M3.1 → M3.2", 0, fpt["A"]),
                    series(c_ev, "flops", "27.6M: M3.3", 1, fpt["C"]),
                ],
                subtitle="FLOPs ≈ (6·N + 6·L·T·C) per token; evaluation excluded",
                y_format="{:.2f}",
            )
        ),
        encoding="utf-8",
    )
    s_tr, c_tr = tr["A"][0] + tr["B"][0], tr["C"][0]
    (out / "train_loss.svg").write_text(
        render(
            Chart(
                "Training loss (EMA of per-step batch loss, α = 0.02)",
                "optimizer step",
                "cross-entropy (nats / token)",
                [
                    Series(
                        "12.4M: M3.1 → M3.2",
                        [r["step"] for r in s_tr if r["step"] >= 600],
                        [
                            v
                            for r, v in zip(s_tr, ema([r["loss"] for r in s_tr]), strict=True)
                            if r["step"] >= 600
                        ],
                    ),
                    Series(
                        "27.6M: M3.3",
                        [r["step"] for r in c_tr if r["step"] >= 600],
                        [
                            v
                            for r, v in zip(c_tr, ema([r["loss"] for r in c_tr]), strict=True)
                            if r["step"] >= 600
                        ],
                        slot=1,
                    ),
                ],
                vlines=[(4800.0, "M3.2 phase →")],
                y_format="{:.2f}",
            )
        ),
        encoding="utf-8",
    )
    (out / "grad_norm.svg").write_text(
        render(
            Chart(
                "Gradient norm (EMA, before clipping at 1.0)",
                "optimizer step",
                "global L2 norm",
                [
                    Series(
                        "12.4M: M3.1",
                        [r["step"] for r in tr["A"][0]],
                        ema([r["grad_norm"] for r in tr["A"][0]]),
                    ),
                    Series(
                        "27.6M: M3.3",
                        [r["step"] for r in c_tr],
                        ema([r["grad_norm"] for r in c_tr]),
                        slot=1,
                    ),
                ],
                y_min=0.0,
                y_max=2.0,
                hlines=[(1.0, "clip")],
            )
        ),
        encoding="utf-8",
    )
    result["plots"] = ["val_vs_tokens.svg", "val_vs_flops.svg", "train_loss.svg", "grad_norm.svg"]

    # ---- stories (pre-registered prompts) ------------------------------------------ #
    spec = json.loads((out / "entity_prompts.json").read_text())
    vocab = {w.lower() for w in WORD.findall(tok.decode(train_np[:20_000_000].tolist()))}
    index = KGramIndex(train_np, [8, 16, 32, 48])
    items: list[dict[str, Any]] = []
    for pr in spec["prompts"]:
        for seed in spec["seeds"]:
            for k, model in models.items():
                s = sample(
                    model,
                    tok,
                    pr["prompt"],
                    seed,
                    device,
                    max_new=spec["decoding"]["max_new_tokens"],
                )
                ids = np.asarray(tok.encode(pr["prompt"]) + s.pop("new_token_ids"))
                s.update(story_metrics(s["text"], vocab), **index.overlap(ids))
                items.append(
                    {
                        "model": k,
                        "prompt_id": pr["id"],
                        "category": pr["category"],
                        "facts": pr["facts"],
                        **s,
                    }
                )
    result["stories"] = items
    summ = {}
    for k in runs:
        rows = [r for r in items if r["model"] == k]
        summ[k] = {
            # k-gram fractions are None for stories shorter than k tokens: skip those
            m: statistics.fmean([float(r[m]) for r in rows if r[m] is not None])
            for m in (
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
        summ[k]["max_longest_match_at_least"] = max(r["longest_match_at_least"] for r in rows)
        summ[k]["n"] = len(rows)
    result["story_summary"] = summ

    order = list(range(len(items)))
    random.Random(20261114).shuffle(order)
    key: dict[str, Any] = {}
    packet = [
        "# M3.3 blind packet\n\nRate every story with `rubric.md`. Model identities "
        "are hidden; facts to check are listed per story.\n"
    ]
    for n, i in enumerate(order, 1):
        sid = f"S{n:03d}"
        it = items[i]
        key[sid] = {
            "model": it["model"],
            "prompt_id": it["prompt_id"],
            "category": it["category"],
            "seed": it["seed"],
        }
        facts = "; ".join(it["facts"])
        packet.append(f"\n## {sid}\nFacts: {facts}\n\n```text\n{it['prompt']}{it['text']}\n```\n")
    atomic_write_text(out / "blind_packet.md", "".join(packet))
    atomic_write_json(out / "blind_key.json", key)
    md = ["# M3.3 stories (labelled; read after blind scoring)\n"]
    for it in sorted(items, key=lambda r: (r["prompt_id"], r["seed"], r["model"])):
        md.append(
            f"\n### {it['prompt_id']} · seed {it['seed']} · {LABELS[it['model']]}\n\n"
            f"```text\n{it['prompt']}{it['text']}\n```\n"
        )
    atomic_write_text(out / "stories.md", "".join(md))

    # ---- extraction probe ---------------------------------------------------------------- #
    eot = tok.eot_id

    def starts(arr: np.ndarray) -> list[int]:
        e = np.flatnonzero(arr == eot)
        s = np.concatenate([[0], e[:-1] + 1])
        return [int(a) for a, b in zip(s, e, strict=True) if b - a >= 96]

    probe: dict[str, Any] = {}
    for split, arr in (("train", train_np), ("val", val_np)):
        cands = starts(arr[:20_000_000] if split == "train" else arr)
        chosen = np.random.default_rng(11).choice(cands, 64, replace=False)
        prompts = torch.tensor(np.stack([arr[c : c + 32] for c in chosen]).astype(np.int64))
        truth = np.stack([arr[c + 32 : c + 96] for c in chosen]).astype(np.int64)
        for k, model in models.items():
            gen = generate(model, prompts.to(device), 64, greedy=True)[:, 32:].cpu().numpy()
            pref = [
                int(np.argmin(np.append(g == t, False))) for g, t in zip(gen, truth, strict=True)
            ]
            probe[f"{k}/{split}"] = {
                "mean": statistics.fmean(pref),
                "max": max(pref),
                "full_64": sum(x == 64 for x in pref),
            }
    result["extraction_probe"] = probe
    result["wall_s"] = time.perf_counter() - t_start
    atomic_write_json(out / "eval_m3_3.json", result)
    log.info("wrote %s (%.0f s); %d stories in the blind packet", out, result["wall_s"], len(items))


if __name__ == "__main__":
    main()

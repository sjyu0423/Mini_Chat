"""M6.2 learning curves, overlaid on M4.1 (both from their metrics.jsonl).

Usage: python scripts/plot_m6_2.py runs/<m6_2 run> docs/reports/m6_2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mini_chatgpt.analysis.svgplot import Chart, Series, ema, render

M41 = Path("runs/m4_1_general_s_bpe8192-20261114-190514")


def load(run: Path) -> dict[str, list[dict[str, Any]]]:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    return {e: [r for r in rows if r["event"] == e] for e in ("train", "eval", "memory")}


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    new, old = load(run), load(M41)

    def save(name: str, chart: Chart) -> None:
        (out / name).write_text(render(chart), encoding="utf-8")

    def tok(rows: list[dict[str, Any]]) -> list[float]:
        return [r["tokens_seen"] / 1e6 for r in rows]

    def val(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [r.get(f"{key}_exact", r[key]) for r in rows]

    save("loss.svg", Chart(
        "Training and validation loss: M6.2 (29.7M) vs M4.1 (14.0M)",
        "training tokens (millions)", "loss (nats/token)",
        [Series("M4.1 train (EMA)", tok(old["train"]), ema([r["loss"] for r in old["train"]]),
                slot=0, opacity=0.6),
         Series("M4.1 validation (exact)", tok(old["eval"]), val(old["eval"], "val_loss"),
                slot=0, markers=True, dash="4 3"),
         Series("M6.2 train (EMA)", tok(new["train"]), ema([r["loss"] for r in new["train"]]),
                slot=1, opacity=0.6),
         Series("M6.2 validation (exact)", tok(new["eval"]), val(new["eval"], "val_loss"),
                slot=1, markers=True, dash="4 3")],
        subtitle="same data, tokenizer, order, schedule and 157.3M tokens; only capacity differs",
        y_min=2.5, y_max=6.0, height=420))  # fmt: skip
    later = [r for r in new["eval"] if r["step"] >= 800]
    later_old = [r for r in old["eval"] if r["step"] >= 800]
    save("val_bpb.svg", Chart(
        "Validation bits per byte", "training tokens (millions)", "bits per UTF-8 byte",
        [Series("M4.1 (14.0M)", tok(later_old), val(later_old, "val_bpb"), slot=0, markers=True),
         Series("M6.2 (29.7M)", tok(later), val(later, "val_bpb"), slot=1, markers=True)],
        subtitle="exact, full validation split; from step 800", height=360))  # fmt: skip
    save("lr.svg", Chart("Learning rate (identical schedules)", "step", "learning rate",
                         [Series("lr", [r["step"] for r in new["train"]],
                                 [r["lr"] for r in new["train"]], slot=1)],
                         height=280, y_format="{:.1e}"))  # fmt: skip
    save("grad_norm.svg", Chart(
        "Gradient norm before clipping (EMA)", "step", "global L2 norm",
        [Series("M4.1", [r["step"] for r in old["train"]],
                ema([r["grad_norm"] for r in old["train"]]), slot=0),
         Series("M6.2", [r["step"] for r in new["train"]],
                ema([r["grad_norm"] for r in new["train"]]), slot=1)],
        y_min=0.0, y_max=2.0, height=300))  # fmt: skip
    save("throughput.svg", Chart(
        "Training throughput (EMA)", "step", "tokens per second",
        [Series("M4.1", [r["step"] for r in old["train"]],
                ema([r["tokens_per_s"] for r in old["train"]]), slot=0),
         Series("M6.2", [r["step"] for r in new["train"]],
                ema([r["tokens_per_s"] for r in new["train"]]), slot=1)],
        y_min=0.0, height=300))  # fmt: skip
    if new["memory"]:
        save("memory.svg", Chart(
            "MPS driver memory pool", "step", "GB",
            [Series("M4.1", [r["step"] for r in old["memory"]],
                    [r["mps_driver_allocated"] / 1e9 for r in old["memory"]], slot=0),
             Series("M6.2", [r["step"] for r in new["memory"]],
                    [r["mps_driver_allocated"] / 1e9 for r in new["memory"]], slot=1)],
            y_min=0.0, height=300))  # fmt: skip
    keys = ("step", "tokens_seen", "lr", "train_loss", "val_loss_exact", "val_bpb_exact",
            "val_ppl", "is_best")  # fmt: skip
    points = {
        "m6_2": [{k: r.get(k) for k in keys} for r in new["eval"]],
        "m4_1": [{k: r.get(k) for k in keys} for r in old["eval"]],
    }
    (out / "curve_points.json").write_text(json.dumps(points, indent=1) + "\n")


if __name__ == "__main__":
    main()

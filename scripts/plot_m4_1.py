"""Learning curves for the M4.1 report from a run's metrics.jsonl.

Usage: python scripts/plot_m4_1.py runs/<m4_1 run> docs/reports/m4_1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mini_chatgpt.analysis.svgplot import Chart, Series, ema, render


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["event"] == "train"]
    evals = [r for r in rows if r["event"] == "eval"]
    mem = [r for r in rows if r["event"] == "memory"]
    tok_m = [r["tokens_seen"] / 1e6 for r in train]

    def save(name: str, chart: Chart) -> None:
        (out / name).write_text(render(chart), encoding="utf-8")

    loss = [r["loss"] for r in train]
    ev_x = [r["tokens_seen"] / 1e6 for r in evals]
    save("loss.svg", Chart(
        "M4.1 training and validation loss", "training tokens (millions)", "loss (nats/token)",
        [Series("train (per step)", tok_m, loss, slot=0, width=0.6, opacity=0.35),
         Series("train (EMA 0.02)", tok_m, ema(loss), slot=0),
         Series("validation (exact, full split)", ev_x,
                [r.get("val_loss_exact", r["val_loss"]) for r in evals], slot=1, markers=True)],
        subtitle="6L/6H/384, 13.99M parameters; single pass over FineWeb-Edu + Cosmopedia v2",
        y_min=2.0, y_max=9.0, height=420))  # fmt: skip
    later = [r for r in evals if r["step"] >= 800]
    save("val_bpb.svg", Chart(
        "Validation bits per byte", "training tokens (millions)", "bits per UTF-8 byte",
        [Series("validation (exact)", [r["tokens_seen"] / 1e6 for r in later],
                [r.get("val_bpb_exact", r["val_bpb"]) for r in later], slot=1, markers=True)],
        subtitle="tokenizer-independent; from step 800", height=360))  # fmt: skip
    save("lr.svg", Chart("Learning rate", "step", "learning rate",
                         [Series("lr", [r["step"] for r in train], [r["lr"] for r in train])],
                         subtitle="200-step warmup to 6e-4, cosine to 6e-5 at step 9,600",
                         height=300, y_format="{:.1e}"))  # fmt: skip
    save("grad_norm.svg", Chart(
        "Gradient norm (before clipping at 1.0)", "step", "global L2 norm",
        [Series("per step", [r["step"] for r in train], [r["grad_norm"] for r in train],
                width=0.6, opacity=0.35),
         Series("EMA 0.02", [r["step"] for r in train], ema([r["grad_norm"] for r in train]))],
        y_min=0.0, y_max=3.0, height=320))  # fmt: skip
    tps = [r["tokens_per_s"] for r in train]
    save("throughput.svg", Chart(
        "Training throughput", "step", "tokens per second",
        [Series("per step", [r["step"] for r in train], tps, width=0.6, opacity=0.35),
         Series("EMA 0.02", [r["step"] for r in train], ema(tps))],
        y_min=0.0, height=300))  # fmt: skip
    if mem:
        key = "mps_driver_allocated"
        save("memory.svg", Chart(
            "MPS memory", "step", "GB",
            [Series("driver pool", [r["step"] for r in mem], [r[key] / 1e9 for r in mem]),
             Series("allocated tensors", [r["step"] for r in mem],
                    [r["mps_current_allocated"] / 1e9 for r in mem], slot=1)],
            y_min=0.0, height=300))  # fmt: skip
    summary: dict[str, Any] = {"evals": [{k: r.get(k) for k in (
        "step", "tokens_seen", "lr", "train_loss", "val_loss_exact", "val_bpb_exact",
        "val_ppl", "is_best")} for r in evals]}  # fmt: skip
    (out / "curve_points.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()

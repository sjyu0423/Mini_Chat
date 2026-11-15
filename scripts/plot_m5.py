"""Learning curves for the M5 SFT report from a run's metrics.jsonl.

Usage: python scripts/plot_m5.py runs/<m5 run> docs/reports/m5 [title prefix, default "M5"]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mini_chatgpt.analysis.svgplot import Chart, Series, ema, render


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else "M5"
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["event"] == "train"]
    evals = [r for r in rows if r["event"] == "eval"]
    ep = [r["epoch"] for r in train]
    ev_ep = [r["epoch"] for r in evals]

    def save(name: str, chart: Chart) -> None:
        (out / name).write_text(render(chart), encoding="utf-8")

    loss = [r["loss"] for r in train]
    save(
        "loss.svg",
        Chart(
            f"{name} SFT assistant-token loss",
            "epoch",
            "loss (nats per assistant token)",
            [
                Series("train (logged steps)", ep, loss, slot=0, width=0.6, opacity=0.35),
                Series("train (EMA)", ep, ema(loss, 0.1), slot=0),
                Series(
                    "validation (exact, 860 conversations)",
                    ev_ep,
                    [r["loss"] for r in evals],
                    slot=1,
                    markers=True,
                ),
            ],
            subtitle="epoch 0 = base model; assistant tokens + <|end|> only",
            height=380,
        ),
    )
    save(
        "end_of_turn.svg",
        Chart(
            "End-of-turn prediction on validation",
            "epoch",
            "fraction",
            [
                Series(
                    "recall: <|end|> targets predicted as <|end|> (argmax)",
                    ev_ep,
                    [r["end_recall"] for r in evals],
                    slot=1,
                    markers=True,
                ),
                Series(
                    "precision: argmax <|end|> that was correct",
                    ev_ep,
                    [r["end_precision"] for r in evals],
                    slot=2,
                    markers=True,
                ),
            ],
            y_min=0.0,
            y_max=1.0,
            height=320,
        ),
    )
    save(
        "lr.svg",
        Chart(
            "Learning rate",
            "step",
            "learning rate",
            [Series("lr", [r["step"] for r in train], [r["lr"] for r in train])],
            subtitle="100-step warmup to 1e-4, cosine to 1e-5",
            height=280,
            y_format="{:.1e}",
        ),
    )
    save(
        "throughput_memory.svg",
        Chart(
            "Throughput and MPS memory",
            "step",
            "value",
            [
                Series(
                    "thousand real tokens / s",
                    [r["step"] for r in train],
                    [r["real_tokens_per_s"] / 1e3 for r in train],
                    width=0.6,
                    opacity=0.5,
                ),
                Series(
                    "MPS driver pool (GB)",
                    [r["step"] for r in train],
                    [r["mps_driver_allocated"] / 1e9 for r in train],
                    slot=1,
                ),
            ],
            y_min=0.0,
            height=300,
        ),
    )
    (out / "curve_points.json").write_text(
        json.dumps(
            {
                "evals": [
                    {
                        k: r.get(k)
                        for k in (
                            "step",
                            "epoch",
                            "target_tokens_seen",
                            "loss",
                            "ppl",
                            "end_recall",
                            "end_precision",
                            "is_best",
                        )
                    }
                    for r in evals
                ]
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

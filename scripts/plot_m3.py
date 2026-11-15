"""Plots for the M3 research report from the benchmark JSON files.

Usage: python scripts/plot_m3.py docs/reports/m3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mini_chatgpt.analysis.svgplot import Chart, Series, render


def main() -> None:
    out = Path(sys.argv[1])
    tok = json.loads((out / "tokenizer_benchmark.json").read_text())
    mod = json.loads((out / "model_benchmark.json").read_text())

    def rows(train_set: str) -> list[dict[str, Any]]:
        return sorted((t for t in tok["tokenizers"] if t["train_set"] == train_set),
                      key=lambda t: t["vocab_size"])  # fmt: skip

    series = []
    for slot, corpus in enumerate(("tinystories", "fineweb_edu")):
        for train_set, dash in ((corpus, None), ("mixed", "6 4")):
            r = rows(train_set)
            label = f"{corpus} val, {'own-corpus' if train_set == corpus else 'mixed'} tokenizer"
            series.append(Series(label,
                                 [t["vocab_size"] for t in r],
                                 [t["eval"][corpus]["bytes_per_token"] for t in r],
                                 slot=slot, markers=True, dash=dash))  # fmt: skip
    (out / "tokenizer_bytes_per_token.svg").write_text(render(Chart(
        "BPE compression vs. vocabulary size", "vocabulary size",
        "UTF-8 bytes per token (higher = fewer tokens)", series,
        subtitle="validation documents only; tokenizers fitted on train documents",
        y_min=2.0, height=420)), encoding="utf-8")  # fmt: skip

    res = [r for r in mod["results"] if r["measured"].get("ok")]

    def group(g: str, key: str) -> tuple[list[float], list[float], list[dict[str, Any]]]:
        base = "S_v4096_t512_b32"  # shared point of every sweep
        picked = [r for r in res if r["case"]["group"] == g or r["case"]["name"] == base]
        rr = sorted(picked, key=lambda r: r["case"][key])
        rr = [r for r in rr if r["case"]["n_embd"] == 384 and r["case"]["dropout"] == 0]
        return [r["case"][key] for r in rr], [r["measured"]["train_tokens_per_s"] for r in rr], rr

    x, y, _ = group("context", "context_length")
    (out / "throughput_vs_context.svg").write_text(render(Chart(
        "Training throughput vs. context length (S: 6L/6H/384, V = 4,096)",
        "context length (tokens)", "training tokens / s", [Series("measured", x, y, markers=True)],
        subtitle="constant 16,384 tokens per batch; MPS, float32, SDPA, dropout 0",
        y_min=0.0, y_format="{:,.0f}")), encoding="utf-8")  # fmt: skip
    x, y, _ = group("vocab", "vocab_size")
    (out / "throughput_vs_vocab.svg").write_text(render(Chart(
        "Training throughput vs. vocabulary size (S: 6L/6H/384, ctx 512, batch 32)",
        "vocabulary size", "training tokens / s", [Series("measured", x, y, markers=True)],
        subtitle="65 = the Shakespeare character vocabulary; MPS, float32, dropout 0",
        y_min=0.0, y_format="{:,.0f}")), encoding="utf-8")  # fmt: skip
    x, _, rr = group("batch", "batch_size")
    (out / "memory_vs_batch.svg").write_text(render(Chart(
        "Memory vs. batch size (S, V = 4,096, ctx 512)", "batch size (sequences)", "GB",
        [Series("measured: MPS allocated after forward (dropout 0)", x,
                [r["measured"]["mps_allocated_after_forward_bytes"] / 1e9 for r in rr],
                markers=True),
         Series("analytic estimate (assumes dropout masks)", x,
                [r["estimate"]["total_bytes_est"] / 1e9 for r in rr], slot=1, dash="6 4",
                markers=True)],
        subtitle="the estimate matches the dropout-0.2 measurement within 9% (6.6 vs 6.0 GB)",
        y_min=0.0)), encoding="utf-8")  # fmt: skip
    print("plots written to", out)


if __name__ == "__main__":
    main()

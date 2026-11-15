"""M4 compute planning: bounded synthetic MPS microbenchmarks for 12M-100M models.

Each case runs in its own subprocess through the M3 worker in
``scripts/benchmark_models.py --worker`` (random weights, random token ids,
float32, SDPA, AdamW + clipping, dropout 0), strictly one case at a time. Nothing
is trained on real data and nothing is written to ``runs/``.

The families keep a head dimension of 64:

* ``S``  6 layers, 6 heads, width 384   (the M3.1 model)
* ``M``  8 layers, 8 heads, width 512   (the M3.3 model)
* ``X`` 10 layers, 10 heads, width 640  (~50M class)
* ``L`` 12 layers, 12 heads, width 768  (GPT-2-small shape, ~100M stretch)

Groups: model size at context 512 and V=8192; context length at a constant
8,192 tokens per micro-batch; vocabulary 4,096 vs 8,192; micro-batch size.

Usage: python scripts/benchmark_m4_compute.py --out docs/reports/m4 [--only NAME ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "scripts" / "benchmark_models.py"
ARCH = {"S": (6, 6, 384), "M": (8, 8, 512), "X": (10, 10, 640), "L": (12, 12, 768)}


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    context_length: int
    batch_size: int
    dropout: float = 0.0


def make(group: str, arch: str, v: int, t: int, b: int) -> Case:
    layer, head, embd = ARCH[arch]
    return Case(f"{arch}_v{v}_t{t}_b{b}", group, layer, head, embd, v, t, b)


CASES: list[Case] = [
    # model size, context 512, V=8192
    make("size", "S", 8192, 512, 32),
    make("size", "M", 8192, 512, 16),
    make("size", "X", 8192, 512, 8),
    make("size", "X", 8192, 512, 16),
    make("size", "L", 8192, 512, 4),
    make("size", "L", 8192, 512, 8),
    # context length at 8,192 tokens per micro-batch
    make("context", "M", 8192, 256, 32),
    make("context", "M", 8192, 1024, 8),
    make("context", "X", 8192, 1024, 8),
    make("context", "X", 8192, 1024, 4),
    make("context", "L", 8192, 1024, 4),
    # vocabulary
    make("vocab", "S", 4096, 512, 32),
    make("vocab", "M", 4096, 512, 16),
    # micro-batch size (same tokens per optimizer step via accumulation)
    make("batch", "S", 8192, 512, 16),
    make("batch", "M", 8192, 512, 8),
    make("batch", "M", 8192, 512, 32),
]


def estimate(case: Case) -> dict[str, Any]:
    from mini_chatgpt.config import ModelConfig

    cfg = ModelConfig(vocab_size=case.vocab_size, context_length=case.context_length,
                      n_layer=case.n_layer, n_head=case.n_head, n_embd=case.n_embd)  # fmt: skip
    p = cfg.estimate_parameters()
    n = int(p["total"])
    return {
        "params_total": n,
        "params_non_embedding": int(p["non_embedding"]),
        "params_embedding": int(p["token_embedding"] + p["position_embedding"]),
        "param_grad_adam_bytes": 16 * n,
        # 6N for the matmuls with weights + 6*L*T*C for attention scores/values
        "train_flops_per_token": 6 * n + 6 * case.n_layer * case.context_length * case.n_embd,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--only", nargs="*", help="case names to run (default: all)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--memory-fraction", type=float, default=0.85)
    args = p.parse_args()

    cases = [c for c in CASES if not args.only or c.name in args.only]
    results = []
    for case in cases:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(WORKER), "--worker", json.dumps(asdict(case)),
             "--warmup", str(args.warmup), "--iters", str(args.iters),
             "--memory-fraction", str(args.memory_fraction)],
            capture_output=True, text=True, cwd=ROOT, timeout=900,
        )  # fmt: skip
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        measured: dict[str, Any]
        if line:
            measured = json.loads(line[7:])
        else:
            err_lines = proc.stderr.strip().splitlines() or ["?"]
            measured = {"ok": False, "error": err_lines[-1]}
        est = estimate(case)
        if measured.get("ok"):
            measured["achieved_tflops"] = (
                est["train_flops_per_token"] * measured["train_tokens_per_s"] / 1e12
            )
        row = {"case": asdict(case), "estimate": est, "measured": measured,
               "wall_s": time.perf_counter() - t0}  # fmt: skip
        results.append(row)
        m = measured
        msg = (f"{est['params_total'] / 1e6:6.1f}M  {m['train_step_ms_median']:6.0f} ms/step  "
               f"{m['train_tokens_per_s']:8,.0f} tok/s  {m['achieved_tflops']:.2f} TFLOP/s  "
               f"{m['mps_allocated_after_forward_bytes'] / 1e9:5.2f} GB alloc  "
               f"{m['mps_driver_after_bytes'] / 1e9:5.2f} GB pool"
               if m.get("ok") else f"FAILED: {m.get('error')}")  # fmt: skip
        print(f"{case.name:18s} {msg}", flush=True)
    if args.out:
        import torch

        from mini_chatgpt.training.checkpoint import atomic_write_json

        args.out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out / "compute_benchmark.json", {
            "torch": torch.__version__, "device": "mps", "dtype": "float32",
            "attention": "sdpa", "warmup": args.warmup, "iters": args.iters,
            "memory_fraction_cap": args.memory_fraction,
            "recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
            "results": results,
        })  # fmt: skip


if __name__ == "__main__":
    main()

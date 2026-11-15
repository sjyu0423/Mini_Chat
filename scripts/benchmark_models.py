"""Bounded MPS microbenchmarks of candidate GPT configurations (synthetic data).

Every case runs in its **own subprocess** so allocator pools and caches cannot
leak between measurements, and cases run strictly one at a time. Inside a case:
random weights (seeded), random token ids, dropout as given, float32, SDPA.

Per case it measures, after ``--warmup`` untimed iterations:
* forward latency in eval mode under ``torch.no_grad`` (median of ``--iters``)
* train-step latency: forward + backward + clip + AdamW step (median)
* tokens/s for the train step
* ``mps_current_allocated`` right after the training forward pass (activations
  live) and the Metal driver pool after the run

A per-process memory cap (``--memory-fraction`` of the recommended maximum)
turns an over-large case into a clean error instead of swapping.

The analytic estimates (parameters, optimizer state, activations) are written
next to the measurements so the two can be compared but never confused.

Usage: python scripts/benchmark_models.py --out docs/reports/m3 [--only S_v4096_t512_b32]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


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


ARCH = {"S": (6, 6, 384), "M": (8, 8, 512), "L": (12, 12, 768)}


def make(group: str, arch: str, v: int, t: int, b: int, dropout: float = 0.0) -> Case:
    layer, head, embd = ARCH[arch]
    suffix = f"_d{dropout:g}" if dropout else ""
    return Case(f"{arch}_v{v}_t{t}_b{b}{suffix}", group, layer, head, embd, v, t, b, dropout)


CASES: list[Case] = [
    # model size at a common setting
    make("model", "S", 4096, 512, 32),
    make("model", "M", 4096, 512, 16),
    make("model", "M", 4096, 512, 32),
    make("model", "L", 4096, 512, 8),
    make("model", "L", 8192, 1024, 4),
    make("model", "M", 4096, 1024, 8),
    # vocabulary size (65 = the Shakespeare char vocabulary)
    make("vocab", "S", 65, 512, 32),
    make("vocab", "S", 2048, 512, 32),
    make("vocab", "S", 8192, 512, 32),
    make("vocab", "S", 16384, 512, 32),
    # context length at a constant 16,384 tokens per batch
    make("context", "S", 4096, 128, 128),
    make("context", "S", 4096, 256, 64),
    make("context", "S", 4096, 1024, 16),
    # batch size at context 512
    make("batch", "S", 4096, 512, 4),
    make("batch", "S", 4096, 512, 8),
    make("batch", "S", 4096, 512, 16),
    # dropout overhead
    make("dropout", "S", 4096, 512, 32, dropout=0.2),
]


# --------------------------------------------------------------------------- #
# Analytic estimates (fp32). Calibrated against M2: gpt_10m at B64 x T256 gives
# 4.3 GB predicted vs 4.05 GB measured after the forward pass.
# --------------------------------------------------------------------------- #
def estimate(case: Case) -> dict[str, Any]:
    from mini_chatgpt.config import ModelConfig

    cfg = ModelConfig(vocab_size=case.vocab_size, context_length=case.context_length,
                      n_layer=case.n_layer, n_head=case.n_head, n_embd=case.n_embd)  # fmt: skip
    p = cfg.estimate_parameters()
    n = p["total"]
    b, t, c, h, v, layers = (case.batch_size, case.context_length, case.n_embd, case.n_head,
                             case.vocab_size, case.n_layer)  # fmt: skip
    per_layer_linear = 74 * b * t * c  # ~18 saved fp32 tensors of width C + 2 byte masks
    per_layer_attn = 9 * b * h * t * t  # scores + probs (fp32) + dropout mask, if materialized
    logits = 12 * b * t * v  # logits, log-softmax output, and their gradient
    acts = layers * (per_layer_linear + per_layer_attn) + logits + 8 * b * t * c
    return {
        "params_total": n,
        "params_non_embedding": p["non_embedding"],
        "params_embedding": p["token_embedding"] + p["position_embedding"],
        "weights_bytes": 4 * n,
        "grads_bytes": 4 * n,
        "adam_state_bytes": 8 * n,
        "param_grad_opt_bytes": 16 * n,
        "activations_bytes_est": acts,
        "activations_attention_share": layers * per_layer_attn / acts,
        "activations_logits_share": logits / acts,
        "total_bytes_est": 16 * n + acts,
        "train_flops_per_token": 6 * n + 12 * layers * t * c,
    }


# --------------------------------------------------------------------------- #
def run_case(case: Case, warmup: int, iters: int, fraction: float) -> dict[str, Any]:
    import torch

    from mini_chatgpt.config import ModelConfig
    from mini_chatgpt.model.gpt import GPT

    dev = torch.device("mps")
    torch.mps.set_per_process_memory_fraction(fraction)
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=case.vocab_size, context_length=case.context_length,
                      n_layer=case.n_layer, n_head=case.n_head, n_embd=case.n_embd,
                      dropout=case.dropout)  # fmt: skip
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.99))
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, case.vocab_size, (case.batch_size, case.context_length), generator=g)
    y = torch.randint(0, case.vocab_size, (case.batch_size, case.context_length), generator=g)
    x, y = x.to(dev), y.to(dev)

    def train_step() -> float:
        torch.mps.synchronize()
        t0 = time.perf_counter()
        _, loss = model(x, y)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.mps.synchronize()
        return time.perf_counter() - t0

    @torch.no_grad()
    def fwd() -> float:
        torch.mps.synchronize()
        t0 = time.perf_counter()
        model(x)
        torch.mps.synchronize()
        return time.perf_counter() - t0

    model.train()
    for _ in range(warmup):
        train_step()
    train_times = [train_step() for _ in range(iters)]
    # memory with the graph alive: one more forward in train mode, then probe
    _, loss = model(x, y)
    torch.mps.synchronize()
    after_fwd = torch.mps.current_allocated_memory()
    assert loss is not None
    loss.backward()
    opt.zero_grad(set_to_none=True)
    del loss
    model.eval()
    for _ in range(warmup):
        fwd()
    fwd_times = [fwd() for _ in range(iters)]
    tokens = case.batch_size * case.context_length
    med = statistics.median(train_times)
    return {
        "train_step_ms_median": 1000 * med,
        "train_step_ms_min": 1000 * min(train_times),
        "train_step_ms_max": 1000 * max(train_times),
        "forward_ms_median": 1000 * statistics.median(fwd_times),
        "train_tokens_per_s": tokens / med,
        "forward_tokens_per_s": tokens / statistics.median(fwd_times),
        "mps_allocated_after_forward_bytes": int(after_fwd),
        "mps_driver_after_bytes": int(torch.mps.driver_allocated_memory()),
        "tokens_per_batch": tokens,
        "params_measured": model.num_parameters(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--only", nargs="*", help="case names to run (default: all)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--memory-fraction", type=float, default=0.85)
    p.add_argument("--worker", help=argparse.SUPPRESS)  # internal: run one case as JSON
    args = p.parse_args()

    if args.worker:
        case = Case(**json.loads(args.worker))
        try:
            res = {"ok": True, **run_case(case, args.warmup, args.iters, args.memory_fraction)}
        except RuntimeError as exc:  # MPS out-of-memory and similar
            res = {"ok": False, "error": str(exc).splitlines()[0][:300]}
        print("RESULT " + json.dumps(res))
        return

    cases = [c for c in CASES if not args.only or c.name in args.only]
    results = []
    for case in cases:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", json.dumps(asdict(case)),
             "--warmup", str(args.warmup), "--iters", str(args.iters),
             "--memory-fraction", str(args.memory_fraction)],
            capture_output=True, text=True, cwd=ROOT, timeout=600,
        )  # fmt: skip
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        measured: dict[str, Any]
        if line:
            measured = json.loads(line[7:])
        else:
            err_lines = proc.stderr.strip().splitlines() or ["?"]
            measured = {"ok": False, "error": err_lines[-1]}
        row = {"case": asdict(case), "estimate": estimate(case), "measured": measured,
               "wall_s": time.perf_counter() - t0}  # fmt: skip
        results.append(row)
        m = measured
        msg = (f"{m['train_step_ms_median']:.0f} ms/step, {m['train_tokens_per_s']:,.0f} tok/s, "
               f"{m['mps_allocated_after_forward_bytes'] / 1e9:.2f} GB"
               if m.get("ok") else f"FAILED: {m.get('error')}")  # fmt: skip
        print(f"{case.name:28s} {msg}", flush=True)
    if args.out:
        import torch

        from mini_chatgpt.training.checkpoint import atomic_write_json

        args.out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out / "model_benchmark.json", {
            "torch": torch.__version__, "device": "mps", "dtype": "float32",
            "attention": "sdpa", "warmup": args.warmup, "iters": args.iters,
            "memory_fraction_cap": args.memory_fraction,
            "recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
            "results": results,
        })  # fmt: skip


if __name__ == "__main__":
    main()

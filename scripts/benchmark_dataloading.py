"""Is batch loading a bottleneck? Time random-window batches from a uint16 memmap.

Creates a synthetic token file of ``--tokens`` uint16 ids in a temporary
directory (deleted afterwards), then times sampling ``--batch`` x ``--context``
windows (x and y), converting to int64 and copying to the device, which is what
a memmap-backed loader for a large BPE corpus would do per step.

Usage: python scripts/benchmark_dataloading.py --tokens 100000000 --out docs/reports/m3
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_chatgpt.training.checkpoint import atomic_write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=100_000_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokens.bin"
        t0 = time.perf_counter()
        rng.integers(0, 4096, args.tokens, dtype=np.uint16).tofile(path)
        results["write_s"] = time.perf_counter() - t0
        data = np.memmap(path, dtype=np.uint16, mode="r")
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            starts = rng.integers(0, len(data) - args.context - 1, args.batch)
            win = np.stack([data[s : s + args.context + 1] for s in starts]).astype(np.int64)
            x = torch.from_numpy(win[:, :-1]).to(dev, non_blocking=True)
            y = torch.from_numpy(win[:, 1:]).to(dev, non_blocking=True)
            if dev.type == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
            del x, y
        med = statistics.median(times[10:])
        tokens = args.batch * args.context
        p95 = sorted(times)[int(0.95 * len(times))]
        results.update(
            {
                "tokens_in_file": args.tokens,
                "file_bytes": args.tokens * 2,
                "batch": args.batch,
                "context": args.context,
                "device": str(dev),
                "batch_ms_median": 1000 * med,
                "batch_ms_p95": 1000 * p95,
                "tokens_per_s": tokens / med,
            }
        )
    print(results)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out / "dataloading_benchmark.json", results)


if __name__ == "__main__":
    main()

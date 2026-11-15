"""Preflight for a new (larger) model configuration on an existing dataset.

1. Historical checkpoints still load (character-level runs: exact val recomputed).
2. The new model gives finite logits, loss and gradients (every parameter).
3. Causal attention: changing future tokens never changes earlier logits, and the
   SDPA and explicit attention backends agree.
4. Optimizer initialization: decay/no-decay groups cover every parameter once;
   betas, weight decay and the first learning rate are as configured.
5. A short MPS stability run of the real configuration (gradient accumulation
   included). Nothing is written to the run directory.
6. Throughput and MPS memory from that run.
7. Free system memory and the MPS recommended maximum.
8. Checkpoint save/load round trip (weights, optimizer, sampler).

Usage: python scripts/preflight_scaling.py configs/<new>.toml --historical runs/<a> runs/<b> ...
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.dataset import get_batch
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.training.checkpoint import atomic_write_json, best_checkpoint_path
from mini_chatgpt.training.optim import learning_rate
from mini_chatgpt.training.run import make_run_dir, setup_logging
from mini_chatgpt.training.seed import make_generator
from mini_chatgpt.training.trainer import Trainer

log = setup_logging()


def historical(runs: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cpu = torch.device("cpu")
    for run in runs:
        cfg = ExperimentConfig.from_dict(json.loads((run / "config.json").read_text()))
        data = load_prepared(processed_dir(cfg.data))
        t = Trainer(cfg, data.splits.train, data.splits.val, cpu)
        meta = t.load_state(best_checkpoint_path(run))
        row: dict[str, Any] = {"step": meta.step, "tokenizer": cfg.data.tokenizer}
        if cfg.data.tokenizer == "char":  # small enough to recompute exactly on CPU
            row["val_exact_recomputed"] = t.full_loss(data.splits.val)
            row["val_exact_logged"] = meta.val_loss_exact
        else:
            x, y = get_batch(
                data.splits.val, 2, cfg.model.context_length, generator=make_generator(0)
            )
            t.model.eval()
            with torch.no_grad():
                _, loss = t.model(x, y)
            assert loss is not None
            row["batch_loss_finite"] = bool(torch.isfinite(loss))
            row["val_exact_logged"] = meta.val_loss_exact
        out[run.name] = row
        log.info("historical %s: %s", run.name, row)
    return out


def free_memory() -> dict[str, Any]:
    text = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"System-wide memory free percentage: (\d+)%", text)
    total = int(
        subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
    )
    pct = int(m[1]) if m else None
    return {
        "physical_bytes": total,
        "free_percent": pct,
        "free_bytes_est": total * pct // 100 if pct is not None else None,
        "mps_recommended_max_bytes": torch.mps.recommended_max_memory(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--historical", type=Path, nargs="*", default=[])
    p.add_argument("--stability-steps", type=int, default=100)
    args = p.parse_args()
    cfg = ExperimentConfig.from_toml(args.config)
    device = select_device(cfg.training.device)
    checks: dict[str, Any] = {"config": str(args.config)}
    checks["1_historical_checkpoints"] = historical(args.historical)

    data = load_prepared(processed_dir(cfg.data))
    checks["dataset"] = {
        k: data.meta[k]
        for k in (
            "tokenizer_fingerprint",
            "train_sha256",
            "val_sha256",
            "num_train_tokens",
            "num_val_tokens",
        )
    }
    t = Trainer(cfg, data.splits.train, data.splits.val, device)
    n_params = t.model.num_parameters()
    checks["parameters"] = {"model": n_params, "analytic": cfg.model.estimate_parameters()}

    # 2. finite logits, loss, gradients
    x, y = get_batch(
        data.splits.train,
        cfg.training.batch_size,
        cfg.model.context_length,
        generator=make_generator(123),
        device=device,
    )
    t.model.train()
    logits, loss = t.model(x, y)
    assert loss is not None
    loss.backward()
    grads = {n: p.grad for n, p in t.model.named_parameters()}
    checks["2_finite"] = {
        "initial_loss": float(loss),
        "ln_vocab": math.log(cfg.model.vocab_size),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "all_grads_finite_nonzero": all(
            g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
            for g in grads.values()
        ),
        "grad_norm": float(
            torch.linalg.vector_norm(
                torch.stack([g.norm() for g in grads.values() if g is not None])
            )
        ),
    }
    t.optimizer.zero_grad(set_to_none=True)

    # 3. causality and backend agreement
    t.model.eval()
    with torch.no_grad():
        base = t.model(x[:4])[0]
        x2 = x[:4].clone()
        cut = cfg.model.context_length // 2
        x2[:, cut:] = torch.randint(
            0, cfg.model.vocab_size, x2[:, cut:].shape, generator=make_generator(7)
        ).to(device)
        changed = t.model(x2)[0]
        t.model.set_attention_backend("explicit")
        expl = t.model(x[:4])[0]
        t.model.set_attention_backend("sdpa")
    checks["3_causal"] = {
        "prefix_max_abs_diff_when_future_changes": float(
            (base[:, :cut] - changed[:, :cut]).abs().max()
        ),
        "suffix_changed": bool((base[:, cut:] - changed[:, cut:]).abs().max() > 1e-3),
        "sdpa_vs_explicit_max_abs_diff": float((base - expl).abs().max()),
    }

    # 4. optimizer initialization
    groups = t.optimizer.param_groups
    ids = [id(q) for g in groups for q in g["params"]]
    checks["4_optimizer"] = {
        "groups": [
            {
                "n_tensors": len(g["params"]),
                "n_params": sum(q.numel() for q in g["params"]),
                "weight_decay": g["weight_decay"],
                "betas": list(g["betas"]),
                "eps": g["eps"],
            }
            for g in groups
        ],
        "covers_every_parameter_once": len(ids) == len(set(ids)) == len(list(t.model.parameters())),
        "lr_step0": learning_rate(0, cfg.training),
        "lr_peak_step": learning_rate(cfg.training.warmup_steps - 1, cfg.training),
        "lr_final_step": learning_rate(cfg.training.max_steps - 1, cfg.training),
        "optimizer_state_empty_at_init": len(t.optimizer.state) == 0,
    }

    # 5./6. stability, throughput, memory (fresh trainer: identical to the real run's start)
    t = Trainer(cfg, data.splits.train, data.splits.val, device)
    rows = [t.train_step() for _ in range(args.stability_steps)]
    losses = [r["loss"] for r in rows]
    checks["5_stability"] = {
        "steps": len(rows),
        "all_finite": all(math.isfinite(v) for v in losses),
        "loss_first": losses[0],
        "loss_last10_mean": statistics.fmean(losses[-10:]),
        "grad_norm_step1": rows[0]["grad_norm"],
        "grad_norm_max_after_step10": max(r["grad_norm"] for r in rows[10:]),
        "lr_last": rows[-1]["lr"],
    }
    med = statistics.median(r["step_time_s"] for r in rows[5:])
    checks["6_throughput_memory"] = {
        "median_step_s": med,
        "tokens_per_s": t.tokens_per_step / med,
        "projected_train_minutes": cfg.training.max_steps * med / 60,
        "memory_max_bytes": t.stats.peak_memory,
    }

    # 7. free memory
    checks["7_free_memory"] = free_memory()

    # 8. checkpoint round trip
    with tempfile.TemporaryDirectory() as tmp:
        path = t.save(tmp)
        r = Trainer(cfg, data.splits.train, data.splits.val, device)
        r.load_state(path)
        t.model.eval()
        r.model.eval()
        with torch.no_grad():
            d = float((t.model(x[:2])[0] - r.model(x[:2])[0]).abs().max())
        so, ro = t.optimizer.state_dict()["state"], r.optimizer.state_dict()["state"]
        checks["8_checkpoint"] = {
            "logit_max_abs_diff": d,
            "step": r.step,
            "optimizer_identical": all(
                torch.equal(so[k]["exp_avg"].cpu(), ro[k]["exp_avg"].cpu()) for k in so
            ),
            "sampler_identical": torch.equal(
                t.data_generator.get_state(), r.data_generator.get_state()
            ),
            "checkpoint_bytes": path.stat().st_size,
        }

    c = checks
    passed = (
        all(v.get("batch_loss_finite", True) for v in c["1_historical_checkpoints"].values())
        and all(
            abs(v["val_exact_recomputed"] - v["val_exact_logged"]) < 1e-4
            for v in c["1_historical_checkpoints"].values()
            if "val_exact_recomputed" in v
        )
        and c["parameters"]["model"] == c["parameters"]["analytic"]["total"]
        and c["2_finite"]["logits_finite"]
        and c["2_finite"]["all_grads_finite_nonzero"]
        and c["3_causal"]["prefix_max_abs_diff_when_future_changes"] < 1e-4
        and c["3_causal"]["suffix_changed"]
        and c["3_causal"]["sdpa_vs_explicit_max_abs_diff"] < 1e-3
        and c["4_optimizer"]["covers_every_parameter_once"]
        and c["5_stability"]["all_finite"]
        and c["5_stability"]["loss_last10_mean"] < c["5_stability"]["loss_first"] - 1.0
        and c["8_checkpoint"]["logit_max_abs_diff"] == 0.0
        and c["8_checkpoint"]["optimizer_identical"]
        and c["8_checkpoint"]["sampler_identical"]
    )
    checks["passed"] = bool(passed)
    out = make_run_dir(cfg.run_dir, f"{cfg.name}_preflight")
    atomic_write_json(out / "preflight.json", checks)
    print(json.dumps(checks, indent=1, default=str))
    print("PREFLIGHT", "PASSED" if passed else "FAILED", out)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

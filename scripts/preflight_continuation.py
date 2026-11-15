"""Preflight checks before continuing a run as a new training phase.

1. The parent checkpoint (best.pt) and the step checkpoint it should equal have
   identical weights and produce identical logits.
2. The exact validation loss recomputed from the restored state matches the value
   logged by the parent run.
3. Dataset hashes (verified on load), tokenizer fingerprints (dataset, parent run,
   continuation config) and data configs agree.
4. AdamW state (exp_avg, exp_avg_sq, step) is restored exactly.
5. The continuation LR is constant at the configured value for every step.
6. Batch sampling resumes exactly: replaying the parent's 4,800 batch draws from the
   seed reproduces the restored sampler state and the next batch.
7. A bounded MPS stability run from the restored state (no files written).

Usage: python scripts/preflight_continuation.py runs/<parent> configs/<continuation>.toml
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.dataset import get_batch
from mini_chatgpt.data.prepare import file_sha256, load_prepared, processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.tokenizer import load_tokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json, load_checkpoint
from mini_chatgpt.training.optim import learning_rate
from mini_chatgpt.training.run import make_run_dir, read_metrics, setup_logging
from mini_chatgpt.training.seed import make_generator
from mini_chatgpt.training.trainer import Trainer

log = setup_logging()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent_run", type=Path)
    p.add_argument("config", type=Path)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--stability-steps", type=int, default=100)
    args = p.parse_args()
    parent: Path = args.parent_run
    ckpt = parent / "checkpoints" / args.checkpoint
    cfg = ExperimentConfig.from_toml(args.config)
    device = select_device(cfg.training.device)
    data = load_prepared(processed_dir(cfg.data))  # verifies bin SHA-256, ids, fingerprint
    checks: dict[str, Any] = {
        "parent_checkpoint": str(ckpt),
        "parent_checkpoint_sha256": file_sha256(ckpt),
    }

    # 1. best.pt == step checkpoint
    state, meta = load_checkpoint(ckpt)
    step = int(state["step"])
    twin = parent / "checkpoints" / f"step_{step:08d}.pt"
    tstate, _ = load_checkpoint(twin)
    same_w = all(torch.equal(state["model"][k], tstate["model"][k]) for k in state["model"])
    t_best = Trainer(cfg, data.splits.train, data.splits.val, device)
    t_best.load_state(ckpt, continuation=True)
    t_twin = Trainer(cfg, data.splits.train, data.splits.val, device)
    t_twin.load_state(twin, continuation=True)
    x, _ = get_batch(
        data.splits.val, 8, cfg.model.context_length, generator=make_generator(99), device=device
    )
    t_best.model.eval()
    t_twin.model.eval()
    with torch.no_grad():
        logit_diff = float((t_best.model(x)[0] - t_twin.model(x)[0]).abs().max())
    checks["1_best_equals_step_checkpoint"] = {
        "step": step,
        "twin": twin.name,
        "weights_identical": same_w,
        "max_abs_logit_diff": logit_diff,
        "optimizer_identical": all(
            torch.equal(state["optimizer"]["state"][k][n], tstate["optimizer"]["state"][k][n])
            for k in state["optimizer"]["state"]
            for n in ("exp_avg", "exp_avg_sq")
        ),
        "sampler_identical": torch.equal(state["data_generator"], tstate["data_generator"]),
    }

    # 2. exact validation loss
    t0 = time.perf_counter()
    val = t_best.full_loss(data.splits.val)
    logged = next(
        r["val_loss_exact"]
        for r in read_metrics(parent / "metrics.jsonl")
        if r["event"] == "eval" and int(r["step"]) == step
    )
    checks["2_validation_loss"] = {
        "recomputed": val,
        "logged_by_parent": logged,
        "abs_diff": abs(val - logged),
        "seconds": time.perf_counter() - t0,
    }

    # 3. fingerprints and configs
    parent_cfg = ExperimentConfig.from_dict(meta.config)
    ptok = load_tokenizer(parent / "tokenizer.json")
    checks["3_fingerprints"] = {
        "dataset_tokenizer": data.meta["tokenizer_fingerprint"],
        "parent_run_tokenizer": getattr(ptok, "fingerprint", lambda: None)(),
        "loaded_tokenizer": data.tokenizer.fingerprint()
        if hasattr(data.tokenizer, "fingerprint")
        else None,
        "train_bin_sha256": data.meta["train_sha256"],
        "val_bin_sha256": data.meta["val_sha256"],
        "model_config_equal": parent_cfg.model == cfg.model,
        "data_config_equal": parent_cfg.data == cfg.data,
    }
    diffs = [
        f
        for f in cfg.training.__dataclass_fields__
        if getattr(parent_cfg.training, f) != getattr(cfg.training, f)
    ]
    checks["3_fingerprints"]["training_config_differences"] = {
        f: [getattr(parent_cfg.training, f), getattr(cfg.training, f)] for f in diffs
    }

    # 4. optimizer state
    restored = t_best.optimizer.state_dict()
    opt_ok = all(
        torch.equal(state["optimizer"]["state"][k][n].cpu(), restored["state"][k][n].cpu())
        for k in state["optimizer"]["state"]
        for n in ("exp_avg", "exp_avg_sq", "step")
    )
    steps_in_opt = {float(v["step"]) for v in restored["state"].values()}
    checks["4_optimizer"] = {
        "moments_and_steps_identical": opt_ok,
        "adam_step_counts": sorted(steps_in_opt),
        "param_groups": [
            {k: v for k, v in g.items() if k != "params"} for g in restored["param_groups"]
        ],
    }

    # 5. constant LR
    lrs = [learning_rate(s, cfg.training) for s in range(step, cfg.training.max_steps)]
    checks["5_learning_rate"] = {
        "min": min(lrs),
        "max": max(lrs),
        "n_steps": len(lrs),
        "all_equal_configured": all(lr == cfg.training.learning_rate for lr in lrs),
        "parent_lr_at_last_update": learning_rate(step - 1, parent_cfg.training),
    }

    # 6. batch sampling replay
    g = make_generator(cfg.training.seed)
    for _ in range(step):
        get_batch(data.splits.train, cfg.training.batch_size, cfg.model.context_length, generator=g)
    replay_state_equal = torch.equal(g.get_state(), t_best.data_generator.get_state())
    g_copy = torch.Generator()
    g_copy.set_state(t_best.data_generator.get_state())
    xr, _ = get_batch(
        data.splits.train, cfg.training.batch_size, cfg.model.context_length, generator=g
    )
    xc, _ = get_batch(
        data.splits.train, cfg.training.batch_size, cfg.model.context_length, generator=g_copy
    )
    checks["6_batch_sampling"] = {
        "replayed_draws": step,
        "sampler_state_equals_replay": replay_state_equal,
        "next_batch_equals_replay": bool(torch.equal(xr, xc)),
    }

    # 7. bounded stability run (no run_dir: nothing is written)
    t_best.model.train()
    rows = [t_best.train_step() for _ in range(args.stability_steps)]
    losses = [r["loss"] for r in rows]
    parent_last = [
        r["loss"]
        for r in read_metrics(parent / "metrics.jsonl")
        if r["event"] == "train" and step - 100 < int(r["step"]) <= step
    ]
    checks["7_stability"] = {
        "steps": len(rows),
        "all_finite": all(math.isfinite(v) for v in losses),
        "loss_mean": statistics.fmean(losses),
        "parent_last_100_mean": statistics.fmean(parent_last),
        "grad_norm_max": max(r["grad_norm"] for r in rows),
        "grad_norm_median": statistics.median(r["grad_norm"] for r in rows),
        "lr_values": sorted({r["lr"] for r in rows}),
        "median_step_s": statistics.median(r["step_time_s"] for r in rows[5:]),
        "memory_max_bytes": t_best.stats.peak_memory,
    }

    c = checks
    passed = (
        c["1_best_equals_step_checkpoint"]["weights_identical"]
        and c["1_best_equals_step_checkpoint"]["max_abs_logit_diff"] == 0.0
        and c["2_validation_loss"]["abs_diff"] < 1e-3
        and c["3_fingerprints"]["dataset_tokenizer"]
        == c["3_fingerprints"]["parent_run_tokenizer"]
        == c["3_fingerprints"]["loaded_tokenizer"]
        and c["3_fingerprints"]["model_config_equal"]
        and c["3_fingerprints"]["data_config_equal"]
        and c["4_optimizer"]["moments_and_steps_identical"]
        and c["5_learning_rate"]["all_equal_configured"]
        and c["6_batch_sampling"]["sampler_state_equals_replay"]
        and c["6_batch_sampling"]["next_batch_equals_replay"]
        and c["7_stability"]["all_finite"]
        and abs(c["7_stability"]["loss_mean"] - c["7_stability"]["parent_last_100_mean"]) < 0.1
    )
    checks["passed"] = bool(passed)
    out = make_run_dir(cfg.run_dir, f"{cfg.name}_preflight")
    atomic_write_json(out / "preflight.json", checks)
    print(json.dumps(checks, indent=1, default=str))
    if not passed:
        raise SystemExit("preflight FAILED")


if __name__ == "__main__":
    main()

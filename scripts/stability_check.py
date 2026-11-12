"""Bounded pre-flight check of a full training config (no checkpoints written).

1. LR schedule: every step's learning rate is compared against the closed-form
   warmup + cosine formula, over the *full* ``max_steps``.
2. Stability: the first ``--steps`` optimizer steps of the real config (the
   schedule is not shortened, so this covers warmup and the peak LR). It fails
   on NaN/Inf, on a gradient-norm explosion, or if the loss does not fall.
3. Timing and memory at the real batch size, for a wall-clock estimate.

Usage: python scripts/stability_check.py --config configs/gpt_10m_m2.toml --steps 300
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from typing import Any

from mini_chatgpt.config import ExperimentConfig, TrainingConfig
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.training.checkpoint import atomic_write_json
from mini_chatgpt.training.optim import learning_rate
from mini_chatgpt.training.run import make_run_dir, setup_logging
from mini_chatgpt.training.trainer import NonFiniteError, Trainer, memory_stats


def reference_lr(step: int, t: TrainingConfig) -> float:
    """Independent restatement of the schedule documented in training_plan.md."""
    if step < t.warmup_steps:
        return t.learning_rate * (step + 1) / t.warmup_steps
    horizon = t.lr_schedule_steps if t.lr_schedule_steps is not None else t.max_steps
    if step >= horizon:
        return t.min_learning_rate
    progress = (step - t.warmup_steps) / max(1, horizon - t.warmup_steps)
    return t.min_learning_rate + 0.5 * (t.learning_rate - t.min_learning_rate) * (
        1 + math.cos(math.pi * progress)
    )


def check_schedule(t: TrainingConfig) -> dict[str, Any]:
    h = t.schedule_horizon
    lrs = [learning_rate(s, t) for s in range(max(t.max_steps, h) + 1)]
    max_err = max(abs(lr - reference_lr(s, t)) for s, lr in enumerate(lrs))
    warm = lrs[: t.warmup_steps]
    decay = lrs[t.warmup_steps : h]
    return {
        "max_abs_error_vs_formula": max_err,
        "warmup_monotonic_increasing": all(a < b for a, b in itertools.pairwise(warm)),
        "decay_monotonic_non_increasing": all(a >= b for a, b in itertools.pairwise(decay)),
        "peak_lr": max(lrs),
        "peak_step": lrs.index(max(lrs)),
        "lr_at": {
            s: lrs[s] for s in (0, 49, 99, 100, 1000, 2550, 4000, 4999, 5000) if s <= len(lrs)
        },
        "schedule_horizon": h,
        "min_lr_reached": lrs[h],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/gpt_10m_m2.toml")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-grad-norm", type=float, default=50.0, help="explosion threshold")
    args = p.parse_args()
    log = setup_logging()
    cfg = ExperimentConfig.from_toml(args.config)
    t = cfg.training

    sched = check_schedule(t)
    log.info("schedule: %s", json.dumps(sched))
    ok_sched = (
        sched["max_abs_error_vs_formula"] < 1e-12
        and sched["warmup_monotonic_increasing"]
        and sched["decay_monotonic_non_increasing"]
        and math.isclose(sched["peak_lr"], t.learning_rate)
        and math.isclose(sched["min_lr_reached"], t.min_learning_rate)
    )

    data = load_prepared(processed_dir(cfg.data))
    device = select_device(t.device)
    trainer = Trainer(cfg, data.splits.train, data.splits.val, device)  # no run_dir: no files
    init = trainer.estimate_loss()
    log.info("initial eval: %s", init)

    rows: list[dict[str, float]] = []
    failure = None
    t0 = time.perf_counter()
    try:
        for _ in range(args.steps):
            m = trainer.train_step()
            rows.append(m)
            if m["grad_norm"] > args.max_grad_norm:
                failure = f"grad norm {m['grad_norm']:.2f} > {args.max_grad_norm} at {trainer.step}"
                break
            if trainer.step % 50 == 0 or trainer.step == 1:
                log.info(
                    "step %d | loss %.4f | lr %.3e | gnorm %.3f | %.0f ms",
                    trainer.step, m["loss"], m["lr"], m["grad_norm"], 1000 * m["step_time_s"],
                )  # fmt: skip
    except NonFiniteError as exc:
        failure = str(exc)
    wall = time.perf_counter() - t0
    end = trainer.estimate_loss()

    losses = [r["loss"] for r in rows]
    gnorms = [r["grad_norm"] for r in rows]
    times = [r["step_time_s"] for r in rows[10:]]
    lr_err = max(abs(r["lr"] - reference_lr(i, t)) for i, r in enumerate(rows))
    first, last = statistics.fmean(losses[:10]), statistics.fmean(losses[-10:])
    median_step = statistics.median(times) if times else math.nan
    eval_s = 2 * t.eval_steps * median_step * 0.4  # forward-only ~ 1/3 of a train step
    result: dict[str, Any] = {
        "config": args.config,
        "device": str(device),
        "steps": trainer.step,
        "schedule": sched,
        "schedule_ok": ok_sched,
        "trained_lr_max_abs_error": lr_err,
        "initial_eval": init,
        "end_eval": end,
        "train_loss_first10_mean": first,
        "train_loss_last10_mean": last,
        "grad_norm": {
            "max": max(gnorms),
            "max_step": gnorms.index(max(gnorms)) + 1,
            "last10_mean": statistics.fmean(gnorms[-10:]),
            "clipped_fraction": sum(g > t.grad_clip for g in gnorms) / len(gnorms),
        },
        "median_step_s": median_step,
        "tokens_per_s": trainer.tokens_per_step / median_step,
        "wall_s": wall,
        "memory_max_bytes": trainer.stats.peak_memory,
        "memory_now_bytes": memory_stats(device),
        "estimated_full_run_min": (
            t.max_steps * median_step + (t.max_steps // t.eval_interval) * eval_s
        )
        / 60,
        "failure": failure,
    }
    result["passed"] = bool(
        ok_sched and failure is None and lr_err < 1e-12 and last < first - 0.5
        and end["val"] < init["val"] - 0.5
    )  # fmt: skip
    run_dir = make_run_dir(cfg.run_dir, f"{cfg.name}_stability")
    atomic_write_json(run_dir / "stability.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("stability check FAILED; see stability.json")


if __name__ == "__main__":
    main()

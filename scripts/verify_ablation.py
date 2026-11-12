"""Pre-flight validity checks for a controlled A/B training comparison.

Checks that two configs differ only where intended, and that their trainers start
from the same state:

1. identical initial weights (bitwise)
2. identical first training batch (drawn from a copy of each data generator)
3. identical initial exact validation loss
4. A's LR sequence equals the reference run's for steps 0..max_steps (exact)
5. B's LR reaches min_learning_rate at its schedule horizon
6. identical optimizer hyperparameters, weight decay, accumulation, warmup
7. a flattened config diff (A vs B, and A vs reference)
8. seeds and environment

Usage:
    python scripts/verify_ablation.py configs/gpt_10m_m2_1_a.toml configs/gpt_10m_m2_1_b.toml \
        --reference runs/gpt_10m_m2-20261112-174425
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.dataset import get_batch
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import describe_device, select_device
from mini_chatgpt.training.checkpoint import atomic_write_json, git_commit_hash, git_is_dirty
from mini_chatgpt.training.optim import learning_rate
from mini_chatgpt.training.run import make_run_dir, read_metrics, setup_logging
from mini_chatgpt.training.trainer import Trainer


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(a: ExperimentConfig, b: ExperimentConfig) -> dict[str, list[Any]]:
    fa, fb = flatten(a.to_dict()), flatten(b.to_dict())
    return {
        k: [fa.get(k), fb.get(k)] for k in sorted(fa.keys() | fb.keys()) if fa.get(k) != fb.get(k)
    }


def first_batch(t: Trainer) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.set_state(t.data_generator.get_state())  # copy: the trainer's stream is untouched
    tc = t.config.training
    return get_batch(t.train_data, tc.batch_size, t.config.model.context_length, generator=gen)


def optimizer_hparams(t: Trainer) -> list[dict[str, Any]]:
    return [{k: v for k, v in g.items() if k != "params"} for g in t.optimizer.param_groups]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config_a", type=Path)
    p.add_argument("config_b", type=Path)
    p.add_argument("--reference", type=Path, help="reference run directory (e.g. M2)")
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    log = setup_logging()

    a, b = ExperimentConfig.from_toml(args.config_a), ExperimentConfig.from_toml(args.config_b)
    ref = (
        ExperimentConfig.from_dict(json.loads((args.reference / "config.json").read_text()))
        if args.reference
        else None
    )
    data = load_prepared(processed_dir(a.data))
    device = select_device(args.device)
    ta = Trainer(a, data.splits.train, data.splits.val, device)
    tb = Trainer(b, data.splits.train, data.splits.val, device)
    checks: dict[str, Any] = {}

    sa = {k: v.cpu() for k, v in ta.model.state_dict().items()}
    sb = {k: v.cpu() for k, v in tb.model.state_dict().items()}
    checks["identical_initial_weights"] = sa.keys() == sb.keys() and all(
        torch.equal(sa[k], sb[k]) for k in sa
    )
    xa, ya = first_batch(ta)
    xb, yb = first_batch(tb)
    checks["identical_first_batch"] = bool(torch.equal(xa, xb) and torch.equal(ya, yb))
    checks["first_batch_shape"] = list(xa.shape)
    checks["first_batch_offsets_hash"] = int(xa[:, 0].sum())
    va, vb = ta.full_loss(data.splits.val), tb.full_loss(data.splits.val)
    checks["initial_exact_val"] = {"a": va, "b": vb}
    checks["identical_initial_exact_val"] = va == vb

    lr_a = [learning_rate(s, a.training) for s in range(a.training.max_steps + 1)]
    lr_b = [learning_rate(s, b.training) for s in range(b.training.max_steps + 1)]
    if ref is not None:
        lr_ref = [learning_rate(s, ref.training) for s in range(a.training.max_steps + 1)]
        checks["a_lr_equals_reference_steps_0_to_max"] = lr_a == lr_ref
        checks["a_lr_max_abs_diff_vs_reference"] = max(
            abs(x - y) for x, y in zip(lr_a, lr_ref, strict=True)
        )
        logged = {
            int(r["step"]): r["lr"]
            for r in read_metrics(args.reference / "metrics.jsonl")
            if r["event"] == "train" and r["step"] <= a.training.max_steps
        }
        # The logged lr at step s is the LR used for the update s-1 -> s.
        checks["a_lr_equals_reference_logged"] = all(lr_a[s - 1] == v for s, v in logged.items())
        checks["reference_initial_exact_val"] = next(
            r["val_loss_exact"] for r in read_metrics(args.reference / "metrics.jsonl")
            if r["event"] == "eval" and r["step"] == 0
        )  # fmt: skip
        tr = Trainer(ref, data.splits.train, data.splits.val, device)
        sr = {k: v.cpu() for k, v in tr.model.state_dict().items()}
        checks["identical_initial_weights_vs_reference"] = all(
            torch.equal(sa[k], sr[k]) for k in sa
        )
        xr, yr = first_batch(tr)
        checks["identical_first_batch_vs_reference"] = bool(
            torch.equal(xa, xr) and torch.equal(ya, yr)
        )
        checks["config_diff_a_vs_reference"] = config_diff(ref, a)
    hb = b.training.schedule_horizon
    checks["b_lr_at_horizon"] = lr_b[hb]
    checks["b_lr_reaches_min_at_horizon"] = math.isclose(
        lr_b[hb], b.training.min_learning_rate, rel_tol=0, abs_tol=1e-15
    )
    checks["b_lr_last_update"] = lr_b[b.training.max_steps - 1]
    checks["a_lr_last_update"] = lr_a[a.training.max_steps - 1]
    checks["a_lr_at_stop"] = lr_a[a.training.max_steps]
    checks["warmup_identical"] = lr_a[: a.training.warmup_steps] == lr_b[: b.training.warmup_steps]

    ta_t, tb_t = a.training, b.training
    same = (
        "weight_decay", "gradient_accumulation_steps", "warmup_steps", "learning_rate",
        "min_learning_rate", "beta1", "beta2", "grad_clip", "batch_size", "seed", "dtype",
        "device", "max_steps", "eval_steps", "eval_interval", "exact_val_eval",
    )  # fmt: skip
    checks["identical_training_settings"] = {k: getattr(ta_t, k) == getattr(tb_t, k) for k in same}
    checks["identical_model_and_data"] = a.model == b.model and a.data == b.data
    checks["identical_optimizer_hparams"] = optimizer_hparams(ta) == optimizer_hparams(tb)
    checks["optimizer_hparams"] = optimizer_hparams(ta)
    checks["config_diff_a_vs_b"] = config_diff(a, b)
    intended = {"name", "description", "training.lr_schedule_steps"}
    checks["only_intended_a_b_differences"] = set(checks["config_diff_a_vs_b"]) <= intended

    env = {
        **asdict(describe_device(device)),
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "seed": a.training.seed,
        "data_generator_seed": a.training.seed,
        "eval_generator_seed": a.training.seed + 1_000_003,
        "init": "torch.manual_seed(seed) on CPU, then .to(device)",
    }
    must_pass = [
        "identical_initial_weights", "identical_first_batch", "identical_initial_exact_val",
        "b_lr_reaches_min_at_horizon", "warmup_identical", "identical_model_and_data",
        "identical_optimizer_hparams", "only_intended_a_b_differences",
    ]  # fmt: skip
    if ref is not None:
        must_pass += [
            "a_lr_equals_reference_steps_0_to_max", "a_lr_equals_reference_logged",
            "identical_initial_weights_vs_reference", "identical_first_batch_vs_reference",
        ]  # fmt: skip
    passed = all(checks[k] for k in must_pass) and all(
        checks["identical_training_settings"].values()
    )
    configs = {
        "a": str(args.config_a),
        "b": str(args.config_b),
        "reference": str(args.reference) if args.reference else None,
    }
    result = {
        "passed": passed,
        "must_pass": must_pass,
        "checks": checks,
        "environment": env,
        "configs": configs,
    }
    run_dir = make_run_dir(a.run_dir, "m2_1_verification")
    atomic_write_json(run_dir / "verification.json", result)
    print(json.dumps(result, indent=2, default=str))
    log.info("verification %s -> %s", "PASSED" if passed else "FAILED", run_dir)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

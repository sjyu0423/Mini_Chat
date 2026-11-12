"""Critical learning checks on the real gpt_10m model (bounded, ~1 minute on MPS).

1. Overfit one fixed batch of real Shakespeare text.
2. Gradient norms reach embeddings, attention, MLP, LayerNorm, and the tied head.
3. Checkpoint save/load preserves predictions.
4. Resumed training vs uninterrupted training.

Usage: python scripts/sanity_checks.py [--device mps] [--overfit-steps 150]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.dataset import get_batch
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.training.checkpoint import atomic_write_json, latest_checkpoint
from mini_chatgpt.training.run import make_run_dir, setup_logging
from mini_chatgpt.training.seed import make_generator, set_seed
from mini_chatgpt.training.trainer import Trainer, load_model


def overfit(
    cfg: ExperimentConfig, train: torch.Tensor, device: torch.device, steps: int, lr: float
) -> Any:
    set_seed(0)
    model = GPT(dataclasses.replace(cfg.model, dropout=0.0)).to(device)
    x, y = get_batch(train, 4, cfg.model.context_length, generator=make_generator(0), device=device)
    # Constant LR, no warmup. Note: at lr=1e-3 without warmup the model sits on the
    # unigram plateau (~3.2) for >100 steps on both CPU and MPS; 3e-4 memorizes quickly.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        _, loss = model(x, y)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {
        "batch": "4 x 256 real tokens",
        "lr": lr,
        "steps": steps,
        "loss_first": losses[0],
        "loss_at": {s: losses[s - 1] for s in (10, 25, 50, 100, 150, steps) if s <= steps},
        "loss_last": losses[-1],
        "reduction_factor": losses[0] / losses[-1],
    }


def gradient_flow(cfg: ExperimentConfig, train: torch.Tensor, device: torch.device) -> Any:
    set_seed(0)
    model = GPT(cfg.model).to(device)
    x, y = get_batch(train, 8, cfg.model.context_length, generator=make_generator(1), device=device)
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    groups = {
        "token_embedding_and_tied_head": ["wte.weight"],
        "position_embedding": ["wpe.weight"],
        "attention_qkv": [f"blocks.{i}.attn.c_attn.weight" for i in range(cfg.model.n_layer)],
        "attention_out_proj": [f"blocks.{i}.attn.c_proj.weight" for i in range(cfg.model.n_layer)],
        "mlp_fc": [f"blocks.{i}.mlp.c_fc.weight" for i in range(cfg.model.n_layer)],
        "mlp_proj": [f"blocks.{i}.mlp.c_proj.weight" for i in range(cfg.model.n_layer)],
        "layernorms": [n for n, _ in model.named_parameters() if ".ln_" in n or "ln_f" in n],
    }
    params = dict(model.named_parameters())
    out: dict[str, Any] = {}
    for group, names in groups.items():
        norms = [float(params[n].grad.norm().item()) for n in names]  # type: ignore[union-attr]
        out[group] = {"min_grad_norm": min(norms), "max_grad_norm": max(norms)}
    out["all_params_have_finite_nonzero_grad"] = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0
        for p in model.parameters()
    )
    return out


def checkpoint_and_resume(
    cfg: ExperimentConfig, train: torch.Tensor, val: torch.Tensor, device: torch.device, steps: int
) -> Any:
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(
            cfg.training,
            batch_size=8,
            max_steps=steps,
            warmup_steps=5,
            eval_interval=steps,
            eval_steps=5,
            checkpoint_interval=steps // 2,
            log_interval=steps,
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        full = Trainer(cfg, train, val, device, Path(tmp) / "full")
        full.fit()
        # A second uninterrupted run measures the backend's own run-to-run noise.
        full2 = Trainer(cfg, train, val, device)
        full2.fit()
        part = Trainer(cfg, train, val, device, Path(tmp) / "part")
        part.fit(max_steps=steps // 2)
        ckpt = latest_checkpoint(Path(tmp) / "part")
        assert ckpt is not None

        # Save/load preserves predictions.
        loaded, _ = load_model(ckpt, device)
        part.model.eval()
        x, _ = get_batch(val, 4, cfg.model.context_length, generator=make_generator(5))
        x = x.to(device)
        with torch.no_grad():
            pred_diff = float((loaded(x)[0] - part.model(x)[0]).abs().max().item())

        torch.manual_seed(12345)  # scramble global RNG; resume must restore it
        resumed = Trainer(cfg, train, val, device, Path(tmp) / "part")
        resumed.load_state(ckpt)
        resumed.fit()

        diffs = [
            float((a.float() - b.float()).abs().max().item())
            for a, b in zip(
                full.model.state_dict().values(), resumed.model.state_dict().values(), strict=True
            )
        ]
        noise = [
            float((a.float() - b.float()).abs().max().item())
            for a, b in zip(
                full.model.state_dict().values(), full2.model.state_dict().values(), strict=True
            )
        ]
        loss_diffs = [
            abs(a - b)
            for a, b in zip(
                full.loss_history[steps // 2 :], resumed.loss_history[steps // 2 :], strict=True
            )
        ]
        return {
            "steps": steps,
            "dropout": cfg.model.dropout,
            "checkpoint_reload_max_abs_logit_diff": pred_diff,
            "resume_max_abs_param_diff": max(diffs),
            "resume_bit_exact": max(diffs) == 0.0,
            "uninterrupted_rerun_max_abs_param_diff": max(noise),
            "uninterrupted_rerun_bit_exact": max(noise) == 0.0,
            "resume_max_abs_train_loss_diff": max(loss_diffs),
            "full_final_loss": full.loss_history[-1],
            "resumed_final_loss": resumed.loss_history[-1],
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/gpt_10m.toml")
    p.add_argument("--device", default="mps")
    p.add_argument("--overfit-steps", type=int, default=200)
    p.add_argument("--overfit-lr", type=float, default=3e-4)
    p.add_argument("--resume-steps", type=int, default=20)
    args = p.parse_args()
    log = setup_logging()
    cfg = ExperimentConfig.from_toml(args.config)
    data = load_prepared(processed_dir(cfg.data))
    device = select_device(args.device)
    train, val = data.splits.train, data.splits.val

    results: dict[str, Any] = {"device": str(device), "config": args.config}
    log.info("1/3 overfitting a single batch")
    results["overfit_single_batch"] = overfit(
        cfg, train, device, args.overfit_steps, args.overfit_lr
    )
    log.info("2/3 gradient flow")
    results["gradient_flow"] = gradient_flow(cfg, train, device)
    log.info("3/3 checkpoint + resume")
    results["checkpoint_resume"] = checkpoint_and_resume(cfg, train, val, device, args.resume_steps)
    results["uniform_loss_ln_v"] = math.log(cfg.model.vocab_size)

    run_dir = make_run_dir(cfg.run_dir, "sanity")
    atomic_write_json(run_dir / "sanity.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

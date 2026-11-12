"""Train a GPT on prepared data.

Every override is applied to the config *before* the run starts and is saved in
the run's ``config.json``, so runs remain fully described by their directory.

Examples:
    python scripts/train.py --config configs/gpt_10m.toml --batch-size 8 --max-steps 100 \
        --warmup-steps 10 --eval-interval 50 --device mps
    python scripts/train.py --resume runs/<run>/checkpoints/step_00000100.pt --max-steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.prepare import load_prepared, processed_dir
from mini_chatgpt.device import describe_device, select_device
from mini_chatgpt.inference.generate import generate_text
from mini_chatgpt.training.checkpoint import atomic_write_json, load_checkpoint
from mini_chatgpt.training.run import make_run_dir, setup_logging, write_run_metadata
from mini_chatgpt.training.trainer import Trainer

TRAINING_OVERRIDES = (
    "batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "warmup_steps",
    "learning_rate",
    "min_learning_rate",
    "eval_interval",
    "eval_steps",
    "log_interval",
    "checkpoint_interval",
    "device",
    "seed",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/gpt_10m.toml")
    p.add_argument("--resume", type=Path, help="checkpoint to resume (uses its saved config)")
    p.add_argument("--name", help="run name (default: config name)")
    p.add_argument("--attention", choices=["sdpa", "explicit"])
    for key in TRAINING_OVERRIDES:
        kind: Any = str if key == "device" else float if "learning_rate" in key else int
        p.add_argument(f"--{key.replace('_', '-')}", type=kind, dest=key)
    p.add_argument("--warmup-skip", type=int, default=10, help="steps excluded from timing stats")
    p.add_argument("--full-eval", action="store_true", help="exact loss over all windows at end")
    p.add_argument("--sample-tokens", type=int, default=300)
    p.add_argument("--sample-prompt", default="ROMEO:")
    return p.parse_args()


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    changes = {k: getattr(args, k) for k in TRAINING_OVERRIDES if getattr(args, k) is not None}
    training = dataclasses.replace(cfg.training, **changes)
    model = cfg.model
    if args.attention:
        model = dataclasses.replace(model, attention=args.attention)
    name = args.name or cfg.name
    return dataclasses.replace(cfg, training=training, model=model, name=name)


def main() -> None:
    args = parse_args()
    log = setup_logging()

    if args.resume:
        _, meta = load_checkpoint(args.resume)
        cfg = ExperimentConfig.from_dict(meta.config)
        if args.max_steps is not None:
            cfg = dataclasses.replace(
                cfg, training=dataclasses.replace(cfg.training, max_steps=args.max_steps)
            )
        run_dir = args.resume.parent.parent
    else:
        cfg = apply_overrides(ExperimentConfig.from_toml(args.config), args)

    data = load_prepared(processed_dir(cfg.data))
    if data.tokenizer.vocab_size != cfg.model.vocab_size:
        raise SystemExit(
            f"config vocab_size {cfg.model.vocab_size} != tokenizer {data.tokenizer.vocab_size}"
        )
    device = select_device(cfg.training.device)

    if not args.resume:
        run_dir = make_run_dir(cfg.run_dir, cfg.name)
        write_run_metadata(run_dir, cfg, describe_device(device))
        data.tokenizer.save(run_dir / "tokenizer.json")  # generation uses this exact vocab
    trainer = Trainer(cfg, data.splits.train, data.splits.val, device, run_dir)
    log.info("parameters: %s | device: %s", f"{trainer.model.num_parameters():,}", device)
    if args.resume:
        trainer.load_state(args.resume)

    initial = trainer.estimate_loss() if trainer.step == 0 else None
    if initial:
        log.info("initial eval: train %.4f | val %.4f", initial["train"], initial["val"])
    final = trainer.fit()

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "device": str(device),
        "parameters": trainer.model.num_parameters(),
        "steps": trainer.step,
        "tokens_seen": trainer.tokens_seen,
        "tokens_per_step": trainer.tokens_per_step,
        "first_step_train_loss": trainer.loss_history[0] if trainer.loss_history else None,
        "last_step_train_loss": trainer.loss_history[-1] if trainer.loss_history else None,
        "initial_eval": initial,
        "final_eval": final,
        "timing": trainer.stats.timing_summary(skip=args.warmup_skip),
        "peak_memory_bytes": trainer.stats.peak_memory,
    }
    timing = summary["timing"]
    if timing:
        timing["tokens_per_s_from_mean"] = trainer.tokens_per_step / timing["mean_step_s"]
        timing["tokens_per_s_from_median"] = trainer.tokens_per_step / timing["median_step_s"]
    if args.full_eval:
        summary["full_val_loss"] = trainer.full_loss(data.splits.val)
        summary["full_val_bits_per_char"] = summary["full_val_loss"] / math.log(2)
    summary["sample"] = generate_text(
        trainer.model,
        data.tokenizer,
        args.sample_prompt,
        args.sample_tokens,
        temperature=0.8,
        top_k=40,
        seed=cfg.training.seed,
    )
    atomic_write_json(Path(run_dir) / f"summary_step{trainer.step}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

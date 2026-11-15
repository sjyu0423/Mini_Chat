"""Train a GPT on prepared data.

Every override is applied to the config *before* the run starts and is saved in
the run's ``config.json``, so runs remain fully described by their directory.

Examples:
    python scripts/train.py --config configs/gpt_10m.toml --batch-size 8 --max-steps 100 \
        --warmup-steps 10 --eval-interval 50 --device mps
    python scripts/train.py --resume runs/<run>/checkpoints/step_00000100.pt
    python scripts/train.py --resume-latest runs/<run>

Resuming keeps the saved config, including ``max_steps`` (the cosine schedule
depends on it). Changing it requires ``--extend-schedule``.

SIGINT/SIGTERM finish the current step, write a checkpoint, and exit. A NaN/Inf
loss or gradient norm stops the run without touching the weights; the last
periodic checkpoint stays valid and ``failure.json`` describes what happened.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import signal
import sys
from pathlib import Path
from typing import Any

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.prepare import file_sha256, load_prepared, processed_dir
from mini_chatgpt.device import describe_device, select_device
from mini_chatgpt.inference.generate import generate_text
from mini_chatgpt.training.checkpoint import (
    atomic_write_json,
    best_checkpoint_path,
    latest_valid_checkpoint,
    load_checkpoint,
)
from mini_chatgpt.training.metrics import bits_per_token, perplexity
from mini_chatgpt.training.run import (
    export_csv,
    make_run_dir,
    setup_logging,
    write_run_metadata,
)
from mini_chatgpt.training.trainer import NonFiniteError, Trainer

TRAINING_OVERRIDES = (
    "batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "lr_schedule_steps",
    "warmup_steps",
    "learning_rate",
    "min_learning_rate",
    "eval_interval",
    "eval_steps",
    "log_interval",
    "checkpoint_interval",
    "keep_last_checkpoints",
    "device",
    "seed",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/gpt_10m.toml")
    p.add_argument("--resume", type=Path, help="checkpoint to resume (uses its saved config)")
    p.add_argument(
        "--resume-latest",
        type=Path,
        metavar="RUN_DIR",
        help="resume from the newest loadable step_*.pt in RUN_DIR",
    )
    p.add_argument(
        "--continue-from",
        type=Path,
        metavar="CKPT",
        help="start a NEW training phase (new run dir, --config's schedule) from CKPT",
    )
    p.add_argument(
        "--extend-schedule",
        action="store_true",
        help="allow --max-steps to differ from the checkpoint (changes the LR schedule)",
    )
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

    if args.resume_latest:
        args.resume = latest_valid_checkpoint(args.resume_latest)
        if args.resume is None:
            raise SystemExit(f"no loadable checkpoint in {args.resume_latest}")
    if args.resume:
        _, meta = load_checkpoint(args.resume)
        cfg = ExperimentConfig.from_dict(meta.config)
        if args.max_steps is not None and args.max_steps != cfg.training.max_steps:
            if not args.extend_schedule:
                raise SystemExit(
                    f"--max-steps {args.max_steps} != the run's {cfg.training.max_steps}; "
                    "that changes the LR schedule. Pass --extend-schedule to do it anyway."
                )
            log.warning("changing max_steps %d -> %d", cfg.training.max_steps, args.max_steps)
            cfg = dataclasses.replace(
                cfg, training=dataclasses.replace(cfg.training, max_steps=args.max_steps)
            )
        run_dir = args.resume.parent.parent
    else:
        cfg = apply_overrides(ExperimentConfig.from_toml(args.config), args)
    if args.continue_from and args.resume:
        raise SystemExit("--continue-from and --resume are mutually exclusive")

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
    if data.meta.get("val_bytes_per_token"):
        trainer.val_bytes_per_token = float(data.meta["val_bytes_per_token"])
    log.info("parameters: %s | device: %s | run: %s",
             f"{trainer.model.num_parameters():,}", device, run_dir)  # fmt: skip
    if args.resume:
        trainer.load_state(args.resume)
    if args.continue_from:
        parent = args.continue_from.resolve()
        if parent.parent.parent == Path(run_dir).resolve():
            raise SystemExit("continuation must write to a new run directory")
        parent_tok = parent.parent.parent / "tokenizer.json"
        if parent_tok.exists() and hasattr(data.tokenizer, "fingerprint"):
            from mini_chatgpt.tokenizer import load_tokenizer

            ptok = load_tokenizer(parent_tok)
            if getattr(ptok, "fingerprint", lambda: None)() != data.tokenizer.fingerprint():
                raise SystemExit("parent run tokenizer differs from the dataset tokenizer")
        parent_sha = file_sha256(parent)
        pmeta = trainer.load_state(parent, continuation=True)
        atomic_write_json(Path(run_dir) / "continuation.json", {
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": parent_sha,
            "parent_step": pmeta.step,
            "parent_config": pmeta.config,
            "phases": trainer.phases,
            "tokenizer_fingerprint": data.meta.get("tokenizer_fingerprint"),
            "dataset_train_sha256": data.meta.get("train_sha256"),
            "dataset_val_sha256": data.meta.get("val_sha256"),
        })  # fmt: skip
        log.info("continuing from %s (step %d) as phase %d: %s lr %.2e -> step %d",
                 parent, pmeta.step, len(trainer.phases), cfg.training.lr_schedule,
                 cfg.training.learning_rate, cfg.training.max_steps)  # fmt: skip

    def on_signal(signum: int, _frame: object) -> None:
        log.warning("received %s: stopping after the current step", signal.Signals(signum).name)
        trainer.request_stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        final = trainer.fit()
    except NonFiniteError as exc:
        log.error("%s", exc)
        atomic_write_json(
            Path(run_dir) / "failure.json",
            {"error": str(exc), "step": trainer.step, "loss_tail": trainer.loss_history[-20:]},
        )
        export_csv(Path(run_dir) / "metrics.jsonl")
        sys.exit(2)

    csvs = export_csv(Path(run_dir) / "metrics.jsonl")
    initial = trainer.eval_history[0] if trainer.eval_history else None
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "device": str(device),
        "parameters": trainer.model.num_parameters(),
        "steps": trainer.step,
        "phases": trainer.phases,
        "max_steps": cfg.training.max_steps,
        "completed": trainer.step >= cfg.training.max_steps,
        "tokens_seen": trainer.tokens_seen,
        "tokens_per_step": trainer.tokens_per_step,
        "first_step_train_loss": trainer.loss_history[0] if trainer.loss_history else None,
        "last_step_train_loss": trainer.loss_history[-1] if trainer.loss_history else None,
        "initial_eval": initial,
        "final_eval": final,
        "best_metric": trainer.best_metric,
        "best_val_loss": trainer.best_val_loss,
        "best_step": trainer.best_step,
        "best_checkpoint": str(best_checkpoint_path(run_dir)),
        "timing_this_process": trainer.stats.timing_summary(skip=args.warmup_skip),
        "memory_max_bytes": trainer.stats.peak_memory,
        "csv": {k: str(v) for k, v in csvs.items()},
    }
    timing = summary["timing_this_process"]
    if timing:
        timing["tokens_per_s_from_mean"] = trainer.tokens_per_step / timing["mean_step_s"]
        timing["tokens_per_s_from_median"] = trainer.tokens_per_step / timing["median_step_s"]
    if args.full_eval:
        loss = trainer.full_loss(data.splits.val)
        summary["full_val_loss"] = loss
        summary["full_val_ppl"] = perplexity(loss)
        summary["full_val_bits_per_char"] = bits_per_token(loss)
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
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

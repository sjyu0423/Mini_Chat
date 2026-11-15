"""Run M5 supervised fine-tuning from the M4.1 base checkpoint.

Writes a new run directory ``runs/<name>-<timestamp>/`` with ``sft_config.json``,
``tokenizer.json`` (copy, fingerprint-checked), ``metrics.jsonl``,
``checkpoints/{best,final}.pt`` and ``summary.json``. The base checkpoint is only read.

Usage: python scripts/train_sft.py --config configs/sft/m5_sft_chat.toml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mini_chatgpt.chat.format import ChatTokens
from mini_chatgpt.chat.sft import SFTConfig, SFTTrainer, load_split
from mini_chatgpt.data.prepare import file_sha256
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    cfg = SFTConfig.from_toml(args.config)
    tok = ByteBPETokenizer.load(Path(cfg.data_dir) / "tokenizer.json")
    if tok.fingerprint() != cfg.expected_fingerprint:
        raise SystemExit("tokenizer fingerprint differs from the base model's")
    base_sha = file_sha256(cfg.base_checkpoint)
    run_dir = Path(cfg.run_dir) / f"{cfg.name}-{datetime.now():%Y%m%d-%H%M%S}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(cfg.data_dir) / "tokenizer.json", run_dir / "tokenizer.json")
    atomic_write_json(
        run_dir / "sft_config.json",
        {**asdict(cfg), "base_checkpoint_sha256": base_sha, "torch": torch.__version__},
    )
    trainer = SFTTrainer(
        cfg,
        run_dir=run_dir,
        device=torch.device(cfg.device),
        train=load_split(cfg.data_dir, "train"),
        val=load_split(cfg.data_dir, "val"),
        end_id=ChatTokens.from_tokenizer(tok).end,
    )
    logging.info(
        "run dir %s; %d steps (%d per epoch x %d epochs)",
        run_dir,
        trainer.max_steps,
        trainer.steps_per_epoch,
        cfg.epochs,
    )
    summary = trainer.fit()
    summary["base_checkpoint_sha256_after"] = file_sha256(cfg.base_checkpoint)
    summary["base_checkpoint_unchanged"] = summary["base_checkpoint_sha256_after"] == base_sha
    atomic_write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()

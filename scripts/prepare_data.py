"""Prepare a dataset, print statistics, and run reproducibility checks.

Usage: python scripts/prepare_data.py [--config configs/gpt_10m.toml] [--no-download]
"""

from __future__ import annotations

import argparse
import json

import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.dataset import get_batch
from mini_chatgpt.data.prepare import build, load_prepared, processed_dir, save
from mini_chatgpt.data.sources import load_text
from mini_chatgpt.training.run import setup_logging
from mini_chatgpt.training.seed import make_generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gpt_10m.toml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    log = setup_logging()

    cfg = ExperimentConfig.from_toml(args.config)
    prepared = build(cfg.data, allow_download=not args.no_download)
    out = save(prepared, processed_dir(cfg.data))
    log.info("wrote %s", out)
    print(json.dumps(prepared.meta, indent=2, ensure_ascii=False))

    # ---- reproducibility checks ------------------------------------------ #
    text = load_text(cfg.data.dataset, cfg.data.data_dir, allow_download=False)
    tok = prepared.tokenizer
    assert tok.decode(tok.encode(text)) == text, "round trip failed"
    log.info("check: full-corpus encode/decode round trip OK (%d chars)", len(text))

    again = build(cfg.data, allow_download=False)
    assert again.meta["train_sha256"] == prepared.meta["train_sha256"]
    assert again.meta["val_sha256"] == prepared.meta["val_sha256"]
    log.info("check: rebuilding produces identical train/val splits")

    loaded = load_prepared(out)
    assert torch.equal(loaded.splits.train, prepared.splits.train)
    assert torch.equal(loaded.splits.val, prepared.splits.val)
    log.info("check: saved .bin files reload and match hashes")

    ctx, bsz = cfg.model.context_length, cfg.training.batch_size
    x1, y1 = get_batch(loaded.splits.train, bsz, ctx, generator=make_generator(cfg.training.seed))
    x2, y2 = get_batch(loaded.splits.train, bsz, ctx, generator=make_generator(cfg.training.seed))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.equal(x1[:, 1:], y1[:, :-1])
    log.info("check: seeded batches are identical and targets are inputs shifted by one")
    log.info("sample batch[0] input : %r", tok.decode(x1[0, :60].tolist()))
    log.info("sample batch[0] target: %r", tok.decode(y1[0, :60].tolist()))


if __name__ == "__main__":
    main()

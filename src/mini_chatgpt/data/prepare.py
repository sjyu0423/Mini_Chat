"""Prepare a dataset: load text, build tokenizer, encode, split, and save.

Output (``<data_dir>/processed/<dataset>_<tokenizer>/``)::

    train.bin       uint16 token ids (little-endian)
    val.bin         uint16 token ids
    tokenizer.json  tokenizer vocabulary
    meta.json       statistics, hashes, and provenance
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_chatgpt.config import DataConfig
from mini_chatgpt.data.dataset import TokenSplits, split_tokens
from mini_chatgpt.data.sources import get_source, load_text, sha256_bytes
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.tokenizer.char import CharTokenizer
from mini_chatgpt.training.checkpoint import atomic_write_bytes, atomic_write_json

_TOKEN_DTYPE = np.dtype("<u2")  # uint16 covers vocabularies up to 65,535


def processed_dir(cfg: DataConfig) -> Path:
    return Path(cfg.data_dir) / "processed" / f"{cfg.dataset}_{cfg.tokenizer}"


def tensor_sha256(t: torch.Tensor) -> str:
    return sha256_bytes(t.numpy().astype(_TOKEN_DTYPE).tobytes())


def text_statistics(text: str, top_k: int = 10) -> dict[str, Any]:
    counts = Counter(text)
    words = text.split()
    return {
        "num_chars": len(text),
        "num_bytes_utf8": len(text.encode("utf-8")),
        "num_lines": text.count("\n"),
        "num_words_whitespace": len(words),
        "num_unique_chars": len(counts),
        "most_common_chars": [[c, n] for c, n in counts.most_common(top_k)],
    }


@dataclass(frozen=True)
class PreparedData:
    tokenizer: CharTokenizer | ByteBPETokenizer
    splits: TokenSplits
    meta: dict[str, Any]


def build(cfg: DataConfig, *, allow_download: bool = True) -> PreparedData:
    """Pure pipeline (no writes besides the one-time raw download)."""
    if cfg.tokenizer != "char":
        raise NotImplementedError(f"tokenizer {cfg.tokenizer!r} is not implemented yet")
    text = load_text(cfg.dataset, cfg.data_dir, allow_download=allow_download)
    # Vocabulary is built from the full corpus. For char-level Shakespeare this
    # only affects which symbols exist, not any statistics the model sees.
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train, val = split_tokens(ids, cfg.val_fraction)
    source = get_source(cfg.dataset)
    meta = {
        "dataset": cfg.dataset,
        "source_url": source.url,
        "source_sha256": source.sha256,
        "license": source.license,
        "tokenizer": cfg.tokenizer,
        "vocab_size": tok.vocab_size,
        "val_fraction": cfg.val_fraction,
        "num_tokens": ids.numel(),
        "num_train_tokens": train.numel(),
        "num_val_tokens": val.numel(),
        "train_sha256": tensor_sha256(train),
        "val_sha256": tensor_sha256(val),
        "text_stats": text_statistics(text),
    }
    return PreparedData(tok, TokenSplits(train, val), meta)


def save(prepared: PreparedData, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    if prepared.tokenizer.vocab_size > np.iinfo(_TOKEN_DTYPE).max + 1:
        raise ValueError("vocabulary too large for uint16 storage")
    atomic_write_bytes(
        out / "train.bin", prepared.splits.train.numpy().astype(_TOKEN_DTYPE).tobytes()
    )
    atomic_write_bytes(out / "val.bin", prepared.splits.val.numpy().astype(_TOKEN_DTYPE).tobytes())
    prepared.tokenizer.save(out / "tokenizer.json")
    atomic_write_json(out / "meta.json", prepared.meta)
    return out


def file_sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_prepared(out_dir: str | Path, *, verify: bool = True) -> PreparedData:
    """Load saved splits and verify them against the recorded hashes.

    Character-level datasets (M0-M2) are read into int64 tensors as before.
    BPE datasets (M3.1+) are memory-mapped as uint16 (zero copy; batches are
    converted to int64 per step), and additionally checked for the tokenizer
    fingerprint and for token ids within the vocabulary.
    """
    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    if meta.get("tokenizer") == "bpe":
        bpe = ByteBPETokenizer.load(out / "tokenizer.json")
        if bpe.fingerprint() != meta["tokenizer_fingerprint"]:
            raise RuntimeError("tokenizer.json fingerprint does not match meta.json")
        if bpe.vocab_size != meta["vocab_size"]:
            raise RuntimeError("tokenizer.json vocab_size does not match meta.json")
        tensors = []
        for name in ("train", "val"):
            path = out / f"{name}.bin"
            if verify and file_sha256(path) != meta[f"{name}_sha256"]:
                raise RuntimeError(f"{name}.bin does not match meta.json hash; re-run prepare")
            mm = np.memmap(path, dtype=_TOKEN_DTYPE, mode="c")  # copy-on-write: never mutates
            if mm.size != meta[f"num_{name}_tokens"]:
                raise RuntimeError(f"{name}.bin has {mm.size} tokens, meta says otherwise")
            if verify and int(mm.max()) >= bpe.vocab_size:
                raise RuntimeError(f"{name}.bin contains ids outside the vocabulary")
            tensors.append(torch.from_numpy(mm))
        return PreparedData(bpe, TokenSplits(tensors[0], tensors[1]), meta)

    tok = CharTokenizer.load(out / "tokenizer.json")
    splits = []
    for name in ("train", "val"):
        arr = np.fromfile(out / f"{name}.bin", dtype=_TOKEN_DTYPE)
        t = torch.from_numpy(arr.astype(np.int64))
        if tensor_sha256(t) != meta[f"{name}_sha256"]:
            raise RuntimeError(f"{name}.bin does not match meta.json hash; re-run prepare")
        splits.append(t)
    if tok.vocab_size != meta["vocab_size"]:
        raise RuntimeError("tokenizer.json vocab_size does not match meta.json")
    return PreparedData(tok, TokenSplits(splits[0], splits[1]), meta)

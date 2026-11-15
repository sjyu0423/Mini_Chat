import json
from pathlib import Path

from mini_chatgpt.tokenizer.base import Tokenizer
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.tokenizer.char import CharTokenizer


def load_tokenizer(path: str | Path) -> CharTokenizer | ByteBPETokenizer:
    """Load a saved tokenizer of either type, dispatching on its ``type`` field."""
    kind = json.loads(Path(path).read_text(encoding="utf-8")).get("type")
    if kind == "char":
        return CharTokenizer.load(path)
    if kind == "byte_bpe":
        return ByteBPETokenizer.load(path)
    raise ValueError(f"unknown tokenizer type {kind!r} in {path}")


__all__ = ["ByteBPETokenizer", "CharTokenizer", "Tokenizer", "load_tokenizer"]

"""Character-level tokenizer: the simplest possible baseline.

The vocabulary is the sorted set of unique characters in the training corpus,
so ids are deterministic for a given text. Unknown characters raise an error
rather than mapping to a hidden <unk> token, which keeps the baseline honest.
"""

from __future__ import annotations

import json
from pathlib import Path

from mini_chatgpt.training.checkpoint import atomic_write_text


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        if not chars:
            raise ValueError("vocabulary must not be empty")
        if any(len(c) != 1 for c in chars):
            raise ValueError("every vocabulary entry must be a single character")
        if len(set(chars)) != len(chars):
            raise ValueError("vocabulary contains duplicate characters")
        self.chars: list[str] = list(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids: list[int]) -> str:
        n = self.vocab_size
        for i in ids:
            if not 0 <= i < n:
                raise ValueError(f"token id {i} out of range [0, {n})")
        return "".join(self.chars[i] for i in ids)

    # ---- persistence ------------------------------------------------------ #

    def to_dict(self) -> dict[str, object]:
        return {"type": "char", "chars": self.chars}

    def save(self, path: str | Path) -> None:
        atomic_write_text(path, json.dumps(self.to_dict(), ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("type") != "char":
            raise ValueError(f"not a char tokenizer file: {path}")
        return cls(list(raw["chars"]))

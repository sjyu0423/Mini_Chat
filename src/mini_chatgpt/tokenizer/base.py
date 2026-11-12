"""Tokenizer interface shared by all tokenizers (char now, BPE later)."""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

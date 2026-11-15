"""Byte-level byte-pair-encoding (BPE) tokenizer, implemented from scratch.

Design
------
* **Base alphabet = the 256 byte values.** Any Unicode string is encoded as its
  UTF-8 bytes, so there is no unknown token and ``decode(encode(s)) == s`` for
  every ``str``.
* **Pre-tokenization.** Before merging, text is split into chunks by a regular
  expression (contractions, optionally-space-prefixed letter runs, 1-3 digit
  groups, punctuation runs, whitespace). Merges never cross chunk boundaries,
  which keeps tokens linguistically sensible ("dog" vs "dog."). The final
  ``[\\s\\S]`` alternative guarantees the chunks always cover the input.
* **Training** counts chunk frequencies, then repeatedly merges the most frequent
  adjacent pair of token ids. Ties are broken by the smallest ``(left, right)``
  id pair, so training is fully deterministic for a given corpus. Pair counts are
  updated incrementally (only words containing the merged pair are touched) and
  the maximum is found with a lazily invalidated heap.
* **Special tokens** get ids after all merges. They are never produced from
  ordinary text unless explicitly allowed: ``encode("<|endoftext|>")`` yields the
  bytes of that string, while ``encode(..., allowed_special="all")`` yields the
  single special id. This prevents text from injecting control tokens.
* **Persistence** is a JSON file with the pattern, merges, special tokens, and a
  SHA-256 fingerprint of all three; loading verifies the fingerprint.

``vocab_size`` counts everything: 256 byte tokens + merges + special tokens.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from mini_chatgpt.training.checkpoint import atomic_write_text

FORMAT_VERSION = 1
DEFAULT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"  # English contractions
    r"| ?[^\W\d_]+"  # letters (Unicode-aware), optional leading space
    r"| ?\d{1,3}"  # numbers in groups of at most three digits
    r"| ?(?:[^\s\w]|_)+"  # punctuation / symbols (and underscores)
    r"|\s+(?!\S)"  # whitespace not followed by a non-space (keeps " word" intact)
    r"|\s+"  # remaining whitespace
    r"|[\s\S]"  # safety net: every character is covered
)
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<|endoftext|>",)
AllowedSpecial = Literal["none", "all"] | set[str] | frozenset[str]


def _merge_word(word: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    a, b = pair
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class ByteBPETokenizer:
    """Byte-level BPE. Construct with :meth:`train` or :meth:`load`."""

    def __init__(
        self,
        merges: Sequence[tuple[int, int]],
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        pattern: str = DEFAULT_PATTERN,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.merges: list[tuple[int, int]] = [(int(a), int(b)) for a, b in merges]
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        self.ranks: dict[tuple[int, int], int] = {}
        for rank, (a, b) in enumerate(self.merges):
            new_id = 256 + rank
            if not (0 <= a < new_id and 0 <= b < new_id):
                raise ValueError(f"merge {rank} {(a, b)} refers to an undefined token")
            if (a, b) in self.ranks:
                raise ValueError(f"duplicate merge {(a, b)}")
            self.ranks[(a, b)] = rank
            self.vocab.append(self.vocab[a] + self.vocab[b])
        if len(set(special_tokens)) != len(special_tokens) or any(not s for s in special_tokens):
            raise ValueError("special tokens must be unique, non-empty strings")
        base = 256 + len(self.merges)
        self.special_tokens: dict[str, int] = {s: base + i for i, s in enumerate(special_tokens)}
        self.special_ids: dict[int, str] = {i: s for s, i in self.special_tokens.items()}
        self._special_regex = (
            re.compile(
                "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
            )
            if special_tokens
            else None
        )
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_id(self) -> int | None:
        return self.special_tokens.get("<|endoftext|>")

    def pretokenize(self, text: str) -> list[str]:
        return self._regex.findall(text)

    # ---- training ---------------------------------------------------------- #
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        pattern: str = DEFAULT_PATTERN,
        min_frequency: int = 2,
    ) -> ByteBPETokenizer:
        """Learn ``vocab_size - 256 - len(special_tokens)`` merges from ``texts``.

        Raises ``ValueError`` if the corpus runs out of pairs occurring at least
        ``min_frequency`` times before the target size is reached.
        """
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size {vocab_size} is smaller than 256 bytes + "
                f"{len(special_tokens)} special tokens"
            )
        regex = re.compile(pattern)
        chunk_counts: Counter[str] = Counter()
        n_docs = 0
        n_bytes = 0
        sha = hashlib.sha256()
        for text in texts:
            n_docs += 1
            encoded = text.encode("utf-8")
            n_bytes += len(encoded)
            sha.update(hashlib.sha256(encoded).digest())
            chunk_counts.update(regex.findall(text))

        # Deterministic word order (sorted chunk strings) -> deterministic ids.
        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk in sorted(chunk_counts):
            words.append(list(chunk.encode("utf-8")))
            freqs.append(chunk_counts[chunk])

        pair_counts: dict[tuple[int, int], int] = {}
        where: dict[tuple[int, int], set[int]] = {}
        for wi, (w, f) in enumerate(zip(words, freqs, strict=True)):
            for pair in itertools.pairwise(w):
                pair_counts[pair] = pair_counts.get(pair, 0) + f
                where.setdefault(pair, set()).add(wi)
        heap = [(-c, a, b) for (a, b), c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges:
            best: tuple[int, int] | None = None
            while heap:
                neg, a, b = heapq.heappop(heap)
                if pair_counts.get((a, b), 0) == -neg and -neg > 0:
                    best = (a, b)
                    break
            if best is None or pair_counts[best] < min_frequency:
                raise ValueError(
                    f"corpus supports only {len(merges)} merges with frequency >= "
                    f"{min_frequency}; requested {n_merges}"
                )
            new_id = 256 + len(merges)
            merges.append(best)
            changed: set[tuple[int, int]] = set()
            for wi in sorted(where.pop(best, ())):
                old = words[wi]
                new = _merge_word(old, best, new_id)
                if len(new) == len(old):
                    continue  # stale index entry: the word no longer has the pair
                f = freqs[wi]
                for pair in itertools.pairwise(old):
                    pair_counts[pair] -= f
                    changed.add(pair)
                for pair in itertools.pairwise(new):
                    pair_counts[pair] = pair_counts.get(pair, 0) + f
                    where.setdefault(pair, set()).add(wi)
                    changed.add(pair)
                words[wi] = new
            pair_counts.pop(best, None)
            for pair in changed:
                c = pair_counts.get(pair, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, pair[0], pair[1]))
                else:
                    pair_counts.pop(pair, None)

        meta = {
            "trained_on_documents": n_docs,
            "trained_on_bytes": n_bytes,
            "trained_on_sha256": sha.hexdigest(),
            "unique_pretokens": len(chunk_counts),
            "min_frequency": min_frequency,
        }
        return cls(merges, special_tokens, pattern, meta)

    # ---- encoding ---------------------------------------------------------- #
    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            best_rank = None
            best_pair = None
            for pair in itertools.pairwise(ids):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None or best_rank is None:
                break
            ids = _merge_word(ids, best_pair, 256 + best_rank)
        out = tuple(ids)
        if len(self._cache) >= 500_000:
            self._cache.clear()
        self._cache[chunk] = out
        return out

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode treating every character (including special-token text) as data."""
        out: list[int] = []
        for chunk in self._regex.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allowed_special: AllowedSpecial = "none") -> list[int]:
        """Encode ``text``. Special tokens are recognized only if ``allowed_special``
        is ``"all"`` or a set containing them; otherwise they are plain text."""
        if allowed_special == "none" or self._special_regex is None:
            return self.encode_ordinary(text)
        allowed = set(self.special_tokens) if allowed_special == "all" else set(allowed_special)
        unknown = allowed - set(self.special_tokens)
        if unknown:
            raise ValueError(f"unknown special tokens: {sorted(unknown)}")
        out: list[int] = []
        pos = 0
        for m in self._special_regex.finditer(text):
            if m.group() not in allowed:
                continue
            out.extend(self.encode_ordinary(text[pos : m.start()]))
            out.append(self.special_tokens[m.group()])
            pos = m.end()
        out.extend(self.encode_ordinary(text[pos:]))
        return out

    # ---- decoding ---------------------------------------------------------- #
    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        parts: list[bytes] = []
        n_ordinary = 256 + len(self.merges)
        for i in ids:
            if 0 <= i < n_ordinary:
                parts.append(self.vocab[i])
            elif i in self.special_ids:
                parts.append(self.special_ids[i].encode("utf-8"))
            else:
                raise ValueError(f"token id {i} out of range [0, {self.vocab_size})")
        return b"".join(parts)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids to text. Invalid UTF-8 (possible for arbitrary model output that
        ends mid-character) is replaced with U+FFFD; encoded text always round-trips."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        return self.decode_bytes([token_id])

    # ---- persistence ------------------------------------------------------- #
    def fingerprint(self) -> str:
        canon = json.dumps(
            {"pattern": self.pattern, "merges": self.merges,
             "special_tokens": list(self.special_tokens)},
            separators=(",", ":"),
        )  # fmt: skip
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "byte_bpe",
            "version": FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "pattern": self.pattern,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
            "fingerprint": self.fingerprint(),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        atomic_write_text(path, json.dumps(self.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ByteBPETokenizer:
        if raw.get("type") != "byte_bpe":
            raise ValueError("not a byte-level BPE tokenizer file")
        if raw.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer format version {raw.get('version')}")
        specials = sorted(raw["special_tokens"], key=lambda s: raw["special_tokens"][s])
        tok = cls([tuple(m) for m in raw["merges"]], specials, raw["pattern"],
                  raw.get("metadata"))  # fmt: skip
        if tok.special_tokens != raw["special_tokens"]:
            raise ValueError("special-token ids do not match the merge count")
        if tok.vocab_size != raw["vocab_size"] or tok.fingerprint() != raw["fingerprint"]:
            raise ValueError("tokenizer file is inconsistent (vocab size or fingerprint)")
        return tok

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

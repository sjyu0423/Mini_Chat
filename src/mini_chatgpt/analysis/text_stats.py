"""Surface statistics of generated text: words, speaker labels, verbatim overlap.

These measure *form* (does the output use real words, real speaker names,
copied passages?), not meaning. A high valid-word rate does not imply that the
model understands anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z' ]*[A-Za-z]):$", re.MULTILINE)


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def vocabulary(text: str) -> set[str]:
    return set(words(text))


def speakers(text: str) -> set[str]:
    """Speaker labels: lines consisting only of a capitalized name and a colon."""
    return set(SPEAKER_RE.findall(text))


@dataclass(frozen=True)
class WordStats:
    n_words: int
    valid_fraction: float  # words found in the reference vocabulary
    valid_fraction_len4: float  # same, restricted to words of >= 4 letters
    n_len4: int


def word_stats(text: str, vocab: set[str]) -> WordStats:
    ws = words(text)
    long = [w for w in ws if len(w) >= 4]
    return WordStats(
        n_words=len(ws),
        valid_fraction=sum(w in vocab for w in ws) / len(ws) if ws else 0.0,
        valid_fraction_len4=sum(w in vocab for w in long) / len(long) if long else 0.0,
        n_len4=len(long),
    )


@dataclass(frozen=True)
class SpeakerStats:
    n_labels: int
    known_fraction: float  # labels that are real speakers in the reference text
    distinct: int


def speaker_stats(text: str, known: set[str]) -> SpeakerStats:
    labels = SPEAKER_RE.findall(text)
    return SpeakerStats(
        n_labels=len(labels),
        known_fraction=sum(lbl in known for lbl in labels) / len(labels) if labels else 0.0,
        distinct=len(set(labels)),
    )


@dataclass(frozen=True)
class Match:
    length: int
    start: int  # offset in the query text
    text: str


def longest_match(query: str, corpus: str) -> Match:
    """Longest substring of ``query`` that occurs verbatim in ``corpus``.

    Linear number of ``in`` checks: a match at offset i of length L implies a
    match of length >= L-1 at offset i+1, so L never has to restart from 0.
    """
    best = Match(0, 0, "")
    length = 0
    n = len(query)
    for i in range(n):
        length = max(length - 1, 0)
        while i + length < n and query[i : i + length + 1] in corpus:
            length += 1
        if length > best.length:
            best = Match(length, i, query[i : i + length])
    return best


def ngram_overlap(query: str, corpus: str, n: int) -> float:
    """Fraction of the query's length-``n`` character windows that occur in corpus."""
    windows = [query[i : i + n] for i in range(len(query) - n + 1)]
    if not windows:
        return 0.0
    return sum(w in corpus for w in windows) / len(windows)

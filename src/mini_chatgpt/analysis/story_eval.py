"""Shared evaluation helpers for story models (M3.1, M3.2).

* ``exact_loss``: exact token-weighted loss of a model over a token stream.
* ``KGramIndex``: exact verbatim-overlap index (sorted 64-bit k-gram hashes) over
  a training token stream, for memorization checks.
* ``story_metrics``: measured surface statistics of a generated story.
* ``sample``: fixed-seed generation that stops at ``<|endoftext|>``.
* ``entity_check``: heuristic fact-consistency check for prompts that bind
  objects or traits to named characters.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.inference.generate import generate
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.run import LOGGER_NAME
from mini_chatgpt.training.trainer import Trainer

log = logging.getLogger(LOGGER_NAME)
TEMPERATURE, TOP_K = 0.7, 40
MAX_NEW = 300
LADDER = [8, 16, 32, 48]  # each k-gram index over ~98M tokens is ~0.8 GB
WORD = re.compile(r"[A-Za-z']+")
HASH_B = np.uint64(1_000_003)


# --------------------------------------------------------------------------- #
def exact_loss(
    model: torch.nn.Module, data: torch.Tensor, cfg: ExperimentConfig, device: torch.device
) -> float:
    t = Trainer(cfg, data, data, device)
    t.model = model  # type: ignore[assignment]
    return t.full_loss(data)


def kgram_hashes(tokens: np.ndarray, k: int) -> np.ndarray:
    """Polynomial hash (mod 2^64) of every length-k window."""
    n = tokens.size - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    t = tokens.astype(np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * HASH_B + t[j : j + n]
    out: np.ndarray = h
    return out


class KGramIndex:
    """Sorted k-gram hashes of the training stream for several k (built lazily)."""

    def __init__(self, tokens: np.ndarray, ladder: list[int]) -> None:
        self.index: dict[int, np.ndarray] = {}
        for k in ladder:
            t0 = time.perf_counter()
            self.index[k] = np.unique(kgram_hashes(tokens, k))
            log.info(
                "k=%d index: %d unique windows (%.0f s)",
                k,
                self.index[k].size,
                time.perf_counter() - t0,
            )

    def hits(self, query: np.ndarray, k: int) -> np.ndarray:
        q = kgram_hashes(query, k)
        if q.size == 0:
            return np.zeros(0, dtype=bool)
        idx = self.index[k]
        pos = np.searchsorted(idx, q)
        pos[pos == idx.size] = 0
        found: np.ndarray = idx[pos] == q
        return found

    def overlap(self, query: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        longest = 0
        for k in LADDER:
            h = self.hits(query, k)
            out[f"frac_{k}gram_in_train"] = float(h.mean()) if h.size else None
            if h.size and h.any():
                longest = k
        out["longest_match_at_least"] = longest  # largest ladder k with any verbatim hit
        return out


def story_metrics(text: str, vocab: set[str]) -> dict[str, Any]:
    words = [w.lower() for w in WORD.findall(text)]
    tri = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    seen: set[tuple[str, ...]] = set()
    rep = 0
    for g in tri:
        rep += g in seen
        seen.add(g)
    bi = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
    common = {
        "Once",
        "One",
        "The",
        "They",
        "He",
        "She",
        "It",
        "But",
        "So",
        "When",
        "Then",
        "A",
        "Mom",
        "Dad",
        "I",
        "You",
        "We",
        "His",
        "Her",
        "In",
        "After",
        "There",
        "This",
        "That",
        "Yes",
        "No",
        "Oh",
        "As",
        "At",
        "And",
        "Every",
        "Finally",
        "From",
        "What",
    }
    names = sorted(set(re.findall(r"\b([A-Z][a-z]+)\b", text)) - common)
    return {
        "words": len(words),
        "valid_word_fraction": sum(w in vocab for w in words) / len(words) if words else 0.0,
        "word_trigram_repetition": rep / len(tri) if tri else 0.0,
        "distinct_2": len(set(bi)) / len(bi) if bi else 0.0,
        "sentences": len(re.findall(r"[.!?]", text)),
        "capitalized_names": names,
    }


def sample(
    model: GPT,
    tok: ByteBPETokenizer,
    prompt: str,
    seed: int,
    device: torch.device,
    greedy: bool = False,
    max_new: int = MAX_NEW,
) -> dict[str, Any]:
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model,
        idx,
        max_new,
        seed=seed,
        greedy=greedy,
        temperature=1.0 if greedy else TEMPERATURE,
        top_k=None if greedy else TOP_K,
    )
    new = out[0, len(ids) :].tolist()
    eot = tok.eot_id
    completed = eot in new
    if completed:
        new = new[: new.index(eot)]
    return {
        "prompt": prompt,
        "seed": seed,
        "greedy": greedy,
        "new_token_ids": new,
        "text": tok.decode(new),
        "completed_with_eot": completed,
        "new_tokens": len(new),
    }


SENT = re.compile(r"[^.!?\n]+[.!?]?")


def entity_check(text: str, names: list[str], facts: list[tuple[str, str]]) -> dict[str, Any]:
    """Heuristic consistency of ``facts`` = [(name, attribute word), ...] in ``text``.

    For every sentence that mentions an attribute word together with exactly one of
    the prompt's ``names``, the mention *supports* the fact if that name is the
    fact's owner and *conflicts* otherwise. Sentences naming several characters are
    ambiguous and not counted. Also reports which prompt names the continuation
    keeps using and which new capitalized names it introduces.
    """
    sents = [s for s in SENT.findall(text) if s.strip()]
    owner = {attr: name for name, attr in facts}
    support = conflict = 0
    examples: list[str] = []
    for s in sents:
        present = [n for n in names if re.search(rf"\b{re.escape(n)}\b", s)]
        for attr, who in owner.items():
            if len(present) == 1 and re.search(rf"\b{re.escape(attr)}\b", s, flags=re.IGNORECASE):
                if present[0] == who:
                    support += 1
                else:
                    conflict += 1
                    if len(examples) < 3:
                        examples.append(s.strip())
    caps = set(re.findall(r"\b([A-Z][a-z]+)\b", text))
    stop = {"Once", "One", "The", "They", "He", "She", "It", "But", "So", "When", "Then",
            "A", "I", "You", "We", "His", "Her", "In", "After", "There", "This", "That",
            "Yes", "No", "Oh", "As", "At", "And", "Every", "Finally", "From", "What", "Mom",
            "Dad", "Mommy", "Daddy", "Grandma", "Grandpa", "Let", "Can", "Do", "Don", "Wow",
            "Thank", "Look", "Come", "Please", "Hello", "Hi", "Why", "How", "Who", "Where",
            "If", "Now", "Their", "Our", "My", "Your", "Its", "These", "Together", "Suddenly",
            "Later", "Soon", "Inside", "Okay", "Sure", "Sorry", "Good", "Great", "Ok"}  # fmt: skip
    new_names = sorted(caps - set(names) - stop)
    return {
        "names_kept": [n for n in names if re.search(rf"\b{re.escape(n)}\b", text)],
        "names_kept_fraction": sum(bool(re.search(rf"\b{re.escape(n)}\b", text)) for n in names)
        / len(names),
        "new_capitalized_names": new_names,
        "fact_supports": support,
        "fact_conflicts": conflict,
        "conflict_examples": examples,
    }

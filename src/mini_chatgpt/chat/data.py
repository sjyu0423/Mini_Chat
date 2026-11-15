"""Conversational SFT data (M5): OASST2 + Dolly-15k -> normalized conversations.

Deterministic and pure-Python (raw files are JSONL / gzipped JSONL):

* ``CHAT_SOURCES``: the two files pinned to a dataset commit with size and SHA-256.
* ``oasst2_conversations``: one conversation per English ``ready_for_export`` tree,
  following the best-ranked valid reply at every turn (``best_path``).
* ``dolly_conversations``: one single-turn conversation per record.
* ``normalize_text`` / ``malformed_reason``: cleaning and rejection rules.
* ``dedup_key`` / ``split_of_group``: exact deduplication and a group-hash split so a
  conversation tree (or a repeated prompt) can never straddle train and validation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mini_chatgpt.chat.format import Message


@dataclass(frozen=True)
class ChatSource:
    name: str
    repo: str
    revision: str
    path: str
    size: int
    sha256: str
    license: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.path}"

    @property
    def local_name(self) -> str:
        return f"{self.name}__{self.path.replace('/', '__')}"


CHAT_SOURCES: tuple[ChatSource, ...] = (
    ChatSource(
        name="oasst2",
        repo="OpenAssistant/oasst2",
        revision="179dd21fc55192153d94adb0e0ce8f69e222bf75",
        path="2023-11-05_oasst2_ready.trees.jsonl.gz",
        size=54_370_156,
        sha256="7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b",
        license="Apache 2.0",
    ),
    ChatSource(
        name="dolly15k",
        repo="databricks/databricks-dolly-15k",
        revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        path="databricks-dolly-15k.jsonl",
        size=13_085_339,
        sha256="2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
        license="CC BY-SA 3.0 (Copyright 2023 Databricks, Inc.; includes Wikipedia excerpts)",
    ),
)


@dataclass
class Conversation:
    source: str
    group: str  # split / leakage unit: OASST tree id, Dolly record index
    messages: list[Message]
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MANY_BLANK = re.compile(r"\n{3,}")
_CITATION = re.compile(r"\[\d+\]")  # Wikipedia citation markers, removed per the Dolly card
_WS = re.compile(r"\s+")


def normalize_text(text: str, *, strip_citations: bool = False) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _CONTROL.sub("", t)
    if strip_citations:
        t = _CITATION.sub("", t)
    t = _TRAILING_WS.sub("\n", t)
    t = _MANY_BLANK.sub("\n\n", t)
    return t.strip()


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum("LATIN" in unicodedata.name(c, "") for c in letters) / len(letters)


def malformed_reason(messages: list[Message]) -> str | None:
    """Why a normalized conversation is unusable, or None."""
    if len(messages) < 2:
        return "no_complete_exchange"
    for i, m in enumerate(messages):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return "roles_not_alternating"
        if not m["content"]:
            return "empty_message"
        if "�" in m["content"]:
            return "replacement_character"
    if messages[-1]["role"] != "assistant":
        return "does_not_end_with_assistant"
    if latin_share(" ".join(m["content"] for m in messages)) < 0.9:
        return "not_english_script"
    return None


def dedup_key(messages: list[Message]) -> str:
    """Exact duplicate key: the first user message, lowercased with collapsed
    whitespace. Two conversations that open with the same request are duplicates
    for leakage purposes even if the replies differ."""
    first = _WS.sub(" ", messages[0]["content"].lower()).strip()
    return hashlib.sha256(first.encode("utf-8")).hexdigest()


def split_of_group(group: str, val_percent: int = 5) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


# --------------------------------------------------------------------------- #
# OASST2
# --------------------------------------------------------------------------- #
def _valid_oasst(m: dict[str, Any]) -> bool:
    return (
        not m.get("deleted", False)
        and m.get("review_result") is not False
        and not m.get("synthetic", False)
        and m.get("lang") == "en"
        and bool((m.get("text") or "").strip())
    )


def _rank_key(m: dict[str, Any]) -> tuple[float, str]:
    rank = m.get("rank")
    return (float(rank) if rank is not None else float("inf"), str(m["message_id"]))


def best_path(root: dict[str, Any]) -> list[dict[str, Any]]:
    """Root-to-leaf path that takes, at every node, the valid reply with the lowest
    (best) human ranking; unranked replies come after ranked ones, ties broken by
    message id. Stops at the first node with no valid reply."""
    path: list[dict[str, Any]] = []
    node: dict[str, Any] | None = root if _valid_oasst(root) else None
    while node is not None:
        path.append(node)
        replies = [r for r in node.get("replies", []) if _valid_oasst(r)]
        node = min(replies, key=_rank_key) if replies else None
    return path


def oasst2_conversations(path: Path, stats: dict[str, int]) -> Iterator[Conversation]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            stats["trees"] = stats.get("trees", 0) + 1
            root = tree["prompt"]
            if root.get("lang") != "en":
                stats["non_english_root"] = stats.get("non_english_root", 0) + 1
                continue
            if tree.get("tree_state") != "ready_for_export":
                stats["not_ready"] = stats.get("not_ready", 0) + 1
                continue
            nodes = best_path(root)
            if nodes and nodes[-1]["role"] == "prompter":
                nodes = nodes[:-1]  # a trailing unanswered user turn is not a target
            messages: list[Message] = [
                {
                    "role": "user" if n["role"] == "prompter" else "assistant",
                    "content": normalize_text(n["text"]),
                }
                for n in nodes
            ]
            yield Conversation(
                source="oasst2",
                group=f"oasst2:{tree['message_tree_id']}",
                messages=messages,
                meta={"tree_id": tree["message_tree_id"], "turns": len(messages)},
            )


# --------------------------------------------------------------------------- #
# Dolly-15k
# --------------------------------------------------------------------------- #
def dolly_user_text(instruction: str, context: str) -> str:
    """Instruction, then the reference passage (if any) after a blank line."""
    instruction = normalize_text(instruction)
    context = normalize_text(context, strip_citations=True)
    return f"{instruction}\n\n{context}" if context else instruction


def dolly_conversations(path: Path, stats: dict[str, int]) -> Iterator[Conversation]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            stats["records"] = stats.get("records", 0) + 1
            messages: list[Message] = [
                {"role": "user", "content": dolly_user_text(row["instruction"], row["context"])},
                {
                    "role": "assistant",
                    "content": normalize_text(row["response"], strip_citations=True),
                },
            ]
            yield Conversation(
                source="dolly15k",
                group=f"dolly15k:{i}",
                messages=messages,
                meta={"index": i, "category": row["category"], "has_context": bool(row["context"])},
            )

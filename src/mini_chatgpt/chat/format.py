"""Chat serialization, assistant-only loss masking and prompt construction (M5).

The format reuses the tokenizer's reserved special tokens (ids fixed in M4.1; the
vocabulary is never changed) and is byte-for-byte the template the frozen M4 suite
(``general_eval.chat_ids``) already uses::

    <|endoftext|>
    <|user|>\\n{user text}<|end|>\\n
    <|assistant|>\\n{assistant text}<|end|>\\n
    <|user|>\\n{user text}<|end|>\\n
    ...

* ``<|endoftext|>`` once at the start: the document separator the base model saw
  between pretraining documents, so a conversation starts like a fresh document.
* ``<|user|>`` / ``<|assistant|>`` / ``<|system|>`` open a message; ``\\n`` follows the
  header. ``<|end|>`` closes every message; ``\\n`` separates messages.
* Message text is encoded with ``allowed_special="none"``: a literal "<|end|>" typed by
  a user is ordinary bytes and can never forge a structural token.

Loss mask: a token is a training *target* iff it belongs to an assistant message's
content or is the ``<|end|>`` that closes an assistant message. Headers, the newline
after a header, user/system text, user ``<|end|>`` and separators condition the model
but are never predicted. Inference supplies exactly those non-target tokens itself
(``build_prompt``) and generation stops when the model emits ``<|end|>``.

Next-token alignment: for ids ``s[0..n-1]`` the model reads ``x = s[:-1]`` and
predicts ``y = s[1:]``; the label at input position t is ``s[t+1]`` if ``s[t+1]`` is a
target, else ``IGNORE`` (-100, skipped by ``F.cross_entropy``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer

Role = Literal["system", "user", "assistant"]
ROLES: tuple[Role, ...] = ("system", "user", "assistant")
IGNORE = -100
END = "<|end|>"
EOT = "<|endoftext|>"
STRUCTURAL = ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>")


class Message(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True)
class ChatTokens:
    """Ids of the structural tokens (read from the tokenizer, never hard-coded)."""

    eot: int
    user: int
    assistant: int
    system: int
    end: int
    newline: tuple[int, ...]

    @classmethod
    def from_tokenizer(cls, tok: ByteBPETokenizer) -> ChatTokens:
        sp = tok.special_tokens
        missing = [t for t in STRUCTURAL if t not in sp]
        if missing:
            raise ValueError(f"tokenizer lacks chat special tokens {missing}")
        return cls(
            eot=sp[EOT],
            user=sp["<|user|>"],
            assistant=sp["<|assistant|>"],
            system=sp["<|system|>"],
            end=sp[END],
            newline=tuple(tok.encode("\n")),
        )

    def header(self, role: Role) -> int:
        return {"system": self.system, "user": self.user, "assistant": self.assistant}[role]

    @property
    def structural(self) -> frozenset[int]:
        return frozenset({self.eot, self.user, self.assistant, self.system, self.end})


def validate_messages(messages: Sequence[Message], *, require_final_assistant: bool) -> None:
    """Optional leading system message, then strictly alternating user/assistant
    starting with user; no empty messages."""
    if not messages:
        raise ValueError("empty conversation")
    body = list(messages[1:] if messages[0]["role"] == "system" else messages)
    if not body:
        raise ValueError("conversation has only a system message")
    for i, m in enumerate(body):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            raise ValueError(f"message {i}: expected role {expected}, got {m['role']}")
    for m in messages:
        if m["role"] not in ROLES:
            raise ValueError(f"unknown role {m['role']!r}")
        if not m["content"].strip():
            raise ValueError("empty message")
    if require_final_assistant and body[-1]["role"] != "assistant":
        raise ValueError("conversation must end with an assistant message")


def encode_message(
    tok: ByteBPETokenizer, ct: ChatTokens, m: Message
) -> tuple[list[int], list[bool]]:
    """``<|role|> \\n content <|end|> \\n`` and its target mask."""
    content = tok.encode(m["content"])  # specials in user text stay ordinary bytes
    ids = [ct.header(m["role"]), *ct.newline, *content, ct.end, *ct.newline]
    is_asst = m["role"] == "assistant"
    mask = [False] * (1 + len(ct.newline)) + [is_asst] * (len(content) + 1)
    mask += [False] * len(ct.newline)
    return ids, mask


def encode_conversation(
    tok: ByteBPETokenizer, messages: Sequence[Message]
) -> tuple[list[int], list[bool]]:
    """Token ids of a complete conversation and a same-length target mask."""
    validate_messages(messages, require_final_assistant=True)
    ct = ChatTokens.from_tokenizer(tok)
    ids, mask = [ct.eot], [False]
    for m in messages:
        mi, mm = encode_message(tok, ct, m)
        ids += mi
        mask += mm
    return ids, mask


def labels_from_mask(ids: Sequence[int], mask: Sequence[bool]) -> tuple[list[int], list[int]]:
    """``(x, y)`` with ``x = ids[:-1]`` and ``y[t] = ids[t+1]`` if that token is a target,
    else ``IGNORE``."""
    if len(ids) != len(mask) or len(ids) < 2:
        raise ValueError("ids and mask must have the same length >= 2")
    x = list(ids[:-1])
    y = [i if m else IGNORE for i, m in zip(ids[1:], mask[1:], strict=True)]
    return x, y


def fit_conversation(
    tok: ByteBPETokenizer, messages: Sequence[Message], max_tokens: int
) -> tuple[list[Message], str]:
    """Deterministic context policy for *training* data.

    The full serialized sequence (``len(ids)``, i.e. ``len(x) + 1``) must be at most
    ``max_tokens + 1`` so that ``x`` fits the context window. If it does not, drop
    trailing messages back to the longest prefix that ends with a complete assistant
    message and fits. Messages are never cut mid-text (a cut reply would teach the model
    to answer without ending). Returns ``(kept, status)`` with status ``"full"``,
    ``"truncated"`` or ``"dropped"`` (not even the first exchange fits).
    """
    ids, _ = encode_conversation(tok, messages)
    if len(ids) <= max_tokens + 1:
        return list(messages), "full"
    msgs = list(messages)
    while True:
        # remove the last assistant message and the user message before it
        msgs = msgs[:-2]
        if not msgs or msgs[-1]["role"] != "assistant":
            return [], "dropped"
        ids, _ = encode_conversation(tok, msgs)
        if len(ids) <= max_tokens + 1:
            return msgs, "truncated"


def build_prompt(
    tok: ByteBPETokenizer,
    history: Sequence[Message],
    *,
    max_prompt_tokens: int | None = None,
) -> tuple[list[int], int]:
    """Inference prompt: the serialized history (ending with a user message) followed by
    an open ``<|assistant|>\\n`` header. Identical to ``general_eval.chat_ids``.

    If ``max_prompt_tokens`` is given and the prompt is longer, the oldest complete
    user/assistant exchanges are dropped (a leading system message is kept) until it
    fits. Returns ``(ids, dropped_messages)``; raises if even the latest user message
    alone does not fit.
    """
    validate_messages(history, require_final_assistant=False)
    if history[-1]["role"] != "user":
        raise ValueError("the last message of a prompt must be from the user")
    ct = ChatTokens.from_tokenizer(tok)
    system = [history[0]] if history[0]["role"] == "system" else []
    turns = list(history[len(system) :])
    dropped = 0
    while True:
        ids = [ct.eot]
        for m in [*system, *turns]:
            ids += encode_message(tok, ct, m)[0]
        ids += [ct.assistant, *ct.newline]
        if max_prompt_tokens is None or len(ids) <= max_prompt_tokens:
            return ids, dropped
        if len(turns) <= 1:
            raise ValueError(
                f"the latest user message alone needs {len(ids)} tokens, more than the "
                f"{max_prompt_tokens} available; shorten it"
            )
        turns = turns[2:]  # drop the oldest user + assistant pair
        dropped += 2


def display_text(tok: ByteBPETokenizer, ids: Sequence[int]) -> str:
    """Decode generated ids for display: structural tokens removed, ends stripped."""
    special = set(tok.special_ids)
    return tok.decode([i for i in ids if i not in special]).strip()

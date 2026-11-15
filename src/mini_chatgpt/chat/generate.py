"""Conversational generation with end-of-turn stopping (M5).

``chat_reply`` builds the prompt from the message history (``format.build_prompt``),
samples one token at a time and stops for exactly one recorded reason:

* ``"end_of_turn"``: the model produced ``<|end|>`` — the only *successful* stop.
* ``"max_tokens"``: ``max_new_tokens`` reached first — the reply is **truncated**.
* ``"structural_token"``: the model produced another structural token
  (``<|user|>``, ``<|assistant|>``, ``<|system|>``, ``<|endoftext|>``) — a format error;
  generation stops there and the token is not shown.
* ``"context_full"``: prompt + reply reached the 512-token window.

Optional repetition penalty (Keskar et al., 2019, CTRL): before sampling, the logit of
every token already present in *this reply* is divided by ``penalty`` if positive and
multiplied by it if negative. Structural tokens are never penalized, so the penalty
cannot discourage ``<|end|>``. ``penalty = 1.0`` disables it.

Optional nucleus (top-p) sampling (M6.1, ``apply_top_p``), applied after the penalty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch

from mini_chatgpt.chat.format import ChatTokens, Message, build_prompt, display_text
from mini_chatgpt.inference.generate import sample_next
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.seed import make_generator

StopReason = Literal["end_of_turn", "max_tokens", "structural_token", "context_full"]


@dataclass
class ChatReply:
    text: str  # display text, structural tokens removed
    ids: list[int]  # generated ids, excluding the stop token
    stop_reason: StopReason
    prompt_tokens: int
    dropped_history_messages: int = 0
    stop_token: int | None = None
    settings: dict[str, float | int | bool | None] = field(default_factory=dict)

    @property
    def ended_naturally(self) -> bool:
        return self.stop_reason == "end_of_turn"

    @property
    def truncated(self) -> bool:
        return self.stop_reason in ("max_tokens", "context_full")


def apply_repetition_penalty(
    logits: torch.Tensor, previous: Sequence[int], penalty: float, exempt: frozenset[int]
) -> torch.Tensor:
    """logits: (V,). Returns a penalized copy (CTRL rule) for tokens in ``previous``."""
    if penalty == 1.0 or not previous:
        return logits
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    ids = sorted(set(previous) - exempt)
    if not ids:
        return logits
    out = logits.clone()
    idx = torch.tensor(ids, device=logits.device)
    vals = out[idx]
    out[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return out


def apply_top_p(logits: torch.Tensor, top_p: float | None, temperature: float) -> torch.Tensor:
    """Nucleus filter (Holtzman et al., 2020). logits: (V,). Keeps the smallest set of
    tokens whose probability under ``softmax(logits / temperature)`` sums to at least
    ``top_p`` (always at least the most likely token); the rest get ``-inf``. Applied after
    the repetition penalty and before temperature/top-k sampling, so the kept set is the
    one the sampler would draw from."""
    if top_p is None or top_p >= 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError("top_p must be in (0, 1]")
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_p, order = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    drop_sorted = (cum - sorted_p) >= top_p  # mass before this token already reaches top_p
    drop = torch.zeros_like(drop_sorted)
    drop[order] = drop_sorted
    return logits.masked_fill(drop, float("-inf"))


@torch.no_grad()
def chat_reply(
    model: GPT,
    tok: ByteBPETokenizer,
    history: Sequence[Message],
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int | None = 40,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    top_p: float | None = None,
    seed: int | None = None,
) -> ChatReply:
    ct = ChatTokens.from_tokenizer(tok)
    ctx = model.config.context_length
    # leave room for at least min(max_new_tokens, ctx // 2) new tokens
    room = min(max_new_tokens, ctx // 2)
    prompt, dropped = build_prompt(tok, history, max_prompt_tokens=ctx - room)
    device = next(model.parameters()).device
    gen = make_generator(seed) if seed is not None else None
    was = model.training
    model.eval()
    ids = list(prompt)
    new: list[int] = []
    reason: StopReason = "max_tokens"
    stop_token: int | None = None
    try:
        for _ in range(max_new_tokens):
            if len(ids) >= ctx:
                reason = "context_full"
                break
            logits, _ = model(torch.tensor([ids], device=device))
            last = apply_repetition_penalty(logits[0, -1], new, repetition_penalty, ct.structural)
            if not greedy:
                last = apply_top_p(last, top_p, temperature)
            nxt = int(
                sample_next(
                    last[None], temperature=temperature, top_k=top_k, greedy=greedy, generator=gen
                ).item()
            )
            if nxt == ct.end:
                reason, stop_token = "end_of_turn", nxt
                break
            if nxt in ct.structural:
                reason, stop_token = "structural_token", nxt
                break
            ids.append(nxt)
            new.append(nxt)
    finally:
        model.train(was)
    return ChatReply(
        text=display_text(tok, new),
        ids=new,
        stop_reason=reason,
        prompt_tokens=len(prompt),
        dropped_history_messages=dropped,
        stop_token=stop_token,
        settings={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "greedy": greedy,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "seed": seed,
        },
    )

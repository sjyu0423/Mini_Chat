"""Autoregressive sampling.

At each step the context is cropped to the last ``context_length`` tokens,
the model produces logits for the final position, and the next token is
chosen by greedy argmax or by sampling from softmax(logits / temperature),
optionally restricted to the top-k logits.

Sampling draws from a CPU ``torch.Generator`` so outputs for a given seed do
not depend on the device's RNG.
"""

from __future__ import annotations

import torch

from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.base import Tokenizer
from mini_chatgpt.training.seed import make_generator


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: (B, V) -> next ids (B, 1) on the logits' device."""
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy=True for argmax decoding)")
    logits = logits.float() / temperature
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1).cpu()
    ids = torch.multinomial(probs, num_samples=1, generator=generator)
    return ids.to(logits.device)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """Extend ``idx`` (B, T0) by ``max_new_tokens``; returns (B, T0 + max_new_tokens)."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if idx.ndim != 2 or idx.size(1) == 0:
        raise ValueError("idx must be (B, T) with T >= 1")
    generator = make_generator(seed) if seed is not None else None
    ctx = model.config.context_length
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            window = idx[:, -ctx:]  # crop to the model's context window
            logits, _ = model(window)
            nxt = sample_next(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                greedy=greedy,
                generator=generator,
            )
            idx = torch.cat([idx, nxt], dim=1)
    finally:
        model.train(was_training)
    return idx


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    seed: int | None = None,
) -> str:
    ids = tokenizer.encode(prompt) if prompt else [0]
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model, idx, max_new_tokens, temperature=temperature, top_k=top_k, greedy=greedy, seed=seed
    )
    return tokenizer.decode(out[0].tolist()) if prompt else tokenizer.decode(out[0, 1:].tolist())

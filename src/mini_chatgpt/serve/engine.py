"""Model engine for the local chat demo (M6).

Loads one checkpoint and its tokenizer once, keeps the model in eval mode, and runs one
generation at a time behind a lock: on a 16 GB Mac, concurrent generations would only
compete for the same GPU and multiply memory use, so requests queue instead.

Generation itself is the M5 implementation (``chat.generate.chat_reply``): same prompt
serialization, end-of-turn stopping, repetition penalty and history trimming.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.chat.format import Message
from mini_chatgpt.chat.generate import chat_reply
from mini_chatgpt.device import select_device
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.trainer import load_model


@dataclass(frozen=True)
class GenerationDefaults:
    """M5 recommended settings (97.5% natural end-of-turn on chat_v1)."""

    temperature: float = 0.7
    top_k: int = 40
    repetition_penalty: float = 1.2
    max_new_tokens: int = 200


@dataclass(frozen=True)
class Limits:
    """Hard request limits enforced by the API."""

    max_message_chars: int = 2000
    max_history_messages: int = 40
    max_total_chars: int = 16000
    min_new_tokens: int = 1
    max_new_tokens: int = 256
    min_temperature: float = 0.1
    max_temperature: float = 1.5
    queue_timeout_s: float = 60.0


@dataclass
class GenerationResult:
    reply: str
    stop_reason: str
    new_tokens: int
    prompt_tokens: int
    dropped_history_messages: int
    duration_ms: float
    tokens_per_s: float
    seed: int
    settings: dict[str, Any] = field(default_factory=dict)


class EngineBusyError(RuntimeError):
    """Raised when a request waited longer than the queue timeout."""


class ChatEngine:
    def __init__(
        self,
        model: GPT,
        tokenizer: ByteBPETokenizer,
        *,
        info: dict[str, Any] | None = None,
        defaults: GenerationDefaults | None = None,
        limits: Limits | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.defaults = defaults or GenerationDefaults()
        self.limits = limits or Limits()
        self.info = info or {}
        self.device = next(model.parameters()).device
        self._lock = threading.Lock()
        self.requests_served = 0

    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        expected_fingerprint: str | None = None,
        device: str = "auto",
    ) -> ChatEngine:
        """Load weights + tokenizer and verify the tokenizer fingerprint.

        Defaults: ``<run>/tokenizer.json`` and the fingerprint recorded in
        ``<run>/sft_config.json`` (written by ``scripts/train_sft.py``).
        """
        t0 = time.perf_counter()
        checkpoint = Path(checkpoint)
        run_dir = checkpoint.resolve().parent.parent
        tok_path = Path(tokenizer_path) if tokenizer_path else run_dir / "tokenizer.json"
        if expected_fingerprint is None and (run_dir / "sft_config.json").exists():
            sft = json.loads((run_dir / "sft_config.json").read_text())
            expected_fingerprint = sft.get("expected_fingerprint") or None
        tokenizer = ByteBPETokenizer.load(tok_path)
        if expected_fingerprint and tokenizer.fingerprint() != expected_fingerprint:
            raise RuntimeError(
                f"tokenizer {tok_path} has fingerprint {tokenizer.fingerprint()}, "
                f"expected {expected_fingerprint}"
            )
        dev = select_device(device)
        model, cfg = load_model(checkpoint, dev)
        if tokenizer.vocab_size != cfg.model.vocab_size:
            raise RuntimeError("tokenizer vocabulary does not match the checkpoint")
        if dev.type == "mps":
            torch.mps.synchronize()
        info = {
            "checkpoint": str(checkpoint),
            "name": cfg.name,
            "parameters": sum(p.numel() for p in model.parameters()),
            "context_length": cfg.model.context_length,
            "n_layer": cfg.model.n_layer,
            "n_head": cfg.model.n_head,
            "n_embd": cfg.model.n_embd,
            "vocab_size": cfg.model.vocab_size,
            "tokenizer_fingerprint": tokenizer.fingerprint(),
            "device": str(dev),
            "load_seconds": round(time.perf_counter() - t0, 3),
        }
        return cls(model, tokenizer, info=info)

    # ------------------------------------------------------------------ #
    def memory(self) -> dict[str, int]:
        if self.device.type != "mps":
            return {}
        return {
            "mps_current_allocated": int(torch.mps.current_allocated_memory()),
            "mps_driver_allocated": int(torch.mps.driver_allocated_memory()),
        }

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """One assistant reply to ``messages`` (ends with the user's message)."""
        d = self.defaults
        settings = {
            "temperature": d.temperature if temperature is None else temperature,
            "top_k": d.top_k,
            "repetition_penalty": d.repetition_penalty,
            "max_new_tokens": d.max_new_tokens if max_new_tokens is None else max_new_tokens,
        }
        used_seed = secrets.randbelow(2**31) if seed is None else seed
        if not self._lock.acquire(timeout=self.limits.queue_timeout_s):
            raise EngineBusyError("the model is busy; try again shortly")
        try:
            t0 = time.perf_counter()
            r = chat_reply(
                self.model,
                self.tokenizer,
                messages,
                max_new_tokens=int(settings["max_new_tokens"]),
                temperature=float(settings["temperature"]),
                top_k=int(settings["top_k"]),
                repetition_penalty=float(settings["repetition_penalty"]),
                seed=used_seed,
            )
            if self.device.type == "mps":
                torch.mps.synchronize()
            dt = time.perf_counter() - t0
            if self.device.type == "mps":
                # Each new sequence length caches MPS buffers (see M5 report); release
                # them so memory stays flat across many requests.
                torch.mps.empty_cache()
            self.requests_served += 1
        finally:
            self._lock.release()
        return GenerationResult(
            reply=r.text,
            stop_reason=r.stop_reason,
            new_tokens=len(r.ids),
            prompt_tokens=r.prompt_tokens,
            dropped_history_messages=r.dropped_history_messages,
            duration_ms=round(dt * 1000, 1),
            tokens_per_s=round(len(r.ids) / dt, 1) if dt > 0 else 0.0,
            seed=used_seed,
            settings=settings,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.info,
            "defaults": asdict(self.defaults),
            "limits": asdict(self.limits),
            "requests_served": self.requests_served,
            "memory": self.memory(),
        }

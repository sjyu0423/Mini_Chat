"""GPT-2-style decoder-only Transformer, written from PyTorch primitives.

Notation: B = batch, T = sequence length (<= context_length), C = n_embd,
H = n_head, D = C / H (head dim), V = vocab_size.

    x0      = E_tok[ids] + E_pos[0..T-1]                       (B, T, C)
    for each block l:
        x  = x + Attn_l(LN_1(x))
        x  = x + MLP_l(LN_2(x))
    logits  = LN_f(x) @ E_tok^T            (tied LM head)      (B, T, V)

Causal self-attention per head h:

    Q, K, V = x W_q, x W_k, x W_v                              (B, H, T, D)
    S       = Q K^T / sqrt(D)                                  (B, H, T, T)
    S[i, j] = -inf  for j > i      (token i may only attend to positions <= i)
    A       = softmax(S, dim=-1)
    Y       = A V                                              (B, H, T, D)

Heads are concatenated back to (B, T, C) and projected by W_o.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from torch import nn

from mini_chatgpt.config import AttentionBackend, ModelConfig

INIT_STD = 0.02


def explicit_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference attention. q, k, v: (B, H, T, D); mask: (T, T) bool, True = keep.

    Returns (output (B, H, T, D), attention weights (B, H, T, T)).
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout
        self.backend: AttentionBackend = config.attention
        # Fused Q, K, V projection: C -> 3C.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.is_residual_projection = True  # type: ignore[assignment]
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Lower-triangular causal mask; a buffer (not a parameter, not saved).
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)  # 3 x (B, T, C)
        # (B, T, C) -> (B, H, T, D)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        if self.backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True
            )
        else:
            mask = self.causal_mask[:t, :t]  # type: ignore[index]
            y, _ = explicit_causal_attention(q, k, v, mask, self.attn_dropout)

        y = y.transpose(1, 2).contiguous().view(b, t, c)  # concatenate heads
        out: torch.Tensor = self.resid_dropout(self.c_proj(y))
        return out


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.c_proj.is_residual_projection = True  # type: ignore[assignment]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.dropout(self.c_proj(F.gelu(self.c_fc(x))))
        return out


class Block(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        out: torch.Tensor = x + self.mlp(self.ln_2(x))
        return out


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.context_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight  # one shared (V, C) tensor
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """GPT-2 initialization.

        * Linear / Embedding weights ~ N(0, 0.02^2); biases = 0; LayerNorm = (1, 0).
        * Residual output projections (attn.c_proj, mlp.c_proj) use
          std = 0.02 / sqrt(2 * n_layer): each block adds two residual branches,
          so this keeps the residual-stream variance roughly constant with depth.
        * With tied embeddings the shared tensor is initialized once (as wte).
        """
        resid_std = INIT_STD / math.sqrt(2 * self.config.n_layer)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.lm_head and self.config.tie_embeddings:
                    continue
                std = resid_std if getattr(module, "is_residual_projection", False) else INIT_STD
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def num_parameters(self, *, non_embedding: bool = False) -> int:
        """Unique parameters (tied weights counted once)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def set_attention_backend(self, backend: AttentionBackend) -> None:
        if backend not in ("sdpa", "explicit"):
            raise ValueError(f"unknown attention backend {backend!r}")
        for block in self.blocks:
            block.attn.backend = backend  # type: ignore[union-attr]

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx, targets: (B, T) int64 with T <= context_length.

        Returns logits (B, T, V) and, if targets are given, the mean next-token
        cross-entropy  L = -(1 / BT) * sum_{b,t} log softmax(logits[b, t])[targets[b, t]].
        """
        if idx.ndim != 2:
            raise ValueError(f"idx must be (B, T), got shape {tuple(idx.shape)}")
        _, t = idx.shape
        if t > self.config.context_length:
            raise ValueError(
                f"sequence length {t} exceeds context_length {self.config.context_length}; "
                "crop the input (generation does this automatically)"
            )
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T, C) + (T, C) broadcast
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

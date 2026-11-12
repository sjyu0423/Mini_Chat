"""GPT-style decoder-only Transformer (see ``gpt.py`` for equations and shapes)."""

from mini_chatgpt.model.gpt import GPT, explicit_causal_attention

__all__ = ["GPT", "explicit_causal_attention"]

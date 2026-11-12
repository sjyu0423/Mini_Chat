import pytest
import torch

from mini_chatgpt.inference import generate, generate_text, sample_next
from mini_chatgpt.model import GPT
from mini_chatgpt.tokenizer import CharTokenizer
from tests.conftest import tiny_model_config


def model(**kw):
    torch.manual_seed(0)
    return GPT(tiny_model_config(**kw))


def test_greedy_matches_argmax():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.0]])
    assert sample_next(logits, greedy=True).tolist() == [[1], [0]]


def test_top_k_restricts_support():
    logits = torch.tensor([[5.0, 4.0, 3.0, -1.0, -2.0]])
    g = torch.Generator().manual_seed(0)
    draws = {sample_next(logits, top_k=2, generator=g).item() for _ in range(200)}
    assert draws <= {0, 1} and len(draws) == 2


def test_low_temperature_approaches_greedy():
    logits = torch.tensor([[1.0, 1.5, 0.0]])
    g = torch.Generator().manual_seed(0)
    assert all(sample_next(logits, temperature=1e-3, generator=g).item() == 1 for _ in range(50))


def test_invalid_sampling_args():
    with pytest.raises(ValueError):
        sample_next(torch.zeros(1, 3), temperature=0.0)
    with pytest.raises(ValueError):
        sample_next(torch.zeros(1, 3), top_k=0)


def test_generate_length_and_prompt_preserved():
    m = model()
    idx = torch.tensor([[1, 2, 3]])
    out = generate(m, idx, 10, seed=0)
    assert out.shape == (1, 13)
    assert out[0, :3].tolist() == [1, 2, 3]
    assert int(out.max()) < 17


def test_generate_beyond_context_crops():
    m = model()  # context_length 16
    out = generate(m, torch.zeros(2, 10, dtype=torch.long), 40, top_k=5, seed=1)
    assert out.shape == (2, 50)


def test_generate_seed_reproducible_and_restores_mode():
    m = model(dropout=0.2).train()
    idx = torch.tensor([[1]])
    a = generate(m, idx, 20, seed=3)
    b = generate(m, idx, 20, seed=3)
    c = generate(m, idx, 20, seed=4)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert m.training  # restored after generation


def test_greedy_generation_is_deterministic():
    m = model()
    idx = torch.tensor([[4, 5]])
    assert torch.equal(generate(m, idx, 15, greedy=True), generate(m, idx, 15, greedy=True))


def test_generate_text():
    tok = CharTokenizer.from_text("abcdefghijklmnopq")  # 17 chars = vocab_size
    text = generate_text(model(), tok, "abc", 12, seed=0)
    assert text.startswith("abc") and len(text) == 15

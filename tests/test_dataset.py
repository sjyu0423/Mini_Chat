import pytest
import torch

from mini_chatgpt.data.dataset import get_batch, iter_sequential_batches, split_tokens
from mini_chatgpt.training.seed import make_generator


def stream(n=1000):
    return torch.arange(n, dtype=torch.long)


# ---- splitting ----------------------------------------------------------- #


def test_split_is_contiguous_disjoint_and_complete():
    tokens = stream(1000)
    train, val = split_tokens(tokens, 0.1)
    assert train.numel() == 900 and val.numel() == 100
    assert torch.equal(torch.cat([train, val]), tokens)
    assert train[-1] + 1 == val[0]


def test_split_is_deterministic():
    a = split_tokens(stream(), 0.2)
    b = split_tokens(stream(), 0.2)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_bad_fraction(frac):
    with pytest.raises(ValueError):
        split_tokens(stream(), frac)


def test_split_rejects_empty_result():
    with pytest.raises(ValueError):
        split_tokens(stream(1), 0.5)


# ---- random batches ------------------------------------------------------ #


def test_batch_shapes_and_dtype():
    x, y = get_batch(stream(), 4, 16, generator=make_generator(0))
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == y.dtype == torch.long


def test_targets_are_next_tokens():
    tokens = torch.randint(0, 50, (500,), generator=make_generator(1))
    x, y = get_batch(tokens, 8, 32, generator=make_generator(2))
    assert torch.equal(x[:, 1:], y[:, :-1])
    # Every (x[t], y[t]) pair must be adjacent in the original stream.
    ar = stream(500)
    xi, yi = get_batch(ar, 8, 32, generator=make_generator(2))
    assert torch.equal(yi, xi + 1)


def test_windows_stay_in_bounds():
    tokens = stream(20)
    for seed in range(50):
        x, y = get_batch(tokens, 4, 19, generator=make_generator(seed))
        assert int(y.max()) <= 19 and int(x.min()) >= 0


def test_same_seed_same_batch_different_seed_differs():
    t = stream(10_000)
    a = get_batch(t, 8, 32, generator=make_generator(7))
    b = get_batch(t, 8, 32, generator=make_generator(7))
    c = get_batch(t, 8, 32, generator=make_generator(8))
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], c[0])


def test_stream_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        get_batch(stream(16), 1, 16)


# ---- sequential eval batches -------------------------------------------- #


def test_sequential_batches_cover_each_target_once():
    tokens = stream(101)
    ys = []
    for x, y in iter_sequential_batches(tokens, 3, 10):
        assert torch.equal(y, x + 1)
        ys.append(y.flatten())
    covered = torch.cat(ys)
    assert torch.equal(covered, torch.arange(1, 101))

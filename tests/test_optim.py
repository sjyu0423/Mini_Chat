import itertools
import math

import pytest

from mini_chatgpt.config import TrainingConfig
from mini_chatgpt.model import GPT
from mini_chatgpt.training.optim import build_optimizer, learning_rate, parameter_groups
from tests.conftest import tiny_model_config


def cfg(**kw):
    base = dict(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10, max_steps=110)
    base.update(kw)
    return TrainingConfig(**base)


def test_warmup_is_linear():
    c = cfg()
    assert learning_rate(0, c) == pytest.approx(1e-4)
    assert learning_rate(4, c) == pytest.approx(5e-4)
    assert learning_rate(9, c) == pytest.approx(1e-3)


def test_cosine_decay_endpoints_and_midpoint():
    c = cfg()
    assert learning_rate(10, c) == pytest.approx(1e-3)
    assert learning_rate(60, c) == pytest.approx(1e-4 + 0.5 * 9e-4)
    assert learning_rate(110, c) == pytest.approx(1e-4)
    assert learning_rate(10_000, c) == pytest.approx(1e-4)


def test_schedule_is_monotone_after_warmup():
    c = cfg()
    lrs = [learning_rate(s, c) for s in range(10, 111)]
    assert all(a >= b for a, b in itertools.pairwise(lrs))
    assert all(math.isfinite(x) for x in lrs)


def test_no_warmup():
    assert learning_rate(0, cfg(warmup_steps=0)) == pytest.approx(1e-3)


def test_parameter_groups_split_and_cover_everything_once():
    m = GPT(tiny_model_config())
    decay, no_decay = parameter_groups(m, 0.1)
    d, nd = decay["params"], no_decay["params"]
    assert all(p.dim() >= 2 for p in d) and all(p.dim() < 2 for p in nd)
    assert len({id(p) for p in d + nd}) == len(list(m.parameters()))
    assert sum(p.numel() for p in d + nd) == m.num_parameters()
    opt = build_optimizer(m, cfg())
    assert opt.param_groups[1]["weight_decay"] == 0.0

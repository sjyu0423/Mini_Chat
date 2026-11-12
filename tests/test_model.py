import math
from pathlib import Path

import pytest
import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.model import GPT, explicit_causal_attention
from tests.conftest import requires_mps, tiny_model_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Numerical tolerances for explicit vs SDPA attention (documented in docs/architecture.md).
CPU_ATOL, CPU_RTOL = 1e-5, 1e-4
MPS_ATOL, MPS_RTOL = 1e-4, 1e-3


def make(seed=0, **kw):
    torch.manual_seed(seed)
    return GPT(tiny_model_config(**kw))


def ids(b=3, t=16, v=17, seed=1):
    return torch.randint(0, v, (b, t), generator=torch.Generator().manual_seed(seed))


# ---- shapes / counts ---------------------------------------------------- #


def test_output_shapes_and_loss():
    m = make()
    x = ids()
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 17)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    logits, loss = m(x)
    assert loss is None


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.toml")), ids=lambda p: p.stem)
def test_parameter_count_matches_analytic_formula(path):
    cfg = ExperimentConfig.from_toml(path).model
    with torch.device("meta"):  # no memory allocated, even for gpt_100m
        m = GPT(cfg)
    est = cfg.estimate_parameters()
    assert m.num_parameters() == est["total"]
    assert m.num_parameters(non_embedding=True) == est["non_embedding"]


def test_gpt_10m_parameter_count():
    cfg = ExperimentConfig.from_toml(CONFIG_DIR / "gpt_10m.toml").model
    with torch.device("meta"):
        assert GPT(cfg).num_parameters() == 10_770_816


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("tie", [True, False])
def test_parameter_count_variants(bias, tie):
    cfg = tiny_model_config(bias=bias, tie_embeddings=tie)
    assert GPT(cfg).num_parameters() == cfg.estimate_parameters()["total"]


def test_weight_tying():
    m = make()
    assert m.lm_head.weight is m.wte.weight
    assert m.lm_head.weight.data_ptr() == m.wte.weight.data_ptr()
    untied = make(tie_embeddings=False)
    assert untied.lm_head.weight is not untied.wte.weight
    # Tied weights survive a state_dict round trip.
    m2 = GPT(tiny_model_config())
    m2.load_state_dict(m.state_dict())
    assert m2.lm_head.weight is m2.wte.weight
    assert torch.equal(m2.wte.weight, m.wte.weight)


def test_initialization_statistics():
    torch.manual_seed(0)
    cfg = tiny_model_config(n_embd=256, n_layer=4, n_head=4)
    m = GPT(cfg)
    assert m.wte.weight.std().item() == pytest.approx(0.02, rel=0.05)
    resid = 0.02 / math.sqrt(2 * cfg.n_layer)
    assert m.blocks[0].mlp.c_proj.weight.std().item() == pytest.approx(resid, rel=0.05)
    assert m.blocks[0].attn.c_proj.weight.std().item() == pytest.approx(resid, rel=0.05)
    assert m.blocks[0].mlp.c_fc.weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert torch.all(m.blocks[0].attn.c_attn.bias == 0)
    assert torch.all(m.ln_f.weight == 1) and torch.all(m.ln_f.bias == 0)


def test_deterministic_initialization():
    a, b, c = make(seed=5), make(seed=5), make(seed=6)
    for (ka, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(va, vb), ka
    assert not torch.equal(a.wte.weight, c.wte.weight)


def test_initial_loss_is_near_uniform():
    # Not a strict requirement, but small-std init should start close to ln(V).
    m = make().eval()
    _, loss = m(ids(), ids(seed=2))
    assert abs(loss.item() - math.log(17)) < 0.5


# ---- causality ------------------------------------------------------------ #


def test_explicit_attention_weights_are_causal_and_normalized():
    q, k, v = (torch.randn(2, 3, 8, 4) for _ in range(3))
    mask = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    y, w = explicit_causal_attention(q, k, v, mask)
    assert y.shape == (2, 3, 8, 4)
    assert torch.all(w.triu(diagonal=1) == 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 8))
    # First position can only attend to itself.
    assert torch.allclose(y[:, :, 0], v[:, :, 0])


@pytest.mark.parametrize("backend", ["explicit", "sdpa"])
def test_future_tokens_do_not_affect_past_logits(backend):
    m = make(attention=backend, dropout=0.1).eval()
    x = ids(b=2)
    for cut in (0, 5, 15):
        x2 = x.clone()
        x2[:, cut + 1 :] = (x2[:, cut + 1 :] + 3) % 17  # change only the future
        l1, _ = m(x)
        l2, _ = m(x2)
        assert torch.allclose(l1[:, : cut + 1], l2[:, : cut + 1], atol=1e-6)
        if cut < 15:
            assert not torch.allclose(l1[:, cut + 1 :], l2[:, cut + 1 :])


# ---- backend agreement ------------------------------------------------------ #


def _compare_backends(device, atol, rtol):
    m = make(seed=3).to(device)
    x, y = ids().to(device), ids(seed=4).to(device)
    grads, outs = {}, {}
    for backend in ("explicit", "sdpa"):
        m.zero_grad(set_to_none=True)
        m.set_attention_backend(backend)
        logits, loss = m(x, y)
        loss.backward()
        outs[backend] = logits.detach().cpu()
        grads[backend] = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters()}
    torch.testing.assert_close(outs["explicit"], outs["sdpa"], atol=atol, rtol=rtol)
    for name in grads["explicit"]:
        torch.testing.assert_close(
            grads["explicit"][name], grads["sdpa"][name], atol=atol, rtol=rtol, msg=name
        )


def test_attention_backends_agree_cpu():
    _compare_backends(torch.device("cpu"), CPU_ATOL, CPU_RTOL)


@requires_mps
def test_attention_backends_agree_mps():
    _compare_backends(torch.device("mps"), MPS_ATOL, MPS_RTOL)


def test_state_dict_is_backend_independent():
    a = make(attention="explicit")
    b = make(attention="sdpa")
    assert a.state_dict().keys() == b.state_dict().keys()


# ---- gradients / numerics ---------------------------------------------------- #


def test_gradients_finite_and_reach_every_parameter():
    m = make(seed=7)
    _, loss = m(ids(), ids(seed=3))
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    # Explicitly check the major components named in the M1 spec.
    names = {n for n, _ in m.named_parameters()}
    for required in ("wte.weight", "wpe.weight", "blocks.0.attn.c_attn.weight",
                     "blocks.1.mlp.c_fc.weight", "ln_f.weight"):  # fmt: skip
        assert required in names


# ---- context boundaries --------------------------------------------------------- #


def test_context_length_boundary():
    m = make()
    m(ids(t=16))  # exactly context_length is fine
    m(ids(t=1))
    with pytest.raises(ValueError, match="exceeds context_length"):
        m(ids(t=17))
    with pytest.raises(ValueError):
        m(torch.zeros(16, dtype=torch.long))


@requires_mps
def test_mps_forward_backward_smoke():
    m = make().to("mps")
    x = ids().to("mps")
    logits, loss = m(x, x)
    loss.backward()
    assert logits.device.type == "mps" and torch.isfinite(loss).item()
    cpu = make()
    torch.testing.assert_close(m(x)[0].cpu(), cpu(x.cpu())[0], atol=MPS_ATOL, rtol=MPS_RTOL)

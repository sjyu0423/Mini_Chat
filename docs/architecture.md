# Architecture

Implementation: [`src/mini_chatgpt/model/gpt.py`](../src/mini_chatgpt/model/gpt.py).
Only PyTorch primitives are used (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
`F.gelu`, `F.softmax`, `F.scaled_dot_product_attention`). No pretrained weights or
model libraries.

## 1. Model

Notation: B batch, T sequence length (≤ context length T_max), C = `n_embd`,
H = `n_head`, D = C/H, V = `vocab_size`, L = `n_layer`.

```
ids (B,T) ──► wte[ids] (B,T,C) + wpe[0..T-1] (T,C) ──► dropout
          ──► L × Block ──► LayerNorm_f ──► lm_head (C→V, weight = wteᵀ) ──► logits (B,T,V)

Block (pre-LayerNorm):
    x = x + Dropout(W_o · Attn(LN₁(x)))
    x = x + Dropout(W₂ · GELU(W₁ · LN₂(x)))          W₁: C→4C, W₂: 4C→C
```

### Causal self-attention

Fused projection `c_attn: C → 3C` gives Q, K, V, reshaped to (B, H, T, D):

```
S = Q Kᵀ / √D                         (B,H,T,T)
S[i,j] = −∞   for j > i               causal mask
A = softmax(S, dim=−1)                 rows sum to 1, upper triangle is 0
Y = Dropout(A) V                       (B,H,T,D) → transpose/concat → (B,T,C) → c_proj
```

Two interchangeable backends (`ModelConfig.attention`, or `model.set_attention_backend`).
Both use the same parameters and state_dict:

| backend | implementation | role |
|---|---|---|
| `explicit` | `explicit_causal_attention`: matmul, scale, `masked_fill` with a `tril` bool buffer (non-persistent), softmax, dropout, matmul; also returns A | reference, debugging, inspecting attention weights |
| `sdpa` (default) | `F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p)` | training |

Agreement (dropout = 0, tested in `tests/test_model.py` for logits **and all
parameter gradients**): CPU `atol=1e-5, rtol=1e-4`, MPS `atol=1e-4, rtol=1e-3`.
With dropout > 0 the backends draw different masks, so they agree only in distribution.

On MPS, SDPA with `dropout_p > 0` raises `NotImplementedError` when autograd is
off (e.g. a train-mode forward under `torch.no_grad()`). With autograd on (training),
it works. Evaluation and generation call `model.eval()`, which sets `dropout_p = 0`,
so this never triggers in our code paths.

### Loss

For logits z ∈ ℝ^{B×T×V} and targets y = ids shifted left by one:

```
L = −(1/BT) Σ_{b,t} log softmax(z[b,t])[y[b,t]]       (nats/token)
perplexity = exp(L),   bits/char = L / ln 2
```

A uniform predictor scores ln V = ln 65 ≈ 4.174. This is a reference point, not a
target for the initial loss. Measured: 4.12 to 4.21 at initialization.

### Initialization (GPT-2)

- Linear and Embedding weights: N(0, 0.02²). Biases: 0. LayerNorm: γ = 1, β = 0.
- Residual output projections (`attn.c_proj`, `mlp.c_proj`): std 0.02/√(2L).
  Each block adds two residual branches, so this keeps the variance of the
  residual stream roughly independent of depth.
- Tied embeddings: `lm_head.weight` *is* `wte.weight` (same tensor). It is
  initialized once, as an embedding, and counted and weight-decayed once.
- Initialization runs on the CPU RNG, then the model moves to the device, so a
  given seed produces identical weights on CPU and MPS.

## 2. Parameter counts (actual = analytic)

Per block: `12C² + 13C` with bias, `12C² + 2C` without. Embeddings: `(V + T_max)·C`.
Final LayerNorm: `2C` (`C` without bias). The LM head adds nothing when tied.
`tests/test_model.py` checks `GPT.num_parameters()` against
`ModelConfig.estimate_parameters()` for every config in `configs/`, using meta
tensors so no memory is allocated.

| config | L / H / C | T_max | bias | **actual params** | non-embedding | if V = 50,257 |
|---|---|---|---|---|---|---|
| `debug` | 2 / 2 / 64 | 64 | yes | 108,352 | 100,096 | 3.3M |
| `gpt_10m` | 6 / 6 / 384 | 256 | yes | **10,770,816** | 10,647,552 | 30.0M |
| `gpt_30m` | 10 / 8 / 512 | 512 | yes | **31,820,288** | 31,524,864 | 57.5M |
| `gpt_100m` | 14 / 12 / 768 | 1024 | no | **99,949,056** | 99,112,704 | 138.5M |

## 3. Code map

```
config.py            typed configs (incl. attention backend); analytic param count
device.py            auto = CUDA → MPS → CPU; explicit unavailable device raises
model/gpt.py         GPT, Block, MLP, CausalSelfAttention, explicit_causal_attention
training/optim.py    AdamW param groups, warmup + cosine LR (pure function of step)
training/trainer.py  Trainer: step, accumulation, clipping, NaN/Inf guard, eval,
                     logging, timing, memory probes, checkpoint save/restore
training/checkpoint.py  atomic writes, paths, metadata sidecars, weights_only loads
inference/generate.py   greedy / temperature / top-k sampling, context cropping
scripts/             train, generate, sanity_checks, prepare_data, show_config, detect_hardware
```

## 4. Data and batching (from M0)

The split is contiguous: the first 90% of tokens train and the last 10% validate.
Each example is `x = s[i:i+T]`, `y = s[i+1:i+T+1]`. Training samples offsets from a
dedicated CPU generator. Exact validation (`Trainer.full_loss`) walks non-overlapping
windows, so each target is scored once. The tail shorter than one window is dropped.

# Experiments

A log of every experiment run, newest first. Each entry links to its run
directory (`runs/<name>-<timestamp>/`, which is git-ignored) and records enough
to reproduce it.

## Conventions

- One config file per experiment family in `configs/`; changes are made by
  editing a copy, never by unlogged command-line overrides.
- Each run directory holds `config.json`, `environment.json` (device, versions,
  git commit), `metrics.jsonl`, and `checkpoints/`.
- Commit the code before a run you intend to keep, so `git_commit` in
  `environment.json` identifies it.
- Report val loss from the full sequential pass over the validation split.
- Change one variable at a time; note the hypothesis before running.

## Entry template

```
### <date> — <short title>
- Hypothesis:
- Config: configs/<file>.toml (+ diff vs baseline)
- Run dir: runs/<name>-<timestamp>
- Commit: <hash>
- Hardware / device / dtype:
- Steps, tokens seen, wall-clock:
- Results: train loss, val loss, val bits/char, tokens/sec, peak memory
- Samples:
- Conclusion / next step:
```

## Log

### 2026-11-12 — M1 smoke benchmark: gpt_10m, 100 steps, MPS

- Command: `scripts/train.py --config configs/gpt_10m.toml --name bench_10m_b8
  --batch-size 8 --max-steps 100 --warmup-steps 10 --eval-interval 50 --eval-steps 20
  --checkpoint-interval 100 --device mps --full-eval`
- Run dir: `runs/bench_10m_b8-20261112-111500`. Code uncommitted (no commit hash yet).
- Hardware: Apple M6, MPS, float32, SDPA attention, dropout 0.2.
- Parameters: 10,770,816 (equal to the analytic count).
- Loss (nats/char): first step 4.142 → last step 2.561. Fixed-batch eval
  train 4.122 / val 4.127 → train 2.497 / val 2.505. Exact full-val loss
  2.5056 (3.61 bits/char).
- Timing (steps 11 to 100): median 66.7 ms, mean 66.9 ms, min/max 65.9/69.4 ms.
  30.7k tokens/s at 2,048 tokens/step.
- Memory: MPS allocated 0.64 GB after forward and 0.20 GB after the step;
  driver 1.19 GB.
- Reproducibility: three identical invocations gave the same losses to ~1e-8.
- Sample (T = 0.8, top-k 40, seed 1337): shows speaker-label structure, line
  breaks, and English-like character statistics, but no words yet:
  `ROMEO:\nTE: an thede.\n\nENI, lrolound wathindof poute t, bee wist ic wis ter; ...`
- Extra 30-step measurements: batch 64 gives 507 ms/step and 32.3k tok/s with
  4.49 GB driver memory. Explicit attention at batch 8 gives 68.1 ms/step and
  30.1k tok/s.
- Conclusion: the engine is stable and learning on MPS. Throughput is
  compute-saturated at batch 8, so M2 can use batch 64 for better gradient
  estimates at almost no throughput cost.

### 2026-11-12 — M1 critical learning checks (`scripts/sanity_checks.py`, MPS)

- **Single-batch overfit** (gpt_10m, 4 × 256 real tokens, AdamW lr 3e-4, no
  dropout): loss 4.210 → 1.107 (step 50) → 0.0155 (step 100) → 0.0058 (step 200),
  a 720× reduction.
  - Finding: at lr 1e-3 with *no warmup*, the loss sits on the unigram plateau
    (~3.23) for over 100 steps on both CPU and MPS before dropping. This is
    optimizer dynamics, not a backend bug. Warmup in the real schedule avoids it.
- **Gradient flow**: every parameter has a finite, non-zero gradient. Gradient
  norms at init: tied embedding/head 2.14, position embedding 0.56, QKV
  0.52 to 0.90, attention output 1.74 to 3.08, MLP fc 0.57 to 0.92,
  MLP proj 2.21 to 4.26, LayerNorms 0.01 to 0.10.
- **Checkpoint reload**: maximum absolute logit difference 0.0.
- **Resume vs uninterrupted** (20 steps, interrupted at 10, dropout 0.2): maximum
  parameter difference 7.6e-5. Two *uninterrupted* MPS runs differ by 6.7e-5, so
  resume adds no error beyond MPS nondeterminism. Final losses 2.91161 vs 2.91160.
  On CPU, resume is bit-exact (pytest).

### 2026-11-12 — M0 data pipeline verification (no training)

- Dataset: Tiny Shakespeare, SHA-256 `86c4e6aa…c565ed`, 1,115,394 bytes (ASCII).
- Char tokenizer: vocab 65; full-corpus encode/decode round trip is exact.
- Split (val_fraction = 0.1): 1,003,854 train / 111,540 val tokens.
  Train hash `6ec30560…ac2626f`, val hash `d37d30cc…e311204b1`; rebuilding
  reproduces both.
- Seeded batches are identical across calls, and targets equal inputs shifted by one.
- Uniform-prediction reference loss: ln 65 = 4.174 nats (6.02 bits/char).

## Planned experiments (subject to approval per milestone)

1. M2 baseline: full `gpt_10m` (5,000 steps, batch 64), about 50 min. Track the
   val-loss minimum and the overfitting onset.
2. Precision: verify bf16 autocast on MPS (agreement and speed) before enabling it.
3. Architecture ablations: pre-LN vs post-LN, learned vs rotary positions,
   GELU vs SwiGLU, LayerNorm vs RMSNorm, with/without bias, tied vs untied embeddings.
4. Optimization: LR sweeps, warmup length, batch size vs LR, weight decay, dropout.
5. Scaling: loss vs non-embedding params and tokens across 1M–100M models on a
   larger corpus with BPE.
6. Chat: supervised fine-tuning on dialogue data, then preference/RL methods.

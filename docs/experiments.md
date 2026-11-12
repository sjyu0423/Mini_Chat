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

### 2026-11-12 — M2 baseline: gpt_10m, 5,000 steps, MPS (full report: [reports/m2_pretraining.md](reports/m2_pretraining.md))

- Hypothesis: the M1 engine trains the full gpt_10m config stably and reaches
  ~1.5 nats/char validation loss. With about 82 epochs over 1M characters, the
  model is expected to overfit before step 5000.
- Config: `configs/gpt_10m_m2.toml`, which is `gpt_10m.toml` plus bookkeeping only
  (checkpoint every 250, keep all, `save_best`, `exact_val_eval`, `device = "mps"`,
  `log_interval = 50`).
- Run dir: `runs/gpt_10m_m2-20261112-174425`. Code: `d49fa80` plus the uncommitted
  M2 working tree.
- Pre-flight: `scripts/stability_check.py --steps 300` passed. The LR schedule
  matched the formula exactly, val fell 4.12 → 1.95, the gradient norm settled at
  about 0.55, and the projected time was 48.6 min.
- Hardware: Apple M6, MPS, float32, SDPA, batch 64 × 256, no accumulation, seed 1337.
- Steps / tokens / wall-clock: 5,000 steps, 81,920,000 tokens (81.6 epochs).
  Total 47.3 min: 43.2 min of steps, ~11.5 s per eval/checkpoint.
- Throughput: median 514 ms/step, 31.9k tok/s. A transient dip to about 28k tok/s
  at steps 2550–2700 coincides with an unrelated `website/` project being
  created in the repo (likely a package install; not verified).
- Memory: MPS allocated after forward 4.05 GB, Metal pool 5.57 GB, process RSS
  0.57 GB. Disk: 2.5 GB.
- Stability: no non-finite values. Gradient norm 10.5 at step 1, then a median of
  about 0.4. 0.12% of post-warmup steps were clipped.
- Results (exact full-split losses, nats/char):

  | | train | val | gap | val ppl | val bpc |
  |---|---|---|---|---|---|
  | init | 4.1230 | 4.1250 | +0.002 | 61.9 | 5.951 |
  | **best, step 1750** | 1.0928 | **1.4736** | +0.381 | 4.36 | 2.126 |
  | final, step 5000 | 0.5862 | 1.7402 | +1.154 | 5.70 | 2.511 |

- Overfitting: exact val is flat over steps 1250–1750, then rises monotonically
  (1.487 at 2000, 1.551 at 3000, 1.740 at 5000) while train loss keeps falling.
- Emergence (temperature 0.7, top-k 40): speaker-label layout appears at step 250,
  and ≥4-letter words are 84% valid at step 500 and 95% at step 1000. There are no
  sustained semantics at any step.
- Memorization: best-model samples have at most 28-char verbatim training matches,
  against 24 chars in held-out text. Final-model samples reach 55 chars. In the
  extraction probe, the matched prefix on training passages is 2.7 (best) vs 5.4
  (final), and 2.0–2.1 on val passages. No passage was reproduced for 64+ chars.
- Conclusion / next step: the infrastructure and training are sound. The schedule is
  too long for this dataset. Proposed next (needs approval): the same config with
  `max_steps = 2000` and eval every 100 steps.

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

1. ~~M2 baseline~~ (done; see the log above). Next: `max_steps = 2000` so the cosine
   anneals before the overfitting onset, eval every 100 steps.
2. Precision: verify bf16 autocast on MPS (agreement and speed) before enabling it.
3. Architecture ablations: pre-LN vs post-LN, learned vs rotary positions,
   GELU vs SwiGLU, LayerNorm vs RMSNorm, with/without bias, tied vs untied embeddings.
4. Optimization: LR sweeps, warmup length, batch size vs LR, weight decay, dropout.
5. Scaling: loss vs non-embedding params and tokens across 1M–100M models on a
   larger corpus with BPE.
6. Chat: supervised fine-tuning on dialogue data, then preference/RL methods.

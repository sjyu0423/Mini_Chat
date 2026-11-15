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

### 2026-11-15 — M6.3 conversational SFT of the 29.7M model vs M5 (report: [reports/m6_3_conversational_scaling.md](reports/m6_3_conversational_scaling.md))

- Hypothesis: the M6.2 29.7M base model, fine-tuned with the unchanged M5 recipe, is a better
  assistant than M5. Pre-registered rules (`reports/m6_3/preregistration.md`, SHA `563dd005…`):
  prompt-level paired bootstrap + sign test, >= 5-point gain, blind review, regression checks.
- Config `configs/sft/m6_3_sft_30m.toml` (only base checkpoint/name differ; attempt 2 also releases
  the MPS cache every step — addendum). Attempt 1 stopped safely at step 2224 (MPS guard 11.0 GB > 10);
  attempt 2 `runs/m6_3_sft_30m-20261115-001347`: 3,153 steps, 17.3 min, best step 2000.
- Results: assistant val 2.932 vs M5 3.085. Strict success 21.3% vs 18.2% on 75 items (+3.1,
  CI [-2.2, +8.9], p = 0.72); factual/instruction/multi-turn ~0, entity +16.7 (n = 10, n.s.).
  Blind review (20): 1 vs 1 correct, M6.3 more fluent (fewer malformed). End-of-turn 97.8% vs 95.2%,
  repetition 2.6% vs 1.9%, latency 160 vs 146 ms, FineWeb-Edu retention 1.400 vs 1.477 bits/byte.
- Conclusion: not promoted; M5 stays the default. Fluency/format improve with scale before knowledge.

### 2026-11-15 — M6.2 capacity scaling: 29.7M vs 14.0M at equal tokens (report: [reports/m6_2_model_scaling.md](reports/m6_2_model_scaling.md))

- Hypothesis: doubling capacity (8L/8H/512, 29,676,544 params) on the identical M4.1 data,
  tokenizer, single-pass order, schedule and 157,286,400 tokens improves the base model.
- Config: `configs/m6_2_general_m_bpe8192.toml` (only n_layer/n_head/n_embd, name differ from M4.1).
  Data and model preflights + single-batch overfit passed.
- Run dir: `runs/m6_2_general_m_bpe8192-20261115-001655`. 205.4 min, 13.2k tok/s, 6.56 GB pool,
  3.00e16 FLOPs (2.09x M4.1), no instabilities; best = final (step 9600).
- Results: val exact 3.4769 -> **3.3122** (1.264 -> 1.204 bits/byte); held-out bpb -4.1% to -6.6%
  on all six sets (incl. Wikipedia/TinyStories OOD). Facts 22 -> 23/40, QA 12 -> 13, code MC 7 -> 9,
  code gen 2 -> 4, grammar 30 -> 29: none significant (paired McNemar). Continuation loops 63% -> 38%.
  No memorization (0 samples with 32-gram train match).
- Conclusion: clear equal-token LM gain at ~2x cost; not a compute-optimality result. M6.3: SFT
  M6.2 with the unchanged M5 recipe and pre-registered comparison against M5.

### 2026-11-14 — M6.1 chat quality: decoding grid + length-capped SFT (report: [reports/m6_1_chat_quality.md](reports/m6_1_chat_quality.md))

- Hypotheses: (a) a different decoding setting improves chat behaviour of the M5 model;
  (b) removing assistant replies over 100 words from SFT (24.9% of replies, 60.8% of target
  tokens) improves conciseness, instruction following and consistency.
- Eval: frozen chat_v1 (40) + new frozen chat_diag_v1 (50, hashed before any output), seeds
  11/22/33, paired McNemar tests; pre-registered decoding selection rule.
- Decoding (M5): default T 0.7 / k 40 / penalty 1.2 remains best eligible (95.2% natural end,
  1.9% repetitive, 41/225 both_ok); top-p 0.9 ties; penalty 1.3 and T 0.5 do not help.
- SFT `configs/sft/m6_1_sft_cap100.toml` (only the data differs from M5; init M4.1):
  run `runs/m6_1_sft_cap100-20261114-185455`, 2,373 steps, 4.6 min. Natural end 99.6% vs 95.2%
  (p = 0.002), median 21 vs 31 words, 2x faster; content not improved on reading; general
  text +0.05 bits/byte vs M5; M5 lower validation loss on both splits.
- Conclusion: keep M5 + default decoding deployed. Remaining failures are knowledge/capacity;
  M6.2 should scale pretraining (30M) and reuse the M5 SFT recipe.

### 2026-11-14 — M5 conversational SFT: OASST2 + Dolly-15k, assistant-only loss, from M4.1 (report: [reports/m5_conversational_training.md](reports/m5_conversational_training.md))

- Hypothesis: a short supervised fine-tune on human-written conversations teaches the M4.1 base
  model the chat format and end-of-turn stopping; it is not expected to add knowledge.
- Data: OASST2 @ `179dd21f` (English best-ranked path per tree, synthetic=false) + Dolly-15k @
  `bdd27f4d`; cleaned, decontaminated against both frozen suites, deduplicated, group-hash split,
  whole-message context policy. 16,813 train / 860 val conversations; 2.01M assistant-target tokens.
- Config: `configs/sft/m5_sft_chat.toml`. M4.1 weights, fresh AdamW, lr 1e-4 (warmup 100, cosine to
  1e-5), batch 16 conversations, 3 epochs = 3,153 steps, clip 1.0; padding to multiples of 64,
  `torch.mps.empty_cache()` every 25 steps.
- Attempt 1 stalled at step ~260 (MPS pool 2.2 → 13.6 GB, swap); kept as
  `runs/m5_sft_chat-20261114-174453_attempt1_stalled`. Fixed run: `runs/m5_sft_chat-20261114-175334`,
  8.0 min, 16.9k real tok/s, pool ≤ 6.5 GB; step-250 val reproduced exactly (3.1590).
- Results: val assistant loss 3.438 → **3.085** (best step 2750; final 3.086). Natural `<|end|>` on
  chat_v1: base 0% → SFT 55% greedy / 77% sampled / **97.5% sampled + penalty 1.2**. Repetitive
  replies (sampled) 28% → 2.5% with penalty. Strict factual QA 0–2 correct (of 15–45). Held-out
  bits/byte +6–16% (forgetting). M4 chat section format 0/12 → 6/12.
- Conclusion: chat structure and stopping learned; knowledge not. Recommended inference defaults:
  T 0.7, top-k 40, repetition penalty 1.2. Ready for an M6 demo interface with honest framing.

### 2026-11-14 — M4.1 general pretraining: FineWeb-Edu 70% + Cosmopedia v2 30%, 14.0M params, single pass (report: [reports/m4_1_general_pretraining.md](reports/m4_1_general_pretraining.md))

- Hypothesis: the M4 plan (6L/6H/384, BPE 8192 with chat tokens, 157M tokens seen once)
  yields a stable general-domain base model for conversational post-training.
- Config: `configs/m4_1_general_s_bpe8192.toml`, 13,989,888 params, new
  `data_order = "single_pass"` (each 512-token window presented exactly once; 0 repeats).
  - Data: pinned FineWeb-Edu shard 013 + Cosmopedia v2 file 0 (25% pool), cleaned,
    deduplicated across sources, hash-split, decontaminated; 157,290,975 train tokens.
  - Tokenizer fit on train only, fingerprint `e6be77cc…`; data and model preflights passed.
- Run dir: `runs/m4_1_general_s_bpe8192-20261114-190514`. 109.9 min, 24.6k tok/s, 4.38 GB
  pool, 1.43e16 FLOPs, no instabilities; every eval a new best.
- Results: val exact 3.4769 nats/token, ppl 32.4, **1.264 bits/byte** (best = final).
  Held-out bpb FineWeb-Edu 1.335, Cosmopedia 0.993, Wikipedia (OOD) 1.660, vs 3.09 / 2.98 /
  3.75 for M3.3; TinyStories 1.169 vs 0.508.
- Suite (acc_norm): facts 22/40 (M3.3 10/40), QA 12/30 (chance-level), grammar 30/30, code
  MC 7/20, greedy code 2/10, chat format 0/12 (base model).
- Failure: sampled text loops (63% of continuations >30% repeated 4-grams vs 0% for M3.3)
  and never emits end-of-text. No 32-gram memorization.
- Conclusion: usable base checkpoint for M5; M5 must teach stopping (`<|end|>`), and
  decoding safeguards should be evaluated separately.

### 2026-11-14 — M3.3 capacity scaling: 27.6M vs 12.4M at equal tokens (report: [reports/m3_3_model_scaling.md](reports/m3_3_model_scaling.md))

- Hypothesis: 2.2× parameters improve loss and entity tracking at equal training
  tokens.
- Config: `configs/tinystories_m_bpe4096.toml`, 8L/8H/512, 27,579,392 params.
  - Same data order as M3.1: 2 × 16 micro-batches, sampler offsets verified
    identical.
  - Same tokenizer, schedule, seed and optimizer.
  - Preflight passed all 8 checks, including exact causality (prefix difference
    0.0).
- Run dir: `runs/ts_m_bpe4096-20261114-042121`. 109.5 min, 13.6k tok/s, 5.62 GB
  pool, no instabilities.
- Results (exact val, nats/token):

  | | val |
  |---|---|
  | M3.1 @4800 | 1.5455 |
  | M3.2 @7200 | 1.5147 |
  | **M3.3 @4800** | **1.4436** |

  M3.3 bits/byte 0.526. M3.3 led M3.1 by about 0.1 at every step.
- FLOPs (estimated): 6.4e15 / 9.6e15 / 1.40e16. At A's compute, M3.3's
  un-annealed checkpoints trail A; it passes B near B's compute.
- Blind rubric: 21 pre-registered prompts × 3 seeds × 3 models, 189 stories, one
  rater.
  - Totals 5.76 / 5.49 / 6.06 out of 10.
  - C − A = +0.30 [−0.32, 0.90], p = 0.31.
  - Entity 1.19 / 1.05 / 1.19.
  - Contradictions 33% / 37% / 30%.
- Conclusion: capacity clearly helps cross-entropy but not detectably entity
  tracking. Equal-compute efficiency is unresolved without compute-matched
  schedules.

### 2026-11-13 — M3.2 extended TinyStories pretraining: continuation at constant LR (report: [reports/m3_2_extended_pretraining.md](reports/m3_2_extended_pretraining.md))

- Hypothesis: more training of the 12.4M M3.1 model lowers validation loss and
  improves narrative and character consistency.
- Setup: phase 2 continued from M3.1 `best.pt` (step 4800, SHA-256 `039f527f…`),
  with a constant LR of 6e-5 for 2,400 steps to step 7200. Config:
  `configs/tinystories_s_bpe4096_m3_2.toml`.
  - Restored exactly: weights, AdamW moments, RNG and sampler state.
  - Preflight passed all 7 checks; recomputed val = 1.5454597872770313, as logged.
- Run dir: `runs/ts_s_bpe4096_m3_2-20261113-190046`. 29.2 min, 26.7k tok/s, 6.75 GB
  pool. Totals: 118.0M presentations = 1.20 epochs.
- Results:
  - val 1.5455 → **1.5147** (0.5633 → 0.5521 bits/byte), improving at every 400-step
    eval;
  - holdout 1.5148;
  - seen-slice gap +0.017 → +0.020;
  - extraction probe flat (2.16 → 2.00).
- Narrative: blind entity-prompt rubric 49.4% (M3.1) vs 45.2% (M3.2), paired 7 vs 5,
  p = 0.77. Verified fact contradictions 2 vs 6 of 12 (p ≈ 0.19). No detectable
  improvement.
- Conclusion: −2.0% loss for +52% compute, with no visible coherence gain. Next
  (needs approval): the 27.6M model on the same data.

### 2026-11-13 — M3.1 TinyStories V2 pretraining, BPE 4,096, 12.4M params (report: [reports/m3_1_tinystories_pretraining.md](reports/m3_1_tinystories_pretraining.md))

- Hypothesis: a 12.4M model trained for about 1 h on unique TinyStories tokens
  learns coherent short stories, without the overfitting seen in M2.
- Data:
  - `roneneldan/TinyStories@f54c09fd`;
  - 8 × 50 MB verified byte ranges of the V2 train file, giving 482,562 stories and
    98,200,365 tokens;
  - the official V2 validation file (SHA-256 verified): 27,630 stories, 5,577,780
    tokens;
  - a 1% hash holdout.

  Deduplication removed 0 stories, exact or near, including any overlap with
  validation. Planted-duplicate recall was 100%.
- Tokenizer: BPE-4096 trained on train stories only (fingerprint `906bf77c…`), at
  3.956 bytes/token. 60 merged tokens are used fewer than 10 times.
- Config: `configs/tinystories_s_bpe4096.toml`.
  - Model: 6L/6H/384, context 512, dropout 0.
  - Optimizer: AdamW (0.9, 0.95), wd 0.1, lr 6e-4 with warmup 200, cosine to
    6e-5 at 4,800 steps, batch 32 × 512.
- Run dir: `runs/ts_s_bpe4096-20261113-172905`. Code: `8d8d7c0` plus the
  uncommitted M3 and M3.1 work.
- Results:
  - Steps, tokens and wall-clock: 4,800 steps, 78.6M presentations (0.80 epoch),
    56.5 min.
  - Throughput and memory: 26.2k tok/s, 4.37 GB allocated, 6.66 GB pool.
  - Validation: best = final (step 4800), val **1.5455** nats/token = 0.5633
    bits/byte (ppl 4.69).
  - Generalization: holdout 1.5453, train slice 1.5286.
  - One clipped gradient spike (step 4450, norm 2.12); no instability.
- Samples: grammatical, self-terminating stories (100% end with EOT); character
  drift and weak causality; subjective rubric 69%.
- Memorization: 16-gram overlap 2.9%, against 2.7% for real validation stories.
  Extraction probe: 2.2 vs 1.6 tokens.
- Next (needs approval): about 2 h on the same data (horizon 9,600), then the
  27.6M model.

### 2026-11-13 — M3 dataset research, BPE tokenizer and MPS microbenchmarks (report: [reports/m3_dataset_research.md](reports/m3_dataset_research.md))

- No model training. Bounded samples were fetched through the HF dataset-viewer API:
  - TinyStories `default/train`: 6,000 stories, 5.3 MB, SHA-256 `49bccd67…`;
  - FineWeb-Edu `sample-10BT`: 4,000 documents, 19.0 MB, SHA-256 `e6cacd68…`.

  Manifests are in `data/raw/samples/*/manifest.json`.
- Tokenizers: byte-level BPE at vocabularies of 2,048, 4,096 and 8,192, fitted on
  hash-split training documents (`runs/m3_tokenizers-20261113-165404`). Validation
  bytes per token:

  | tokenizer | TinyStories | FineWeb-Edu |
  |---|---|---|
  | TinyStories | 3.59 / 3.92 / 4.09 | |
  | FineWeb-Edu | | 2.93 / 3.38 / 3.81 |

- Model microbenchmarks (synthetic data, one subprocess per case):
  - S 12.4M: 26.2k tok/s at 4.4 GB;
  - M 27.6M: 13.7k tok/s;
  - L 88.6M: 4.85k tok/s.

  Throughput is saturated from batch 8. Going from vocabulary 4k to 8k costs −6%,
  and context 512→1024 costs −23%.
- Data loading takes 0.36 ms per 16k-token batch, so computation is the bottleneck.
- Next (needs approval): M3.1, S on about 100M TinyStories-V2 tokens for about 1 h.

### 2026-11-12 — M2.1 LR-schedule ablation: horizon 5000 vs 2000, both stopped at 2000 (report: [reports/m2_1_lr_ablation.md](reports/m2_1_lr_ablation.md))

- Hypothesis: annealing the LR to lr_min by step 2000 (B) beats early stopping on
  M2's 5000-step schedule (A). A should reproduce M2 up to step 2000.
- Configs: `configs/gpt_10m_m2_1_a.toml` (horizon 5000) and
  `configs/gpt_10m_m2_1_b.toml` (horizon 2000). The only A/B difference is
  `lr_schedule_steps`. Both run 2000 steps and evaluate every 100.
- Validity, checked before training (`scripts/verify_ablation.py`): identical
  weights, first batch, and initial exact val (4.124992488992625) for A, B and M2;
  A's LRs equal M2's exactly for steps 0–2000; B reaches 1e-4 at step 2000.
- Run dirs: `runs/gpt_10m_m2_1_a-20261112-185842` and
  `runs/gpt_10m_m2_1_b-20261112-191929`. Code: `8d6143f` plus uncommitted M2.1 changes.
- Hardware: Apple M6, MPS, float32, seed 1337. About 20.6 min wall-clock each,
  32.5–32.9k tok/s, 5.57 GB MPS pool, no instabilities.
- Results (exact val, nats/char):

  | | best [step] | val at 2000 | gap at 2000 |
  |---|---|---|---|
  | M2 | 1.4736 [1750] | 1.4869 | +0.445 |
  | A | 1.4703 [1800] | 1.4865 | +0.445 |
  | B | **1.4638** [1800] | **1.4672** | **+0.405** |

- Reproduction: A − M2 at shared evals is at most 0.0021, and the per-step losses
  match to about 5e-8 for steps 1–100 before drifting.
- Conclusion: early stopping explains about 93% of the final-checkpoint gain over M2.
  The compressed schedule adds −0.019 at the final checkpoint and −0.0065 at the best
  one. That is about 3× the MPS-noise bound, but from a single seed, so it is not
  significant.
- Next: repeat A and B with three or more seeds.

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

1. ~~M2 baseline~~ and ~~M2.1 schedule ablation~~ (done; see the log above). Next:
   a repeated-seed A/B study (three or more seeds each) before adopting a compressed
   schedule.
2. Precision: verify bf16 autocast on MPS (agreement and speed) before enabling it.
3. Architecture ablations: pre-LN vs post-LN, learned vs rotary positions,
   GELU vs SwiGLU, LayerNorm vs RMSNorm, with/without bias, tied vs untied embeddings.
4. Optimization: LR sweeps, warmup length, batch size vs LR, weight decay, dropout.
5. Scaling: loss vs non-embedding params and tokens across 1M–100M models on a
   larger corpus with BPE.
6. Chat: supervised fine-tuning on dialogue data, then preference/RL methods.

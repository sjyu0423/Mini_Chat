# Training plan

## Optimization (implemented in M1)

- **AdamW** (β₁, β₂ from config, ε = 1e-8). Weight decay applies to tensors with
  ≥ 2 dimensions (Linear weights and the tied embedding). It never applies to
  biases or LayerNorm.
- **LR schedule** (`training/optim.py`, a pure function of the step):
  linear warmup `lr_max·(s+1)/W` for s < W, then cosine
  `lr_min + ½(lr_max − lr_min)(1 + cos(π(s−W)/(H−W)))`, then held at lr_min.
  H is the schedule horizon, `lr_schedule_steps` (M2.1). It defaults to `max_steps`
  and is independent of how long training runs:
  - `H > max_steps` stops early on an unchanged trajectory.
  - `H < max_steps` holds lr_min after H.
- **Gradient accumulation**: each optimizer step runs `gradient_accumulation_steps`
  micro-batches, and each loss is scaled by 1/n. Tests check that the gradient
  equals that of a single large batch.
- **Gradient clipping**: global L2 norm, applied to the parameters directly.
- **NaN/Inf guard**: mean loss and gradient norm are checked before
  `optimizer.step()`. A non-finite value raises `NonFiniteError` and the weights
  are left untouched. There is one host sync per step.
- **Precision**: float32 only. The trainer rejects `bfloat16`/`float16` until
  they are explicitly verified on MPS. There is no silent device or precision
  fallback.
- **Evaluation**: `estimate_loss` runs `eval_steps` random batches per split
  from a freshly seeded generator, so every evaluation sees the same batches.
  `full_loss` gives the exact token-weighted loss over the whole split.

## Checkpoint semantics

Each periodic checkpoint is `runs/<run>/checkpoints/step_XXXXXXXX.pt` plus a
`.json` metadata sidecar. Both are written atomically (temp file, fsync,
`os.replace`) and loaded with `torch.load(weights_only=True)`. A checkpoint contains:

| key | restores |
|---|---|
| `model` | weights (the tied tensor is stored once and re-tied on load) |
| `optimizer` | AdamW moments and step counts (moved to the parameters' device on load) |
| `step`, `tokens_seen` | the loop position. The LR schedule is derived from `step`. |
| `best_val_loss`, `best_step` | best-checkpoint bookkeeping (M2) |
| `loss_history`, `eval_history` | every per-step train loss and every eval record so far (M2), so a resumed trainer holds the whole run |
| `data_generator` | the batch-sampling RNG |
| `rng` | torch CPU RNG, MPS/CUDA RNG (dropout), and Python `random` |
| metadata | full `ExperimentConfig`, `kind`, git commit, torch version, losses (incl. exact val), timestamp |

**Checkpoint kinds** (`metadata.kind`, added in M2):

| kind | file | written when |
|---|---|---|
| `periodic` | `step_*.pt` | every `checkpoint_interval` steps, and at the end of a partial `fit(max_steps=...)` |
| `final` | `step_<max_steps>.pt` | the scheduled last step. Never deleted by retention. |
| `interrupt` | `step_*.pt` | after SIGINT/SIGTERM: the current step finishes, is saved, and the process exits |
| `best` | `best.pt` | whenever the selection metric improves at an evaluation (`save_best`). The selection metric is the exact full-split validation loss when `exact_val_eval = true`, otherwise the fixed-batch estimate. Step 0 is evaluated but never saved as best. |

**Retention** (`keep_last_checkpoints`): after each save, the oldest `periodic` and
`interrupt` checkpoints beyond the newest N are deleted with their sidecars. `0` keeps
everything. `final` checkpoints and `best.pt` are never deleted.

**Resuming** (`scripts/train.py --resume <ckpt>` or `--resume-latest <run_dir>`, which
picks the newest `step_*.pt` that actually loads):

- An identical model config is required (a mismatch raises).
- The saved `max_steps` is kept. A different `--max-steps` is refused unless
  `--extend-schedule` is passed, because it changes the cosine schedule.
- Steps trained after the last checkpoint and before a crash are retrained and logged
  again. CSV export keeps the last record per `(event, step)`.
- If `best.pt` is newer than the resume point, a warning is logged. It is replaced only if
  the resumed run beats the restored `best_val_loss`.
- M1 checkpoints (without histories, `best_step`, `kind`, or the new config keys) still
  load. Missing fields take their defaults (tested).

A NaN/Inf loss or gradient norm raises `NonFiniteError` before `optimizer.step()`.
`train.py` then writes `failure.json` (step and recent losses), exports the CSVs, and
exits with code 2. Nothing is checkpointed, so the last periodic checkpoint stays the
latest valid state.

**Determinism:** on CPU, an interrupted-and-resumed run is bit-identical to an
uninterrupted one (tested, with dropout on, for both checkpoint-based and
`request_stop` interruption). On MPS, even two uninterrupted runs differ by about
1e-5 in parameters after 20 steps, because MPS kernels are not bit-reproducible.
Resume differs from an uninterrupted run by the same amount, so resume is as exact as
the backend allows.

## Logging and exports (M2)

`metrics.jsonl` in the run directory is append-only and flushed per record:

| event | frequency | fields |
|---|---|---|
| `train` | every optimizer step | `loss`, `lr` (the LR used for this update), `grad_norm` (pre-clip), `step_time_s`, `tokens_per_s`, `tokens_seen` |
| `eval` | step 0, every `eval_interval`, last step | `train_loss`, `val_loss` (fixed-batch estimates), `val_loss_exact`, perplexities, bits/char, `gap`, `lr`, `best_val_loss`, `is_best` |
| `memory` | every `log_interval` | `process_rss_peak`, `mps_current_allocated`, `mps_driver_allocated` (or CUDA peaks) |

At the end of a run (or on a NaN stop), `train_metrics.csv`, `eval_metrics.csv`, and
`memory_metrics.csv` are exported from it. The console log shows progress
(`step/max`, %, ETA), LR, gradient norm, step time, and throughput.

**Memory counters.** CUDA has true allocator peaks (`max_memory_allocated`). MPS in
PyTorch 2.15 has none, so three complementary values are recorded:

- `mps_current_allocated`, sampled right after the last forward pass (activations
  live), after each step, and after each eval. Its maximum is a lower bound on the
  true peak.
- `mps_driver_allocated`, the Metal allocator pool. It only grows unless
  `empty_cache()` is called, so its maximum over samples is close to the peak
  reserved memory.
- `process_rss_peak` from `getrusage`. This is a true OS high-water mark for the
  process's resident memory. Unified-memory GPU buffers are not necessarily counted
  in it.

## Hardware budget

The machine is an Apple M6 with 16 GiB unified memory and a 12-core GPU, using
MPS with PyTorch 2.15.0. Budget roughly 10 to 12 GiB for training.

Measured for `gpt_10m` (float32, SDPA):

| batch × ctx | median step | tokens/s | MPS driver memory | allocated (post-forward) |
|---|---|---|---|---|
| 8 × 256 | 66.9 ms | 30.7k | 1.19 GB | 0.64 GB |
| 64 × 256 | 507 ms | 32.3k | 4.49 GB | n/a (pre-probe run) |
| 8 × 256, explicit attention | 68.1 ms | 30.1k | 1.19 GB | n/a |
| 64 × 256, M2 full run (5,000 steps, incl. evals in pool) | 514 ms | 31.9k | 5.57 GB | 4.05 GB |

Throughput is already saturated at batch 8: batch 64 adds only about 5%.
That is about 6 · 10.8M · 31k ≈ 2 TFLOP/s of fp32 training compute.

Projections for larger configs (not measured; activation-bound, so verify
before use):

- `gpt_30m`: about 3× the FLOPs per token → ~10k tokens/s.
- `gpt_100m`: about 10× → ~3k tokens/s. 16 bytes/param of optimizer state
  alone is 1.6 GB, and a 1024 context needs small micro-batches.

## Dataset strategy

1. **Tiny Shakespeare, char-level** (current): 1,003,854 train and 111,540 val tokens.
2. **Byte-level BPE**: implemented in M3 (`tokenizer/bpe.py`). It is trained on the
   training split only.
3. **Larger corpus**: researched in M3. M3.1 proposal: TinyStories V2, about 100M
   tokens from bounded byte ranges, with manifests, hashes, deduplication and
   hash-based splits. FineWeb-Edu follows. Needs approval before any download.
4. **Dialogue data** for supervised fine-tuning, then preference/RL experiments.

Compute-optimal training wants about 20 tokens/parameter, which for `gpt_10m` is
about 200M tokens. Tiny Shakespeare has 1M, so char-level runs are multi-epoch and
limited by overfitting. Dropout and early stopping on validation loss matter here.

## Evaluation metrics

| metric | definition |
|---|---|
| train / val loss | mean cross-entropy (nats/token). The fixed-batch estimate is used during training; the exact full pass at the end. |
| bits/char | loss / ln 2 |
| perplexity | exp(loss) |
| generalization gap | val − train |
| tokens/s, median step time | measured after warmup steps, with device sync |
| memory | `torch.mps.current_allocated_memory` (sampled after the last forward, when activations are live, and after the step) and `driver_allocated_memory` (the allocator pool) |
| samples | fixed prompt, temperature 0.8, top-k 40, fixed seed |

## M2 result and next step

The full `gpt_10m` run (5,000 steps, batch 64) took 47.3 min. The best exact val
loss was **1.4736 at step 1750**; the final value was 1.7402. The model overfits
after about 29 epochs. Details are in [reports/m2_pretraining.md](reports/m2_pretraining.md).

Implications for planning:

- On Tiny Shakespeare at this scale, the useful budget is about 2,000 steps
  (~33M tokens, ~33 epochs), not 5,000. Proposed M3 candidate (needs approval):
  `max_steps = 2000`, eval every 100. About 19 min.
- Checkpoint storage: 129 MB each. Keeping every 250-step snapshot costs 2.5 GB per
  5k-step run. Use `keep_last_checkpoints` for runs where emergence analysis is not
  needed.
- Eval overhead with `exact_val_eval` plus 50-batch estimates is about 11.5 s per
  eval, about 8% of the run at a 250-step interval.

## M2.1 result (LR-schedule ablation)

With 2000 steps, a horizon of 2000 (B) gave best val 1.4638 and final 1.4672. M2's
horizon of 5000, stopped at 2000 (A), gave 1.4703 and 1.4865. Early stopping explains
most of the gain over M2's final checkpoint. The compressed schedule mainly makes the
last checkpoint nearly as good as the best. It is a single seed, so the planned next
step is a repeated-seed A/B study. Details are in
[reports/m2_1_lr_ablation.md](reports/m2_1_lr_ablation.md).

## M3 measurements relevant to planning (BPE, MPS, float32)

| model | params (V 4,096) | tok/s | memory after forward | tokens per hour |
|---|---|---|---|---|
| S 6/6/384, T512 B32 | 12.42M | 26.2k (24.4k with dropout 0.2) | 4.4 GB (6.0 GB) | about 85M |
| M 8/8/512, T512 B16 | 27.58M | 13.7k | 4.0 GB | about 44M |
| L 12/12/768, T512 B8 | 88.59M | 4.85k | 5.2 GB | about 16M |

Character-level throughput does not transfer to BPE. Per byte of text, BPE-4096 is
about 3.5× faster on TinyStories. Details are in
[reports/m3_dataset_research.md](reports/m3_dataset_research.md).

## M3.1 result (first BPE model)

TinyStories V2, BPE 4,096, 12.4M parameters: in 56.5 min (4,800 steps × 16,384
tokens, 0.8 epoch) validation reached 1.5455 nats/token (0.563 bits/byte) and was
still falling, with no overfitting. The planned step budget matched the actual
time to within 3%. The next planned experiment is a longer run on the same data,
then the 27.6M model. Details are in
[reports/m3_1_tinystories_pretraining.md](reports/m3_1_tinystories_pretraining.md).

## M3.2 result (continuation phase)

Continuing M3.1 for 2,400 steps at a constant 6e-5 lowered validation from 1.5455 to
1.5147 nats/token (−2.0%) for +52% compute, with no overfitting and no measurable
narrative gain. Continuation phases are explicit:
- `lr_schedule = "constant"` sets the schedule;
- `scripts/train.py --continue-from` writes to a new run directory and
  `continuation.json`;
- the checkpoint `phases` field records each phase.

Recommendation: scale to the 27.6M model next. Details are in
[reports/m3_2_extended_pretraining.md](reports/m3_2_extended_pretraining.md).

## M3.3 result (capacity scaling)

27.6M parameters at equal tokens (4,800 steps) reached val 1.4436 nats/token, against
1.5455 for 12.4M. It runs at 13.6k tok/s, so a 4,800-step run takes about 110 min,
using a 5.6 GB pool at micro-batch 16 × 2. At matched compute the intermediate
checkpoints trail the small model, so the next step is runs with schedules sized
to compute. Details are in [reports/m3_3_model_scaling.md](reports/m3_3_model_scaling.md).

## Known limitations

- Only float32 is supported. bf16 and mixed precision are unverified on MPS and
  rejected by the trainer.
- MPS is not bit-deterministic, so resume there is exact only to about 1e-5.
  CPU resume is bit-exact.
- Timing statistics cover the current process only. Loss and eval histories are
  checkpointed (M2), so after a resume they cover the whole run.
- MPS has no peak-allocated counter in this PyTorch version. See "Memory
  counters" above for what is measured instead.
- On MPS, SDPA with dropout fails under `torch.no_grad()` in train mode. Our
  code never does this, but external callers could.
- BPE streams pack documents with `<|endoftext|>` but do not mask attention across
  document boundaries.
- Generation recomputes the full context for every token (no KV cache). This is
  fine at 10M parameters and a 256 context, but slow for long outputs.
- Best-checkpoint saving exists (M2), but there is no early stopping: training
  always runs to `max_steps`.
- The character vocabulary is built from the whole corpus (see M0 notes).
- Run directory timestamps are in UTC.

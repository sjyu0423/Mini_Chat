# Training plan

## Optimization (implemented in M1)

- **AdamW** (β₁, β₂ from config, ε = 1e-8). Weight decay applies to tensors with
  ≥ 2 dimensions (Linear weights and the tied embedding). It never applies to
  biases or LayerNorm.
- **LR schedule** (`training/optim.py`, a pure function of the step):
  linear warmup `lr_max·(s+1)/W` for s < W, then cosine
  `lr_min + ½(lr_max − lr_min)(1 + cos(π(s−W)/(S−W)))`, then held at lr_min.
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

Each checkpoint is `runs/<run>/checkpoints/step_XXXXXXXX.pt` plus a `.json`
metadata sidecar. Both are written atomically (temp file, fsync, `os.replace`)
and loaded with `torch.load(weights_only=True)`. A checkpoint contains:

| key | restores |
|---|---|
| `model` | weights (the tied tensor is stored once and re-tied on load) |
| `optimizer` | AdamW moments and step counts (moved to the parameters' device on load) |
| `step`, `tokens_seen`, `best_val_loss` | the loop position. The LR schedule is derived from `step`. |
| `data_generator` | the batch-sampling RNG |
| `rng` | torch CPU RNG, MPS/CUDA RNG (dropout), and Python `random` |
| metadata | full `ExperimentConfig`, git commit, torch version, losses, timestamp |

Resuming requires an identical model config (a mismatch raises). If the training
config differs, for example a larger `max_steps`, the trainer logs a warning:
the run continues, but the cosine schedule changes shape.

**Determinism:** on CPU, an interrupted-and-resumed run is bit-identical to an
uninterrupted one (tested, with dropout on). On MPS, even two uninterrupted runs
differ by about 1e-5 in parameters after 20 steps, because MPS kernels are not
bit-reproducible. Resume differs from an uninterrupted run by the same amount,
so resume is as exact as the backend allows.

## Hardware budget

The machine is an Apple M6 with 16 GiB unified memory and a 12-core GPU, using
MPS with PyTorch 2.15.0. Budget roughly 10 to 12 GiB for training.

Measured for `gpt_10m` (float32, SDPA):

| batch × ctx | median step | tokens/s | MPS driver memory | allocated (post-forward) |
|---|---|---|---|---|
| 8 × 256 | 66.9 ms | 30.7k | 1.19 GB | 0.64 GB |
| 64 × 256 | 507 ms | 32.3k | 4.49 GB | n/a (pre-probe run) |
| 8 × 256, explicit attention | 68.1 ms | 30.1k | 1.19 GB | n/a |

Throughput is already saturated at batch 8: batch 64 adds only about 5%.
That is about 6 · 10.8M · 31k ≈ 2 TFLOP/s of fp32 training compute.

Projections for larger configs (not measured; activation-bound, so verify
before use):

- `gpt_30m`: about 3× the FLOPs per token → ~10k tokens/s.
- `gpt_100m`: about 10× → ~3k tokens/s. 16 bytes/param of optimizer state
  alone is 1.6 GB, and a 1024 context needs small micro-batches.

## Dataset strategy

1. **Tiny Shakespeare, char-level** (current): 1,003,854 train and 111,540 val tokens.
2. **Byte-level BPE** implemented in-repo and trained on the training split only.
3. **Larger permissive corpus**, sharded, with documented licenses, pinned hashes,
   and deduplication. Needs approval before any large download.
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

## M2 recommendation (pending approval)

Run the full `configs/gpt_10m.toml` as written: batch 64 × 256, 5,000 steps
(~82M tokens, ~82 epochs), lr 1e-3 → 1e-4, warmup 100, dropout 0.2, eval every 250.

- **Estimated cost**: 5,000 × 0.507 s ≈ 42 min of training, plus about 6 min of
  evaluation (20 evals × 100 forward passes). About 50 min wall-clock in total.
- **Memory**: ~4.5 GB driver memory.
- **Disk**: 129 MB per checkpoint (fp32 weights plus Adam moments). Five
  checkpoints come to about 650 MB.
- **Before running**: add best-validation checkpoint retention, and keep the
  Mac awake (`caffeinate -i`).
- **Expectation**: public reference runs of this exact configuration report
  validation loss around 1.5 nats/char. We should treat that as a target to
  check, not an assumption.

## Known limitations (M1)

- Only float32 is supported. bf16 and mixed precision are unverified on MPS and
  rejected by the trainer.
- MPS is not bit-deterministic, so resume there is exact only to about 1e-5.
  CPU resume is bit-exact.
- `loss_history` and timing statistics cover the current process only. After a
  resume, the full history is in `metrics.jsonl`.
- Memory "peak" values are probes taken after the last forward pass and after
  each step, not a true allocator high-water mark. MPS has no peak-allocated
  counter in this PyTorch version.
- On MPS, SDPA with dropout fails under `torch.no_grad()` in train mode. Our
  code never does this, but external callers could.
- Generation recomputes the full context for every token (no KV cache). This is
  fine at 10M parameters and a 256 context, but slow for long outputs.
- No best-validation checkpoint retention or early stopping yet.
- The character vocabulary is built from the whole corpus (see M0 notes).
- Run directory timestamps are in UTC.

# mini_chatgpt

A GPT-style, decoder-only language model built and trained **from randomly
initialized weights**, for learning and research: architecture, pretraining,
scaling laws, optimization, and later conversational fine-tuning and RL.

- No pretrained weights, no external LLM APIs, no high-level model frameworks.
- PyTorch is used for tensors, autograd, and optimizers. The Transformer is
  written by hand (`src/mini_chatgpt/model/gpt.py`).
- Target hardware: an Apple Silicon Mac mini with 16 GB unified memory (MPS backend).

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 Foundation | project layout, configs, device selection, char tokenizer, data pipeline, training utilities, tests, docs | done |
| M1 GPT + training engine | hand-written GPT (explicit and SDPA attention), AdamW + warmup/cosine, accumulation, clipping, NaN guard, checkpoint/resume, sampling, 100-step MPS benchmark | **done (awaiting review)** |
| M2 | full gpt_10m char-level baseline run (~50 min) | proposed |
| later | BPE tokenizer, larger corpus, scaling study, chat fine-tuning, RL | planned |

## Confirmed hardware (from `scripts/detect_hardware.py`, 2026-11-12)

| | |
|---|---|
| Chip | Apple M6 |
| Unified memory | 16 GiB |
| CPU cores | 12 (2 "Super" + 4 Performance + 6 Efficiency) |
| GPU cores | 12 (Metal 4) |
| OS | macOS 27.0.1 (arm64) |
| Python | 3.15.2 (Homebrew), in `.venv` via `uv` |
| PyTorch | 2.15.0 — MPS built: yes, MPS available: yes, CUDA: no |

## Setup

```bash
uv venv --python 3.15 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

(`python -m venv .venv && .venv/bin/pip install -e ".[dev]"` works too.)

## Usage

```bash
# Hardware + backend report (optionally saved under runs/, which is git-ignored)
.venv/bin/python scripts/detect_hardware.py --json runs/hardware.json

# Download Tiny Shakespeare (~1.1 MB, SHA-256 pinned), tokenize, split, run reproducibility checks
.venv/bin/python scripts/prepare_data.py --config configs/gpt_10m.toml

# Print configs, selected device, and parameter counts
.venv/bin/python scripts/show_config.py configs/*.toml

# Critical learning checks on gpt_10m: overfit one batch, gradient flow, checkpoint/resume (~1 min)
.venv/bin/python scripts/sanity_checks.py --device mps

# Bounded smoke benchmark (100 steps, ~10 s on M6); writes runs/<name>-<UTC timestamp>/
.venv/bin/python scripts/train.py --config configs/gpt_10m.toml --name bench_10m_b8 \
    --batch-size 8 --max-steps 100 --warmup-steps 10 --eval-interval 50 --eval-steps 20 \
    --checkpoint-interval 100 --device mps --full-eval

# Resume (continues step count, optimizer, and RNG state)
.venv/bin/python scripts/train.py --resume runs/<run>/checkpoints/step_00000100.pt --max-steps 200

# Sample from a checkpoint (uses the run's saved tokenizer)
.venv/bin/python scripts/generate.py runs/<run>/checkpoints/step_00000100.pt \
    --prompt "ROMEO:" --max-new-tokens 300 --temperature 0.8 --top-k 40 --seed 1

# Quality checks
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
```

## Layout

```
src/mini_chatgpt/
  config.py            typed, validated Model/Training/Data/Experiment configs (TOML/JSON)
  device.py            MPS/CUDA/CPU selection, dtype resolution, device info
  data/sources.py      dataset provenance, license notes, pinned hashes, download
  data/dataset.py      contiguous train/val split, random + sequential next-token batches
  data/prepare.py      text -> tokenizer -> ids -> split -> train.bin/val.bin/meta.json
  tokenizer/           Tokenizer protocol, CharTokenizer
  model/gpt.py         GPT, blocks, explicit + SDPA causal attention, GPT-2 init
  training/seed.py     global seeding + dedicated data generators
  training/metrics.py  running means, perplexity, bits/token, throughput
  training/checkpoint.py  atomic writes, checkpoint paths, metadata sidecars
  training/run.py      run directories, JSONL metrics log, console logging
  training/optim.py    AdamW param groups, warmup + cosine LR
  training/trainer.py  training engine: accumulation, clipping, eval, checkpoint/resume
  inference/generate.py  greedy / temperature / top-k sampling
configs/               debug, gpt_10m, gpt_30m, gpt_100m (TOML)
scripts/               train, generate, sanity_checks, prepare_data, show_config, detect_hardware
tests/                 pytest suite (no network access needed)
docs/                  architecture, training plan, experiment log
data/, runs/           git-ignored (downloads and experiment outputs)
```

## M1 results at a glance (gpt_10m, MPS, float32)

| | |
|---|---|
| parameters | 10,770,816 (equal to the analytic count) |
| 100-step smoke run, batch 8 × 256 | val loss 4.127 → 2.505. Median step 66.7 ms, 30.7k tokens/s, 1.19 GB MPS driver memory. |
| single-batch overfit | loss 4.21 → 0.006 in 200 steps |
| resume | bit-exact on CPU. On MPS, within the backend's own run-to-run noise (~1e-5). |

Known limitations are listed in [docs/training_plan.md](docs/training_plan.md) and [docs/architecture.md](docs/architecture.md).

## Docs

- [docs/architecture.md](docs/architecture.md): model, attention, initialization, parameter counts
- [docs/training_plan.md](docs/training_plan.md): optimization, checkpoint semantics, measured hardware budget, limitations, M2 plan
- [docs/experiments.md](docs/experiments.md): experiment log and conventions

## Data licensing

Tiny Shakespeare: Shakespeare's plays (public domain), as packaged in the
MIT-licensed [karpathy/char-rnn](https://github.com/karpathy/char-rnn)
repository. The file is not committed; `prepare_data.py` downloads it to
`data/raw/` and verifies its SHA-256.

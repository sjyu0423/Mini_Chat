# mini_chatgpt

A GPT-style, decoder-only language model built and trained **from randomly
initialized weights**, for learning and research: architecture, tokenizer, pretraining,
conversational fine-tuning, evaluation, and a local chat app (v1.0).

- No pretrained weights, no external LLM APIs, no high-level model frameworks.
- PyTorch is used for tensors, autograd, and optimizers. The Transformer is
  written by hand (`src/mini_chatgpt/model/gpt.py`).
- Target hardware: an Apple Silicon Mac mini with 16 GB unified memory (MPS backend).

## Project overview (v1.0)

- **[Overview](docs/showcase/overview.md)**: a one-page summary of what was built, what it can and cannot do, and the key 14M vs 30M finding.
- **[Technical walkthrough](docs/showcase/walkthrough.md)**: the model, tokenizer, pretraining, chat fine-tuning, Apple Silicon issues, evaluation method and limitations.
- **[Release report](docs/reports/m7_portfolio_release.md)**: the v1.0 release manifest, verification and commit plan.
- Every number in the showcase is traced to a report in [`docs/showcase/facts.json`](docs/showcase/facts.json).

## Try the chat demo (M6)

A local web app for talking to the model (14M parameters, trained from scratch here). It runs
entirely on this machine, with no external AI API. Responses are often wrong; see the
[M6 report](docs/reports/m6_chat_demo.md).

```bash
uv pip install --python .venv/bin/python -e ".[serve]"   # FastAPI + uvicorn
npm --prefix apps/chat-web install && npm --prefix apps/chat-web run build
.venv/bin/python scripts/serve_chat.py                    # then open http://127.0.0.1:8000
```

The command needs the M5 checkpoint (`runs/m5_sft_chat-20261114-175334/checkpoints/best.pt`,
Git-ignored, 168 MB). It is produced by the M4.1 and M5 commands under Usage below.

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 Foundation | project layout, configs, device selection, char tokenizer, data pipeline, training utilities, tests, docs | done |
| M1 GPT + training engine | hand-written GPT (explicit and SDPA attention), AdamW + warmup/cosine, accumulation, clipping, NaN guard, checkpoint/resume, sampling, 100-step MPS benchmark | done |
| M2 First pretraining experiment | best/final checkpoints, retention, histories, CSV export, stability check; full 5,000-step gpt_10m run; evaluation, emergence and memorization analysis | done |
| M2.1 LR-schedule ablation | decoupled cosine horizon (`lr_schedule_steps`); A (M2 schedule, stop at 2000) vs B (horizon 2000); validity checks, comparison report | done |
| M3 Dataset research + BPE | TinyStories / FineWeb-Edu research, bounded samples, in-repo byte-level BPE, tokenizer comparison (2k/4k/8k), model sizing, MPS microbenchmarks, data budget | done |
| M3.1 TinyStories pretraining | verified byte-range acquisition, dedup, BPE-4096, memmap pipeline; 12.4M-param model, 4,800 steps (~1 h); story/memorization evaluation | done |
| M3.2 Extended pretraining | explicit continuation phases (`--continue-from`, constant LR); +2,400 steps at 6e-5; blind entity-consistency evaluation | done |
| M3.3 Model scaling | 27.6M model (8L/8H/512), equal tokens to M3.1; pre-registered 21-prompt × 3-seed blind evaluation of 3 models; FLOP accounting | done |
| M4 General-data research | FineWeb-Edu / Cosmopedia / chat-data research, frozen 202-item evaluation suite, BPE-8192 with chat tokens, compute plan | done |
| M4.1 General pretraining | pinned FineWeb-Edu 70% + Cosmopedia v2 30%, single-pass 157.3M tokens, 14.0M-param base model (110 min); 1.264 bits/byte | done |
| M5 Conversational SFT | chat format on reserved tokens, assistant-only loss, OASST2 + Dolly-15k SFT (8 min), stop-aware generation, repetition penalty, `scripts/chat.py` | done |
| M6 Local chat demo | FastAPI inference service (localhost, validated, serialized) + React/TypeScript web UI (`apps/chat-web`), tests, benchmark | done |
| M6.1 Chat quality | decoding grid (incl. top-p), SFT data audit, controlled length-capped SFT; M5 + default decoding kept | done |
| M6.2 Model scaling | 29.7M base model (8L/8H/512) on the identical M4.1 data/recipe, 157.3M tokens (205 min); 1.204 vs 1.264 bits/byte | done |
| M6.3 Conversational scaling | M5 SFT recipe on the 29.7M model, pre-registered blind paired comparison; not better at content, M5 stays served | done |
| M7 Portfolio release | showcase docs, in-app project page and recorded-conversation fallback, verified inference-only export of both models | **done (awaiting review)** |
| v1.0 | training frozen: M5 (13,989,888 params) is the default chatbot; M6.3 (29,676,544) is kept for research | |
| later | scaling, data mixing for SFT, preference tuning / RL | deferred until after v1.0 |

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

# Resume (continues step count, optimizer, RNG state, and loss histories; keeps max_steps)
.venv/bin/python scripts/train.py --resume-latest runs/<run>
.venv/bin/python scripts/train.py --resume runs/<run>/checkpoints/step_00000100.pt

# M2: pre-flight check of the full config (LR schedule + first 300 steps, ~2.5 min)
.venv/bin/python scripts/stability_check.py --config configs/gpt_10m_m2.toml --steps 300

# M2: full baseline (~50 min on M6); keep the Mac awake
caffeinate -i .venv/bin/python scripts/train.py --config configs/gpt_10m_m2.toml --full-eval

# M2: evaluate best + final checkpoints, samples, emergence/memorization stats, SVG plots
.venv/bin/python scripts/analyze_run.py runs/<run> --out docs/reports/m2
.venv/bin/python scripts/memorization_probe.py runs/<run> --out docs/reports/m2

# M3: bounded dataset samples (dataset-viewer API; manifest + hashes), stats, tokenizer + model benchmarks
.venv/bin/python scripts/fetch_samples.py roneneldan/TinyStories --requests 60 --out data/raw/samples/tinystories
.venv/bin/python scripts/fetch_samples.py HuggingFaceFW/fineweb-edu --config sample-10BT --requests 40 \
    --out data/raw/samples/fineweb_edu_10bt
.venv/bin/python scripts/sample_stats.py --corpus tinystories=data/raw/samples/tinystories \
    --corpus fineweb_edu=data/raw/samples/fineweb_edu_10bt --out docs/reports/m3
.venv/bin/python scripts/benchmark_tokenizers.py --corpus tinystories=data/raw/samples/tinystories \
    --corpus fineweb_edu=data/raw/samples/fineweb_edu_10bt --vocab 2048 4096 8192 --out docs/reports/m3
.venv/bin/python scripts/benchmark_models.py --out docs/reports/m3      # ~10 min, one case at a time
.venv/bin/python scripts/benchmark_dataloading.py --out docs/reports/m3
.venv/bin/python scripts/plot_m3.py docs/reports/m3

# M3.1: TinyStories V2 (pinned revision, 8 verified 50 MB byte ranges, dedup, BPE-4096, memmap streams)
.venv/bin/python scripts/prepare_tinystories.py --ranges 8 --range-mb 50 --vocab 4096
caffeinate -i .venv/bin/python scripts/train.py --config configs/tinystories_s_bpe4096.toml \
    --sample-prompt "Once upon a time" --sample-tokens 200
.venv/bin/python scripts/evaluate_tinystories.py runs/<run> --out docs/reports/m3_1
.venv/bin/python scripts/generate.py runs/<run>/checkpoints/best.pt \
    --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.7 --top-k 40 --seed 1

# M3.2: continue a run as a NEW phase (new run dir; constant LR; full state restored)
.venv/bin/python scripts/preflight_continuation.py runs/<m3_1_run> configs/tinystories_s_bpe4096_m3_2.toml
caffeinate -i .venv/bin/python scripts/train.py --config configs/tinystories_s_bpe4096_m3_2.toml \
    --continue-from runs/<m3_1_run>/checkpoints/best.pt
.venv/bin/python scripts/evaluate_m3_2.py --parent runs/<m3_1_run> --run runs/<m3_2_run> --out docs/reports/m3_2
.venv/bin/python scripts/evaluate_m3_2.py --unblind docs/reports/m3_2   # after filling blind_scores.json

# M3.3: capacity scaling (preflight, train, blind evaluation of M3.1 / M3.2 / M3.3)
.venv/bin/python scripts/preflight_scaling.py configs/tinystories_m_bpe4096.toml --historical runs/<run> ...
caffeinate -i .venv/bin/python scripts/train.py --config configs/tinystories_m_bpe4096.toml
.venv/bin/python scripts/evaluate_m3_3.py --m31 runs/<m3_1> --m32 runs/<m3_2> --m33 runs/<m3_3> --out docs/reports/m3_3
.venv/bin/python scripts/evaluate_m3_3.py --unblind docs/reports/m3_3   # after filling blind_scores.json

# M2.1: verify an A/B pair is a controlled comparison, then compare finished runs
.venv/bin/python scripts/verify_ablation.py configs/gpt_10m_m2_1_a.toml configs/gpt_10m_m2_1_b.toml \
    --reference runs/<m2_run>
.venv/bin/python scripts/compare_runs.py --run M2=runs/<m2_run> --run A=runs/<a_run> \
    --run B=runs/<b_run> --out docs/reports/m2_1

# M4.1: general pretraining data (pinned FineWeb-Edu + Cosmopedia v2 files), preflight, train, evaluate
.venv/bin/python scripts/prepare_general.py --download && .venv/bin/python scripts/prepare_general.py
.venv/bin/python scripts/preflight_m4_1_data.py configs/m4_1_general_s_bpe8192.toml
caffeinate -dims .venv/bin/python scripts/train.py --config configs/m4_1_general_s_bpe8192.toml
.venv/bin/python scripts/evaluate_m4_1.py runs/<m4_1_run> --out docs/reports/m4_1

# M5: conversational SFT (OASST2 + Dolly-15k), preflight, train, evaluate, chat
.venv/bin/python scripts/prepare_chat.py --download
.venv/bin/python scripts/preflight_sft.py configs/sft/m5_sft_chat.toml --out docs/reports/m5/preflight.json
caffeinate -dims .venv/bin/python scripts/train_sft.py --config configs/sft/m5_sft_chat.toml
.venv/bin/python scripts/evaluate_m5.py runs/<m5_run> --out docs/reports/m5
.venv/bin/python scripts/chat.py runs/<m5_run>/checkpoints/best.pt --repetition-penalty 1.2

# M7: inference-only release (weights + tokenizer, verified identical) and showcase data
.venv/bin/python scripts/export_release.py          # -> runs/release/mini_chatgpt-v1.0/, docs/release/v1.0_manifest.json
.venv/bin/python scripts/serve_chat.py --checkpoint runs/release/mini_chatgpt-v1.0/m5/checkpoints/model.pt
.venv/bin/python scripts/build_showcase_data.py      # recorded replies + facts for the web app
VITE_STATIC_DEMO=1 npm --prefix apps/chat-web run build   # static, no-backend build (recorded replies only)

# M6: local chat demo (API + web UI on one port), its benchmark, and frontend checks
.venv/bin/python scripts/serve_chat.py            # http://127.0.0.1:8000
.venv/bin/python scripts/benchmark_chat_server.py
npm --prefix apps/chat-web run typecheck && npm --prefix apps/chat-web run lint && npm --prefix apps/chat-web test

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
  data/prepare.py      text -> tokenizer -> ids -> split -> train.bin/val.bin/meta.json;
                       verified loading (char: int64; BPE: uint16 memmap + fingerprint/hash checks)
  data/tinystories.py  TinyStories V2: verified byte-range fetch, story recovery, MinHash dedup
  tokenizer/           Tokenizer protocol, CharTokenizer, ByteBPETokenizer (byte-level BPE, stdlib only)
  model/gpt.py         GPT, blocks, explicit + SDPA causal attention, GPT-2 init
  training/seed.py     global seeding + dedicated data generators
  training/metrics.py  running means, perplexity, bits/token, throughput
  training/checkpoint.py  atomic writes, checkpoint paths, metadata sidecars
  training/run.py      run directories, JSONL metrics log, console logging
  training/optim.py    AdamW param groups, warmup + cosine LR
  training/trainer.py  training engine: accumulation, clipping, eval, best/final checkpoints,
                       retention, histories, graceful stop, checkpoint/resume
  inference/generate.py  greedy / temperature / top-k sampling
  analysis/svgplot.py  dependency-free SVG line charts for reports
  analysis/text_stats.py  word/speaker statistics, verbatim-overlap (memorization) checks
  analysis/story_eval.py  exact losses, k-gram training-overlap index, story metrics, entity checks
  analysis/general_eval.py  frozen general-language suite scorer (M4)
  data/general.py      FineWeb-Edu + Cosmopedia pipeline: pinned sources, cleaning, dedup, splits (M4.1)
  chat/                chat format + loss masks, OASST2/Dolly pipeline, SFT trainer, chat generation (M5)
  serve/               local chat demo: model engine + FastAPI app (M6)
apps/chat-web/         React + TypeScript (Vite) web UI for the local chat demo (M6)
evals/                 frozen evaluation suites (general_v1, chat_v1) with locked hashes
configs/               debug, gpt_10m, gpt_10m_m2, gpt_10m_m2_1_{a,b}, tinystories_s_bpe4096(_m3_2),
                       tinystories_m_bpe4096,
                       gpt_30m, gpt_100m
scripts/               train, generate, sanity_checks, stability_check, analyze_run, memorization_probe,
                       verify_ablation, compare_runs, fetch_samples, sample_stats,
                       benchmark_tokenizers, benchmark_models, benchmark_dataloading, plot_m3,
                       prepare_tinystories, evaluate_tinystories, preflight_continuation,
                       evaluate_m3_2, preflight_scaling, evaluate_m3_3,
                       prepare_data, show_config, detect_hardware,
                       prepare_general, evaluate_m4_1 (M4.1), prepare_chat, preflight_sft, train_sft,
                       evaluate_m5, chat (M5), serve_chat, benchmark_chat_server (M6)
tests/                 pytest suite (no network access needed)
docs/                  architecture, training plan, experiment log, reports/ (per-milestone)
data/, runs/           git-ignored (downloads and experiment outputs)
```

## M1 results at a glance (gpt_10m, MPS, float32)

| | |
|---|---|
| parameters | 10,770,816 (equal to the analytic count) |
| 100-step smoke run, batch 8 × 256 | val loss 4.127 → 2.505. Median step 66.7 ms, 30.7k tokens/s, 1.19 GB MPS driver memory. |
| single-batch overfit | loss 4.21 → 0.006 in 200 steps |
| resume | bit-exact on CPU. On MPS, within the backend's own run-to-run noise (~1e-5). |

## M2 results at a glance (gpt_10m, 5,000 steps, batch 64 × 256, MPS float32)

| | |
|---|---|
| best val loss (exact) | **1.4736 nats/char** at step 1750 (2.126 bits/char, perplexity 4.36) |
| final val loss | 1.7402 (train 0.586, gap +1.15): strong overfitting after about 29 epochs |
| run | 81.9M tokens in 47.3 min, 31.9k tok/s, 5.57 GB MPS pool, no instabilities |
| samples | real words, speaker labels and verse layout; locally grammatical, no sustained meaning |
| memorization | no long regurgitation; training-specific copying grows as the model overfits |

Full report: [docs/reports/m2_pretraining.md](docs/reports/m2_pretraining.md).

**M2.1 (LR-schedule ablation, 2,000 steps each):** compressing the cosine horizon to
2,000 gave best val **1.4638** and final 1.4672. M2's schedule stopped at 2,000 gave
1.4703 and 1.4865. Early stopping explains most of the gain over M2's final
checkpoint. This is a single seed, not significant. Report:
[docs/reports/m2_1_lr_ablation.md](docs/reports/m2_1_lr_ablation.md).

Known limitations are listed in [docs/training_plan.md](docs/training_plan.md) and [docs/architecture.md](docs/architecture.md).

## Docs

- [docs/architecture.md](docs/architecture.md): model, attention, initialization, parameter counts
- [docs/training_plan.md](docs/training_plan.md): optimization, checkpoint semantics, measured hardware budget, logging, limitations, M2 result
- [docs/experiments.md](docs/experiments.md): experiment log and conventions
- [docs/reports/m2_pretraining.md](docs/reports/m2_pretraining.md): M2 report (config, metrics, plots, samples, analysis, limitations)
- [docs/reports/m2_1_lr_ablation.md](docs/reports/m2_1_lr_ablation.md): M2.1 LR-schedule ablation report
- [docs/reports/m3_dataset_research.md](docs/reports/m3_dataset_research.md): M3 dataset research, BPE tokenizer, model sizing and MPS benchmarks
- [docs/reports/m3_1_tinystories_pretraining.md](docs/reports/m3_1_tinystories_pretraining.md): M3.1 TinyStories pretraining report
- [docs/reports/m3_2_extended_pretraining.md](docs/reports/m3_2_extended_pretraining.md): M3.2 extended pretraining (continuation) report
- [docs/reports/m3_3_model_scaling.md](docs/reports/m3_3_model_scaling.md): M3.3 model-capacity scaling report
- [docs/reports/m4_general_pretraining_research.md](docs/reports/m4_general_pretraining_research.md): M4 general-data research and frozen evaluation suite
- [docs/reports/m4_1_general_pretraining.md](docs/reports/m4_1_general_pretraining.md): M4.1 general pretraining report (14M base model)
- [docs/reports/m5_conversational_training.md](docs/reports/m5_conversational_training.md): M5 conversational post-training (SFT) report
- [docs/reports/m6_chat_demo.md](docs/reports/m6_chat_demo.md): M6 local chat demo (architecture, setup, screenshots, transcripts, performance)
- [docs/reports/m6_3_conversational_scaling.md](docs/reports/m6_3_conversational_scaling.md): M6.3 SFT of the 29.7M model vs M5, pre-registered blind evaluation, promotion decision
- [docs/reports/m6_2_model_scaling.md](docs/reports/m6_2_model_scaling.md): M6.2 capacity scaling, 29.7M vs 14.0M base model at equal tokens
- [docs/reports/m6_1_chat_quality.md](docs/reports/m6_1_chat_quality.md): M6.1 decoding grid, SFT data audit, length-capped SFT experiment
- [docs/portfolio_summary.md](docs/portfolio_summary.md): concise M0–M5 technical summary

## M3 at a glance

| | |
|---|---|
| recommended data | TinyStories V2 (GPT-4 only), ~100M tokens via bounded byte ranges; FineWeb-Edu next |
| tokenizer | in-repo byte-level BPE; 4,096 for TinyStories (3.92 bytes/token), 8,192 for FineWeb-Edu (3.81) |
| model | 6L/6H/384, ctx 512, V 4,096: 12,417,024 params; 26.2k tok/s, 4.4 GB on MPS (measured) |
| budget | ~85M tokens/hour; model compute is the bottleneck (batch loading 0.36 ms vs 626 ms/step) |

Report: [docs/reports/m3_dataset_research.md](docs/reports/m3_dataset_research.md).

## M3.1 results at a glance (TinyStories V2, BPE 4,096, 12,417,024 params, MPS float32)

| | |
|---|---|
| data | 98.2M unique train tokens (482,562 stories, 400 MB of verified byte ranges); official V2 validation (5.58M tokens) |
| run | 4,800 steps x 16,384 tokens = 78.6M presentations (0.80 epoch), 56.5 min, 26.2k tok/s, 6.66 GB MPS pool |
| best = final | val **1.5455 nats/token = 0.563 bits/byte** (ppl 4.69); holdout 1.5453; no overfitting |
| stories | grammatical, self-terminating stories; weak character consistency and causality (rubric 69%) |
| memorization | 16-gram overlap with training equals that of real validation stories (2.9% vs 2.7%) |

Report: [docs/reports/m3_1_tinystories_pretraining.md](docs/reports/m3_1_tinystories_pretraining.md).

**M3.2 (continuation, +2,400 steps at constant 6e-5, 29 min):** val 1.5455 → **1.5147**
nats/token (0.552 bits/byte), no overfitting (holdout = val), but no measurable gain in
narrative/entity consistency in a blind comparison (45% vs 49%, p = 0.77). Report:
[docs/reports/m3_2_extended_pretraining.md](docs/reports/m3_2_extended_pretraining.md).

**M3.3 (27.6M params, 4,800 steps, equal tokens to M3.1, 110 min):** val **1.4436**
nats/token (0.526 bits/byte) vs 1.5455 (M3.1) and 1.5147 (M3.2). Blind 189-story
rubric: 6.06 vs 5.76 / 5.49 out of 10 (not significant; entity consistency unchanged).
Report: [docs/reports/m3_3_model_scaling.md](docs/reports/m3_3_model_scaling.md).

## Data licensing

Tiny Shakespeare: Shakespeare's plays (public domain), as packaged in the
MIT-licensed [karpathy/char-rnn](https://github.com/karpathy/char-rnn)
repository. The file is not committed; `prepare_data.py` downloads it to
`data/raw/` and verifies its SHA-256.

TinyStories V2 (CDLA-Sharing-1.0; generated by GPT-4; pinned to dataset commit
`f54c09fd`) is used as a 400 MB bounded, git-ignored subset in M3.1; only hashes,
statistics and manifests are committed. FineWeb-Edu (ODC-By 1.0 plus Common
Crawl's Terms of Use) is used only as a bounded research sample. Provenance and
licensing notes are in the M3 and M3.1 reports.

M4.1 pretrains on one pinned FineWeb-Edu file (ODC-By 1.0 + Common Crawl ToU) and one
pinned Cosmopedia v2 file from `HuggingFaceTB/smollm-corpus` (ODC-By 1.0; text generated by
Mixtral-8x7B-Instruct). M5 fine-tunes on OASST2 (`OpenAssistant/oasst2`, Apache 2.0,
human-written; synthetic messages excluded) and Dolly-15k (`databricks/databricks-dolly-15k`,
CC BY-SA 3.0, © Databricks, includes Wikipedia excerpts). All raw and processed data stays in
the git-ignored `data/` directory and is not redistributed; see the M4.1 and M5 reports.

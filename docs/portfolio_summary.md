# mini_chatgpt — technical summary (M0–M6.3)

A small conversational language model built **from scratch**: our own transformer, tokenizer, data
pipelines, training engine, evaluation suites and chat post-training. It runs on a single Apple
Silicon Mac mini (16 GB, PyTorch MPS, float32). No pretrained weights, pretrained tokenizers or
hosted AI APIs were used at any stage. Every number below comes from a logged, reproducible run;
the detailed reports are in `docs/reports/`.

## Milestones

| milestone | what was built | key result |
|---|---|---|
| **M0** data foundation | Tiny Shakespeare pipeline (pinned SHA-256), character tokenizer, deterministic split and batches | exact round trip; reproducible split hashes |
| **M1** model and engine | GPT-2-style decoder (pre-LN, causal self-attention, tied embeddings), AdamW with warmup + cosine, clipping, checkpoint/resume, metrics | 10.77M params = analytic count; single-batch overfit 4.21 → 0.006; bit-exact resume on CPU |
| **M2** first pretraining | 5,000-step character-level run, plots, memorization probes | best val 1.474 nats/char at step 1,750, then measurable overfitting (81 epochs over 1 MB) |
| **M2.1** schedule ablation | controlled A/B with verified identical start and data order | early stopping explains ~93% of the gain; schedule effect within single-seed noise |
| **M3** data and tokenizer research | byte-level BPE (own implementation), dataset survey, MPS benchmarks | BPE-4096 at 3.9 bytes/token; model sizing for a 1-hour budget |
| **M3.1–M3.3** TinyStories | MinHash deduplication, memmap streams, continuation, 12.4M vs 27.6M scaling, blind rubric | val 1.546 → 1.444 nats/token with 2.2× parameters; self-terminating stories |
| **M4** general-data research | frozen 202-item evaluation suite (hash-locked), FineWeb-Edu / Cosmopedia analysis, compute plan | recommended 70/30 mix, BPE-8192 with reserved chat tokens, 14M model |
| **M4.1** general pretraining | pinned-source data pipeline (cleaning, cross-source dedup, decontamination), single-pass data order | 13,989,888 params, 157.3M tokens each seen once, 110 min; 1.264 bits/byte; 2.3–3× better than TinyStories models on web text |
| **M5** chat post-training | chat format on reserved special tokens, assistant-only masked loss, SFT trainer, stop-aware generation, repetition penalty, conversational eval | 0% → 77–98% replies that end on their own; repetition 28% → 2.5% with penalty; 8-minute training run |
| **M6** local chat demo | FastAPI inference service (localhost, validated, one generation at a time) + React/TypeScript web UI, tests, benchmark | ~1 s startup, median 74 ms per reply, flat memory over 160+ requests |
| **M6.1** chat quality | decoding grid incl. top-p, SFT data audit, controlled length-capped SFT | defaults confirmed; shorter SFT replies improved stopping but not answer quality |
| **M6.2** model scaling | 29.7M model on the identical data, tokenizer, order and 157.3M tokens | 1.204 vs 1.264 bits/byte on validation; better on all six held-out sets; benchmarks unchanged |
| **M6.3** conversational scaling | same SFT on the 29.7M model, pre-registered paired + blind evaluation | more fluent, better retention, but 21% vs 18% strict success (n.s.); smaller model kept in service |

## What the final model can and cannot do

- **Can:** hold a chat turn in the assistant role, write short fluent English, produce lists and
  short explanations, stop by itself, use words from earlier turns.
- **Cannot:** reliably answer factual questions. A strict grade gives 0–2 of 15 factual questions
  right, and answers are often circular or confidently wrong. Reasoning across turns is shallow.
  It knows only English.
- This is the expected behaviour of a 14M-parameter model trained on ~157M tokens, orders of
  magnitude smaller than production assistants.

## Engineering highlights

- **Reproducibility:**
  - every dataset is pinned to a commit and SHA-256;
  - every stream, split and tokenizer is fingerprinted;
  - evaluation suites are hash-locked;
  - runs are deterministic from a seed, and a rerun reproduced a validation loss exactly.
- **Correctness checks before every long run:**
  - causality tests, and gradient-flow and finite-value checks;
  - overfit tests and checkpoint round trips;
  - masked-loss verification;
  - re-serialization of stored data.
- **Data hygiene:**
  - exact and MinHash near-duplicate removal;
  - document-hash train/validation/holdout splits;
  - n-gram decontamination against the evaluation suites;
  - leakage checks between splits.
- **Honest evaluation:**
  - Wilson confidence intervals;
  - base-vs-tuned comparisons on identical prompts and seeds;
  - truncation never counted as success;
  - a strict re-grade where automatic checks overcount;
  - failures documented alongside successes.
- **Apple-Silicon specifics:** diagnosed an MPS memory blow-up caused by variable-length batches.
  The MPS buffer pool grew from 2 to 13.6 GB while live tensors stayed at 0.2 GB. Fixed with
  shape-bounded padding, periodic cache release and a memory guard.

## Custom code vs PyTorch

| ours | PyTorch |
|---|---|
| transformer architecture and initialization, generation (temperature, top-k, repetition penalty, end-of-turn stopping) | tensors, autograd, MPS backend |
| byte-level BPE tokenizer (training, special tokens, fingerprint) | `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, GELU |
| data pipelines: cleaning, dedup, splits, decontamination, memmap streams, chat serialization and loss masks | `scaled_dot_product_attention` (verified against our explicit attention) |
| training loops: schedule, accumulation, clipping, checkpoint/resume, single-pass ordering, SFT with masked loss | `AdamW`, `clip_grad_norm_`, `cross_entropy` |
| evaluation suites and scorers, bits per byte, memorization index, plots (own SVG renderer) | `torch.save` / `torch.load` |

## Numbers at a glance

| | |
|---|---|
| final model | 6 layers, 6 heads, width 384, context 512, 13,989,888 parameters |
| tokenizer | byte-level BPE, 8,192 ids, chat tokens `<|user|>` `<|assistant|>` `<|system|>` `<|end|>` |
| pretraining | 157.3M tokens (FineWeb-Edu 70%, Cosmopedia v2 30%), 9,600 steps, 110 min, 1.43e16 FLOPs |
| chat tuning | 16,813 human-written conversations (OASST2 + Dolly-15k), 3 epochs, 8 min |
| scaling study | 29.7M variant (M6.2/M6.3): better language modelling, same practical answer quality |
| tests | 300+ Python tests and 20 frontend tests; ruff, mypy (strict), TypeScript and ESLint clean |

## Scaling experiment (M6.2–M6.3), in one paragraph

We scaled the model from 14M to 30M parameters on identical data and repeated the same chat
fine-tuning under a pre-registered, blind, paired evaluation. The larger model models language
measurably better (4–7% fewer bits per byte on held-out text, lower fine-tuning loss, better
retention) and stops more reliably, but it does not answer questions measurably more correctly
(21% vs 18% strict success on 75 items, not significant; one correct reply each in a 20-item blind
review), so the smaller model stayed in production. Fluency and format arrive before knowledge, and
lower loss is not the same as a more useful assistant.


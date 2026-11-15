# mini_chatgpt — a chatbot language model built from scratch

**One-line summary:** I wrote a GPT-style language model, its tokenizer, data pipelines, training
loop, chat fine-tuning and a local web app from the ground up in PyTorch, trained every model from
random weights on a 16 GB Mac mini, and evaluated them with pre-registered, paired, blind tests.

## Why I built it

To understand how a chat assistant actually works by building every layer myself, rather than
calling an API or fine-tuning someone else's model, and to practise the engineering habits real ML
work needs: reproducible data, controlled experiments, honest evaluation.

## What I built

| layer | what it is |
|---|---|
| **Model** | A decoder-only Transformer (GPT-2 style): token and position embeddings, 6 blocks of causal self-attention and MLP with pre-layer-norm and residual connections, and tied input/output embeddings. **13,989,888 parameters.** |
| **Tokenizer** | My own byte-level BPE (byte-pair encoding): it learns 7,931 merges from training text and reserves chat tokens (`<|user|>`, `<|assistant|>`, `<|end|>`). Every tokenizer is fingerprinted, so a model can never load the wrong one. |
| **Pretraining** | 157.3 million tokens of educational web text (FineWeb-Edu) and textbook-style text (Cosmopedia), cleaned, de-duplicated and decontaminated against the test sets. Each token is seen exactly once; the run takes 110 minutes. |
| **Chat fine-tuning** | 16,813 human-written conversations (OASST2, Dolly-15k). The loss counts only the assistant's words, so the model learns to answer *and to stop*. |
| **App** | A local web chat: React + TypeScript frontend and a FastAPI service running the model on the Mac's GPU. It answers in about 0.1 s and never calls an external AI. |

## What it can and cannot do

After fine-tuning, the model holds a chat turn, writes fluent English and ends **97.5%** of its
replies on its own; the base model ended **0%**. It is **not** a reliable assistant: on simple
factual questions it is usually wrong ("The capital of France is Vienna."). For a model orders of
magnitude smaller than modern assistants, that is expected, and the app says so on screen.

## The key experiment: does a bigger model help?

I doubled the model to **29,676,544 parameters** and kept everything else identical: the same data,
the same order and the same fine-tuning.

| | 14M (served) | 30M |
|---|---|---|
| base model: held-out web text, bits per byte (lower is better) | 1.335 | **1.278** |
| after chat fine-tuning: assistant-token validation loss | 3.085 | **2.932** |
| after chat fine-tuning: correct answers on 75 test prompts (strict) | 18.2% | 21.3% |
| difference, 95% interval | | +3.1 points, **−2.2 to +8.9** (p = 0.72) |
| correct replies in a 20-item blind review | 1 | 1 |

**Doubling model parameters improved language-modeling loss and fluency, but did not produce a
statistically clear improvement in factual or conversational correctness.** I had written the
promotion rule down *before* training: a credible gain of at least 5 points. The larger model did
not meet it, so the smaller, faster model stays in production. Lower loss is not the same as a more
useful assistant.

## Engineering highlights

- **Reproducible:** every dataset is pinned to a commit and hash, and evaluation suites are
  hash-locked. A rerun reproduced validation losses to four decimals.
- **Tested:** 300+ Python tests and 20 frontend tests. Before every long run, a preflight checks
  causality, gradients, data alignment, overfitting and checkpoint restore.
- **Apple Silicon:** I diagnosed a GPU-memory blow-up during fine-tuning: the cache grew from 2 to
  13.6 GB because of varying sequence lengths. Shape bucketing, cache release and a memory guard
  fixed it. The guard later stopped a larger run safely instead of letting the machine swap.
- **Honest evaluation:** paired statistics with the prompt as the unit, uncertainty intervals, blind
  review, and failures documented as carefully as successes.

*Every number above is traced to a report in [`facts.json`](facts.json).*

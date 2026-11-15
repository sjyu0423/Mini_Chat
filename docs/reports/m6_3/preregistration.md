# M6.3 pre-registration: M5 (14.0M) vs M6.3 (29.7M SFT) — evaluation and promotion rules

Written before M6.3 training started and before any M6.3 output existed. Its SHA-256 is
recorded in the M6.3 report; the evaluation script implements exactly these rules.

## Models

* **Incumbent (M5):** `runs/m5_sft_chat-20261114-175334/checkpoints/best.pt` (SHA-256 of the
  file recorded at evaluation time), currently served by the chat app.
* **Candidate (M6.3):** the M5 SFT recipe applied to the M6.2 base model
  `runs/m6_2_general_m_bpe8192-20261115-001655/checkpoints/step_00009600.pt`
  (SHA-256 `5312a61336bfef44c7a6ebea81871dc49be2c5c5215c26a8f56d1cd88acd7a00`), best checkpoint
  by assistant-token validation loss on the M5 validation split (the same selection rule M5 used).

## Evaluation data and decoding

* Frozen suites: `evals/chat_v1.json` (40 items, SHA `c5b264d3…`), `evals/chat_diag_v1.json`
  (50 items, SHA `c5e37648…`), `evals/general_v1.json` (SHA `f2265af3…`).
* Production decoding only, identical for both models: temperature 0.7, top-k 40, repetition
  penalty 1.2, max 200 new tokens, seeds 11, 22, 33 (`scripts/m6_1_decoding.py`, setting
  `t0.7_k40_rp1.2`). Both models are re-generated in the same session. No decoding tuning.
* General-language retention: `scripts/evaluate_general.py` with the M5 held-out sets
  (FineWeb-Edu val, Cosmopedia v2 val, Wikipedia OOD, TinyStories val; 400 documents each).

## Unit of analysis

The **prompt (item)** is the unit, not the generation. For each item and model, the item score
is the fraction of its 3 seeds that pass (0, 1/3, 2/3, 1). Paired comparisons use per-item
differences (candidate − incumbent). Three samples of one prompt are never treated as three
independent observations.

## Metrics

| # | metric | definition |
|---|---|---|
| P | **primary: strict content success** | reply ended with `<|end|>` AND passed every check of its item (`both_ok`), over all checked items: chat_v1 33 + diag 42 = **75 items** |
| 1 | factual correctness | primary metric restricted to chat_v1 `qa` + diag `fact` (27 items) |
| 2 | instruction following | chat_v1 `instruction` + diag `instruction` (22 items) |
| 3 | entity tracking | diag `entity` (10 items) |
| 4 | multi-turn context | chat_v1 `multiturn` + diag `multiturn` (16 items) |
| 5 | natural end-of-turn | share of all 270 replies per model ending with `<|end|>` (and per-item paired) |
| 6 | repetition | share of 270 replies with > 30% repeated 4-grams |
| 7 | response length | median words over 270 replies |
| 8 | latency | mean and median ms per reply, same machine and session |
| 9 | retention | held-out bits/byte on the four sets above, and general_v1 MC/grammar/code |

Statistics for P and 1–4: mean per-item score for each model; mean paired difference with a 95%
paired bootstrap interval over items (10,000 resamples, seed 0); exact two-sided sign test on the
items whose scores differ.

## Blind review

For a fixed set of **20 items** — the first 4 items of each of: chat_v1 `qa`, diag `fact`,
diag `instruction`, diag `entity`, diag `multiturn` — the seed-11 replies of both models are
written to an anonymized packet in random A/B order (random seed 2026). The mapping is written to a
separate key file that is not opened until all grades are recorded. Each reply is graded:

* **correct/useful** — answers the request correctly (format AND content);
* **fluent but wrong** — well-formed, readable, but incorrect or off-task (counts as failure);
* **malformed** — broken format, nonsense, loop or truncated (failure).

Plus a preference per item: A, B, or tie (tie when both are equally right or equally wrong). The
reviewer is a single rater (Claude); blinding is procedural.

## Promotion rule (all must hold)

1. **Credible content improvement:** primary metric P — mean paired difference > 0, 95% bootstrap
   interval lower bound > 0, and sign-test p < 0.05.
2. **Practically meaningful:** the candidate's P exceeds M5's by at least **5 percentage points**
   (absolute, item-mean scale).
3. **Confirmed by reading:** in the blind review the candidate has strictly more
   "correct/useful" replies than M5, and more preferences than M5.
4. **No serious regression:**
   * natural end-of-turn ≥ 95% and not more than 2 points below M5;
   * repetitive replies ≤ 5%;
   * no crashes, NaNs or memory-guard aborts in training or evaluation;
   * mean latency ≤ 2× M5's and median latency < 1 s;
   * FineWeb-Edu held-out bits/byte not worse than M5's by more than 0.02.

If any condition fails, **M5 remains the default** and M6.3 is kept as a research artifact. The
served checkpoint is not changed in M6.3 in any case; a promotion is a recommendation that needs
explicit approval.

## Fixed training recipe

M5's SFT recipe unchanged (data `data/processed/chat_m5`, splits, serialization, assistant-only
mask, AdamW β 0.9/0.95, wd 0.1, lr 1e-4 warmup 100 cosine to 1e-5, batch 16, 3 epochs, clip 1.0,
seed 1337, padding to 64, MPS cache release every 25 steps), fresh optimizer, initialized from
the M6.2 checkpoint above. No length cap, no pretraining-data mixing, one run.

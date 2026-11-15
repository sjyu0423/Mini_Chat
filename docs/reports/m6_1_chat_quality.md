# M6.1 — Conversational quality improvement

**Status:** complete, stopped for review. Nothing committed; no 30M training; M7 not started.

**Outcome in one paragraph.** Decoding was tested on 8 settings with fixed prompts and seeds.
The current default (temperature 0.7, top-k 40, repetition penalty 1.2) is still the best by a
pre-registered rule, so the deployed configuration is unchanged. The SFT audit found one concrete,
fixable imbalance: replies over 100 words are 25% of replies but carry 61% of the training signal.
One controlled SFT run removed them. It made replies **end more reliably** (99.6% vs 95.2%, paired
p = 0.002), **shorter** (median 21 vs 31 words) and **2× faster**. It did **not** improve factual
accuracy, instruction following, entity tracking or multi-turn consistency once outputs were read
rather than string-matched, and it cost slightly more general-text ability. **M5 stays deployed.**
The remaining failures are knowledge and reasoning failures of a 14M-parameter model, which no
decoding or SFT change addressed.

## 1. Diagnosis of M5 failures

Source: M5 best checkpoint, default decoding, the frozen `chat_v1` (40 items) plus the new frozen
`chat_diag_v1` (50 items, §4), 3 seeds, 270 replies (`docs/reports/m6_1/decoding_m5.json`).

| failure class | measured (default decoding) | plausible lever |
|---|---|---|
| Factual hallucination | 71/81 factual replies fail (e.g. "The capital of France is Vienna.") | **pretraining/scale** (knowledge is not in the model; SFT only formats it) |
| Simple instruction failure | 55/66 fail ("Answer only yes or no" → "The following are no, unseen.") | SFT data (format examples) + scale |
| Entity / name tracking | 23/30 fail; names become things ("Priya is a French country") | scale/pretraining; multi-turn SFT data |
| History confusion | 35/48 multi-turn items fail; 7/78 copy the previous assistant turn verbatim ("And what is the capital of Italy?" → "The capital of Japan is Tokyo.") | multi-turn SFT data + scale |
| Repetition | 5/270 (1.9%) with penalty 1.2; 27% without | **decoding** (solved in M5) |
| Failed end-of-turn | 13/270 (4.8%) | decoding + SFT (length) |
| Overly long / irrelevant | 57/270 over 80 words; 79/270 restate the question | **SFT data length** |

Decoding can fix repetition and termination, and SFT data can change length and format. Neither
gives the model facts it does not have. M4.1 already showed this, with general-knowledge
multiple-choice at chance level.

## 2. Decoding experiments (M5 checkpoint, unchanged)

`scripts/m6_1_decoding.py` ran 90 prompts × 3 seeds (greedy: 1 pass) with max 200 tokens. Top-p
(nucleus) sampling was added to `chat/generate.py` (`apply_top_p`, tested) and to `scripts/chat.py`
(`--top-p`).

**Selection rule, written into the script before any output existed:**
1. A setting is eligible if natural end-of-turn is at least 95% and repetitive replies are at most
   5%.
2. Eligible settings are ranked by `both_ok`: the reply terminated and passed its content checks.
3. The default changes only if the winner's 95% interval clears the default's.

| setting | natural end | repetitive | both_ok (225 checked) | median words | mean latency | eligible |
|---|---|---|---|---|---|---|
| greedy | 53% | 62% | 8/75 (11%) | 46 | 240 ms | no |
| greedy + pen. 1.2 | 90% | 24% | 16/75 (21%) | 23 | 125 ms | no |
| T 0.7, k 40 | 79% | 27% | 25 (11%) | 35 | 200 ms | no |
| T 0.7, k 40, pen. 1.1 | 90% | 5.2% | 31 (14%) | 39 | 179 ms | no |
| **T 0.7, k 40, pen. 1.2 (default)** | **95.2%** | **1.9%** | **41 (18%) [14, 24]** | 31 | 143 ms | **yes** |
| T 0.7, k 40, pen. 1.3 | 96.3% | 0.4% | 35 (16%) | 33 | 132 ms | yes |
| T 0.7, top-p 0.9, pen. 1.2 | 95.2% | 2.6% | 41 (18%) | 31 | 141 ms | yes |
| T 0.5, k 40, pen. 1.2 | 94.4% | 6.7% | 42 (19%) | 30 | 138 ms | no (repetition) |

There were no empty replies under any setting, and one structural-token stop in total.

**Result:** the default is the best eligible setting, so the deployed decoding is unchanged.
- **Penalty 1.3** removes almost all repetition but lowers content passes (35 vs 41).
- **Top-p 0.9** ties the default exactly, so it adds nothing. It stays available in the CLI.
- **Lower temperature (0.5)** brings repetition back.

## 3. SFT dataset audit (`scripts/m6_1_sft_audit.py` → `sft_audit.json`)

These are measurements over 17,914 assistant replies in the M5 training set.

| finding | evidence | action |
|---|---|---|
| **Long replies dominate the loss** | Replies over 100 words are 24.9% of replies but **60.8% of target tokens**. OASST2 median reply is 112 words, with 23.5% over 200 words. | **tested (§5)** |
| Context-copying tasks | Dolly closed QA, information extraction, summarization and classification copy 79–89% of reply words from the user turn | documented, not changed |
| Restating the question is normal in the data | 39% of replies restate the question ("The capital of France is Paris.") | not a defect; M5's circular answers are missing knowledge |
| Persona and refusal boilerplate | OASST2: "as an AI…" 2.1%, refusals 2.3%, "sorry" 1.6%, URLs 2.2%, code 5.8% | documented, not changed (small) |
| Near-duplicate prompts | 892 MinHash pairs (J ≥ 0.8), mostly templates ("five creative ways to describe a broken *X*") or different questions over the same passage | left in: exact duplicates were already removed |
| Truncation effects (M5 policy) | 1,009 OASST2 conversations cut to earlier turns; 2,454 conversations dropped by the 512-token limit | none |
| End-token balance | `<|end|>` is 0.94% of target tokens (1 per 106 targets) | none |

Sentence-level correctness of training answers cannot be verified at scale without an external
model, which was not used. Reading a fixed-seed sample of 40 conversations (in the audit JSON)
found no systematically wrong answers.

## 4. New frozen diagnostic set `evals/chat_diag_v1.json`

The set was written and hashed (`c5e37648…`, enforced by a test) before any M6.1 output was
generated. It has 50 items:
- 12 strict facts: the right answer must be present, common wrong answers absent, and the reply
  concise;
- 12 format instructions;
- 10 entity- and name-tracking items;
- 8 topic-switch multi-turn items;
- 8 open prompts.

One item's opening question (dm01, "What is the capital of Japan?") also occurs in M5's training
data. The new SFT data was decontaminated against the whole set (1 conversation removed).

## 5. Controlled SFT experiment: replies ≤ 100 words

**Change:** exactly one data rule, in `scripts/prepare_chat_m6_1.py`. Starting from the M5
conversations, a conversation is cut before its first assistant reply over 100 words, and dropped
if that is the first reply. Everything else is identical to M5:
- init from M4.1 (not the M5 checkpoint), with a fresh AdamW optimizer;
- learning rate 1e-4 (cosine), batch 16, 3 epochs, seed 1337;
- the same architecture and tokenizer.

The config `configs/sft/m6_1_sft_cap100.toml` differs from M5's only in name, description and data
directory. The preflight passed.

| | M5 | M6.1 cap100 |
|---|---|---|
| train conversations / target tokens | 16,813 / 2.01M | 12,648 / 0.77M |
| OASST2 conversations / multi-turn conversations | 3,970 / 2,034 | 1,552 / 485 |
| steps / wall-clock | 3,153 / 8.0 min | 2,373 / **4.6 min** |
| best step | 2,750 | 1,500 |

**Confound, inherent to the rule:** most long replies are OASST2's, so the cap also removes 76% of
the multi-turn conversations.

### Results

The settings were the same 90 prompts, the same seeds and the default decoding. Comparisons are
paired McNemar tests on the identical prompt/seed pairs.

| metric | M5 | M6.1 cap100 | paired test |
|---|---|---|---|
| natural end-of-turn (270) | 257 (95.2%) | **269 (99.6%)** | 1 vs 13 discordant, **p = 0.002** |
| repetitive replies (270) | 5 | 4 | p = 1.0 |
| both_ok (225) | 41 (18%) | 52 (23%) | 19 vs 30, p = 0.15 (n.s.) |
| median words / mean latency | 31 / 143 ms | **21 / 76 ms** | — |
| facts: chat_v1 qa + diag fact (81) | 10 | 9 | n.s. |
| instructions: chat_v1 + diag (66) | 11 | 13 | n.s. |
| entity (30) | 7 | 8 | n.s. |
| multi-turn, chat_v1 (24) | 11 | 18 | 1 vs 10, p = 0.012* |
| multi-turn, diag (24) | 2 | 4 | n.s. |

\*Reading the replies shows the chat_v1 multi-turn gain is mostly check artifacts:
- mt01 passes with "My favorite color is red. My favorite color is blue…";
- mt06 passes because a stray "2" appears in digit noise.

Both models give the *same* wrong answers on the same seeds: "The capital of France is Vienna.",
and "The capital of Japan is Tokyo." when asked about Italy.

**Validation loss and general text:**

| | M5 | M6.1 cap100 | M4.1 base |
|---|---|---|---|
| assistant loss, M5 validation split | **3.085** | 3.232 | 3.438 |
| assistant loss, capped validation split | **2.903** | 2.966 | 3.563 |
| FineWeb-Edu bits/byte | **1.476** | 1.526 | 1.335 |
| Wikipedia (OOD) bits/byte | **1.779** | 1.824 | 1.660 |
| facts MC / QA MC / grammar (40/30/30) | 20 / 15 / 29 | 19 / 13 / 27 | 22 / 12 / 30 |
| frozen M4 chat section, format (12) | 6 | 7 | 0 |

M5 has lower loss even on the short-reply validation split: more data helps loss. The capped model
is still better at stopping. This is a direct instance of the brief's warning that validation loss
is not the measure of chat behaviour. The cap also forgot slightly more general text (+0.05
bits/byte).

### Decision

The capped model does not improve the M6.1 targets: repetition, instruction following and
consistency. Its real gains are termination, length and speed. It has a small general-text
regression and no visible content gain in the actual text. **This is not a defensible replacement,
so M5 stays the deployed checkpoint.** The capped checkpoint is kept, documented, and usable with
`scripts/serve_chat.py --checkpoint runs/m6_1_sft_cap100-20261114-185455/checkpoints/best.pt`.

## 6. Before and after outputs (unedited)

Selection is fixed: five M5 transcript items plus the first item of each diagnostic category, at
seed 11 with default decoding and with greedy. Copied by `scripts/m6_1_transcripts.py`; the full
set is in [`m6_1/transcripts.md`](m6_1/transcripts.md). A sample at default decoding, seed 11:

| prompt | M5 | M6.1 cap100 |
|---|---|---|
| What is the largest planet in our solar system? | The largest planet in our solar system is Mercury | The largest planet in our solar system is Mercury |
| *(after "My sister is called Ana and my brother is called Leo.")* What is my brother's name? | The name "Ana" is a word used for wording from the word "le." It refers to a phrase that describes a person's behavior, … | The Emperor is named Leo. |
| Is fire hot? Answer only yes or no. | The following are no, unseen. | No, it is no reason to answer. |
| *(after the Japan exchange)* And what is the capital of Italy? | The capital of Japan is Tokyo. | The capital of Japan is Tokyo. |
| Tell me about the moon. | The moon is a mundane, with no other night skyings. The sun is light orbiting around the Earth's atmosphere. … | The moon is a mundane, known as the Bullmite. The sun is a brickbone that has a carnivores and telescopes. … |

## 7. Integration

- **Deployed checkpoint:** unchanged (M5 best).
- **Deployed decoding:** unchanged. It was confirmed by the measured grid:
  `GenerationDefaults` (T 0.7, k 40, penalty 1.2, 200 tokens).
- **New and reproducible:**
  - top-p sampling (`chat_reply(top_p=…)`, `scripts/chat.py --top-p`, `--top-k 0` to disable
    top-k);
  - the decoding grid script, audit script and capped-data pipeline;
  - the diagnostic suite.
- **Frontend:** unaffected; its tests and build pass, and the local server smoke test passes (§9).

## 8. Remaining limitations

- Facts, entities and instructions remain mostly wrong. The bottleneck is model knowledge and
  capacity, not decoding or SFT formatting.
- Copying the previous answer in multi-turn chats persists in both models.
- The evaluation is small: 90 prompts, string checks that overcount (illustrated in §5), and a
  single reader of outputs.
- The SFT experiment had one seed; the multi-turn data loss is a confound.

## 9. Verification

These checks were run at the end of M6.1:
- pytest: 302 passed;
- ruff, ruff format and mypy: clean;
- frontend: 15 tests, `tsc`, ESLint and the production build all clean;
- the M5 and M6.1 checkpoints load with the right fingerprint;
- a real server smoke test: health ready, a chat reply ended naturally;
- the M4.1 and M5 checkpoints and all earlier artifacts are byte-identical;
- `website/` is unchanged.

## 10. Recommendation for M6.2 (30M model)

Every M6.1 measurement points at **knowledge and capacity**, so M6.2 should spend its budget on
pretraining, not SFT tricks:

1. **A 30M model (8L/8H/512) with more pretraining tokens.** M3.3 showed 2.2× parameters improve loss
   at equal tokens. Keep the single-pass data order and the M4.1 pipeline with more shards (the
   FineWeb-Edu and Cosmopedia quotas have unused capacity). Expect about 2.5–4 h on this Mac. Keep
   the 512 context.
2. **Re-use the M5 SFT recipe as-is** (all conversations, 3 epochs), so the scale effect is measured
   alone. Run the length cap as a second, separately reported variant. It reliably improves stopping
   and speed, and may be worth adopting at larger scale.
3. **Mix about 10–20% pretraining text into SFT** to limit the 7–16% general-text forgetting seen in
   M5 and M6.1. Measure this as its own variable.
4. **Evaluate with the same frozen suites, prompts, seeds and paired tests** used here, plus a strict
   human read of facts and entity items. Decide deployment by pre-registered criteria.

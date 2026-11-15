# M4 report: general-purpose pretraining research

Status: research, dataset engineering and bounded benchmarking only. Nothing was trained.
No full dataset was downloaded. Nothing was committed or pushed.

## Summary

- **Data.** The recommended pretraining mix for M4.1 is **FineWeb-Edu (70%) plus
  Cosmopedia v2 (30%)**.
  - Both are English, permissively licensed (ODC-By), well documented, and measured
    clean in our samples.
  - We need one shard of each, 1.7 GB of downloads at pinned revisions.
  - Wikipedia is kept for held-out evaluation only.
  - For future post-training, the cleanest-licensed small conversational data is
    OASST2 (human-written, Apache 2.0) and Dolly-15k (human-written, CC BY-SA 3.0).
    The SmolTalk family is the best fit for small models, but parts of it are
    Llama 3.1 outputs, which carry naming and attribution conditions.
- **Tokenizer.** Use **8,192 tokens**.
  - On the proposed 70/30 mix it compresses text 13.7% better than 4,096 (3.94 vs
    3.46 bytes/token).
  - Because each token carries more text, models see about the same amount of text
    per second at either size (S model) or 7.5% more (M model), despite slower steps.
  - The cost is 1.6M extra embedding parameters for the small model.
  - The historical TinyStories tokenizer gets 31% fewer bytes per token on this mix
    and must not be reused for general data.
- **Evaluation.** A frozen, hand-written suite (`evals/general_v1.json`, 202 items)
  separates two tracks:
  - next-token tests: held-out bits/byte, knowledge and QA multiple choice,
    grammar minimal pairs, code, generation and repetition metrics;
  - instruction tests: 12 chat conversations, which a base model is not expected
    to pass.

  The M3 TinyStories models score at chance on facts (7/40 and 10/40, chance 10/40)
  and reach 3.05–3.12 bits/byte on web text, which sets the floor.
- **Compute.** MPS sustains 2.1–2.8 TFLOP/s at every size from 12M to 92M
  parameters, so the budget is fixed in FLOPs. In two hours:

  | model | training tokens | tokens per parameter |
  |---|---|---|
  | 14M | 174M | 12.4 |
  | 30M | 93M | 3.1 |
  | 55M | 54M | 1.0 |
  | 92M | 35M | 0.4 |

  The commonly cited compute-optimal ratio is about 20 tokens per parameter, so a
  14M model is the right size for a 1–2 hour run. 50M and 100M models need 40 h
  and 106 h to reach that ratio.
- **Proposed M4.1** (not started; needs approval and one new dependency):

  | | |
  |---|---|
  | model | 6L/6H/384, **13,989,888 parameters** |
  | context | 512 |
  | tokenizer | BPE 8,192 with chat special tokens reserved |
  | training | 9,600 steps × 16,384 tokens = **157.3M tokens**, single pass |
  | time | about **2 h** |
  | compute | 1.43e16 FLOPs, about the same as M3.3 |

  The acceptance criteria separate engineering checks, which must pass, from
  research outcomes, which are reported either way.

---

## 0. Git and repository safety

| item | state |
|---|---|
| branch | `milestone-3-dataset-research` |
| HEAD | `8d8d7c0` (also the tip of `milestone-2-pretraining`; not pushed) |
| staged | nothing |
| uncommitted | all of M3–M3.3: 14 modified tracked files, plus untracked configs, scripts, modules, tests and reports |
| `website/` | untracked; not read, modified or staged |
| stashes / other worktrees | none |

**Branching limitation.** M4 builds on uncommitted M3 code (the BPE tokenizer and
`load_prepared`).
- A separate worktree or a clean branch would not contain that code.
- Switching branches carries the uncommitted files along, so it would give no
  isolation either.

I therefore worked in place on the current branch and kept M4 separable instead:
- almost all M4 work is in **new files**;
- the single exception is a backward-compatible extension of the untracked M3
  script `scripts/fetch_samples.py`. It keeps chat/metadata fields and records the
  dataset revision; earlier behaviour and samples are unchanged.

I did not edit README, `docs/experiments.md` or `docs/training_plan.md`, because
they already carry uncommitted M3 changes.

**Integrity.** I took SHA-256 hashes of all 345 files under `runs/`, `data/`,
`docs/reports/` and `configs/` before starting and rechecked them at the end:
- **every pre-existing file is byte-identical**;
- new files are only the new samples, the new report directory, the new config and
  the new `runs/m4_*` logs and tokenizers (§7).

---

## 1. Dataset research

### 1.1 Candidates

Sizes and revisions come from the Hugging Face Hub API on 2026-11-14. Licenses come
from the dataset cards. **These are summaries of the license texts, not legal
advice.** A public download does not mean training and redistribution are
unrestricted.

**Pretraining candidates:**

| dataset (revision) | content | license | size | bounded option | fit for 27–100M models |
|---|---|---|---|---|---|
| **FineWeb-Edu** `HuggingFaceFW/fineweb-edu` (`87f0914`, v1.4.0) | Common Crawl 2013–2025, filtered by a classifier trained on Llama-3-70B "educational value" scores (score ≥ 3 keeps 8%). English, fastText ≥ 0.65. | **ODC-By 1.0**, plus Common Crawl Terms of Use | 1.3T tokens. `sample-10BT`: 9.67M docs in 14 parquet shards (28.5 GB). | **one shard**, `013_00000.parquet`, 0.54 GB | **Best real-world knowledge per token.** Documents are long (median 710 tokens at V=8192). |
| **Cosmopedia v2** in `HuggingFaceTB/smollm-corpus` (`3ba9d60`) | Synthetic textbooks, stories and WikiHow-style articles written by Mixtral-8x7B-Instruct, seeded with web pages | **ODC-By** (corpus card). Mixtral is Apache 2.0, with no output-use restrictions. | 39.1M docs, 104 shards of 1.17 GB (122 GB) | **one shard**, 1.17 GB (we need about 15% of it) | **Good.** Clean, simple, explanatory prose built for small models. It is synthetic and comes from a single generator, so its style is narrow. |
| Cosmopedia v1 `HuggingFaceTB/cosmopedia` (`0ae6ec6`) | Same method (v1). The card documents 10-gram decontamination against MMLU, HellaSwag, ARC and others. | Apache 2.0 | 25B tokens, 92 GB | shards of 49–282 MB | Superseded by v2; usable as an alternative. |
| Wikipedia `wikimedia/wikipedia` 20231101 (`b04c8d1`) | Encyclopedia text | **CC BY-SA 3.0 / GFDL** (share-alike) | en: 6.4M articles, 11.6 GB. simple: 242k, 0.16 GB. | per-language parquet | High factual density. Share-alike makes redistribution of derived data more constrained, and many stubs are templated (§1.2). **Held-out evaluation only.** |
| FineMath `HuggingFaceTB/finemath` (`e92b25a`) | Math web pages | ODC-By + Common Crawl ToU | 34B tokens | shards | Too specialized for a first general run. |
| Python-Edu in smollm-corpus | Educational Python files from The Stack v2 | per-file licenses (see the-stack-v2) | 7.7M files | **only file IDs are on the Hub**; contents come from Software Heritage | Not practical. Not reproducible without a second source. |
| TinyStories V2 (current) | Synthetic children's stories | CDLA-Sharing-1.0; OpenAI outputs | 2.2 GB | already local | Narrow domain. Kept as a regression held-out set. |
| SlimPajama, Dolma, C4 | Large web mixtures | various | 0.6–3T tokens | — | Too large and too noisy for this scale. SlimPajama metadata could not be read without signing in, so it was not evaluated further. |

**Instruction and chat data (future post-training; not used in M4.1):**

| dataset (revision) | origin | license | size | notes |
|---|---|---|---|---|
| **OASST2** `OpenAssistant/oasst2` (`179dd21`) | human-written conversation trees | **Apache 2.0** | 135k messages; multilingual, filter to English | Cleanest licensing. Quality varies. |
| **Dolly-15k** `databricks/databricks-dolly-15k` (`bdd27f4`) | written by Databricks employees | **CC BY-SA 3.0** | 15k records | Human-written, commercial use allowed, share-alike on the data. Single turn. |
| **smol-smoltalk** `HuggingFaceTB/smol-smoltalk` (`f73fe85`) | mix built for models under 1B (SmolLM2-135M/360M) | Apache 2.0 for HF's own parts; the card says to check each upstream dataset | 485k conversations, 0.97 GB | Best fit for small models. Our sample (§1.2): 60% Smol-Magpie-Ultra (**Llama-3.1-405B outputs**), 11% OpenHermes-50k (upstream license not verified), 42% contain code fences, median assistant reply 1,305 characters (long for a 14–30M model). |
| everyday-conversations `HuggingFaceTB/everyday-conversations-llama3.1-2k` (`14f5432`) | 2.2k short multi-turn chats | Apache 2.0, but **Llama-3.1-70B outputs** | 2.3k, 2 MB | Ideal shape for a tiny model: about 8 messages, median reply 136 characters. |
| Alpaca `tatsu-lab/alpaca` | text-davinci-003 outputs | **CC BY-NC 4.0** | 52k | **Excluded** (non-commercial). |
| Tulu 3 SFT mixture `allenai/tulu-3-sft-mixture` | mixture | ODC-By; "some portions … are non-commercial" (card) | about 940k | **Excluded** for now (mixed licenses; too large). |

**The Llama 3.1 license condition.** Llama 3.1 license §1.b.i says that if a
distributed model was trained with Llama outputs, you "shall also include “Llama” at
the beginning of any such AI model name", and must show "Built with Llama". This
matters only if the post-trained model is distributed. It is a decision for you
before any post-training on SmolTalk or everyday-conversations data.

### 1.2 Bounded samples and measurements

All samples were fetched through the dataset-viewer `/rows` API with seeded offsets
(`scripts/fetch_samples.py`). Each sample has a `manifest.json` with every request,
the dataset revision at fetch time, byte counts and the file's SHA-256. All are
git-ignored under `data/raw/samples/`. The FineWeb-Edu and TinyStories samples are
reused from M3.

| sample | revision | docs | MB | mean / median chars | distinct words | non-ASCII | exact / near-dup pairs (docs) | dup pairs across train/val split | flagged by ≥ 1 heuristic quality rule |
|---|---|---|---|---|---|---|---|---|---|
| FineWeb-Edu `sample-10BT` | (M3) | 4,000 | 19.0 | 4,718 / 2,870 | 83,351 | 0.27% | 0 / 0 | 0 | 14.4% |
| Cosmopedia v2 | `3ba9d60` | 3,000 | 11.2 | 3,719 / 3,420 | 42,787 | 0.05% | 0 / 0 | 0 | 30.8% (mostly the stop-word rule, see below) |
| Wikipedia en | `b04c8d1` | 2,500 | 6.3 | 2,524 / 1,278 | 65,855 | 0.25% | 0 / 10 (16) | 0 | 86% |
| Wikipedia simple | `b04c8d1` | 2,000 | 4.5 | 2,203 / 475 | 29,987 | 0.93% | 2 / 496 (77) | **83** | 87% |
| TinyStories | (M3) | 6,000 | 5.3 | 881 / 790 | 8,511 | 0.13% | 0 / 0 | 0 | 1.9% |
| smol-smoltalk | `f73fe85` | 1,000 conv. | 3.9 | 4.7 messages per conversation | — | — | — | — | 0.1% non-Latin |
| everyday-conversations | `14f5432` | 405 conv. | 0.3 | 7.7 messages per conversation | — | — | — | — | 0% non-Latin |

**How to read these numbers.**

- **Near-duplicates** use MinHash, Jaccard ≥ 0.8, as in M3.
- **Quality rules** (`scripts/m4_data_analysis.py`) are Gopher/C4-style heuristics,
  computed for reporting only:
  - fewer than 50 words;
  - mean word length outside 3–10;
  - alphabetic-word share below 0.8;
  - fewer than half of lines ending in punctuation;
  - duplicate-line share above 0.3;
  - markup share above 5%;
  - stop-word rate below 0.15;
  - Latin-letter share below 0.95.

**Findings:**

1. **FineWeb-Edu is clean and diverse.**
   - Topics: 3,620 domains in 4,000 documents; the top domains are Wikipedia (24),
     reference.com (16) and Britannica (12).
   - Classifier `int_score`: 87% at 3, 13% at 4.
   - Most of its 14% flag rate comes from lists and headings (lines without final
     punctuation, 13.8%).
   - 5% of documents contain boilerplate terms ("cookie", "subscribe", …).
   - The `language_score` minimum is 0.657 (card threshold 0.65), so it is English
     only.
2. **Cosmopedia v2 is the cleanest text**:
   - no markup;
   - 0.05% non-ASCII;
   - every document is complete.

   Its 31% flag rate is an artifact of our crude stop-word rule. Fluent textbook
   prose scores just under 0.15; the first flagged document is 0.149, with the
   opening "Have you ever wondered about how different people lived…". The rule is
   **not usable as a filter** without calibration.

   Composition:
   - formats: 63% textbook, 11% unconditioned-topic textbook, 16% stories of
     several kinds, 5% WikiHow, 3% dialogue;
   - audiences: middle-school 44%, college 31%, general 12%, children 10%;
   - seeds: 78% FineWeb pages.
3. **Wikipedia is not usable as-is.**
   - Contiguous row windows land in runs of templated stubs.
   - Simple English: 35% of articles are under 50 words. The top words include
     "align", "bgcolor" and "socorro" (minor-planet tables), and there are 496
     near-duplicate pairs such as "X is a commune in the Ardèche département…".
   - **83 near-duplicate pairs straddle a document-hash train/validation split.**
     That is direct evidence that a naive split of templated sources leaks, and why
     M4.1 near-deduplicates before splitting.
4. **Cross-source verbatim overlap** (documents sharing any word 13-gram with
   another corpus) is at most 0.2%. In particular, Cosmopedia v2's `text` does not
   copy its FineWeb seeds (0.0%). Only its `prompt` column contains seed text, and
   M4.1 drops that column.
5. **Language.** All text corpora are English. Fewer than 0.33% of documents in any
   sample have under 95% Latin letters. OASST2 (not sampled) is multilingual and
   needs filtering.
6. **Chat data shape.**
   - smol-smoltalk: 78% multi-turn, 17% of conversations have a system message,
     median reply 1,305 characters, and 42% contain code.
   - everyday-conversations: replies are short (median 136 characters, p95 246),
     which matches what a 14–30M model can produce.

### 1.3 Contamination

**Checked:** the 202 suite items against every sample (`ngram_overlap`):
- a document is a hit if it shares a lower-cased word 8-gram with an item;
- items shorter than 8 words are matched whole.

**Hits:**
- FineWeb-Edu: 3 items;
- Cosmopedia: 8;
- Wikipedia: 1 each;
- TinyStories: 0;
- smol-smoltalk: 11;
- everyday-conversations: 2.

**All hits are generic.** They are the stress prompts "Chapter 1" and
"1, 2, 3, 4, 5," and common code idioms such as `if __name__ == "__main__":` and
`for n in numbers: print(n)`. In the chat data, one hit is "What is the capital of
France". **No fact, QA, grammar or explanation item appears verbatim.** Facts such
as "Paris is the capital of France" naturally occur in any general corpus. That is
knowledge, not leakage, and is what the suite is meant to measure.

**Benchmark contamination** (MMLU, ARC and others) in FineWeb-Edu is not addressed
by its card. Cosmopedia v1 documents decontamination; the v2 card in smollm-corpus
does not say. We do not use public benchmarks, so this mainly matters if they are
added later.

**Decontamination rule for M4.1** (pre-registered):
- drop training documents that share a word 8-gram with any suite item of 8 or more
  words;
- shorter items cannot be meaningfully decontaminated and are reported only.

### 1.4 Disk, download and reproducibility

| | download | text (est.) | tokens at V=8192 (est.) | needed for M4.1 |
|---|---|---|---|---|
| FineWeb-Edu `013_00000.parquet` | 540.6 MB | about 0.86 GB | about 225M | 110M train + val/holdout |
| Cosmopedia v2 `train-00000-of-00104.parquet` | 1,174 MB | about 1.4 GB (the parquet also stores long prompts) | about 320M | 47M train + val/holdout |
| processed `uint16` streams | — | — | 157M train + about 4M held out | about 0.33 GB |

**Reproducibility:**
- Files are addressed by **commit SHA** (`/resolve/<revision>/…`) and verified
  against the Hub's LFS SHA-256 (§5, manifest).
- HTTP range requests work (`accept-ranges: bytes`, redirected to the Xet CDN).
- Reading parquet requires **`pyarrow`, which is not installed**. Adding it as an
  optional `data` dependency is a decision for you; the alternatives are slower or
  bigger.
- The extrapolations from rows to tokens in this table are estimates from sample
  averages, not measurements.

### 1.5 Recommendation

**Pretraining: FineWeb-Edu 70% + Cosmopedia v2 30% by tokens.**
- FineWeb-Edu supplies real-world knowledge and diversity.
- Cosmopedia supplies clean, explanatory prose that small models learn from more
  easily.
- 70/30 is a documented starting point, not a tuned value. A ratio ablation is
  future work.

**Excluded from training:**
- Wikipedia: share-alike licensing and templated stubs; kept as an
  out-of-distribution held-out set;
- code: no reproducible source on the Hub;
- TinyStories: domain too narrow; kept as a regression set.

**For post-training** (future, needs a decision):
- start with human-written OASST2 (English) and Dolly-15k;
- add everyday-conversations or smol-smoltalk only if you accept the Llama 3.1
  naming and attribution terms for any distributed model.

---

## 2. Tokenizer research

**Protocol** (`scripts/m4_tokenizer_study.py`):
- Documents are deduplicated and split by `sha256(text)` (10% validation)
  **before** fitting.
- Candidates are trained **only on training documents** of the proposed mix:
  6,312 docs, 27.2 MB, digest `2c36ad8c…`.
- Five special tokens are reserved: `<|endoftext|>`, `<|user|>`, `<|assistant|>`,
  `<|system|>`, `<|end|>`. Reserving the chat tokens now means post-training will not
  need to resize the vocabulary.
- Metrics are on validation documents. Encoding speed is measured on a freshly
  loaded tokenizer.
- The historical tokenizer is read only. Its file and fingerprint (`906bf77c…`) are
  unchanged, and so are the M3 checkpoints.

**Candidates:**

| | fingerprint | fit time |
|---|---|---|
| 4,096 | `6d21ee2b7ac465de…` | 2.7 s |
| 8,192 | `1eb21d0c27556e1f…` | 3.2 s |

Both are saved under `runs/m4_tokenizers/` (git-ignored).

### 2.1 Compression (UTF-8 bytes per token; higher is better)

| domain (val) | **mix 4,096** | **mix 8,192** | gain | historical TinyStories 4,096 |
|---|---|---|---|---|
| FineWeb-Edu | 3.354 | **3.787** | +12.9% | 2.679 |
| Cosmopedia v2 | 3.746 | **4.346** | +16.0% | 2.836 |
| Wikipedia en (not in fit corpus) | 2.860 | 3.230 | +12.9% | 2.403 |
| Wikipedia simple | 2.685 | 3.055 | +13.8% | 2.184 |
| TinyStories | 3.163 | 3.526 | +11.5% | **3.869** |
| chat transcripts (SmolTalk) | 3.482 | 3.922 | +12.6% | 2.756 |
| fenced code | 2.170 | 2.362 | +8.8% | 1.563 |
| **70/30 mix** | 3.463 | **3.939** | **+13.7%** | 2.724 |

**Tokens per document at 8,192:**

| | median | p95 | longer than 512 tokens | longer than 1,024 |
|---|---|---|---|---|
| FineWeb-Edu | 710 | 3,606 | 64% | 34% |
| Cosmopedia | 781 | 1,521 | 89% | 25% |

Most documents therefore span more than one 512-token window (§4.3). Code compresses
poorly with both candidates (2.2–2.4), because there is no code in the fit corpus.

### 2.2 Frequency distribution and rare tokens

| | 4,096 | 8,192 | historical |
|---|---|---|---|
| token types covering 50% / 90% / 99% of mix-val tokens | 202 / 2,099 / 3,535 | 261 / 3,737 / 6,921 | 102 / 858 / 2,269 |
| Zipf slope (ranks 10–1000) | −0.86 | −0.85 | −1.08 |
| merged tokens never used in the 27 MB fit corpus | 6 | 23 | 29 |
| used fewer than 10× / fewer than 100× in the fit corpus | 11 / 83 | 72 / 587 | 265 / 1,279 |
| merged tokens unused on mix validation | 16 | 100 | 296 |
| median fit-corpus count of the last 500 merges | 433 | 148 | 82 |
| tokens that are single non-ASCII bytes (FineWeb-Edu val) | 0.48% | 0.41% | 0.55% |
| exact round trip on every domain | yes | yes | yes |
| encode speed (cold, FineWeb-Edu / Cosmopedia) | 12.7 / 12.5 MB/s | 12.3 / 12.1 MB/s | 13.2 / 13.0 MB/s |

**Rare-token risk at 8,192.** 587 tokens appear fewer than 100 times in 27 MB of text.
In 157M training tokens (about 620 MB), each is expected about 23× more often, so
nearly all embeddings get real gradient. That is the main difference from M3, where
an 8,192 vocabulary on TinyStories left 48% of merges under 10 uses.

**Fit-size sensitivity** (8,192 trained on 25 / 50 / 100% of the fit documents):
mix-val bytes/token is 3.949 / 3.965 / 3.970. Quadrupling the fit corpus gains
0.5%, so the planned fit on about 100 MB of M4.1 training text is ample.

### 2.3 Vocabulary overhead (tied embeddings, context 512)

| model | V = 4,096 | V = 8,192 | embedding share at 8,192 |
|---|---|---|---|
| S 6L/384 | 12,417,024 | 13,989,888 (+1.57M) | 22.5% |
| M 8L/512 | 27,579,392 | 29,676,544 (+2.10M) | 14.1% |
| X 10L/640 | 52,185,600 | 54,807,040 (+2.62M) | 9.6% |
| L 12L/768 | 88,594,944 | 91,740,672 (+3.15M) | 6.9% |

**What matters at equal compute is text per second, not tokens per second.**
Measured train throughput (§4) × mix bytes/token:

| | V = 4,096 | V = 8,192 | change |
|---|---|---|---|
| S | 27,049 tok/s × 3.463 = 93.7 KB/s | 24,135 × 3.939 = **95.1 KB/s** | +1.5% |
| M | 13,640 × 3.463 = 47.2 KB/s | 12,897 × 3.939 = **50.8 KB/s** | +7.5% |

A 512-token window also covers 2.0 KB at 8,192 instead of 1.8 KB.

**Recommendation: 8,192.** It costs nothing in text throughput, gives more context
per window and makes rare-token starvation negligible at our data size. Its
embedding parameters are the price: 22% of the S model's parameters.

---

## 3. General-language benchmark design

### 3.1 Suite `evals/general_v1.json`

- **Frozen.** SHA-256 `f2265af3e47ea291be6eee533e7c0b241e8282c23763e57fe2df063faffd8365`,
  pinned by `tests/test_general_eval.py::test_suite_v1_is_frozen`.
- **Original.** All items were written by hand for this project; none is copied
  from a public benchmark.
- **Scoring** is in `src/mini_chatgpt/analysis/general_eval.py`.
- **Document start.** Every prompt starts after `<|endoftext|>`, as documents do in
  training.

**Next-token track.** This is what a base model can be measured on, with no
instruction following assumed.

| section | items | scoring | chance |
|---|---|---|---|
| `lm_heldout` | data | bits per UTF-8 byte (tokenizer-independent) on held-out FineWeb-Edu, Cosmopedia, Wikipedia-en and TinyStories documents | — |
| 1. `continuation` | 20 | 3 seeded samples each (T = 0.7, top-k 40, 200 tokens). Automatic surface metrics now; blind human rubric later. | — |
| 2. `facts_mc` | 40 | cloze multiple choice ("The capital of France is" → Paris / London / Rome / Madrid). Primary score: log-likelihood per byte (`acc_norm`); raw sum also reported. Answer positions are balanced (10 per slot). | 25% |
| 3. `qa_mc` | 30 | `Question: …\nAnswer:` multiple choice | 25% |
| 4. `explanations` | 12 | generated answers; human rubric against key points and listed misconceptions (e.g. seasons ≠ distance from the Sun) | — |
| 5. `code_mc` | 20 | Python/JS syntax and semantics multiple choice | 25% |
| 5. `code_generation` | 10 | greedy exact-prefix completion (`def square(x): return x *` → ` x`) | ≈ 0 |
| `grammar_pairs` | 30 | minimal pairs; the full-sentence log-probability of the grammatical version must be higher | 50% |
| 7. `repetition_stress` | 8 | loop-inducing prompts plus an unconditional start; surface metrics only | — |

**Surface metrics:**
- distinct-1 and distinct-2;
- **repeated 4-gram fraction**, with the share of samples above 0.3;
- maximum line repeats;
- ended with `<|endoftext|>`;
- empty outputs;
- U+FFFD replacement characters.

**Instruction track.** This is kept separate and expected to fail before
post-training.

| section | items | scoring |
|---|---|---|
| 6. `chat` | 12 multi-turn conversations | Template `<|user|>\n…<|end|>\n<|assistant|>\n…`. Greedy reply of up to 128 tokens. **Format** check: ends with `<|end|>` and never opens a new role. **Content** checks: `include_any` keywords and `max_sentences`. Includes memory across turns ("My name is Lena … What is my name?") and arithmetic follow-ups. |

For a tokenizer without the chat specials, the section is reported as *not
applicable* rather than scored. Message text is always encoded as data, so a user
cannot inject `<|end|>` (tested).

**Leak prevention:**
- the suite is versioned and hash-pinned, and never used for tuning;
- the 8-gram overlap check (§1.3) runs against the full M4.1 training stream at
  data preparation, with the decontamination rule above;
- held-out bits/byte uses document-hash validation splits that never entered
  tokenizer fitting.

**Statistics:**
- accuracies are reported with Wilson 95% intervals;
- model comparisons use the paired items (as in M3.2/M3.3);
- with 30–40 items, only large differences are detectable (±15 percentage points),
  which is acknowledged in advance.

### 3.2 Harness check on existing checkpoints (evaluation only)

`scripts/evaluate_general.py` was run on the M3.1 and M3.3 final checkpoints.
- Held-out sets: 150 validation-split documents per domain.
- Run time: 61 s and 99 s on MPS.
- Output: `docs/reports/m4/general_eval_baseline_*.json`.

| | M3.1 (12.4M, TinyStories) | M3.3 (27.6M, TinyStories) | chance |
|---|---|---|---|
| bits/byte FineWeb-Edu | 3.118 | 3.055 | — |
| bits/byte Cosmopedia v2 | 2.971 | 2.905 | — |
| bits/byte Wikipedia en | 3.869 | 3.798 | — |
| bits/byte TinyStories (original mixture sample) | 0.772 | 0.740 | — |
| facts (acc_norm) | 7/40 = 17.5% [9, 32] | 10/40 = 25% [14, 40] | 25% |
| QA | 12/30 = 40% [25, 58] | 13/30 = 43% [27, 61] | 25% |
| code multiple choice | 5/20 [11, 47]% | 7/20 [18, 57]% | 25% |
| grammar pairs | 26/30 = 87% | 26/30 = 87% | 50% |
| code generation (exact) | 0/10 | 1/10 | — |
| continuation: repeated 4-gram fraction / ended with EOT | 0.087 / 65% | 0.052 / 85% | — |
| chat | not applicable (no chat tokens) | not applicable | — |

**What this confirms:**
1. The harness behaves sensibly.
   - TinyStories models know no facts: 17.5% and 25% against 25% chance.
   - They switch to story mode on any prompt. For example, the code-generation
     outputs are story text.
   - On web text they score 3.05–3.12 bits/byte, worse than a generic compressor;
     they are out of domain.
2. Grammar minimal pairs are easy (87% already). Expect a ceiling at 97–100%. They
   remain useful as a sanity check, not a discriminator.
3. QA is slightly above chance for both. Some items (puppy, spider legs) are
   children's-story knowledge, which is a caveat when comparing to TinyStories
   models.
4. These numbers are the **floor** M4.1 must clear.

---

## 4. Compute planning

### 4.1 Microbenchmarks

**Setup** (`scripts/benchmark_m4_compute.py`, which reuses the M3 subprocess worker):
- synthetic random tokens, random initialization;
- float32, SDPA attention, AdamW with clipping;
- 3 warmup and 8 timed iterations per case, one case at a time;
- a per-process MPS cap of 85% of the recommended maximum, which is 11.84 GiB.

| case | params (non-emb.) | train step | train tok/s | forward tok/s | achieved TFLOP/s | allocated after fwd | MPS pool |
|---|---|---|---|---|---|---|---|
| S 6L/384, V8192, T512, b32 | 14.0M (10.6M) | 679 ms | 24,135 | 114,391 | 2.20 | 4.98 GB | 7.65 GB |
| S b16 | 14.0M | 335 ms | 24,475 | 115,644 | 2.23 | 2.60 GB | 4.37 GB |
| M 8L/512, V8192, T512, b16 | 29.7M (25.2M) | 635 ms | 12,897 | 60,783 | 2.46 | 4.27 GB | 5.49 GB |
| M b8 | 29.7M | 316 ms | 12,981 | 60,764 | 2.47 | 2.37 GB | 3.31 GB |
| M b32 | 29.7M | **out of memory** (9.58 GiB at the cap) | | | | | |
| X 10L/640, V8192, T512, b8 | 54.8M (49.2M) | 542 ms | 7,558 | 34,290 | 2.63 | 3.69 GB | 4.35 GB |
| X b16 | 54.8M | 1,090 ms | 7,513 | 34,073 | 2.62 | 6.50 GB | 8.75 GB |
| L 12L/768, V8192, T512, b4 | 91.7M (85.1M) | 452 ms | 4,535 | 20,883 | 2.62 | 3.44 GB | 4.90 GB |
| L b8 | 91.7M | 855 ms | 4,790 | 20,960 | 2.77 | 5.40 GB | 7.11 GB |
| M, T256, b32 | 29.5M | 543 ms | 15,079 | 62,976 | 2.77 | 3.73 GB | 5.56 GB |
| M, T1024, b8 | 29.9M | 795 ms | 10,304 | 56,116 | 2.11 | 5.35 GB | 6.56 GB |
| X, T1024, b8 | — | **out of memory** | | | | | |
| X, T1024, b4 (added after the OOM) | 55.1M | 671 ms | 6,104 | 32,265 | 2.26 | 4.54 GB | 5.43 GB |
| L, T1024, b4 | 92.1M | 1,063 ms | 3,852 | 19,736 | 2.35 | 6.62 GB | 7.83 GB |
| S, V4096, b32 (M3.1 shape) | 12.4M | 606 ms | 27,049 | 129,997 | 2.21 | 4.41 GB | 6.66 GB |
| M, V4096, b16 (M3.3 shape) | 27.6M | 601 ms | 13,640 | 61,320 | 2.43 | 3.97 GB | 4.55 GB |

FLOPs per token are 6N + 6·L·T·C (the M3.3 convention).

**Calibration against real runs.** M3.3 trained at 13.6k tok/s, matching its
synthetic case (13,640). The data pipeline and gradient accumulation added no
measurable overhead, so these numbers predict real step time well.

**Findings:**
1. **Utilization is nearly flat**, at 2.1–2.8 TFLOP/s from 14M to 92M parameters.
   Bigger models are proportionally slower, not more efficient. The budget is fixed
   in FLOPs: 2 h ≈ 1.6–2.0e16.
2. **Micro-batch size barely matters** once it is at least 4–8 sequences (S: 24.1k
   vs 24.5k tok/s; M: 12.9k vs 13.0k). The effective batch is set by gradient
   accumulation, which costs nothing. Memory scales with the micro-batch.
3. **Memory is not the binding constraint up to 100M** with micro-batches of 4–16.
   Parameters, gradients and Adam take 16 bytes per parameter: 0.22 GB for S, 1.47 GB
   for L. Activations dominate.
4. **Context length.** For M at 8,192 tokens per micro-batch, going from T = 512 to
   1,024 costs 20% throughput (12.9k → 10.3k tok/s). Going down to 256 gains 17%.
   Attention memory grows quadratically: X at 1,024 needs b ≤ 4.
5. **Vocabulary 4,096 → 8,192** costs 11% tok/s for S and 5% for M. This is
   recovered in text throughput (§2.3).

### 4.2 What a 1–2 hour budget buys

| model | params | FLOPs/token | tokens in 60 min | in 120 min | tokens/param at 120 min | time to reach 20 tokens/param |
|---|---|---|---|---|---|---|
| S | 14.0M | 91.0M | 86.9M | 173.8M | **12.4** | 3.2 h |
| M | 29.7M | 190.6M | 46.4M | 92.9M | 3.1 | 12.8 h |
| X | 54.8M | 348.5M | 27.2M | 54.4M | 1.0 | 40 h |
| L | 91.7M | 578.8M | 17.2M | 34.5M | 0.4 | 106 h |

**Compute-optimal size.**
- The Chinchilla rule of thumb (about 20 tokens per parameter, N_opt ≈ √(C/120)) gives
  N_opt ≈ 12M parameters for C ≈ 1.7e16 FLOPs (2 h), with about 240M tokens.
- For a 1-hour budget it gives about 8M.
- The rule was fitted on much larger models, so this is a guide, not a law.

**Our own evidence points the same way.** In M3.3, at about M3.1's compute, the
27.6M model's checkpoints were *worse* than the 12.4M model's (1.64 vs 1.55 at 1.09×
the FLOPs). The larger model only won when given 2.18× the compute. That comparison
was confounded by un-annealed learning rates, so it is not decisive.

**Conclusion for a 1–2 hour budget:**
- the **S class (14M) is the right size**;
- 30M is defensible only with an overnight budget (8–13 h);
- 50M and 100M are not trainable to a useful token count on this machine in hours.
  They are stretch targets for multi-day runs, or for a later machine.

### 4.3 Equal-token vs equal-compute

- **Equal-token** comparisons (as in M3.3) answer "what does capacity buy for the
  same data?". At fixed time they favour bigger models unfairly.
- **Equal-compute** comparisons answer "what should I train with this budget?". Each
  arm then needs its **own** learning-rate schedule ending at its own step count.
  Mid-schedule checkpoints are not valid stand-ins, which was M3.3's limitation.
- **Text-equal** comparisons are a third axis when vocabularies differ: compare
  bytes, not tokens, and report loss in bits per byte.

M4.1 is a single run, so none of these comparisons is made inside it. Its compute
(1.43e16) happens to match M3.3 (1.40e16), so M3.3 is a fair equal-compute reference
point in bits per byte on held-out text.

---

## 5. Proposed M4.1 (not started; requires approval)

| item | proposal |
|---|---|
| **Objective** | First general-purpose base model. Measurable gains on held-out web and textbook text and on knowledge probes over the M3 models at equal compute. **It will not be ChatGPT-like.** A 14M base model after 2 h is expected to produce fluent-looking, often wrong, encyclopedia-style text and to be near chance on many knowledge items. |
| **Data** | FineWeb-Edu `sample-10BT/013_00000.parquet` @ `87f09149…` (sha256 `b393f51f…832de`, 540.6 MB) and Cosmopedia v2 `cosmopedia-v2/train-00000-of-00104.parquet` @ `3ba9d605…` (sha256 `5a257211…1491d`, 1,174 MB). 70/30 by tokens. Single pass, no repetition. Full plan: `docs/reports/m4/m4_1_data_manifest_plan.json`. |
| **Processing** | UTF-8 validation; exact and MinHash near-dedup (as M3.1) **before** splitting; split by document hash (0.5% validation, 0.5% holdout, 99% train); decontamination per §1.3; drop Cosmopedia's `prompt` column. |
| **Manifest** | Per source: revision, file, LFS sha256 (verified after download), rows read, kept and dropped by each rule. Output `train.bin` / `val.bin` / `holdout.bin` with SHA-256, token counts and the tokenizer fingerprint in `meta.json`, verified on load (existing `load_prepared`). |
| **Tokenizer** | Byte-level BPE, V = **8,192**, `DEFAULT_PATTERN`. Specials `<|endoftext|>` (8187), `<|user|>`, `<|assistant|>`, `<|system|>`, `<|end|>` (8188–8191). Fitted on the first 100 MB (sha256 order) of the **M4.1 train split only**. Its fingerprint is fixed at preparation, recorded in the manifest and checked at train time. The research tokenizer (`1eb21d0c…`) was fitted on sample documents and is **not** reused: some of those documents could land in M4.1 validation. |
| **Model** | GPT-2 pre-LN, tied embeddings, learned positions, bias. **6 layers, 6 heads, width 384, context 512, dropout 0: 13,989,888 parameters (10,647,552 non-embedding).** Draft config: `configs/m4_1_general_s_bpe8192.toml` (parses; tested). |
| **Token budget** | 9,600 steps × 16,384 tokens = **157,286,400 tokens** (11.2 per parameter); about 620 MB of text. |
| **Optimizer** | AdamW, β = (0.9, 0.95), weight decay 0.1 on matrices only, clip 1.0. **Unchanged from M3.1**, with no tuning. |
| **LR schedule** | linear warmup over 200 steps to 6e-4, then cosine to 6e-5 at step 9,600 (horizon = run length) |
| **Batch** | 16 × 512 per micro-batch × 2 accumulation steps. Identical sampling semantics to one batch of 32 (verified in M3.3), with a 4.4 GB pool instead of 7.7 GB. |
| **Estimated time** | about 0.67 s/step × 9,600 = **108 min**, plus 12 validation passes (about 2M tokens at about 115k tok/s, roughly 20 s each, ≈ 4 min), plus final evaluations (≈ 5 min). About **2 h**, run under `caffeinate`. |
| **Estimated memory** | MPS pool about 4.4 GB (measured S b16 case). Leaves about 7 GB of headroom within the recommended maximum. |
| **Compute** | 91.0M FLOPs/token × 157.3M = **1.43e16** (1.02× M3.3, 2.2× M3.1) |
| **Validation** | Every 800 steps: exact loss on the full validation split (about 2M tokens, mixed). best.pt is selected on it. At the end: bits/byte on M4.1 val and holdout per source, on Wikipedia-en (out of distribution) and on TinyStories V2 val (regression); the full `general_v1` suite; memorization (k-gram overlap and extraction probe, as M3); 3-seed samples. |
| **Checkpoints** | every 800 steps, all kept (12 × about 170 MB ≈ 2 GB), plus best.pt; atomic writes; resumable (`--resume-latest`); isolated new run directory |
| **Preflight** | The M3.3 8-check preflight (`preflight_scaling.py`): historical checkpoints load, finite gradients, causality, optimizer groups, 100-step stability, throughput and memory, free memory, checkpoint round trip. Plus a data check: manifest hashes, tokenizer fingerprint, no shared document hashes between splits. |

### 5.1 Acceptance criteria (pre-registered)

**Engineering acceptance.** All must pass, or the run is invalid:
1. Downloaded files match the pinned LFS SHA-256.
2. The processed streams match the manifest, and the tokenizer fingerprint matches.
3. Train, validation and holdout share no document hash and no near-duplicate
   cluster.
4. The decontamination counts are recorded.
5. Preflight passes all checks.
6. Training finishes 9,600 steps with no NaN/Inf, no OOM and no restart from step 0.
7. Every checkpoint loads. The final suite run is reproducible: the same checkpoint
   gives identical multiple-choice scores twice.
8. No earlier run directory, dataset or tokenizer changes (hash check).

**Research outcomes.** These are reported whatever they show, with M3.1 and M3.3 as
references:

| metric | M3 floor (M3.3) | expectation (not a guarantee) |
|---|---|---|
| bits/byte, FineWeb-Edu held-out | 3.055 | substantially lower |
| bits/byte, Wikipedia-en (out of distribution) | 3.798 | substantially lower |
| bits/byte, TinyStories | 0.740 | higher (regression: no TinyStories training) |
| facts_mc acc_norm | 10/40 | above chance. "Measurable" means a Wilson lower bound above 25%, i.e. at least 16/40. A null result is possible. |
| qa_mc / code_mc | 13/30, 7/20 | reported with CIs; no claim from differences under about 15 points |
| grammar pairs | 26/30 | likely at the ceiling |
| repetition (repeated 4-gram fraction) | 0.052 | reported; a rise would be a failure signal |
| chat | not applicable | not applicable (no post-training) |

---

## 6. Limitations

- **Samples are small and clustered.** The `/rows` API returns 100 contiguous rows
  per request, so samples cluster by dump or topic (most visibly the Wikipedia stub
  runs). The statistics describe these samples, not the full datasets.
- **Not verified:**
  - full-dataset licenses beyond the cards;
  - the upstream licenses of every smol-smoltalk component;
  - whether FineWeb-Edu shard `013` is a uniform draw (it is the smallest of 14);
  - token-count extrapolations from sample averages (§1.4).
- **The heuristic quality rules are uncalibrated.** The stop-word rule mis-flags
  fluent textbook prose.
- **Tokenizer fit corpus was 27 MB.** The 25–100% sensitivity test suggests little
  gain from more text, but the vocabulary that M4.1 actually ships will differ.
- **Benchmarks are synthetic microbenchmarks.** They are calibrated against one real
  run (M3.3), and the MPS timings vary run to run by a few percent. The training
  times in §5 are estimates.
- **The compute-optimal ratio (about 20 tokens per parameter) is extrapolated** from
  much larger models. Our only in-house equal-compute evidence (M3.3) is confounded
  by schedules.
- **The evaluation suite is small (202 items) and written by one author**, which
  risks stylistic bias.
  - Accuracy differences under about 15 points on 30–40 items are not resolvable.
  - Generation quality still needs a blind human rater; there is no automatic judge.
  - The code sections measure idioms, not programming ability.
  - Grammar pairs will saturate.
- **Short eval items cannot be decontaminated** (common facts and idioms). This is
  by design and reported, not hidden.
- The baseline TinyStories bits/byte here (0.74–0.77) uses the original GPT-3.5 +
  GPT-4 TinyStories sample. It is **not comparable** to the M3 V2 validation numbers
  (0.53–0.56).

## 7. Verification and files

**Checks:**
- **pytest:** 237 passed (225 existing + 12 new in `tests/test_general_eval.py`).
- **Ruff:** clean. **MyPy (strict):** clean.
- **Integrity:** all 345 pre-existing files under `runs/`, `data/`, `docs/reports/`
  and `configs/` are byte-identical (SHA-256) to the snapshot taken before M4. That
  includes the M2–M3.3 run directories, checkpoints, TinyStories data and tokenizer.
- **Historical checkpoints:** the M3.1 and M3.3 final checkpoints load and run
  through the new suite (§3.2).

**New files:**
- `evals/general_v1.json`: frozen suite;
- `src/mini_chatgpt/analysis/general_eval.py`: scorer;
- `scripts/evaluate_general.py`, `scripts/m4_data_analysis.py`,
  `scripts/m4_tokenizer_study.py`, `scripts/benchmark_m4_compute.py`;
- `configs/m4_1_general_s_bpe8192.toml`: draft, not run;
- `tests/test_general_eval.py`;
- `docs/reports/m4/`:
  - `sample_stats.json`, `data_analysis.json`, `tokenizer_study.json`,
    `compute_benchmark.json`;
  - `general_eval_baseline_m3_1_s12m.json`, `general_eval_baseline_m3_3_m28m.json`;
  - `m4_1_data_manifest_plan.json`.

**Modified:** `scripts/fetch_samples.py` (untracked M3 script; backward-compatible
extension, §0).

**Git-ignored local data:**
- `data/raw/samples/{cosmopedia_v2, wikipedia_en, wikipedia_simple, smol_smoltalk, everyday_conversations}`
  (25 MB);
- `runs/m4_tokenizers/`;
- `runs/m4_*.log`.

## Sources

- FineWeb-Edu card and changelog: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu;
  Penedo et al. 2024, https://arxiv.org/abs/2406.17557
- SmolLM-Corpus (Cosmopedia v2, FineWeb-Edu-dedup, Python-Edu):
  https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- Cosmopedia (v1, decontamination section): https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- Wikipedia: https://huggingface.co/datasets/wikimedia/wikipedia
- FineMath: https://huggingface.co/datasets/HuggingFaceTB/finemath
- SmolTalk and smol-smoltalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk,
  https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk; Allal et al. 2025,
  https://arxiv.org/abs/2502.02737
- everyday-conversations: https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k
- OASST2: https://huggingface.co/datasets/OpenAssistant/oasst2
- Dolly-15k: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Alpaca: https://huggingface.co/datasets/tatsu-lab/alpaca
- Tulu 3 SFT mixture: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- Llama 3.1 license: https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE
- ODC-By 1.0: https://opendatacommons.org/licenses/by/1-0/; Common Crawl Terms of Use:
  https://commoncrawl.org/terms-of-use
- Hoffmann et al. 2022 (compute-optimal scaling, "Chinchilla"): https://arxiv.org/abs/2203.15556

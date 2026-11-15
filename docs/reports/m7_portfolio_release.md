# M7 — Portfolio release preparation (mini_chatgpt v1.0)

**Status:** complete, **stopped for review.** Nothing is committed, pushed or published, the
external `website/` is untouched, and the Mac mini is not exposed. Training is frozen. M5
(13,989,888 parameters) is the default chatbot and M6.3 (29,676,544) the research comparison;
both checkpoints are preserved.

## 1. Git and artifact audit

**Repository state:**
- Branch `milestone-3-dataset-research`, HEAD `8d8d7c0` (M2.x, unpushed). Nothing staged, no
  stashes.
- **15 modified tracked files** (+719 / −38 lines) and **265 untracked files** outside `website/` and `.claude/`
  (about 14 MB).

| category | what | in Git? |
|---|---|---|
| source | `src/mini_chatgpt/` (tokenizer/bpe, data/{tinystories,general}, analysis/{story_eval,general_eval}, chat/, serve/) + edits to config, dataset, prepare, trainer, optim, svgplot | yes |
| scripts | 42 new scripts plus 2 edited (prepare, preflight, train, evaluate, plot, serve, export) | yes |
| tests | 9 new test files + 2 edited; 305 Python tests | yes |
| configs | TinyStories (3), M4.1, M6.2, `configs/sft/` (3) | yes |
| frozen eval suites | `evals/general_v1.json`, `chat_v1.json`, `chat_diag_v1.json` (hash-locked by tests) | yes |
| reports | `docs/reports/m3…m7` (Markdown, JSON results, SVG charts, JPG screenshots; largest file 1.8 MB) | yes |
| showcase | `docs/showcase/{overview,walkthrough}.md`, `facts.json`; `docs/release/v1.0_manifest.json` | yes |
| frontend | `apps/chat-web/` sources, configs, tests, lockfile (`node_modules/` and `dist/` ignored by its `.gitignore`) | yes |
| **checkpoints, runs** | `runs/` 14 GB (all training runs, two aborted attempts with NOTEs, the release export, the static demo build) | **no** (`/runs/` ignored) |
| **datasets** | `data/` 2.7 GB (raw parquet/jsonl, processed token streams) | **no** (`/data/` ignored) |
| tooling | `.venv/` (ignored), `.claude/launch.json` (my preview config) | recommend **not** committing `.claude/` |
| unrelated | `website/` (separate "Mini_Chat" site, last updated at M1) | **leave untracked**; not modified (57 files hash-checked) |

**Credentials and personal data:**
- A regex scan for keys and tokens (`sk-`, `hf_`, private keys, `password`/`secret` assignments),
  the user's email and absolute home paths found **nothing real**.
- The only hits were the word "password" inside model-generated text and Python's `secrets`
  module.

**Dataset text inside reports** (licensing note):
- The long strings in JSON reports are almost all *model generations*.
- Real dataset excerpts are few:
  - `docs/reports/m6_1/sft_audit.json`: a fixed sample of 40 OASST2 (Apache 2.0) and Dolly-15k
    (CC BY-SA 3.0) conversations;
  - short (≤ 160 character) dedup examples and one decoded 40-token training window.
- Attribution is in the README "Data licensing" section.
- **Decision for you:** keep the 40-conversation sample (small, attributed; CC BY-SA applies to
  that excerpt) or drop `manual_sample_seed0` before committing.

## 2. Proposed commit plan (not executed)

The 15 modified core files mix changes from M3 to M4.1. For example, `trainer.py` gained the
TinyStories features *and* the single-pass data order. Interactive hunk staging is unavailable, so
I propose grouping by milestone, with the shared core edits in the first commit, rather than
inventing a finer history.

| # | commit | contents |
|---|---|---|
| 1 | M3–M3.3: byte-level BPE, TinyStories pipeline, scaling study | 15 modified core files, tokenizer/bpe, data/tinystories, analysis/story_eval, M3 scripts, TinyStories configs, tests (bpe, tinystories, continuation, eval_stats), `docs/reports/m3*` |
| 2 | M4: general-data research and frozen general_v1 suite | analysis/general_eval, `evals/general_v1.json`, M4 scripts, test_general_eval, `docs/reports/m4*` |
| 3 | M4.1: general pretraining (14M base model) | data/general, prepare_general, preflight_m4_1_data, evaluate_m4_1, M4.1 config, test_general_data, `docs/reports/m4_1*` |
| 4 | M5: conversational SFT | `src/mini_chatgpt/chat/`, `evals/chat_v1.json`, SFT scripts, `configs/sft/m5_sft_chat.toml`, test_chat, `docs/reports/m5*` |
| 5 | M6: local chat demo | `src/mini_chatgpt/serve/`, serve_chat, benchmark_chat_server, `apps/chat-web/`, test_serve, `docs/reports/m6*` |
| 6 | M6.1: decoding grid, SFT audit, length-capped SFT | `evals/chat_diag_v1.json`, M6.1 scripts and config, `docs/reports/m6_1*` |
| 7 | M6.2–M6.3: 30M scaling and conversational comparison | M6.2/M6.3 configs and scripts, test_m6_3_eval, `docs/reports/m6_2*`, `docs/reports/m6_3*` |
| 8 | M7: v1.0 showcase and release manifest | `docs/showcase/`, `docs/release/`, `docs/reports/m7*`, export and showcase scripts, the M7 frontend changes, README, `docs/portfolio_summary.md`, `docs/experiments.md` |

Notes:
- **Branch.** The branch name `milestone-3-dataset-research` no longer describes the work. Consider
  committing on a new branch (e.g. `v1.0-release`) from the current HEAD, then merging to `main`.
  `main` currently ends before M2.
- **Order matters.** The report and test files of later milestones reference earlier code. Each
  commit above was chosen to stay buildable when applied in order, but only the final state has
  been verified by the full test suite.
- **Checks before committing:** `git add` paths explicitly (never `-A`, to keep `website/` and
  `.claude/` out), then `git status`, then the full checks.

## 3. Portfolio content

| file | purpose |
|---|---|
| [`docs/showcase/overview.md`](../showcase/overview.md) | recruiter-friendly one-pager: why, what was built, what it can and cannot do, the key experiment, engineering highlights |
| [`docs/showcase/walkthrough.md`](../showcase/walkthrough.md) | technical walkthrough: Transformer, BPE, pretraining, SFT and masking, Apple Silicon memory issues, 14M vs 30M, evaluation method, the app, limitations, PyTorch vs own code |
| [`docs/showcase/facts.json`](../showcase/facts.json) | 25 numbers, each with the report or JSON file that supports it (all paths checked) |
| `docs/portfolio_summary.md` | updated M0–M6.3 table and the scaling paragraph |
| web app "Project" tab | the overview's key numbers and finding, read from `facts.json` (tested) |

`facts.json` follows the evidence rule of the existing `website/` (`src/content/facts.ts`): no
number without a traceable source. That makes it ready to port into the site later. The site
itself is out of scope here.

## 4. The key experiment story

> Doubling model parameters improved language-modeling loss and fluency, but did not produce a
> statistically clear improvement in factual or conversational correctness.

| | 14M (M5, served) | 30M (M6.3) |
|---|---|---|
| base model, held-out FineWeb-Edu bits/byte | 1.335 | 1.278 |
| SFT assistant-token validation loss | 3.085 | 2.932 |
| strict content success, 75 pre-registered prompts | 18.2% | 21.3% |
| paired difference (prompt = unit), 95% bootstrap CI, sign test | | +3.1 points, [−2.2, +8.9], p = 0.72 |
| blind review (20 pairs): correct replies | 1 | 1 |
| natural end-of-turn / repetition / mean latency | 95.2% / 1.9% / 146 ms | 97.8% / 2.6% / 160 ms |

**Why M5 stays the default despite its higher loss:**
- The promotion rule was written and hashed before training: a credible gain of at least 5 points,
  confirmed by blind reading, with no regressions.
- The 30M model passed every regression check but failed the content criteria. It is more fluent
  and retains more general text, but it is not more often *right*.
- M5 is also half the size and faster.

## 5. Release and reproducibility

`scripts/export_release.py` writes an **inference-only** bundle per model to the Git-ignored
`runs/release/mini_chatgpt-v1.0/{m5,m6_3}/`:
- `checkpoints/model.pt`: weights and config metadata, no optimizer state;
- `tokenizer.json`;
- `sft_config.json`: the expected tokenizer fingerprint.

The committed manifest [`docs/release/v1.0_manifest.json`](../release/v1.0_manifest.json) records
the SHA-256 and size of every file, the source-checkpoint hash, the training commit, the
architecture, the special-token ids and the verification results.

| | M5 (default) | M6.3 (research) |
|---|---|---|
| exported weights | 56.0 MB (from a 168 MB training checkpoint) | 118.7 MB (from 356 MB) |
| all tensors equal to the source | ✓ | ✓ |
| max \|logit difference\| on fixed inputs | 0.0 | 0.0 |
| greedy and seeded-sampling replies to 3 fixed prompts identical | ✓ (6/6) | ✓ (6/6) |
| served through the real FastAPI server | ✓, identical reply to the original checkpoint for the same seed | ✓ (M6.3 verified in M6.3) |

Run the exported model locally:

```bash
uv pip install --python .venv/bin/python -e ".[serve]"
npm --prefix apps/chat-web install && npm --prefix apps/chat-web run build
.venv/bin/python scripts/serve_chat.py --checkpoint runs/release/mini_chatgpt-v1.0/m5/checkpoints/model.pt
# open http://127.0.0.1:8000
```

Distributing the weights, for example as a GitHub release asset, is a publishing decision left for
your approval. The manifest's hashes let anyone verify a downloaded copy.

## 6. Demonstration and fallback

The web app (`apps/chat-web`) now has three views. They were tested on desktop and on a 375 px
mobile viewport, with real screenshots in `docs/reports/m7/screenshots/`.

| view | what it shows |
|---|---|
| **Live chat** | unchanged M6 chat against the local model. If the server is unreachable it offers "See recorded conversations instead". |
| **Recorded** | 8 real, unedited M5 replies from the pre-registered evaluation (production decoding, seed 11), each marked "passed its content check" or "wrong / failed its content check". It includes 2 correct and 6 typical failures, and the matched 30M replies are available on demand. |
| **Project** | the overview's numbers and the key finding, every value read from `facts.json` |

**Static fallback for website visitors.** `VITE_STATIC_DEMO=1 npm --prefix apps/chat-web run build`
produces a backend-free build with only Recorded and Project. It was verified in the browser to
make **zero** `/api` requests and to say "No model runs on this page". The test build is in the
Git-ignored `runs/release/static-demo/`, about 250 KB. Hosting it, for example on GitHub Pages, is
a publishing step left for approval. The Mac mini is never exposed: the server binds to 127.0.0.1.

The recorded replies come from `docs/reports/m6_3/decoding_*.json` through
`scripts/build_showcase_data.py`; they are copied, never edited.

## 7. Verification

| check | result |
|---|---|
| pytest | **305 passed** |
| ruff check, ruff format, mypy (strict) | clean (114 source files) |
| frontend: TypeScript, ESLint, Vitest | clean; **20 tests passed** (5 new: static mode makes no API calls, recorded data, project numbers, fact traceability, unreachable-server fallback) |
| production builds | normal (`dist/`) and static demo both succeed |
| real M5 checkpoint | default server ready (13,989,888 params, fingerprint `e6be77cc`) and answers through `/api/chat` |
| release export | served through the API; identical reply to the original checkpoint |
| browser | live chat, Recorded (with 30M comparison) and Project views on desktop; live chat and Project on mobile; static demo on mobile |
| historical artifacts | all 542 pre-M6.3 artifact files byte-identical; M5 `best.pt` `f31594e2…`, M6.3 `best.pt` `088803ad…` unchanged |
| `website/` | 57 files byte-identical, still untracked |

## 8. Waiting for your approval

1. **Commits** following §2, including the branch choice and the `sft_audit.json` sample decision.
2. **Push** to the remote.
3. **Publishing:**
   - the static demo build;
   - optionally the inference weights as release assets.
4. **Updating the external portfolio site** (`website/` or its repository) with
   `docs/showcase/` and `facts.json`.
5. **Whether `.claude/launch.json`** should be deleted or ignored.

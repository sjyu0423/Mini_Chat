/**
 * Every number shown on the site comes from this file. Each fact names the
 * repository artifacts that support it and the experiment stage it belongs to.
 *
 * Rules:
 * - Only completed, reviewed stages. Live or partial runs are never shown.
 * - Prefer committed evidence; local run outputs under `runs/` are git-ignored
 *   and are cited as corroboration only.
 * - If a value cannot be traced to evidence here, it does not go on the site.
 * - Historical M1 facts stay labeled as M1; they describe the early character-
 *   level teaching model, not the v1.0 served chatbot.
 */

import { site } from '@/config/site'

export type ExperimentStage =
  | 'M1'
  | 'M2'
  | 'M3.1'
  | 'M3.3'
  | 'M4.1'
  | 'M5'
  | 'M6'
  | 'M6.2'
  | 'M6.3'
  | 'M7'
  | 'v1.0'

export interface Evidence {
  /** Path relative to the repository root. */
  path: string
  /** Commit containing the file, or `null` for git-ignored local run output. */
  commit: string | null
  detail: string
}

export interface Fact {
  id: string
  /** Display value. Shown in monospace. */
  value: string
  label: string
  description: string
  stage: ExperimentStage
  evidence: readonly Evidence[]
}

const M1_COMMIT = 'd49fa80'
const M1_COMMIT_SHA = 'd49fa804ad4cb96d89dc3fdb66d1fd12638d6efe'

/** Commit that contains the v1.0 showcase docs and M4.1–M7 reports. */
const V1_COMMIT = '8101752'
const V1_COMMIT_SHA = '8101752960c54a9c5c1bcf986b68812259d8fcf1'

/** GitHub URL for a repository file, pinned to the M1 commit so it never drifts. */
export function m1SourceUrl(path: string): string {
  return `${site.repoUrl}/blob/${M1_COMMIT_SHA}/${path}`
}

/** GitHub URL pinned to the v1.0 release commit. */
export function v1SourceUrl(path: string): string {
  return `${site.repoUrl}/blob/${V1_COMMIT_SHA}/${path}`
}

function evidence(path: string, detail: string, commit: string | null = V1_COMMIT): Evidence {
  return { path, commit, detail }
}

/* ------------------------------------------------------------------------- */
/* Homepage highlights (v1.0)                                                */
/* ------------------------------------------------------------------------- */

export const highlights: readonly Fact[] = [
  {
    id: 'served-parameters',
    value: '13,989,888',
    label: 'parameters in the served model',
    description:
      'The default chatbot: six layers, six attention heads, width 384. Trained from random weights on one Mac mini.',
    stage: 'M4.1',
    evidence: [
      evidence(
        'docs/showcase/facts.json',
        'id m5_parameters: 13,989,888 — parameters (served model).',
      ),
      evidence(
        'docs/reports/m4_1_general_pretraining.md',
        'Model table and parameter count: 13,989,888 (tied embeddings).',
      ),
      evidence(
        'configs/m4_1_general_s_bpe8192.toml',
        'Comment and config for the 14M general-pretraining run.',
      ),
    ],
  },
  {
    id: 'research-parameters',
    value: '29,676,544',
    label: 'parameters in the research model',
    description:
      'A larger comparison model (eight layers, eight heads, width 512) trained on the same data and order. Better at language modeling; not measurably better at correct answers.',
    stage: 'M6.2',
    evidence: [
      evidence(
        'docs/showcase/facts.json',
        'id m63_parameters: 29,676,544 — parameters (30M comparison model).',
      ),
      evidence(
        'docs/reports/m6_2_model_scaling.md',
        'Parameter table: 29,676,544 (= analytic count).',
      ),
    ],
  },
  {
    id: 'pretraining-tokens',
    value: '157,286,400',
    label: 'pretraining token presentations',
    description:
      'FineWeb-Edu and Cosmopedia v2, cleaned and de-duplicated. Each 512-token window was seen exactly once.',
    stage: 'M4.1',
    evidence: [
      evidence(
        'docs/showcase/facts.json',
        'id pretraining_tokens: 157,286,400 — each seen once.',
      ),
      evidence(
        'docs/reports/m4_1_general_pretraining.md',
        '157,286,400 presented, each exactly once (single_pass data order).',
      ),
      evidence('docs/reports/m4_1/preflight_data.json', 'Preflight data budget for the M4.1 run.', null),
    ],
  },
  {
    id: 'vocabulary',
    value: '8,192',
    label: 'byte-level BPE vocabulary',
    description:
      'A custom tokenizer: 256 byte bases, 7,931 merges learned from training text, and five reserved chat tokens.',
    stage: 'M4.1',
    evidence: [
      evidence(
        'docs/showcase/facts.json',
        'id bpe_merges: 7,931 merges → 8,192 ids (256 + 7,931 + 5 special).',
      ),
      evidence(
        'docs/reports/m4_1_general_pretraining.md',
        '§3 Tokenizer: byte-level BPE with 8,192 ids and chat specials.',
      ),
      evidence('configs/m4_1_general_s_bpe8192.toml', 'vocab_size = 8192.'),
    ],
  },
  {
    id: 'tests',
    value: '305 + 20',
    label: 'automated tests at v1.0',
    description:
      '305 Python tests and 20 frontend tests, plus TypeScript, ESLint, ruff and mypy checks on the release branch.',
    stage: 'M7',
    evidence: [
      evidence(
        'docs/showcase/facts.json',
        'id tests: 305 + 20 — Python + frontend automated tests at v1.0.',
      ),
      evidence(
        'docs/reports/m7_portfolio_release.md',
        'pytest 305 passed; frontend Vitest 20 passed.',
      ),
    ],
  },
]

export const highlightsFootnote =
  'Figures are from completed M4.1–M7 runs logged in the repository and catalogued in docs/showcase/facts.json. The 14M model is the local default; the 30M model is a controlled research comparison. This site does not host a live model.'

/* ------------------------------------------------------------------------- */
/* Architecture walkthrough — default 14M (v1.0 served) model                */
/* ------------------------------------------------------------------------- */

const v1ConfigEvidence: Evidence = evidence(
  'configs/m4_1_general_s_bpe8192.toml',
  'vocab_size = 8192, context_length = 512, n_layer = 6, n_head = 6, n_embd = 384; 13,989,888 parameters.',
)

const v1ModelCodeEvidence: Evidence = evidence(
  'src/mini_chatgpt/model/gpt.py',
  'Token + position embeddings; pre-LN blocks; causal attention; MLP C→4C→C with GELU; tied lm_head = wteᵀ.',
)

const v1TokenizerEvidence: Evidence = evidence(
  'docs/reports/m4_1_general_pretraining.md',
  '§3 Tokenizer: byte-level BPE, 8,192 ids, fingerprint e6be77cc…, chat specials reserved.',
)

const v1WalkthroughEvidence: Evidence = evidence(
  'docs/showcase/walkthrough.md',
  'Architecture diagram: 8,192 × 384 embeddings, 6 × [LN → causal MHA (6×64) → MLP], tied output.',
)

const researchConfigEvidence: Evidence = evidence(
  'configs/m6_2_general_m_bpe8192.toml',
  'n_layer = 8, n_head = 8, n_embd = 512, vocab_size = 8192, context_length = 512; 29,676,544 parameters.',
)

/** Default served chatbot (14M). Shown in “Under the hood”. */
export const architectureFacts: readonly Fact[] = [
  {
    id: 'arch-parameters',
    value: '13,989,888',
    label: 'Parameters',
    description: 'Default served model. Analytic count matches the loaded checkpoint.',
    stage: 'M4.1',
    evidence: [v1ConfigEvidence, evidence('docs/showcase/facts.json', 'id m5_parameters.')],
  },
  {
    id: 'arch-layers',
    value: '6',
    label: 'Transformer layers',
    description:
      'Pre-LayerNorm decoder blocks: LayerNorm → causal multi-head attention → residual, then LayerNorm → MLP (384 → 1,536 → 384, GELU) → residual.',
    stage: 'M4.1',
    evidence: [v1ConfigEvidence, v1ModelCodeEvidence, v1WalkthroughEvidence],
  },
  {
    id: 'arch-heads',
    value: '6',
    label: 'Attention heads per layer',
    description: 'Each head is 64 dimensions (384 ÷ 6).',
    stage: 'M4.1',
    evidence: [v1ConfigEvidence],
  },
  {
    id: 'arch-width',
    value: '384',
    label: 'Embedding width',
    description: 'Hidden size C: every position is a 384-dimensional vector through the stack.',
    stage: 'M4.1',
    evidence: [v1ConfigEvidence],
  },
  {
    id: 'arch-context',
    value: '512',
    label: 'Context length',
    description: 'At most 512 tokens of history in one forward pass.',
    stage: 'M4.1',
    evidence: [v1ConfigEvidence],
  },
  {
    id: 'arch-vocab',
    value: '8,192',
    label: 'Vocabulary',
    description:
      'Byte-level BPE: 256 byte bases + 7,931 merges + 5 special chat tokens. Fingerprint e6be77cc…32ce91c8.',
    stage: 'M4.1',
    evidence: [v1TokenizerEvidence, evidence('docs/showcase/facts.json', 'ids bpe_merges, tokenizer_fingerprint.')],
  },
  {
    id: 'arch-attention',
    value: 'causal',
    label: 'Self-attention',
    description:
      'Token i may attend only to positions ≤ i. Scores use QKᵀ/√D with future positions masked to −∞ before softmax.',
    stage: 'M4.1',
    evidence: [v1ModelCodeEvidence, v1WalkthroughEvidence],
  },
  {
    id: 'arch-positions',
    value: 'learned',
    label: 'Positional embeddings',
    description: 'A trained vector per index 0…511, added to the token embedding.',
    stage: 'M4.1',
    evidence: [v1ModelCodeEvidence, v1WalkthroughEvidence],
  },
  {
    id: 'arch-output',
    value: 'tied',
    label: 'Output layer',
    description:
      'The language-model head reuses the token embedding matrix (logits = LN(x) @ E_tokᵀ), so it adds no extra parameters.',
    stage: 'M4.1',
    evidence: [v1ModelCodeEvidence, v1WalkthroughEvidence],
  },
]

/** Research comparison model (30M). Not the served default. */
export const researchArchitectureFacts: readonly Fact[] = [
  {
    id: 'arch-research-parameters',
    value: '29,676,544',
    label: 'Parameters',
    description: 'Controlled scaling study only — same data, tokenizer, and 512 context as the 14M model.',
    stage: 'M6.2',
    evidence: [researchConfigEvidence, evidence('docs/showcase/facts.json', 'id m63_parameters.')],
  },
  {
    id: 'arch-research-shape',
    value: '8 × 8 × 512',
    label: 'Layers × heads × width',
    description: 'Deeper and wider than the default; not deployed as the local chatbot default.',
    stage: 'M6.2',
    evidence: [researchConfigEvidence],
  },
]

export const architectureSources = [
  v1ConfigEvidence.path,
  v1ModelCodeEvidence.path,
  v1TokenizerEvidence.path,
  'docs/showcase/walkthrough.md',
] as const

/* ------------------------------------------------------------------------- */
/* Historical M1 character-level teaching model (optional aside)             */
/* ------------------------------------------------------------------------- */

const m1ConfigEvidence: Evidence = {
  path: 'configs/gpt_10m.toml',
  commit: M1_COMMIT,
  detail: 'vocab_size = 65, context_length = 256, n_layer = 6, n_head = 6, n_embd = 384.',
}

/** Historical M1 numbers for the optional “How it started” aside. */
export const historicalArchitectureFacts: readonly Fact[] = [
  {
    id: 'm1-parameters',
    value: '10,770,816',
    label: 'Parameters (historical)',
    description: 'Early character-level teaching model — not the v1.0 chatbot.',
    stage: 'M1',
    evidence: [
      {
        path: 'docs/experiments.md',
        commit: M1_COMMIT,
        detail: 'M1 smoke benchmark: Parameters 10,770,816.',
      },
      m1ConfigEvidence,
    ],
  },
  {
    id: 'm1-vocab',
    value: '65',
    label: 'Character vocabulary',
    description: 'Tiny Shakespeare characters only; replaced by byte-level BPE for general language.',
    stage: 'M1',
    evidence: [m1ConfigEvidence],
  },
  {
    id: 'm1-context',
    value: '256',
    label: 'Context length',
    description: 'Shorter window than the final 512-token models.',
    stage: 'M1',
    evidence: [m1ConfigEvidence],
  },
]

/** The M1 character vocabulary, in id order (id 0 is a newline, id 1 a space). */
export const vocabulary =
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

export const vocabularyChars: readonly string[] = Array.from(vocabulary)

/** Historical M1 example prompt (character-level). */
export const examplePrompt = 'To be or not to'
export const examplePromptChars: readonly string[] = Array.from(examplePrompt)
export const examplePromptIds: readonly number[] = examplePromptChars.map((char) =>
  vocabulary.indexOf(char),
)

/** @deprecated Prefer historicalArchitectureFacts; kept for any leftover imports. */
export const m1HistoricalHighlights = historicalArchitectureFacts

/* ------------------------------------------------------------------------- */
/* Training journey (W4) — additional verified milestones                    */
/* ------------------------------------------------------------------------- */

export const journeyFacts: readonly Fact[] = [
  {
    id: 'm2-best-val',
    value: '1.474',
    label: 'M2 best validation loss (nats/char)',
    description: 'Character-level Tiny Shakespeare; best at step 1,750 before overfitting.',
    stage: 'M2',
    evidence: [
      evidence('docs/reports/m2/analysis.json', 'exact.best.val_loss ≈ 1.4736; portfolio rounds to 1.474.'),
      evidence('docs/portfolio_summary.md', 'M2: best val 1.474 nats/char at step 1,750.'),
    ],
  },
  {
    id: 'm3-val-scaling',
    value: '1.546 → 1.444',
    label: 'TinyStories val loss (nats/token)',
    description: '12.4M (M3.1) → 27.6M (M3.3) on TinyStories BPE.',
    stage: 'M3.3',
    evidence: [
      evidence('docs/reports/m3_3/eval_m3_3.json', 'exact.A.val.loss ≈ 1.545; exact.C.val.loss ≈ 1.444.'),
      evidence('docs/portfolio_summary.md', 'val 1.546 → 1.444 nats/token with 2.2× parameters.'),
    ],
  },
  {
    id: 'm3-params',
    value: '12.4M / 27.6M',
    label: 'TinyStories model sizes',
    description: 'Analytic totals 12,417,024 and 27,579,392 parameters.',
    stage: 'M3.3',
    evidence: [
      evidence('docs/reports/m3_3/eval_m3_3.json', 'exact.A.params.total / exact.C.params.total.'),
    ],
  },
]

/** Verified 14M vs 30M comparison figures for the scaling panel. */

export interface ScalingRow {
  id: string
  label: string
  m14: string
  m30: string
  detail: string
}

export const scalingHeadline =
  'Doubling parameters improved language-model prediction and fluency, but did not establish a statistically clear improvement in correct answers.'

export const scalingRows: readonly ScalingRow[] = [
  {
    id: 'params',
    label: 'Parameters',
    m14: '13,989,888',
    m30: '29,676,544',
    detail: 'Same 8,192-id tokenizer and 512-token context.',
  },
  {
    id: 'bpb',
    label: 'Base validation (bits/byte)',
    m14: '1.264',
    m30: '1.204',
    detail: 'Lower is better. Same data and single-pass order.',
  },
  {
    id: 'heldout',
    label: 'Held-out FineWeb-Edu (bits/byte)',
    m14: '1.335',
    m30: '1.278',
    detail: 'Tokenizer-independent comparison on held-out web text.',
  },
  {
    id: 'wall',
    label: 'Pretraining wall-clock',
    m14: '109.9 min',
    m30: '205.4 min',
    detail: 'More compute for the larger model (~2.1× FLOPs).',
  },
  {
    id: 'sft-loss',
    label: 'SFT assistant-token val loss',
    m14: '3.085',
    m30: '2.932',
    detail: 'Same SFT recipe after base training.',
  },
  {
    id: 'content',
    label: 'Strict content success (75 prompts)',
    m14: '18.2%',
    m30: '21.3%',
    detail: '+3.1 points; 95% bootstrap CI −2.2 to +8.9; sign test p = 0.72.',
  },
  {
    id: 'blind',
    label: 'Blind review correct replies',
    m14: '1 / 20',
    m30: '1 / 20',
    detail: 'Human/LLM blind grades of matched pairs.',
  },
] as const

export const scalingSources = [
  'docs/showcase/facts.json',
  'docs/reports/m6_2_model_scaling.md',
  'docs/reports/m6_3/comparison.json',
  'docs/reports/m6_3_conversational_scaling.md',
] as const

export const promotionRule =
  'Pre-registered rule: promote the larger model only with a credible gain of at least 5 content points and no regressions. The 30M model did not meet that bar, so M5 (14M) remains the served chatbot.'

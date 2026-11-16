/**
 * Training-journey narrative. Numbers cited here must also appear in facts.ts
 * or curves.ts with evidence paths.
 */

export interface JourneyPhase {
  id: string
  letter: string
  title: string
  changed: string
  why: string
  findings: readonly string[]
  detail: string
  stageLabel: string
}

export const journeyPhases: readonly JourneyPhase[] = [
  {
    id: 'character',
    letter: 'A',
    title: 'Character-level beginnings',
    changed: 'A 65-character vocabulary and a ~10.77M GPT trained on Tiny Shakespeare.',
    why: 'Prove next-token prediction works end-to-end on one Mac mini.',
    findings: [
      'Best validation loss 1.474 nats/char at step 1,750, then clear overfitting.',
      'The model learned Shakespeare-like form without copying long training spans.',
    ],
    detail:
      'Each character was its own token. That made the autoregressive loop easy to see, but a 65-symbol alphabet cannot represent general English or chat structure.',
    stageLabel: 'M0–M2',
  },
  {
    id: 'stories',
    letter: 'B',
    title: 'Learning stories',
    changed: 'Custom byte-level BPE and TinyStories training at 12.4M and 27.6M parameters.',
    why: 'Move beyond characters to word-like pieces and longer fluent text.',
    findings: [
      'Validation loss improved from 1.546 → 1.444 nats/token when scaling 12.4M → 27.6M.',
      'Stories often ended on their own, but entity consistency stayed fragile.',
    ],
    detail:
      'Story fluency arrived before reliable tracking of who owns what. A chatbot needs broader knowledge and conversational format — not only children’s stories.',
    stageLabel: 'M3.1–M3.3',
  },
  {
    id: 'general',
    letter: 'C',
    title: 'General-language pretraining',
    changed:
      'FineWeb-Edu + Cosmopedia v2, 8,192-id BPE, 157,286,400 single-pass tokens, 13,989,888 parameters.',
    why: 'Learn ordinary English from educational web and textbook-style text.',
    findings: [
      'Validation 1.264 bits/byte after ~110 minutes on the Mac mini GPU.',
      'Much better on held-out web text than the TinyStories models — still not a chatbot.',
    ],
    detail:
      'The base model could continue English, but often looped and never ended a reply in the assistant role. Conversational behaviour required a second training stage.',
    stageLabel: 'M4.1',
  },
  {
    id: 'sft',
    letter: 'D',
    title: 'Conversational fine-tuning',
    changed:
      '16,813 OASST2 + Dolly conversations; assistant-only loss; ~8 minutes of SFT on the 14M model.',
    why: 'Teach the assistant role and when to stop.',
    findings: [
      'Natural end-of-turn rose from 0% (base) to about 97.5% under production decoding.',
      'Repetitive sampled replies fell from 28% to 2.5% with repetition penalty 1.2.',
    ],
    detail:
      'SFT fixed format and stopping. It did not create reliable factual knowledge — answers can still be circular or confidently wrong.',
    stageLabel: 'M5',
  },
  {
    id: 'scaling',
    letter: 'E',
    title: 'Scaling investigation',
    changed:
      'A 29,676,544-parameter twin on the same tokenizer, data order, and 157.3M tokens, then the same SFT recipe.',
    why: 'Test whether more parameters buy better answers under a pre-registered rule.',
    findings: [
      'Base validation improved to 1.204 bits/byte (from 1.264) with more compute (~205 vs ~110 minutes).',
      'Strict content success 18.2% → 21.3% (+3.1 pts, 95% CI −2.2 to +8.9, p = 0.72) — not a clear win.',
    ],
    detail:
      'Promotion required a credible ≥5-point gain. The 30M model failed that bar, so the 14M M5 checkpoint remains the served default.',
    stageLabel: 'M6.2–M6.3',
  },
] as const

export const journeyConclusion =
  'Doubling parameters improved language-model prediction and fluency, but did not establish a statistically clear improvement in correct answers. The 14M fine-tuned model stays the local default.'

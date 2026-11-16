/**
 * Project status, grouped so built features, finished results and release
 * notes are never mixed. Statuses follow completed M0–M7 work on v1.0-release.
 */

export type RoadmapStatus = 'done' | 'in-progress' | 'planned'

export interface RoadmapItem {
  text: string
  status: RoadmapStatus
}

export interface RoadmapGroup {
  id: string
  title: string
  items: readonly RoadmapItem[]
}

export const roadmap: readonly RoadmapGroup[] = [
  {
    id: 'built',
    title: 'Built',
    items: [
      { text: 'A GPT-style Transformer written by hand in PyTorch', status: 'done' },
      { text: 'A custom byte-level BPE tokenizer with fingerprinting', status: 'done' },
      { text: 'Data pipelines: cleaning, deduplication, splits and decontamination', status: 'done' },
      { text: 'Pretraining from random initialization on educational web text', status: 'done' },
      { text: 'Supervised conversational fine-tuning with assistant-only loss', status: 'done' },
      { text: 'A local FastAPI + React chatbot (runs on this machine only)', status: 'done' },
    ],
  },
  {
    id: 'results',
    title: 'Results',
    items: [
      {
        text: 'General-language pretraining: 13.99M parameters, 157.3M tokens each seen once',
        status: 'done',
      },
      {
        text: 'Chat fine-tuning: replies that end on their own rose from 0% to about 97.5%',
        status: 'done',
      },
      {
        text: '14M vs 30M scaling: better language-modeling loss, no clear gain in correct answers',
        status: 'done',
      },
      {
        text: '305 Python tests and 20 frontend tests on the v1.0 release branch',
        status: 'done',
      },
    ],
  },
  {
    id: 'release',
    title: 'Release',
    items: [
      { text: 'Code, reports and showcase docs on the v1.0-release branch', status: 'done' },
      { text: 'Local chatbot demo and recorded-conversation fallback', status: 'done' },
      {
        text: 'This public project website (static pages; no hosted live model) — ready for GitHub Pages',
        status: 'in-progress',
      },
    ],
  },
]

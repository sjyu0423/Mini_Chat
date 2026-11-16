export const site = {
  name: 'Mini_Chat',
  tagline: 'A GPT-style language model built and trained from scratch.',
  summary:
    'A small conversational language model: custom Transformer, custom tokenizer, pretraining, chat fine-tuning and a local demo — trained from scratch on a Mac mini.',
  /** Intended GitHub Pages project URL (Vite base `/Mini_Chat/`). */
  pagesUrl: 'https://sjyu0423.github.io/Mini_Chat/',
  repoUrl: 'https://github.com/sjyu0423/Mini_Chat',
  portfolioUrl: 'https://sungju-yu-portfolio.vercel.app/',
} as const

export interface NavItem {
  id: string
  label: string
}

export const navItems: readonly NavItem[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'engineering', label: 'Engineering' },
]

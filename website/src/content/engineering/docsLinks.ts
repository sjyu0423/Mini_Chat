import { site } from '@/config/site'
import { v1SourceUrl } from '@/content/facts'

export interface DocLink {
  id: string
  label: string
  description: string
  href: string
  external: boolean
}

/** Documentation entry points for the Engineering section. Paths verified on v1.0-release. */
export const engineeringDocLinks: readonly DocLink[] = [
  {
    id: 'repo',
    label: 'GitHub repository',
    description: 'Source, configs, tests, and reports for mini_chatgpt v1.0.',
    href: site.repoUrl,
    external: true,
  },
  {
    id: 'overview',
    label: 'Showcase overview',
    description: 'Recruiter-friendly one-pager: what was built and what it cannot do.',
    href: v1SourceUrl('docs/showcase/overview.md'),
    external: true,
  },
  {
    id: 'walkthrough',
    label: 'Technical walkthrough',
    description: 'Transformer, BPE, pretraining, SFT, MPS issues, evaluation, and the app.',
    href: v1SourceUrl('docs/showcase/walkthrough.md'),
    external: true,
  },
  {
    id: 'portfolio',
    label: 'Portfolio summary',
    description: 'Milestone table and engineering highlights across M0–M6.3.',
    href: v1SourceUrl('docs/portfolio_summary.md'),
    external: true,
  },
  {
    id: 'm7',
    label: 'M7 release report',
    description: 'Inference-only export, static demo build, and verification notes.',
    href: v1SourceUrl('docs/reports/m7_portfolio_release.md'),
    external: true,
  },
  {
    id: 'manifest',
    label: 'v1.0 release manifest',
    description: 'Pinned checkpoint hashes, tokenizer fingerprint, and decoding defaults.',
    href: v1SourceUrl('docs/release/v1.0_manifest.json'),
    external: true,
  },
  {
    id: 'readme',
    label: 'Local install & serve',
    description: 'Create the venv, build the chat UI, and run scripts/serve_chat.py on localhost.',
    href: v1SourceUrl('README.md'),
    external: true,
  },
] as const

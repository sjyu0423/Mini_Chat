import { ArrowDown } from 'lucide-react'

import { GitHubMark } from '@/components/icons/GitHubMark'
import { Badge } from '@/components/ui/Badge'
import { LinkButton } from '@/components/ui/Button'
import { Container } from '@/components/ui/Container'
import { Section } from '@/components/ui/Section'
import { site } from '@/config/site'

const technologies = ['Python', 'PyTorch', 'Apple Silicon GPU'] as const

export function Hero() {
  return (
    <Section id="intro" labelledBy="intro-title" className="pt-20 sm:pt-28">
      <Container>
        <Badge tone="accent" dot>
          v1.0 · trained from scratch
        </Badge>

        <h1 id="intro-title" className="mt-8 max-w-4xl text-display text-balance text-fg">
          I built and trained a GPT-style language model from scratch.
        </h1>

        <p className="mt-7 max-w-reading text-lede text-pretty text-fg">
          A custom Transformer, a custom tokenizer, pretraining from random weights, conversational
          fine-tuning, and a working local chatbot — all on a Mac mini, with no pretrained models
          and no hosted AI APIs.
        </p>

        <p className="mt-4 max-w-reading text-body text-pretty text-fg-muted">
          It can hold a short chat turn and write fluent English. Factual answers are often wrong;
          this is an engineering demonstration, not a production assistant.
        </p>

        <div className="mt-10 flex flex-wrap gap-3">
          <LinkButton href={site.repoUrl} newTab>
            <GitHubMark />
            View on GitHub
          </LinkButton>
          <LinkButton href="#overview" variant="secondary">
            Project overview
            <ArrowDown aria-hidden="true" />
          </LinkButton>
        </div>

        <div className="mt-14 flex flex-wrap items-center gap-x-4 gap-y-3 border-t border-border pt-6">
          <p className="text-small text-fg-muted">Built with</p>
          <ul aria-label="Technologies" className="flex flex-wrap gap-2">
            {technologies.map((name) => (
              <li key={name}>
                <Badge>{name}</Badge>
              </li>
            ))}
          </ul>
        </div>
      </Container>
    </Section>
  )
}

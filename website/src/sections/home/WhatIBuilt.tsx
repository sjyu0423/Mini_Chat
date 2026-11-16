import { MessageSquare, Network, Repeat, Wrench, type LucideIcon } from 'lucide-react'

import { Card } from '@/components/ui/Card'
import { Container } from '@/components/ui/Container'
import { Section } from '@/components/ui/Section'
import { SectionHeading } from '@/components/ui/SectionHeading'

interface BuiltPart {
  title: string
  description: string
  icon: LucideIcon
}

const parts: readonly BuiltPart[] = [
  {
    title: 'Data and tokenizer',
    description:
      'Text is cleaned, de-duplicated and split, then encoded with a custom byte-level BPE tokenizer (8,192 ids). The first experiments used a 65-character vocabulary on Shakespeare; the final models predict tokens, not single characters.',
    icon: Wrench,
  },
  {
    title: 'The Transformer',
    description:
      'A GPT-style decoder written layer by layer in PyTorch: embeddings, causal self-attention, feed-forward blocks and tied output weights — no ready-made language-model library.',
    icon: Network,
  },
  {
    title: 'Pretraining and chat fine-tuning',
    description:
      'The model starts from random weights and learns next-token prediction on educational web text, then is fine-tuned on human-written conversations so it answers in the assistant role and learns when to stop.',
    icon: Repeat,
  },
  {
    title: 'Local inference',
    description:
      'A FastAPI service and React UI run the trained weights on this machine. The public website is static documentation; it does not host a live model.',
    icon: MessageSquare,
  },
]

export function WhatIBuilt() {
  return (
    <Section id="what-i-built" labelledBy="what-i-built-title" divider>
      <Container>
        <SectionHeading
          id="what-i-built-title"
          eyebrow="What I built"
          title="An end-to-end system, built from the ground up"
          description="Text preparation → tokenization → Transformer → pretraining → conversational fine-tuning → local inference. Everything is written in Python with PyTorch."
        />

        <ul className="mt-12 grid gap-5 sm:grid-cols-2">
          {parts.map(({ title, description, icon: Icon }) => (
            <li key={title}>
              <Card as="article" className="h-full">
                <Icon aria-hidden="true" className="size-5 text-accent" />
                <h3 className="mt-5 text-title text-fg">{title}</h3>
                <p className="mt-2 text-body text-pretty text-fg-muted">{description}</p>
              </Card>
            </li>
          ))}
        </ul>
      </Container>
    </Section>
  )
}

import type { ReactNode } from 'react'

import { Container } from '@/components/ui/Container'
import { Section } from '@/components/ui/Section'
import { SectionHeading } from '@/components/ui/SectionHeading'

import { HistoricalAside } from './HistoricalAside'
import {
  EmbeddingsVisual,
  GenerationVisual,
  LayersVisual,
  PromptVisual,
  TokenIdsVisual,
} from './StepVisuals'
import { UnderTheHood } from './UnderTheHood'

interface Step {
  title: string
  body: string
  visual: ReactNode
}

const steps: readonly Step[] = [
  {
    title: 'Text becomes a token sequence',
    body: 'The model never “reads” words. Incoming text is serialized into a list of token ids — including chat role markers in the local app — before any neural network runs.',
    visual: <PromptVisual />,
  },
  {
    title: 'Byte-level BPE assigns integer ids',
    body: 'A custom byte-level BPE tokenizer splits text into variable-length pieces and maps each piece to an id from an 8,192-entry vocabulary. The ids below are from the real v1.0 tokenizer.',
    visual: <TokenIdsVisual />,
  },
  {
    title: 'Ids become vectors',
    body: 'Each id selects a learned 384-dimensional token embedding. A learned positional embedding for that index is added, producing the sequence the Transformer will process (at most 512 tokens).',
    visual: <EmbeddingsVisual />,
  },
  {
    title: 'Six Transformer layers build context',
    body: 'Six pre-LayerNorm blocks run in order. Inside each block, six attention heads let every position gather information from earlier positions only — never from the future.',
    visual: <LayersVisual />,
  },
  {
    title: 'Predict the next token, then repeat',
    body: 'A tied output head scores all 8,192 vocabulary ids. Decoding picks one token, appends it, and the loop continues until an end-of-turn token or a length limit.',
    visual: <GenerationVisual />,
  },
]

export function ArchitectureSection() {
  return (
    <Section id="architecture" labelledBy="architecture-title" divider>
      <Container>
        <SectionHeading
          id="architecture-title"
          eyebrow="Architecture"
          title="How the 14M model works, in five steps"
          description="The default Mini_Chat chatbot predicts the next token with a GPT-style Transformer: custom byte-level BPE, 13,989,888 parameters, context 512. The diagrams teach the real architecture; illustrative vectors and masks are labeled when they are not measured weights."
        />

        <ol className="mt-14">
          {steps.map((step, index) => (
            <li
              key={step.title}
              className="group relative grid gap-5 pb-12 pl-12 last:pb-0 sm:pl-14 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-12"
            >
              {index < steps.length - 1 && (
                <span
                  aria-hidden="true"
                  className="absolute top-10 bottom-1 left-4 w-px bg-border"
                />
              )}
              <span
                aria-hidden="true"
                className="absolute top-0 left-0 flex size-8 items-center justify-center rounded-pill border border-border bg-background font-mono text-small text-accent transition-colors group-hover:border-accent"
              >
                {index + 1}
              </span>

              <div className="lg:pt-1">
                <h3 className="text-title text-fg">
                  <span className="sr-only">Step {index + 1}: </span>
                  {step.title}
                </h3>
                <p className="mt-2 max-w-md text-body text-pretty text-fg-muted">{step.body}</p>
              </div>

              <div className="min-w-0">{step.visual}</div>
            </li>
          ))}
        </ol>

        <UnderTheHood />
        <HistoricalAside />
      </Container>
    </Section>
  )
}

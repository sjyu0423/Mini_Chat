import { Container } from '@/components/ui/Container'
import { Section } from '@/components/ui/Section'
import { SectionHeading } from '@/components/ui/SectionHeading'

const steps = [
  'Build the model',
  'Train from scratch',
  'Fine-tune for chat',
  'Run it locally',
] as const

export function Motivation() {
  return (
    <Section id="motivation" labelledBy="motivation-title" divider>
      <Container className="grid gap-10 lg:grid-cols-[5fr_7fr] lg:gap-16">
        <SectionHeading id="motivation-title" eyebrow="Motivation" title="Why did I build this?" />

        <div className="max-w-reading space-y-5 text-lede text-pretty text-fg-muted">
          <p>
            Using AI through an API shows you what it can do, but not how it works. I wanted to
            understand what actually happens inside a language model, so I built one myself
            instead of downloading one.
          </p>
          <p>
            So I did every step myself: preparing text, writing a tokenizer and Transformer,
            pretraining from random weights, fine-tuning for chat, and running a local demo. I kept
            it small enough to train on my own computer, so I could see each part working.
          </p>

          <ol
            aria-label="Project steps"
            className="grid gap-px overflow-hidden rounded-card border border-border bg-border sm:grid-cols-2"
          >
            {steps.map((step, index) => (
              <li key={step} className="flex items-baseline gap-3 bg-surface px-5 py-4">
                <span aria-hidden="true" className="font-mono text-small text-accent">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <span className="text-body text-fg">{step}</span>
              </li>
            ))}
          </ol>
        </div>
      </Container>
    </Section>
  )
}

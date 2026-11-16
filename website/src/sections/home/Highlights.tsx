import { Container } from '@/components/ui/Container'
import { Section } from '@/components/ui/Section'
import { SectionHeading } from '@/components/ui/SectionHeading'
import { highlights, highlightsFootnote } from '@/content/facts'

export function Highlights() {
  return (
    <Section id="highlights" labelledBy="highlights-title" divider>
      <Container>
        <SectionHeading
          id="highlights-title"
          eyebrow="Verified results"
          title="A few numbers"
          description="Taken directly from logged runs in the repository."
        />

        <dl className="mt-12 grid gap-px overflow-hidden rounded-card border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
          {highlights.map((fact) => (
            <div key={fact.id} className="flex flex-col bg-background p-6 sm:p-7">
              <dt className="order-2 mt-3 text-small font-medium text-fg">{fact.label}</dt>
              <dd className="order-1 font-mono text-[1.625rem] leading-tight tracking-tight text-fg tabular-nums">
                {fact.value}
              </dd>
              <dd className="order-3 mt-2 text-small text-pretty text-fg-muted">
                {fact.description}
              </dd>
            </div>
          ))}
        </dl>

        <p className="mt-6 max-w-reading text-small text-pretty text-fg-muted">
          {highlightsFootnote}
        </p>
      </Container>
    </Section>
  )
}

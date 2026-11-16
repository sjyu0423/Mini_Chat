import { Check, Circle, CircleDot, type LucideIcon } from 'lucide-react'

import { Container } from '@/components/ui/Container'
import { Section } from '@/components/ui/Section'
import { SectionHeading } from '@/components/ui/SectionHeading'
import { roadmap, type RoadmapStatus } from '@/content/roadmap'
import { cn } from '@/lib/cn'

const statusStyles: Record<RoadmapStatus, { icon: LucideIcon; label: string; className: string }> = {
  done: { icon: Check, label: 'Done', className: 'text-accent' },
  'in-progress': { icon: CircleDot, label: 'In progress', className: 'text-accent' },
  planned: { icon: Circle, label: 'Planned', className: 'text-fg-muted' },
}

export function StatusRoadmap() {
  return (
    <Section id="status" labelledBy="status-title" divider>
      <Container>
        <SectionHeading
          id="status-title"
          eyebrow="Status"
          title="Where the project is now"
          description="Training is complete through v1.0. The code and local demo are ready; this public site is being synchronized with the finished results."
        />

        <div className="mt-12 grid gap-10 md:grid-cols-3 md:gap-8">
          {roadmap.map((group) => (
            <div key={group.id}>
              <h3 id={`status-${group.id}`} className="border-b border-border pb-3 text-title text-fg">
                {group.title}
              </h3>
              <ul aria-labelledby={`status-${group.id}`} className="mt-4 space-y-3.5">
                {group.items.map((item) => {
                  const status = statusStyles[item.status]
                  const Icon = status.icon
                  return (
                    <li key={item.text} className="flex gap-3 text-body text-pretty text-fg-muted">
                      <Icon
                        aria-hidden="true"
                        className={cn('mt-1 size-4 shrink-0', status.className)}
                      />
                      <span>
                        {item.text}
                        {item.status !== 'done' && (
                          <span
                            className={cn(
                              'ml-2 inline-block text-eyebrow whitespace-nowrap uppercase',
                              status.className,
                            )}
                          >
                            {status.label}
                          </span>
                        )}
                        {item.status === 'done' && <span className="sr-only"> ({status.label})</span>}
                      </span>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 max-w-reading space-y-3 border-t border-border pt-8">
          <h3 className="text-title text-fg">What it can and cannot do</h3>
          <p className="text-body text-pretty text-fg-muted">
            After fine-tuning, the model can reply in a chat turn, write short fluent English, and
            usually stop on its own. It is not a reliable source of facts: simple questions are often
            answered incorrectly, multi-turn tracking is shallow, and it only handles English.
          </p>
          <p className="text-small text-pretty text-fg-muted">
            This website documents the engineering work. It does not run the model in your browser
            and does not host a public chatbot.
          </p>
        </div>
      </Container>
    </Section>
  )
}

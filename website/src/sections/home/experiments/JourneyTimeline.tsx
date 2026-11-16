import { journeyPhases } from '@/content/experiments/phases'

export function JourneyTimeline() {
  return (
    <ol className="mt-12 space-y-4">
      {journeyPhases.map((phase) => (
        <li key={phase.id}>
          <details className="group rounded-card border border-border bg-surface open:bg-surface">
            <summary className="flex cursor-pointer list-none flex-col gap-3 rounded-card px-5 py-5 transition-colors hover:bg-surface-elevated focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent sm:flex-row sm:items-start sm:gap-6 sm:px-7 [&::-webkit-details-marker]:hidden">
              <span
                aria-hidden="true"
                className="flex size-10 shrink-0 items-center justify-center rounded-pill border border-accent/40 bg-background font-mono text-body text-accent"
              >
                {phase.letter}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <span className="text-title text-fg">{phase.title}</span>
                  <span className="font-mono text-small text-fg-muted">{phase.stageLabel}</span>
                </span>
                <span className="mt-2 block text-body text-pretty text-fg-muted">{phase.changed}</span>
                <span className="mt-3 block text-small text-pretty text-fg">
                  <span className="text-fg-muted">Why it mattered: </span>
                  {phase.why}
                </span>
                <ul className="mt-3 space-y-1.5 text-small text-pretty text-fg-muted">
                  {phase.findings.map((finding) => (
                    <li key={finding} className="flex gap-2">
                      <span aria-hidden="true" className="text-accent">
                        ·
                      </span>
                      <span>{finding}</span>
                    </li>
                  ))}
                </ul>
                <span className="mt-3 inline-block text-small text-accent-strong group-open:hidden">
                  More detail
                </span>
              </span>
            </summary>
            <div className="border-t border-border px-5 py-5 sm:px-7 sm:pl-[4.75rem]">
              <p className="max-w-reading text-body text-pretty text-fg-muted">{phase.detail}</p>
            </div>
          </details>
        </li>
      ))}
    </ol>
  )
}

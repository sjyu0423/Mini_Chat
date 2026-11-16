import { ChevronDown, ChevronRight } from 'lucide-react'

import { engineeringModules } from '@/content/engineering/modules'

export function EngineeringModules() {
  return (
    <div>
      <h3 className="text-title text-fg">Technical deep dive</h3>
      <p className="mt-2 max-w-reading text-body text-pretty text-fg-muted">
        Expand a module for equations, tensor shapes, and engineering notes. Architecture and
        Experiments already cover the interactive walkthrough and recorded curves.
      </p>

      <ul className="mt-8 grid gap-4">
        {engineeringModules.map((mod, index) => (
          <li key={mod.id}>
            <details className="group rounded-card border border-border bg-surface open:bg-surface">
              <summary className="flex cursor-pointer list-none items-start justify-between gap-4 rounded-card px-5 py-5 transition-colors hover:bg-surface-elevated focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent sm:px-7 [&::-webkit-details-marker]:hidden">
                <span className="min-w-0">
                  <span className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <span className="font-mono text-small text-accent">
                      {String(index + 1).padStart(2, '0')}
                    </span>
                    <span className="text-title text-fg">{mod.title}</span>
                  </span>
                  <span className="mt-2 block text-small text-pretty text-fg-muted">
                    {mod.summary}
                  </span>
                </span>
                <ChevronDown
                  aria-hidden="true"
                  className="mt-1 size-5 shrink-0 text-fg-muted transition-transform group-open:rotate-180"
                />
              </summary>

              <div className="min-w-0 border-t border-border px-5 py-6 sm:px-7 sm:py-7">
                <ul className="space-y-2.5 text-body text-fg">
                  {mod.bullets.map((bullet) => (
                    <li key={bullet} className="flex gap-3">
                      <span aria-hidden="true" className="mt-2 size-1.5 shrink-0 rounded-pill bg-accent" />
                      <span className="min-w-0 text-pretty [overflow-wrap:anywhere]">{bullet}</span>
                    </li>
                  ))}
                </ul>

                {mod.equation ? (
                  <pre className="mt-6 max-w-full overflow-x-auto rounded-control border border-border bg-background p-4 font-mono text-small leading-relaxed text-fg whitespace-pre-wrap [overflow-wrap:anywhere]">
                    {mod.equation}
                  </pre>
                ) : null}

                <p className="mt-5 max-w-reading text-small text-pretty text-fg-muted">{mod.detail}</p>

                <a
                  href={mod.related}
                  className="mt-5 inline-flex items-center gap-2 text-small text-accent-strong underline-offset-4 transition-colors hover:text-accent hover:underline focus-visible:rounded-control focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  Related section
                  <ChevronRight aria-hidden="true" className="size-3.5" />
                </a>
              </div>
            </details>
          </li>
        ))}
      </ul>
    </div>
  )
}

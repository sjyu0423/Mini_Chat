import { ExternalLink } from 'lucide-react'

import { engineeringDocLinks } from '@/content/engineering/docsLinks'

export function EngineeringDocs() {
  return (
    <div>
      <h3 className="text-title text-fg">Repository documentation</h3>
      <p className="mt-2 max-w-reading text-body text-pretty text-fg-muted">
        Commit-pinned links into the v1.0-release tree. Model weights are not published from this
        site.
      </p>

      <ul className="mt-8 grid gap-4 sm:grid-cols-2">
        {engineeringDocLinks.map((link) => (
          <li key={link.id} className="min-w-0">
            <a
              href={link.href}
              target={link.external ? '_blank' : undefined}
              rel={link.external ? 'noopener noreferrer' : undefined}
              className="flex h-full min-w-0 flex-col rounded-card border border-border bg-surface p-5 transition-colors hover:border-border-strong hover:bg-surface-elevated focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent sm:p-6"
            >
              <span className="flex items-start justify-between gap-3">
                <span className="text-title text-fg">{link.label}</span>
                {link.external ? (
                  <ExternalLink aria-hidden="true" className="mt-1 size-4 shrink-0 text-fg-muted" />
                ) : null}
              </span>
              <span className="mt-2 text-small text-pretty text-fg-muted">{link.description}</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}

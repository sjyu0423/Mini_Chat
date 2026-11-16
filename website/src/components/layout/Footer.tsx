import { ArrowLeft, ArrowUpRight } from 'lucide-react'

import { GitHubMark } from '@/components/icons/GitHubMark'
import { Container } from '@/components/ui/Container'
import { site } from '@/config/site'

import { Wordmark } from './Wordmark'

const linkClassName =
  'inline-flex items-center gap-2 rounded-control py-1 text-small text-fg-muted transition-colors hover:text-fg [&_svg]:size-4'

export function Footer() {
  return (
    <footer className="border-t border-border">
      <Container className="grid gap-10 py-14 sm:py-16 md:grid-cols-[1fr_auto] md:items-end">
        <div className="max-w-sm space-y-3">
          <Wordmark className="text-[1.0625rem]" />
          <p className="text-small text-fg-muted">{site.summary}</p>
        </div>

        <nav aria-label="Footer">
          <ul className="flex flex-col gap-3 sm:flex-row sm:gap-8">
            <li>
              <a href={site.repoUrl} target="_blank" rel="noopener noreferrer" className={linkClassName}>
                <GitHubMark />
                GitHub repository
                <ArrowUpRight aria-hidden="true" />
                <span className="sr-only"> (opens in a new tab)</span>
              </a>
            </li>
            <li>
              <a href={site.portfolioUrl} className={linkClassName}>
                <ArrowLeft aria-hidden="true" />
                Back to Portfolio
              </a>
            </li>
          </ul>
        </nav>
      </Container>
    </footer>
  )
}

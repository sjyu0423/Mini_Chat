import { ChevronDown } from 'lucide-react'

import {
  examplePrompt,
  examplePromptChars,
  examplePromptIds,
  historicalArchitectureFacts,
  m1SourceUrl,
} from '@/content/facts'

import { displayChar, spokenChar } from './charDisplay'

/**
 * Optional historical M1 character-level example.
 * Numbers here must not be read as properties of the v1.0 14M chatbot.
 */
export function HistoricalAside() {
  return (
    <details className="group mt-6 rounded-card border border-border/80 bg-background">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 rounded-card px-5 py-4 transition-colors hover:bg-surface focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent sm:px-7 [&::-webkit-details-marker]:hidden">
        <span>
          <span className="block text-small font-medium uppercase tracking-[0.14em] text-fg-muted">
            Historical
          </span>
          <span className="mt-1 block text-title text-fg">How it started: character-level GPT</span>
          <span className="mt-1 block text-small text-fg-muted">
            Tiny Shakespeare, 65 characters, 10,770,816 parameters, context 256 — an early teaching
            model, not the served chatbot.
          </span>
        </span>
        <ChevronDown
          aria-hidden="true"
          className="size-5 shrink-0 text-fg-muted transition-transform group-open:rotate-180"
        />
      </summary>

      <div className="border-t border-border px-5 py-6 sm:px-7">
        <p className="max-w-reading text-body text-pretty text-fg-muted">
          The first experiments predicted one character at a time from a fixed 65-character
          alphabet. That made the loop easy to see. General-language training later replaced it with
          byte-level BPE and a much larger vocabulary.
        </p>

        <dl className="mt-6 grid gap-x-8 gap-y-5 sm:grid-cols-3">
          {historicalArchitectureFacts.map((fact) => (
            <div key={fact.id}>
              <dt className="text-small text-fg-muted">{fact.label}</dt>
              <dd className="mt-1 font-mono text-[1.0625rem] text-fg">{fact.value}</dd>
              <dd className="mt-1.5 text-small text-pretty text-fg-muted">{fact.description}</dd>
            </div>
          ))}
        </dl>

        <figure className="mt-8 rounded-card border border-border bg-surface p-5">
          <ol
            aria-label={`Historical M1 encoding of “${examplePrompt}”`}
            className="flex flex-wrap gap-1.5"
          >
            {examplePromptChars.map((char, index) => (
              <li
                key={index}
                className="flex w-9 flex-col items-center rounded-control border border-border bg-background py-1.5 font-mono"
              >
                <span aria-hidden="true" className="text-small text-fg">
                  {displayChar(char)}
                </span>
                <span className="sr-only">{spokenChar(char)} becomes </span>
                <span className="text-[0.6875rem] text-accent tabular-nums">
                  {examplePromptIds[index]}
                </span>
              </li>
            ))}
          </ol>
          <figcaption className="mt-4 text-small text-pretty text-fg-muted">
            Historical only: real M1 character ids for “{examplePrompt}”. The v1.0 model does not use
            this vocabulary.
          </figcaption>
        </figure>

        <p className="mt-6 min-w-0 text-small text-fg-muted [overflow-wrap:anywhere]">
          Source:{' '}
          <a
            href={m1SourceUrl('configs/gpt_10m.toml')}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-[0.2rem] font-mono break-all text-accent-strong underline decoration-accent/40 underline-offset-4"
          >
            configs/gpt_10m.toml
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
          .
        </p>
      </div>
    </details>
  )
}

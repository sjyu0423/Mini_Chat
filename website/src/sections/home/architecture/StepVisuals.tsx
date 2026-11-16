import type { ReactNode } from 'react'

import {
  chatPrefixTokens,
  exampleText,
  exampleTokens,
} from '@/content/bpeExample'
import { cn } from '@/lib/cn'

import { displayTokenSurface, spokenTokenSurface } from './charDisplay'

function Panel({ children, caption }: { children: ReactNode; caption?: ReactNode }) {
  return (
    <figure className="rounded-card border border-border bg-surface p-5 sm:p-6">
      {children}
      {caption && (
        <figcaption className="mt-4 text-small text-pretty text-fg-muted">{caption}</figcaption>
      )}
    </figure>
  )
}

export function PromptVisual() {
  return (
    <Panel caption="In the local chatbot, the same question is wrapped with role markers such as <|user|> and <|end|> before it reaches the model. Those markers are special token IDs, not ordinary text.">
      <p className="font-mono text-[1.0625rem] text-fg sm:text-lg">
        {exampleText}
        <span
          aria-hidden="true"
          className="ml-0.5 inline-block h-[1.1em] w-[0.5ch] translate-y-[0.18em] bg-accent/60"
        />
      </p>
      <ol
        aria-label="Chat serialization around the same question"
        className="mt-5 flex flex-wrap gap-1.5"
      >
        {chatPrefixTokens.map((piece, index) => (
          <li
            key={`${String(piece.id)}-${String(index)}`}
            className={cn(
              'rounded-control border px-2 py-1 font-mono text-small',
              piece.surface.startsWith('<|')
                ? 'border-accent/50 bg-accent/10 text-accent-strong'
                : 'border-border bg-background text-fg-muted',
            )}
          >
            <span className="sr-only">
              {spokenTokenSurface(piece.surface)}, id {piece.id}
            </span>
            <span aria-hidden="true">{displayTokenSurface(piece.surface)}</span>
          </li>
        ))}
      </ol>
    </Panel>
  )
}

export function TokenIdsVisual() {
  return (
    <Panel caption="Real IDs from the v1.0 byte-level BPE tokenizer (fingerprint e6be77cc…). Leading spaces are part of the token when present.">
      <ol aria-label={`“${exampleText}” as BPE token IDs`} className="flex flex-wrap gap-2">
        {exampleTokens.map((piece) => (
          <li
            key={piece.id}
            className="flex min-w-[4.5rem] flex-col items-center rounded-control border border-border bg-background px-2 py-2 font-mono"
          >
            <span aria-hidden="true" className="text-body text-fg">
              {displayTokenSurface(piece.surface)}
            </span>
            <span className="sr-only">{spokenTokenSurface(piece.surface)} becomes </span>
            <span className="mt-1 text-small text-accent tabular-nums">{piece.id}</span>
          </li>
        ))}
      </ol>
    </Panel>
  )
}

/** Illustrative embedding diagram — values are schematic, not model weights. */
export function EmbeddingsVisual() {
  return (
    <Panel caption="Illustrative only: the dots stand for a 384-dimensional vector. Real embedding weights are not shown.">
      <div className="space-y-3">
        {exampleTokens.map((piece, index) => (
          <div
            key={piece.id}
            className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 sm:grid-cols-[7rem_minmax(0,1fr)]"
          >
            <div className="font-mono text-small text-fg">
              <span className="sr-only">
                Token {spokenTokenSurface(piece.surface)}, id {piece.id}, position {index}.
              </span>
              <span aria-hidden="true" className="block truncate">
                {displayTokenSurface(piece.surface)}
              </span>
              <span aria-hidden="true" className="text-fg-muted">
                id {piece.id} · pos {index}
              </span>
            </div>
            <div
              role="img"
              aria-hidden="true"
              className="flex h-9 items-center gap-0.5 overflow-hidden rounded-control border border-border bg-background px-2"
            >
              {Array.from({ length: 18 }, (_, dot) => (
                <span
                  key={dot}
                  className="h-3 w-1.5 shrink-0 rounded-[1px] bg-accent/35"
                  style={{ opacity: 0.35 + ((piece.id + index * 7 + dot * 3) % 7) / 12 }}
                />
              ))}
              <span className="ml-2 shrink-0 font-mono text-[0.6875rem] text-fg-muted">ℝ³⁸⁴</span>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-4 font-mono text-small text-fg-muted">
        e = E_tok[id] + E_pos[index] → shape (T, 384), T ≤ 512
      </p>
    </Panel>
  )
}

export function LayersVisual() {
  const labels = exampleTokens.map((piece) => displayTokenSurface(piece.surface))
  const lastIndex = labels.length - 1
  const n = labels.length

  return (
    <Panel caption="Causal mask: when predicting from the final “?”, the model may use earlier tokens, never later ones. Mask cells are schematic — not measured attention weights.">
      <div
        role="img"
        aria-label="The six Transformer layers run in order, one after another. Each layer has six attention heads."
        className="grid grid-cols-6 gap-1.5"
      >
        {Array.from({ length: 6 }, (_, index) => (
          <span
            key={index}
            aria-hidden="true"
            className="flex flex-col items-center rounded-control border border-border-strong bg-surface-elevated py-2.5"
          >
            <span className="text-[0.6875rem] leading-4 text-fg-muted">layer</span>
            <span className="font-mono text-body text-fg">{index + 1}</span>
          </span>
        ))}
      </div>
      <p className="mt-3 font-mono text-small text-fg-muted">6 heads × 64 dims inside each layer</p>

      <div
        role="img"
        aria-label={`Causal attention mask for ${String(n)} tokens. A filled cell means the row token may attend to the column token. Future positions are blocked.`}
        className="mt-5 inline-grid gap-1"
        style={{
          gridTemplateColumns: `auto repeat(${String(n)}, minmax(1.75rem, 2.25rem))`,
        }}
      >
        <span aria-hidden="true" />
        {labels.map((label, col) => (
          <span
            key={`h-${String(col)}`}
            aria-hidden="true"
            className="truncate text-center font-mono text-[0.625rem] text-fg-muted"
          >
            {label}
          </span>
        ))}
        {labels.map((rowLabel, row) => (
          <div key={`r-${String(row)}`} className="contents">
            <span
              aria-hidden="true"
              className="truncate pr-2 text-right font-mono text-[0.625rem] text-fg-muted"
            >
              {rowLabel}
            </span>
            {labels.map((_, col) => {
              const allowed = col <= row
              const isQuery = row === lastIndex && allowed
              return (
                <span
                  key={`${String(row)}-${String(col)}`}
                  aria-hidden="true"
                  className={cn(
                    'flex aspect-square items-center justify-center rounded-[0.25rem] border font-mono text-[0.625rem]',
                    allowed
                      ? isQuery
                        ? 'border-accent bg-accent/20 text-accent-strong'
                        : 'border-border bg-accent/10 text-fg-muted'
                      : 'border-border/60 bg-background text-fg-muted/40',
                  )}
                >
                  {allowed ? '·' : '×'}
                </span>
              )
            })}
          </div>
        ))}
      </div>
    </Panel>
  )
}

export function GenerationVisual() {
  return (
    <Panel caption="Illustration of the generation loop. No next-token probabilities are shown — those depend on the live weights and decoding settings.">
      <ol className="space-y-3 text-small text-fg-muted">
        <li className="rounded-control border border-border bg-background px-3 py-2.5">
          <span className="font-mono text-accent">1.</span> Final LayerNorm + tied LM head →{' '}
          <span className="text-fg">8,192 logits</span> (one score per vocabulary id).
        </li>
        <li className="rounded-control border border-border bg-background px-3 py-2.5">
          <span className="font-mono text-accent">2.</span> Local chat decoding uses temperature{' '}
          <span className="text-fg">0.7</span>, top-k <span className="text-fg">40</span>, and
          repetition penalty <span className="text-fg">1.2</span> by default.
        </li>
        <li className="rounded-control border border-border bg-background px-3 py-2.5">
          <span className="font-mono text-accent">3.</span> The chosen token id is appended to the
          sequence.
        </li>
        <li className="rounded-control border border-border bg-background px-3 py-2.5">
          <span className="font-mono text-accent">4.</span> Repeat until{' '}
          <span className="font-mono text-fg">{'<|end|>'}</span> (id 8191) or a token limit.
        </li>
      </ol>
      <p className="mt-4 font-mono text-small text-fg">
        {exampleText}
        <span className="text-fg-muted"> → … → </span>
        <span className="rounded-[0.3rem] border border-accent px-1.5 py-0.5 text-accent-strong">
          {'<|end|>'}
        </span>
      </p>
    </Panel>
  )
}

import { useId, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import {
  recordedConversations,
  recordedDecoding,
  recordedModels,
  recordedSource,
  type DemoOutcome,
  type RecordedConversation,
  type RecordedReply,
} from '@/content/engineering/recordedDemo'
import { cn } from '@/lib/cn'

const outcomeLabel: Record<DemoOutcome, string> = {
  correct: 'Correct answer',
  limitation: 'Limitation',
  behavior: 'Conversational behavior',
}

const outcomeTone: Record<DemoOutcome, 'accent' | 'neutral'> = {
  correct: 'accent',
  limitation: 'neutral',
  behavior: 'neutral',
}

function promptLabel(conversation: RecordedConversation): string {
  const firstUser = conversation.turns.find((t) => t.role === 'user')
  return firstUser?.content ?? conversation.id
}

function ReplyBlock({
  modelKey,
  reply,
}: {
  modelKey: 'm5' | 'm6_3'
  reply: RecordedReply
}) {
  const model = recordedModels[modelKey]
  return (
    <div className="min-w-0 rounded-card border border-border bg-background p-4 sm:p-5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <p className="text-title text-fg">{model.label}</p>
        <p className="font-mono text-small text-fg-muted">{model.parameters} parameters</p>
      </div>
      <Badge className="mt-3" tone="accent" mono>
        Recorded · not live
      </Badge>
      <pre className="mt-4 max-h-72 max-w-full overflow-auto rounded-control border border-border bg-surface p-3 font-mono text-small leading-relaxed break-words whitespace-pre-wrap text-fg [overflow-wrap:anywhere]">
        {reply.reply}
      </pre>
      <dl className="mt-4 grid gap-2 font-mono text-[0.75rem] text-fg-muted sm:grid-cols-3">
        <div>
          <dt className="text-fg-muted">stop_reason</dt>
          <dd className="text-fg">{reply.stop_reason}</dd>
        </div>
        <div>
          <dt className="text-fg-muted">new_tokens</dt>
          <dd className="text-fg">{String(reply.new_tokens)}</dd>
        </div>
        <div>
          <dt className="text-fg-muted">content_check</dt>
          <dd className="text-fg">
            {reply.content_check === null ? 'n/a' : reply.content_check ? 'pass' : 'fail'}
          </dd>
        </div>
      </dl>
    </div>
  )
}

function firstConversation(): RecordedConversation {
  const first = recordedConversations[0]
  if (first === undefined) {
    throw new Error('recordedConversations must contain at least one verified transcript')
  }
  return first
}

function conversationById(id: string): RecordedConversation {
  for (const conversation of recordedConversations) {
    if (conversation.id === id) return conversation
  }
  return firstConversation()
}

export function RecordedDemo() {
  const selectId = useId()
  const compareId = useId()
  const [selectedId, setSelectedId] = useState(() => firstConversation().id)
  const [compare, setCompare] = useState(false)

  const selected = conversationById(selectedId)

  return (
    <div id="engineering-demo" className="scroll-mt-header">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-reading">
          <h3 className="text-title text-fg">Recorded model demonstration</h3>
          <p className="mt-2 text-body text-pretty text-fg-muted">
            Verbatim replies from the M7 showcase, captured under fixed decoding. Choose a prompt
            to inspect the default 14M (M5) answer; optionally compare the research 30M (M6.3)
            recording when both exist.
          </p>
        </div>
        <Badge tone="accent" mono>
          Static · no API
        </Badge>
      </div>

      <p
        role="note"
        className="mt-6 max-w-reading rounded-card border border-accent/35 bg-accent/10 px-5 py-4 text-small text-pretty text-fg"
      >
        These are real, unedited responses from a small experimental language model. The answers
        may be incorrect or nonsensical. This website displays recorded results; it does not run
        the model. Fluency is not evidence of factual correctness — a confident sentence can still
        be wrong.
      </p>

      <div className="mt-8 rounded-card border border-border-strong bg-surface-elevated p-5 sm:p-7">
        <div className="grid gap-6 lg:grid-cols-[minmax(0,16rem)_minmax(0,1fr)] lg:gap-8">
          <div className="min-w-0 space-y-5">
            <div>
              <label htmlFor={selectId} className="block text-small text-fg-muted">
                Recorded prompt
              </label>
              <select
                id={selectId}
                value={selected.id}
                onChange={(event) => {
                  setSelectedId(event.target.value)
                }}
                className="mt-2 w-full min-w-0 rounded-control border border-border bg-background px-3 py-2.5 text-small text-fg transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                {recordedConversations.map((conversation) => (
                  <option key={conversation.id} value={conversation.id}>
                    {promptLabel(conversation)}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <p className="text-small text-fg-muted">Example type</p>
              <Badge tone={outcomeTone[selected.outcome]}>{outcomeLabel[selected.outcome]}</Badge>
              <p className="text-small text-pretty text-fg-muted">{selected.outcomeNote}</p>
            </div>

            <div className="rounded-control border border-border bg-background/70 p-3 font-mono text-[0.75rem] leading-relaxed text-fg-muted">
              <p>
                T={String(recordedDecoding.temperature)} · top-k={String(recordedDecoding.top_k)} ·
                rp={String(recordedDecoding.repetition_penalty)} · seed=
                {String(recordedDecoding.seed)}
              </p>
              <p className="mt-2 break-words [overflow-wrap:anywhere]">Source: {recordedSource}</p>
            </div>

            <div className="flex items-start gap-3">
              <input
                id={compareId}
                type="checkbox"
                checked={compare}
                onChange={(event) => {
                  setCompare(event.target.checked)
                }}
                className="mt-1 size-4 shrink-0 rounded-sm border-border accent-[var(--color-accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              />
              <label htmlFor={compareId} className="text-small text-pretty text-fg">
                Compare with recorded 30M (M6.3) response
              </label>
            </div>
          </div>

          <div className="min-w-0 space-y-5">
            <div>
              <p className="text-small text-fg-muted">Conversation context</p>
              <ol className="mt-2 space-y-3">
                {selected.turns.map((turn, index) => (
                  <li
                    key={`${selected.id}-${String(index)}`}
                    className={cn(
                      'min-w-0 rounded-control border px-3 py-2.5 text-small break-words whitespace-pre-wrap [overflow-wrap:anywhere]',
                      turn.role === 'user'
                        ? 'border-border bg-background text-fg'
                        : 'border-border/80 bg-surface text-fg-muted',
                    )}
                  >
                    <span className="font-mono text-[0.75rem] uppercase tracking-wide text-fg-muted">
                      {turn.role}
                    </span>
                    <span className="mt-1 block">{turn.content}</span>
                  </li>
                ))}
              </ol>
            </div>

            <div className={cn('grid gap-4', compare && 'lg:grid-cols-2')}>
              <ReplyBlock modelKey="m5" reply={selected.replies.m5} />
              {compare ? <ReplyBlock modelKey="m6_3" reply={selected.replies.m6_3} /> : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

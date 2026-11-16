import { experimentSamples, type ExperimentSample } from '@/content/experiments/samples'
import { cn } from '@/lib/cn'

const kindStyles: Record<ExperimentSample['kind'], string> = {
  progress: 'text-accent-strong',
  limitation: 'text-fg-muted',
  contrast: 'text-accent',
}

const kindLabels: Record<ExperimentSample['kind'], string> = {
  progress: 'Progress',
  limitation: 'Limitation',
  contrast: 'Contrast',
}

export function SamplesPanel() {
  return (
    <div>
      <h3 className="text-title text-fg">Recorded model outputs</h3>
      <p className="mt-2 max-w-reading text-body text-pretty text-fg-muted">
        Short, unedited excerpts from logged experiments. They are examples of behaviour — not a
        claim that every reply looks like this.
      </p>

      <ul className="mt-8 grid gap-5 lg:grid-cols-2">
        {experimentSamples.map((sample) => (
          <li
            key={sample.id}
            className="flex min-w-0 flex-col rounded-card border border-border bg-surface p-5 sm:p-6"
          >
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <h4 className="text-title text-fg">{sample.title}</h4>
              <span className={cn('text-eyebrow uppercase', kindStyles[sample.kind])}>
                {kindLabels[sample.kind]}
              </span>
            </div>
            <p className="mt-2 font-mono text-small text-fg-muted">
              {sample.model} · {sample.experiment}
            </p>
            <p className="mt-4 min-w-0 text-small text-fg-muted">
              <span className="text-fg">Prompt: </span>
              <span className="font-mono break-words whitespace-pre-wrap [overflow-wrap:anywhere]">
                {sample.prompt}
              </span>
            </p>
            <pre className="mt-3 max-h-56 max-w-full overflow-auto rounded-control border border-border bg-background p-3 font-mono text-small leading-relaxed break-words whitespace-pre-wrap text-fg [overflow-wrap:anywhere]">
              {sample.output}
            </pre>
            <p className="mt-3 text-small text-pretty text-fg-muted">{sample.note}</p>
            <p className="mt-2 min-w-0 font-mono text-[0.75rem] break-words text-fg-muted [overflow-wrap:anywhere]">
              Unedited · source {sample.source}
            </p>
          </li>
        ))}
      </ul>
    </div>
  )
}

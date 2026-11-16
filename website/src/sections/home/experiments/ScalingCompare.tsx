import {
  promotionRule,
  scalingHeadline,
  scalingRows,
  scalingSources,
} from '@/content/experiments/scaling'
import { v1SourceUrl } from '@/content/facts'

export function ScalingCompare() {
  return (
    <div className="min-w-0 rounded-card border border-border bg-surface p-5 sm:p-7">
      <h3 className="text-title text-fg">14M vs 30M — the scaling check</h3>
      <p className="mt-3 max-w-reading text-lede text-pretty text-fg">{scalingHeadline}</p>

      <div className="mt-8 min-w-0 overflow-x-auto">
        <table className="w-full min-w-[32rem] border-collapse text-left text-small">
          <caption className="sr-only">
            Side-by-side metrics for the 14M served model and the 30M research model.
          </caption>
          <thead>
            <tr className="border-b border-border text-fg-muted">
              <th scope="col" className="py-3 pr-4 font-medium">
                Metric
              </th>
              <th scope="col" className="py-3 pr-4 font-mono font-medium text-fg">
                14M (served)
              </th>
              <th scope="col" className="py-3 font-mono font-medium text-fg">
                30M (research)
              </th>
            </tr>
          </thead>
          <tbody>
            {scalingRows.map((row) => (
              <tr key={row.id} className="border-b border-border/70 align-top">
                <th scope="row" className="py-3.5 pr-4 font-medium text-fg">
                  {row.label}
                  <span className="mt-1 block font-sans text-[0.8125rem] font-normal text-pretty text-fg-muted">
                    {row.detail}
                  </span>
                </th>
                <td className="py-3.5 pr-4 font-mono tabular-nums text-fg">{row.m14}</td>
                <td className="py-3.5 font-mono tabular-nums text-fg">{row.m30}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-6 max-w-reading text-body text-pretty text-fg-muted">{promotionRule}</p>
      <p className="mt-4 min-w-0 font-mono text-[0.75rem] text-fg-muted [overflow-wrap:anywhere]">
        Sources:{' '}
        {scalingSources.map((path, index) => (
          <span key={path}>
            <a
              href={v1SourceUrl(path)}
              target="_blank"
              rel="noopener noreferrer"
              className="break-all text-accent-strong underline decoration-accent/40 underline-offset-4"
            >
              {path}
              <span className="sr-only"> (opens in a new tab)</span>
            </a>
            {index < scalingSources.length - 1 ? ', ' : '.'}
          </span>
        ))}
      </p>
    </div>
  )
}

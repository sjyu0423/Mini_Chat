import { cn } from '@/lib/cn'

export interface ChartSeries {
  id: string
  label: string
  /** Stroke uses currentColor via className */
  className: string
  /** Dash pattern for series that must remain distinct without color alone */
  dasharray?: string
  points: readonly { x: number; y: number }[]
}

interface LineChartProps {
  title: string
  description: string
  xLabel: string
  yLabel: string
  series: readonly ChartSeries[]
  formatX?: (value: number) => string
  formatY?: (value: number) => string
  source: string
}

const WIDTH = 640
const HEIGHT = 280
const PAD = { top: 16, right: 16, bottom: 44, left: 52 }

export function LineChart({
  title,
  description,
  xLabel,
  yLabel,
  series,
  formatX = (v) => String(v),
  formatY = (v) => v.toFixed(2),
  source,
}: LineChartProps) {
  const all = series.flatMap((s) => s.points)
  const minX = Math.min(...all.map((p) => p.x))
  const maxX = Math.max(...all.map((p) => p.x))
  const minY = Math.min(...all.map((p) => p.y))
  const maxY = Math.max(...all.map((p) => p.y))
  const yPad = (maxY - minY) * 0.08 || 0.05
  const y0 = minY - yPad
  const y1 = maxY + yPad

  const innerW = WIDTH - PAD.left - PAD.right
  const innerH = HEIGHT - PAD.top - PAD.bottom

  const sx = (x: number) => PAD.left + ((x - minX) / (maxX - minX || 1)) * innerW
  const sy = (y: number) => PAD.top + (1 - (y - y0) / (y1 - y0 || 1)) * innerH

  const yTicks = [y0, (y0 + y1) / 2, y1]
  const xTicks = [minX, (minX + maxX) / 2, maxX]

  return (
    <figure className="rounded-card border border-border bg-surface p-5 sm:p-6">
      <figcaption className="mb-4">
        <h3 className="text-title text-fg">{title}</h3>
        <p className="mt-1 max-w-reading text-small text-pretty text-fg-muted">{description}</p>
      </figcaption>

      <div className="min-w-0 overflow-x-auto">
        <svg
          role="img"
          aria-label={`${title}. ${description}`}
          viewBox={`0 0 ${String(WIDTH)} ${String(HEIGHT)}`}
          className="h-auto w-full max-w-full text-fg"
        >
          {yTicks.map((tick) => (
            <g key={tick}>
              <line
                x1={PAD.left}
                x2={WIDTH - PAD.right}
                y1={sy(tick)}
                y2={sy(tick)}
                className="stroke-border"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 8}
                y={sy(tick) + 4}
                textAnchor="end"
                className="fill-fg-muted"
                fontSize={11}
                fontFamily="IBM Plex Mono, ui-monospace, monospace"
              >
                {formatY(tick)}
              </text>
            </g>
          ))}
          {xTicks.map((tick) => (
            <text
              key={tick}
              x={sx(tick)}
              y={HEIGHT - 14}
              textAnchor="middle"
              className="fill-fg-muted"
              fontSize={11}
              fontFamily="IBM Plex Mono, ui-monospace, monospace"
            >
              {formatX(tick)}
            </text>
          ))}
          <text
            x={PAD.left + innerW / 2}
            y={HEIGHT - 2}
            textAnchor="middle"
            className="fill-fg-muted"
            fontSize={11}
          >
            {xLabel}
          </text>
          <text
            x={16}
            y={PAD.top + innerH / 2}
            textAnchor="middle"
            className="fill-fg-muted"
            fontSize={11}
            transform={`rotate(-90 16 ${String(PAD.top + innerH / 2)})`}
          >
            {yLabel}
          </text>

          {series.map((s) => {
            const d = s.points
              .map((p, i) => `${i === 0 ? 'M' : 'L'} ${String(sx(p.x))} ${String(sy(p.y))}`)
              .join(' ')
            return (
              <g key={s.id}>
                <path
                  d={d}
                  fill="none"
                  className={cn('stroke-current', s.className)}
                  strokeWidth={2.25}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  strokeDasharray={s.dasharray}
                />
                {s.points.map((p) => (
                  <circle
                    key={`${s.id}-${String(p.x)}`}
                    cx={sx(p.x)}
                    cy={sy(p.y)}
                    r={3}
                    className={cn('fill-current', s.className)}
                  />
                ))}
              </g>
            )
          })}
        </svg>
      </div>

      <ul className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-small text-fg-muted">
        {series.map((s) => (
          <li key={s.id} className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className={cn('inline-block h-0.5 w-6', s.className, 'bg-current')}
              style={s.dasharray ? { backgroundImage: 'repeating-linear-gradient(90deg, currentColor 0 4px, transparent 4px 8px)', height: 2 } : undefined}
            />
            <span>{s.label}</span>
          </li>
        ))}
      </ul>

      <p className="mt-3 min-w-0 font-mono text-[0.75rem] break-words text-fg-muted [overflow-wrap:anywhere]">
        Source: {source}
      </p>
    </figure>
  )
}

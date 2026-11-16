import type { LucideIcon } from 'lucide-react'
import type { ComponentPropsWithoutRef } from 'react'

import { cn } from '@/lib/cn'

interface BadgeProps extends ComponentPropsWithoutRef<'span'> {
  tone?: 'neutral' | 'accent'
  icon?: LucideIcon
  /** Static status dot; no animation. */
  dot?: boolean
  /** Use for identifiers, parameters and metrics. */
  mono?: boolean
}

export function Badge({
  tone = 'neutral',
  icon: Icon,
  dot = false,
  mono = false,
  className,
  children,
  ...props
}: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 rounded-pill border px-3 py-1 text-small leading-5',
        tone === 'neutral'
          ? 'border-border bg-surface text-fg-muted'
          : 'border-accent/35 bg-accent/10 text-accent-strong',
        mono && 'font-mono text-[0.8125rem]',
        className,
      )}
      {...props}
    >
      {dot && <span aria-hidden="true" className="size-1.5 rounded-pill bg-accent" />}
      {Icon && <Icon aria-hidden="true" className="size-3.5 shrink-0" />}
      {children}
    </span>
  )
}

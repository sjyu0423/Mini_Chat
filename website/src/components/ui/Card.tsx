import type { ComponentPropsWithoutRef } from 'react'

import { cn } from '@/lib/cn'

interface CardProps extends ComponentPropsWithoutRef<'div'> {
  as?: 'div' | 'article'
  tone?: 'default' | 'elevated'
}

export function Card({ as: Element = 'div', tone = 'default', className, ...props }: CardProps) {
  return (
    <Element
      className={cn(
        'rounded-card border p-6 sm:p-7',
        tone === 'default' ? 'border-border bg-surface' : 'border-border-strong bg-surface-elevated',
        className,
      )}
      {...props}
    />
  )
}

import type { ComponentPropsWithoutRef } from 'react'

import { cn } from '@/lib/cn'

interface SectionProps extends ComponentPropsWithoutRef<'section'> {
  id: string
  /** Id of the heading that names this section. */
  labelledBy: string
  spacing?: 'default' | 'compact'
  divider?: boolean
}

export function Section({
  labelledBy,
  spacing = 'default',
  divider = false,
  className,
  ...props
}: SectionProps) {
  return (
    <section
      aria-labelledby={labelledBy}
      className={cn(
        spacing === 'default' ? 'py-section' : 'py-16 sm:py-20',
        divider && 'border-t border-border',
        className,
      )}
      {...props}
    />
  )
}

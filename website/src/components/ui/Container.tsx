import type { ComponentPropsWithoutRef } from 'react'

import { cn } from '@/lib/cn'

interface ContainerProps extends ComponentPropsWithoutRef<'div'> {
  /** `page` for full layouts, `reading` for prose-width content. */
  width?: 'page' | 'reading'
}

export function Container({ width = 'page', className, ...props }: ContainerProps) {
  return (
    <div
      className={cn(
        'mx-auto w-full px-gutter',
        width === 'page' ? 'max-w-page' : 'max-w-reading',
        className,
      )}
      {...props}
    />
  )
}

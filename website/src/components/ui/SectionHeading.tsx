import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

interface SectionHeadingProps {
  id: string
  title: ReactNode
  eyebrow?: ReactNode
  description?: ReactNode
  level?: 1 | 2 | 3
  size?: 'display' | 'headline' | 'title'
  className?: string
}

const titleSizes = {
  display: 'text-display',
  headline: 'text-headline',
  title: 'text-title',
} as const

const descriptionSizes = {
  display: 'mt-6 text-lede',
  headline: 'mt-5 text-lede',
  title: 'mt-3 text-body',
} as const

export function SectionHeading({
  id,
  title,
  eyebrow,
  description,
  level = 2,
  size = 'headline',
  className,
}: SectionHeadingProps) {
  const Heading = level === 1 ? 'h1' : level === 2 ? 'h2' : 'h3'

  return (
    <div className={cn('max-w-reading', className)}>
      {eyebrow && <p className="mb-4 text-eyebrow text-accent uppercase">{eyebrow}</p>}
      <Heading id={id} className={cn(titleSizes[size], 'text-balance text-fg')}>
        {title}
      </Heading>
      {description && (
        <p className={cn(descriptionSizes[size], 'text-pretty text-fg-muted')}>{description}</p>
      )}
    </div>
  )
}

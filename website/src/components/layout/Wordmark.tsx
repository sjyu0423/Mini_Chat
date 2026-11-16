import { cn } from '@/lib/cn'

interface WordmarkProps {
  className?: string
}

export function Wordmark({ className }: WordmarkProps) {
  return (
    <span className={cn('font-mono font-medium tracking-tight text-fg', className)}>
      Mini<span className="text-accent">_</span>Chat
    </span>
  )
}

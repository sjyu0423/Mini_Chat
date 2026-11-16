import { cn } from '@/lib/cn'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost'
export type ButtonSize = 'sm' | 'md'

const base =
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-control font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0'

const variants: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-background hover:bg-accent-strong',
  secondary:
    'border border-border bg-surface text-fg hover:border-border-strong hover:bg-surface-elevated',
  ghost: 'text-fg-muted hover:bg-surface hover:text-fg',
}

const sizes: Record<ButtonSize, string> = {
  sm: 'h-9 px-3.5 text-small',
  md: 'h-11 px-5 text-small',
}

export function buttonClassName(
  variant: ButtonVariant,
  size: ButtonSize,
  className?: string,
): string {
  return cn(base, variants[variant], sizes[size], className)
}

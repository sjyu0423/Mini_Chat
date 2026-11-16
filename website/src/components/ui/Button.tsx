import type { ComponentPropsWithoutRef } from 'react'

import { buttonClassName, type ButtonSize, type ButtonVariant } from './buttonStyles'

interface ButtonProps extends ComponentPropsWithoutRef<'button'> {
  variant?: ButtonVariant
  size?: ButtonSize
}

export function Button({
  variant = 'primary',
  size = 'md',
  type = 'button',
  className,
  ...props
}: ButtonProps) {
  return <button type={type} className={buttonClassName(variant, size, className)} {...props} />
}

interface LinkButtonProps extends ComponentPropsWithoutRef<'a'> {
  href: string
  variant?: ButtonVariant
  size?: ButtonSize
  /** Opens in a new tab and announces that to screen readers. */
  newTab?: boolean
}

export function LinkButton({
  variant = 'primary',
  size = 'md',
  newTab = false,
  className,
  children,
  ...props
}: LinkButtonProps) {
  return (
    <a
      className={buttonClassName(variant, size, className)}
      {...(newTab ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
      {...props}
    >
      {children}
      {newTab && <span className="sr-only"> (opens in a new tab)</span>}
    </a>
  )
}

import { Menu, X } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'

import { GitHubMark } from '@/components/icons/GitHubMark'
import { LinkButton } from '@/components/ui/Button'
import { navItems, site } from '@/config/site'
import { useActiveSection } from '@/hooks/useActiveSection'
import { cn } from '@/lib/cn'

import { Wordmark } from './Wordmark'

const sectionIds = navItems.map((item) => item.id)
const desktopQuery = '(min-width: 48rem)'

export function Navigation() {
  const [menuOpen, setMenuOpen] = useState(false)
  const activeId = useActiveSection(sectionIds)
  const menuId = useId()
  const headerRef = useRef<HTMLElement>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!menuOpen) return

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setMenuOpen(false)
        toggleRef.current?.focus()
      }
    }

    function handlePointerDown(event: PointerEvent) {
      if (event.target instanceof Node && !headerRef.current?.contains(event.target)) {
        setMenuOpen(false)
      }
    }

    const desktop = window.matchMedia(desktopQuery)
    function handleViewportChange(event: MediaQueryListEvent) {
      if (event.matches) setMenuOpen(false)
    }

    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('pointerdown', handlePointerDown)
    desktop.addEventListener('change', handleViewportChange)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('pointerdown', handlePointerDown)
      desktop.removeEventListener('change', handleViewportChange)
    }
  }, [menuOpen])

  return (
    <header ref={headerRef} className="sticky top-0 z-40 border-b border-border bg-background">
      <div className="mx-auto flex h-header w-full max-w-page items-center justify-between gap-6 px-gutter">
        <a
          href="#top"
          aria-label={`${site.name}, back to top`}
          className="-mx-2 rounded-control px-2 py-1 text-[1.0625rem] transition-opacity hover:opacity-80"
        >
          <Wordmark />
        </a>

        <nav aria-label="Primary" className="hidden md:block">
          <ul className="flex items-center gap-1">
            {navItems.map((item) => {
              const active = item.id === activeId
              return (
                <li key={item.id}>
                  <a
                    href={`#${item.id}`}
                    aria-current={active ? 'location' : undefined}
                    className={cn(
                      'relative block rounded-control px-3 py-2 text-small transition-colors',
                      'after:absolute after:inset-x-3 after:-bottom-px after:h-px after:transition-colors',
                      active
                        ? 'text-fg after:bg-accent'
                        : 'text-fg-muted after:bg-transparent hover:text-fg',
                    )}
                  >
                    {item.label}
                  </a>
                </li>
              )
            })}
          </ul>
        </nav>

        <div className="flex items-center gap-2">
          <LinkButton
            href={site.repoUrl}
            newTab
            variant="secondary"
            className="max-sm:w-11 max-sm:px-0"
          >
            <GitHubMark />
            <span className="max-sm:sr-only">GitHub</span>
          </LinkButton>

          <button
            ref={toggleRef}
            type="button"
            aria-expanded={menuOpen}
            aria-controls={menuId}
            onClick={() => {
              setMenuOpen((open) => !open)
            }}
            className="inline-flex size-11 items-center justify-center rounded-control border border-border text-fg transition-colors hover:bg-surface md:hidden"
          >
            <span className="sr-only">Menu</span>
            {menuOpen ? (
              <X aria-hidden="true" className="size-4.5" />
            ) : (
              <Menu aria-hidden="true" className="size-4.5" />
            )}
          </button>
        </div>
      </div>

      <div id={menuId} hidden={!menuOpen} className="border-t border-border md:hidden">
        <nav aria-label="Primary">
          <ul className="mx-auto w-full max-w-page px-gutter py-3">
            {navItems.map((item) => {
              const active = item.id === activeId
              return (
                <li key={item.id}>
                  <a
                    href={`#${item.id}`}
                    aria-current={active ? 'location' : undefined}
                    onClick={() => {
                      setMenuOpen(false)
                    }}
                    className={cn(
                      'flex min-h-12 items-center border-l-2 pl-4 text-body transition-colors',
                      active
                        ? 'border-accent text-fg'
                        : 'border-transparent text-fg-muted hover:text-fg',
                    )}
                  >
                    {item.label}
                  </a>
                </li>
              )
            })}
          </ul>
        </nav>
      </div>
    </header>
  )
}

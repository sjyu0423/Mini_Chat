import { useEffect, useState } from 'react'

const scrollKeys = new Set(['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '])

/**
 * Returns the id of the section the reader is looking at, for navigation highlighting.
 *
 * - While scrolling: the first listed section crossing a thin band just above the
 *   middle of the viewport. At the bottom of the page the last section wins, since
 *   a short final section may never reach the band.
 * - After an in-page link to a listed section is followed: that section stays active
 *   until the reader scrolls themselves. Near the end of the page several short
 *   sections are visible at once, so position alone cannot tell which was chosen.
 *
 * `ids` must be referentially stable.
 */
export function useActiveSection(ids: readonly string[]): string | null {
  const [activeId, setActiveId] = useState<string | null>(null)

  useEffect(() => {
    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((element): element is HTMLElement => element !== null)

    const lastId = elements.at(-1)?.id
    if (lastId === undefined) return

    const intersecting = new Set<string>()
    let chosenId: string | null = null

    const update = () => {
      if (chosenId !== null) {
        setActiveId(chosenId)
        return
      }
      const atBottom =
        window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2
      setActiveId(atBottom ? lastId : (ids.find((id) => intersecting.has(id)) ?? null))
    }

    const release = () => {
      if (chosenId === null) return
      chosenId = null
      update()
    }

    const handleClick = (event: MouseEvent) => {
      const link = event.target instanceof Element ? event.target.closest('a[href^="#"]') : null
      if (!link) return
      const targetId = link.getAttribute('href')?.slice(1) ?? ''
      chosenId = ids.includes(targetId) ? targetId : null
      update()
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (scrollKeys.has(event.key)) release()
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) intersecting.add(entry.target.id)
          else intersecting.delete(entry.target.id)
        }
        update()
      },
      { rootMargin: '-35% 0px -60% 0px' },
    )

    for (const element of elements) observer.observe(element)
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('wheel', release, { passive: true })
    window.addEventListener('touchmove', release, { passive: true })
    window.addEventListener('pointerdown', release)
    window.addEventListener('keydown', handleKeyDown)
    document.addEventListener('click', handleClick)
    return () => {
      observer.disconnect()
      window.removeEventListener('scroll', update)
      window.removeEventListener('wheel', release)
      window.removeEventListener('touchmove', release)
      window.removeEventListener('pointerdown', release)
      window.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('click', handleClick)
    }
  }, [ids])

  return activeId
}

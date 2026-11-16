/** Visible glyph for a single character; whitespace gets a printable stand-in. */
export function displayChar(char: string): string {
  if (char === '\n') return '↵'
  if (char === ' ') return '␣'
  return char
}

/** Screen-reader name for a single character. */
export function spokenChar(char: string): string {
  if (char === '\n') return 'newline'
  if (char === ' ') return 'space'
  return `“${char}”`
}

/** Visible label for a BPE surface string (may include spaces or newlines). */
export function displayTokenSurface(surface: string): string {
  if (surface.startsWith('<|') && surface.endsWith('|>')) return surface
  return surface.replaceAll('\n', '↵').replaceAll(' ', '␣')
}

/** Screen-reader name for a BPE surface string. */
export function spokenTokenSurface(surface: string): string {
  if (surface === '\n') return 'newline'
  if (surface === ' ') return 'space'
  if (surface.startsWith('<|') && surface.endsWith('|>')) return surface
  const readable = surface.replaceAll('\n', ' newline ').replaceAll(' ', ' space ').trim()
  return `“${readable}”`
}

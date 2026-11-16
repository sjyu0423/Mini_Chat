/**
 * Verified encoding fixture for the architecture walkthrough.
 *
 * Source: ByteBPETokenizer.load("runs/release/mini_chatgpt-v1.0/m5/tokenizer.json")
 * Fingerprint matches docs/showcase/facts.json (e6be77cc…32ce91c8).
 * Surfaces come from vocab byte decoding or special-token names — not invented.
 * Verified during W3.5B against the release tokenizer; this typed module is
 * imported by the Architecture walkthrough.
 */

export interface BpeTokenPiece {
  id: number
  surface: string
}

export const bpeTokenizerMeta = {
  tokenizerPath: 'runs/release/mini_chatgpt-v1.0/m5/tokenizer.json',
  fingerprint: 'e6be77ccff4929187ed84ee1b71d15b7654824abd9e23f3ad889c57232ce91c8',
  fingerprintShort: 'e6be77cc…32ce91c8',
  vocabSize: 8192,
  merges: 7931,
  specialTokens: {
    '<|endoftext|>': 8187,
    '<|user|>': 8188,
    '<|assistant|>': 8189,
    '<|system|>': 8190,
    '<|end|>': 8191,
  },
} as const

/** Plain-text walkthrough example. */
export const exampleText = 'What is Python?'

export const exampleTokens: readonly BpeTokenPiece[] = [
  { id: 1667, surface: 'What' },
  { id: 321, surface: ' is' },
  { id: 4714, surface: ' Python' },
  { id: 63, surface: '?' },
]

/** Chat-role serialization around the same question (specials allowed). */
export const chatPrefixText = '<|user|>\nWhat is Python?<|end|>\n'

export const chatPrefixTokens: readonly BpeTokenPiece[] = [
  { id: 8188, surface: '<|user|>' },
  { id: 10, surface: '\n' },
  { id: 1667, surface: 'What' },
  { id: 321, surface: ' is' },
  { id: 4714, surface: ' Python' },
  { id: 63, surface: '?' },
  { id: 8191, surface: '<|end|>' },
  { id: 10, surface: '\n' },
]

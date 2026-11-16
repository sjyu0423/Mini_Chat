import { ChevronDown } from 'lucide-react'

import {
  architectureFacts,
  architectureSources,
  researchArchitectureFacts,
  v1SourceUrl,
} from '@/content/facts'

export function UnderTheHood() {
  return (
    <details className="group mt-14 rounded-card border border-border bg-surface">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 rounded-card px-5 py-5 transition-colors hover:bg-surface-elevated focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent sm:px-7 [&::-webkit-details-marker]:hidden">
        <span>
          <span className="block text-title text-fg">Under the hood</span>
          <span className="mt-1 block text-small text-fg-muted">
            Default 14M served model: shapes, attention equations, and an optional 30M research
            comparison.
          </span>
        </span>
        <ChevronDown
          aria-hidden="true"
          className="size-5 shrink-0 text-fg-muted transition-transform group-open:rotate-180"
        />
      </summary>

      <div className="border-t border-border px-5 py-6 sm:px-7 sm:py-8">
        <dl className="grid gap-x-10 gap-y-7 sm:grid-cols-2 lg:grid-cols-3">
          {architectureFacts.map((fact) => (
            <div key={fact.id}>
              <dt className="text-small text-fg-muted">{fact.label}</dt>
              <dd className="mt-1 font-mono text-[1.0625rem] text-fg">{fact.value}</dd>
              <dd className="mt-1.5 text-small text-pretty text-fg-muted">{fact.description}</dd>
            </div>
          ))}
        </dl>

        <div className="mt-10 rounded-card border border-border bg-background/60 p-5 sm:p-6">
          <h3 className="text-title text-fg">Block equations (from the code)</h3>
          <p className="mt-2 max-w-reading text-small text-pretty text-fg-muted">
            Notation: T ≤ 512 sequence length, C = 384, H = 6 heads, D = 64, V = 8,192.
          </p>
          <ul className="mt-4 space-y-2 font-mono text-small text-fg-muted">
            <li>x₀ = E_tok[ids] + E_pos[0…T−1] → (B, T, C)</li>
            <li>Q, K, V = x W_qkv ; S = QKᵀ / √D ; S[i,j&gt;i] = −∞ ; A = softmax(S)</li>
            <li>x ← x + Attn(LN₁(x)) ; x ← x + MLP(LN₂(x)) with MLP: C → 4C → C (GELU)</li>
            <li>logits = LN_f(x) @ E_tokᵀ → (B, T, V) · weight-tied</li>
          </ul>
        </div>

        <div className="mt-10">
          <h3 className="text-title text-fg">Research comparison (not served)</h3>
          <p className="mt-2 max-w-reading text-small text-pretty text-fg-muted">
            The 30M model used the same tokenizer, context length, and pretraining token budget. It
            is a controlled experiment, not the default local chatbot.
          </p>
          <dl className="mt-5 grid gap-x-10 gap-y-6 sm:grid-cols-2">
            {researchArchitectureFacts.map((fact) => (
              <div key={fact.id}>
                <dt className="text-small text-fg-muted">{fact.label}</dt>
                <dd className="mt-1 font-mono text-[1.0625rem] text-fg">{fact.value}</dd>
                <dd className="mt-1.5 text-small text-pretty text-fg-muted">{fact.description}</dd>
              </div>
            ))}
          </dl>
        </div>

        <p className="mt-8 min-w-0 border-t border-border pt-5 text-small text-fg-muted [overflow-wrap:anywhere]">
          v1.0 default configuration (
          <span className="font-mono">m4_1_general_s_bpe8192</span> / M5 chat). Sources:{' '}
          {architectureSources.map((path, index) => (
            <span key={path}>
              <a
                href={v1SourceUrl(path)}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-[0.2rem] font-mono break-all text-accent-strong underline decoration-accent/40 underline-offset-4 transition-colors hover:text-fg hover:decoration-fg"
              >
                {path}
                <span className="sr-only"> (opens in a new tab)</span>
              </a>
              {index < architectureSources.length - 1 ? ', ' : '.'}
            </span>
          ))}
        </p>
      </div>
    </details>
  )
}

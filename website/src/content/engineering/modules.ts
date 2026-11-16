/**
 * Engineering deep-dive copy. Numbers must match facts.ts / showcase sources.
 * Detailed W3 visuals and W4 curves are linked, not duplicated.
 */

export interface EngineeringModule {
  id: string
  title: string
  summary: string
  bullets: readonly string[]
  detail: string
  equation?: string
  related: string
}

export const engineeringIntro =
  'Building Mini_Chat meant more than wiring a Transformer: custom data and tokenization, training and chat fine-tuning, honest evaluation, and a local inference stack. The modules below are optional depth — the Architecture and Experiments sections already show how the model works and what the runs measured.'

export const engineeringModules: readonly EngineeringModule[] = [
  {
    id: 'tokenization',
    title: 'Tokenization and data',
    summary:
      'A from-scratch byte-level BPE (8,192 ids) with reserved chat tokens, fingerprinted so checkpoints cannot load the wrong vocabulary.',
    bullets: [
      '256 UTF-8 byte bases + 7,931 merges + 5 specials (`<|user|>`, `<|assistant|>`, `<|end|>`, …).',
      'Train / validation / holdout splits by document hash; exact and MinHash de-duplication before split.',
      'FineWeb-Edu + Cosmopedia v2 mixture for general pretraining; n-gram decontamination against eval suites.',
      'Every tokenizer file carries a SHA-256 fingerprint (e6be77cc…32ce91c8).',
    ],
    detail:
      'Ordinary user text is encoded with specials disabled, so typing “<|end|>” cannot inject a control token. See the Architecture walkthrough for a live example of “What is Python?” → real token ids.',
    related: '#architecture',
  },
  {
    id: 'transformer',
    title: 'Transformer implementation',
    summary:
      'A GPT-2-style decoder written from PyTorch primitives: embeddings, causal multi-head attention, pre-LN residuals, and a tied LM head.',
    bullets: [
      'Project code owns model wiring, init, causal mask logic, and generation.',
      'PyTorch supplies tensors, autograd, MPS, Linear / Embedding / LayerNorm, GELU, and SDPA.',
      'Default shape: 6 layers × 6 heads × width 384, context 512, V = 8,192.',
      'Weight tying: logits = LN(x) @ E_tokᵀ (no extra output parameters).',
    ],
    detail:
      'Causal attention uses QKᵀ/√D with future positions masked to −∞ before softmax. Full step-by-step visuals live in Architecture; equations match src/mini_chatgpt/model/gpt.py.',
    equation: 'x ← x + Attn(LN₁(x));  x ← x + MLP(LN₂(x));  logits = LN_f(x) @ E_tokᵀ',
    related: '#architecture',
  },
  {
    id: 'training',
    title: 'Training and post-training',
    summary:
      'Next-token cross-entropy with AdamW, cosine schedule, clipping, and single-pass data order — then assistant-only SFT so the model learns to answer and to stop.',
    bullets: [
      'Pretraining: 157,286,400 token presentations, each seen once (~110 min for 14M).',
      'SFT: 16,813 conversations; loss only on assistant tokens including `<|end|>` (~8 min).',
      'Checkpoints, seeds, and preflights (causality, gradients, overfit, resume) before long runs.',
      'MPS challenge: allocator cache grew 2.2 → 13.6 GB from shape variety; fixed with shape bucketing, cache release, and a memory guard.',
    ],
    detail:
      'Natural end-of-turn rose from 0% to ~97.5% after SFT under production decoding. Fluency and stopping improved; factual accuracy did not. Recorded loss curves are in Experiments.',
    related: '#experiments',
  },
  {
    id: 'inference',
    title: 'Local inference system',
    summary:
      'A localhost FastAPI service loads verified weights and serves a React UI — one generation at a time on Apple Silicon.',
    bullets: [
      'Path: Browser → FastAPI → custom tokenizer → GPT on MPS → tokens → JSON reply.',
      'Checkpoint + tokenizer fingerprint checks before serving; inference-only export in docs/release.',
      'Decoding defaults: temperature 0.7, top-k 40, repetition penalty 1.2; replies report stop_reason.',
      'Context trimmed to the 512-token window; requests are size-limited and serialized.',
    ],
    detail:
      'Median reply latency ~74 ms in the local app. This public website is static: it shows recorded replies and never calls the model API. Run scripts/serve_chat.py locally for live inference.',
    related: '#engineering-demo',
  },
] as const

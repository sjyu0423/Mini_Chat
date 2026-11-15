// Client for the local inference API (scripts/serve_chat.py). No other network calls.

export type Role = "user" | "assistant";
export type StopReason =
  | "end_of_turn"
  | "max_tokens"
  | "structural_token"
  | "context_full"
  | "error";

export interface HistoryMessage {
  role: Role;
  content: string;
}

export interface GenerationSettings {
  temperature?: number;
  max_new_tokens?: number;
  seed?: number;
}

export interface ChatResponse {
  reply: string;
  stop_reason: StopReason;
  new_tokens: number;
  prompt_tokens: number;
  dropped_history_messages: number;
  duration_ms: number;
  tokens_per_s: number;
  seed: number;
  settings: {
    temperature: number;
    top_k: number;
    repetition_penalty: number;
    max_new_tokens: number;
  };
}

export interface ModelInfo {
  name: string;
  parameters: number;
  context_length: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
  vocab_size: number;
  tokenizer_fingerprint: string;
  device: string;
  load_seconds: number;
}

export interface Health {
  status: "loading" | "ready" | "error";
  error?: string;
  model?: ModelInfo;
  defaults?: { temperature: number; top_k: number; repetition_penalty: number; max_new_tokens: number };
  limits?: {
    max_message_chars: number;
    max_new_tokens: number;
    min_new_tokens: number;
    min_temperature: number;
    max_temperature: number;
  };
}

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return "The request was rejected as malformed.";
  } catch {
    /* fall through */
  }
  return `Server error (${res.status})`;
}

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  const res = await fetch("/api/health", { signal });
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  return (await res.json()) as Health;
}

export async function sendChat(
  message: string,
  history: HistoryMessage[],
  settings: GenerationSettings,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, settings }),
    signal,
  });
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  return (await res.json()) as ChatResponse;
}

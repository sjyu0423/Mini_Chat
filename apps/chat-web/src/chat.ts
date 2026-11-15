// Conversation state helpers (pure functions; unit-tested).
import type { ChatResponse, HistoryMessage, StopReason } from "./api";

export interface UserTurn {
  id: string;
  role: "user";
  content: string;
  failed?: boolean;
}

export interface AssistantTurn {
  id: string;
  role: "assistant";
  content: string;
  meta: Pick<
    ChatResponse,
    "stop_reason" | "new_tokens" | "duration_ms" | "tokens_per_s" | "dropped_history_messages" | "seed"
  >;
}

export type Turn = UserTurn | AssistantTurn;

let counter = 0;
export const newId = (): string => `t${Date.now().toString(36)}${(counter++).toString(36)}`;

/**
 * History sent to the server: complete user/assistant pairs, in order, exactly as shown.
 * A user message whose request failed, or a pair whose reply was empty, is not sent
 * (the API requires non-empty alternating pairs); nothing is rewritten.
 */
export function buildHistory(turns: Turn[]): HistoryMessage[] {
  const history: HistoryMessage[] = [];
  for (let i = 0; i < turns.length - 1; i++) {
    const u = turns[i];
    const a = turns[i + 1];
    if (u?.role === "user" && !u.failed && a?.role === "assistant" && a.content.trim()) {
      history.push({ role: "user", content: u.content }, { role: "assistant", content: a.content });
      i++;
    }
  }
  return history;
}

export interface StopLabel {
  text: string;
  tone: "ok" | "warn" | "error";
  title: string;
}

export function stopLabel(reason: StopReason): StopLabel {
  switch (reason) {
    case "end_of_turn":
      return { text: "ended naturally", tone: "ok", title: "The model produced its end-of-turn token." };
    case "max_tokens":
      return {
        text: "cut off at token limit",
        tone: "warn",
        title: "The reply reached the maximum number of new tokens before the model ended it.",
      };
    case "context_full":
      return {
        text: "cut off: context full",
        tone: "warn",
        title: "The 512-token context window filled up before the reply ended.",
      };
    case "structural_token":
      return {
        text: "stopped: format error",
        tone: "warn",
        title: "The model produced a role or document token instead of ending its turn.",
      };
    default:
      return { text: "error", tone: "error", title: "Generation failed." };
  }
}

export function formatDuration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

const STORAGE_KEY = "mini-gpt-chat:v1";

export function loadTurns(storage: Storage | undefined): Turn[] {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as Turn[]) : [];
  } catch {
    return [];
  }
}

export function saveTurns(storage: Storage | undefined, turns: Turn[]): void {
  try {
    if (turns.length) storage?.setItem(STORAGE_KEY, JSON.stringify(turns));
    else storage?.removeItem(STORAGE_KEY);
  } catch {
    /* storage full or unavailable: the conversation still works in memory */
  }
}

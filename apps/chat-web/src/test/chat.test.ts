import { describe, expect, it } from "vitest";
import { buildHistory, formatDuration, loadTurns, saveTurns, stopLabel } from "../chat";
import type { Turn } from "../chat";

const meta = {
  stop_reason: "end_of_turn" as const,
  new_tokens: 3,
  duration_ms: 10,
  tokens_per_s: 300,
  dropped_history_messages: 0,
  seed: 1,
};

describe("buildHistory", () => {
  it("sends complete user/assistant pairs in order, unmodified", () => {
    const turns: Turn[] = [
      { id: "1", role: "user", content: "Hi" },
      { id: "2", role: "assistant", content: "Hello!", meta },
      { id: "3", role: "user", content: "What is Python?" },
      { id: "4", role: "assistant", content: "A language.", meta },
    ];
    expect(buildHistory(turns)).toEqual([
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello!" },
      { role: "user", content: "What is Python?" },
      { role: "assistant", content: "A language." },
    ]);
  });

  it("skips failed user messages and empty replies", () => {
    const turns: Turn[] = [
      { id: "1", role: "user", content: "lost", failed: true },
      { id: "2", role: "user", content: "Hi" },
      { id: "3", role: "assistant", content: "", meta },
      { id: "4", role: "user", content: "Again" },
      { id: "5", role: "assistant", content: "Yes.", meta },
    ];
    expect(buildHistory(turns)).toEqual([
      { role: "user", content: "Again" },
      { role: "assistant", content: "Yes." },
    ]);
  });
});

describe("labels and storage", () => {
  it("distinguishes natural stops from truncation and errors", () => {
    expect(stopLabel("end_of_turn").tone).toBe("ok");
    expect(stopLabel("max_tokens").text).toMatch(/cut off/);
    expect(stopLabel("structural_token").tone).toBe("warn");
    expect(stopLabel("error").tone).toBe("error");
    expect(formatDuration(250)).toBe("250 ms");
    expect(formatDuration(1520)).toBe("1.52 s");
  });

  it("round-trips the conversation through session storage", () => {
    const turns: Turn[] = [{ id: "1", role: "user", content: "Hi" }];
    saveTurns(sessionStorage, turns);
    expect(loadTurns(sessionStorage)).toEqual(turns);
    saveTurns(sessionStorage, []);
    expect(loadTurns(sessionStorage)).toEqual([]);
    sessionStorage.setItem("mini-gpt-chat:v1", "{not json");
    expect(loadTurns(sessionStorage)).toEqual([]);
  });
});

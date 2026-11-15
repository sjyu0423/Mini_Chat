import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getHealth, sendChat } from "./api";
import type { Health } from "./api";
import { buildHistory, loadTurns, newId, saveTurns } from "./chat";
import type { Turn } from "./chat";
import { Composer } from "./components/Composer";
import { MessageList } from "./components/MessageList";
import { AboutPanel, SettingsPanel } from "./components/Panels";
import { ProjectPage } from "./components/ProjectPage";
import { RecordedConversations } from "./components/Recorded";
import { DEFAULT_SETTINGS } from "./settings";
import type { UiSettings } from "./settings";

export const DISCLAIMER =
  "Experimental language model trained from scratch. Responses may be incorrect or nonsensical.";

const storage = typeof window !== "undefined" ? window.sessionStorage : undefined;

/** Static builds (VITE_STATIC_DEMO=1) have no model server: recorded replies + project page only. */
export const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === "1";

type View = "chat" | "recorded" | "project";

export default function App({
  healthPollMs = 1000,
  staticDemo = STATIC_DEMO,
}: {
  healthPollMs?: number;
  staticDemo?: boolean;
}) {
  const [view, setView] = useState<View>(staticDemo ? "recorded" : "chat");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>(() => loadTurns(storage));
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettings] = useState<UiSettings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Poll /api/health until the model is ready (or failed to load). Never in static mode.
  useEffect(() => {
    if (staticDemo) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const h = await getHealth();
        if (stop) return;
        setHealth(h);
        setHealthError(null);
        if (h.status === "loading") timer = setTimeout(poll, healthPollMs);
      } catch {
        if (stop) return;
        setHealthError("Cannot reach the local model server. Is scripts/serve_chat.py running?");
        timer = setTimeout(poll, healthPollMs * 2);
      }
    };
    void poll();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
    };
  }, [healthPollMs, staticDemo]);

  useEffect(() => saveTurns(storage, turns), [turns]);

  useEffect(() => {
    if (staticDemo) document.title = "Mini GPT · recorded demo";
  }, [staticDemo]);

  const ready = health?.status === "ready";
  const maxChars = health?.limits?.max_message_chars ?? 2000;

  const send = useCallback(
    async (text: string) => {
      const userTurn: Turn = { id: newId(), role: "user", content: text };
      const history = buildHistory(turns);
      setTurns((t) => [...t, userTurn]);
      setDraft("");
      setError(null);
      setPending(true);
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const r = await sendChat(
          text,
          history,
          { temperature: settings.temperature, max_new_tokens: settings.maxNewTokens },
          controller.signal,
        );
        setTurns((t) => [
          ...t,
          {
            id: newId(),
            role: "assistant",
            content: r.reply,
            meta: {
              stop_reason: r.stop_reason,
              new_tokens: r.new_tokens,
              duration_ms: r.duration_ms,
              tokens_per_s: r.tokens_per_s,
              dropped_history_messages: r.dropped_history_messages,
              seed: r.seed,
            },
          },
        ]);
      } catch (e) {
        if (controller.signal.aborted) return;
        setTurns((t) => t.map((x) => (x.id === userTurn.id ? { ...x, failed: true } : x)));
        const msg =
          e instanceof ApiError
            ? e.message
            : "Could not reach the local model server. Check that it is still running.";
        setError(msg);
        setDraft(text);
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
          setPending(false);
        }
      }
    },
    [turns, settings],
  );

  const newChat = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPending(false);
    setTurns([]);
    setError(null);
    setDraft("");
  };

  const status =
    healthError ??
    (health?.status === "error"
      ? `The model failed to load: ${health.error ?? "unknown error"}`
      : health?.status === "loading" || !health
        ? "Loading the model…"
        : null);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            mG
          </span>
          <div>
            <h1>Mini GPT</h1>
            <p className="subtitle">
              A 14M-parameter language model built from scratch · {staticDemo ? "recorded demo" : "local demo"}
            </p>
          </div>
        </div>
        <nav className="actions">
          {view === "chat" && (
            <button type="button" onClick={() => setShowSettings((v) => !v)} aria-expanded={showSettings}>
              Settings
            </button>
          )}
          <button type="button" onClick={() => setShowAbout(true)}>
            About
          </button>
          {view === "chat" && (
            <button type="button" className="primary" onClick={newChat}>
              New chat
            </button>
          )}
        </nav>
      </header>

      <nav className="tabs" aria-label="Views">
        {(staticDemo ? (["recorded", "project"] as const) : (["chat", "recorded", "project"] as const)).map(
          (v) => (
            <button
              key={v}
              type="button"
              className={`tab ${view === v ? "tab-active" : ""}`}
              aria-current={view === v ? "page" : undefined}
              onClick={() => setView(v)}
            >
              {v === "chat" ? "Live chat" : v === "recorded" ? "Recorded" : "Project"}
            </button>
          ),
        )}
      </nav>

      <p className="disclaimer" role="note">
        {DISCLAIMER}
      </p>

      {view === "recorded" && (
        <main className="chat">
          <RecordedConversations
            intro={
              staticDemo
                ? "No model runs on this page. These are real, unedited replies recorded from the trained model with its production settings. Run the project locally for live chat."
                : "Real, unedited replies recorded from the trained model with its production settings (for when the live model is not running)."
            }
          />
        </main>
      )}
      {view === "project" && (
        <main className="chat">
          <ProjectPage />
        </main>
      )}

      {view === "chat" && showSettings && (
        <SettingsPanel
          settings={settings}
          onChange={setSettings}
          limits={{
            minT: health?.limits?.min_temperature ?? 0.1,
            maxT: health?.limits?.max_temperature ?? 1.5,
            minTokens: health?.limits?.min_new_tokens ?? 1,
            maxTokens: health?.limits?.max_new_tokens ?? 256,
          }}
        />
      )}

      {view === "chat" && (
        <main className="chat">
          {status && (
            <p className={`status ${healthError || health?.status === "error" ? "status-error" : ""}`} role="status">
              {status}{" "}
              {healthError && (
                <button type="button" className="link" onClick={() => setView("recorded")}>
                  See recorded conversations instead
                </button>
              )}
            </p>
          )}
          <MessageList turns={turns} pending={pending} onExample={(t) => ready && !pending && void send(t)} />
        </main>
      )}

      {view === "chat" && (
      <footer className="footer">
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button type="button" className="link" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}
        <Composer
          disabled={!ready}
          busy={pending}
          maxChars={maxChars}
          onSend={(t) => void send(t)}
          draft={draft}
          setDraft={setDraft}
        />
        <p className="footnote">
          Enter to send · Shift+Enter for a new line · Runs entirely on this computer
          {health?.model ? ` (${health.model.device.toUpperCase()})` : ""} · Conversation is kept only in
          this browser tab.
        </p>
      </footer>
      )}

      {showAbout && <AboutPanel model={health?.model} onClose={() => setShowAbout(false)} />}
    </div>
  );
}

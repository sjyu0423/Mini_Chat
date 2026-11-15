import { useEffect, useRef } from "react";
import type { Turn } from "../chat";
import { formatDuration, stopLabel } from "../chat";

interface Props {
  turns: Turn[];
  pending: boolean;
  onExample: (text: string) => void;
}

const EXAMPLES = [
  "What is Python?",
  "List three fruits.",
  "Why is exercise good for you?",
  "Write a short poem about autumn.",
];

export function MessageList({ turns, pending, onExample }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [turns.length, pending]);

  if (turns.length === 0 && !pending) {
    return (
      <div className="empty">
        <h2>Ask the model something</h2>
        <p>
          It is a tiny model trained from scratch, so expect fluent but often wrong answers. Try
          one of these:
        </p>
        <div className="examples">
          {EXAMPLES.map((e) => (
            <button key={e} type="button" className="chip" onClick={() => onExample(e)}>
              {e}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <ol className="messages" aria-label="Conversation" aria-live="polite">
      {turns.map((t) => (
        <li key={t.id} className={`msg msg-${t.role}`} data-testid={`msg-${t.role}`}>
          <div className="who">{t.role === "user" ? "You" : "Model"}</div>
          <div className="bubble">
            {t.content ? (
              <p className="text">{t.content}</p>
            ) : (
              <p className="text muted">(empty reply)</p>
            )}
            {t.role === "user" && t.failed && <p className="failed">Not answered (request failed)</p>}
            {t.role === "assistant" && <Meta turn={t} />}
          </div>
        </li>
      ))}
      {pending && (
        <li className="msg msg-assistant" data-testid="pending">
          <div className="who">Model</div>
          <div className="bubble">
            <p className="typing" role="status">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
              <span className="sr-only">Generating…</span>
            </p>
          </div>
        </li>
      )}
      <div ref={endRef} />
    </ol>
  );
}

function Meta({ turn }: { turn: Extract<Turn, { role: "assistant" }> }) {
  const s = stopLabel(turn.meta.stop_reason);
  return (
    <div className="meta">
      <span className={`badge badge-${s.tone}`} title={s.title}>
        {s.text}
      </span>
      <span>{turn.meta.new_tokens} tokens</span>
      <span>{formatDuration(turn.meta.duration_ms)}</span>
      <span>{turn.meta.tokens_per_s.toFixed(0)} tok/s</span>
      {turn.meta.dropped_history_messages > 0 && (
        <span title="Older messages did not fit the 512-token context window and were not shown to the model.">
          {turn.meta.dropped_history_messages} older messages not seen
        </span>
      )}
    </div>
  );
}

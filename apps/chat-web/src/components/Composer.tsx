import { useState } from "react";
import type { KeyboardEvent } from "react";

interface Props {
  disabled: boolean;
  busy: boolean;
  maxChars: number;
  onSend: (text: string) => void;
  draft: string;
  setDraft: (text: string) => void;
}

export function Composer({ disabled, busy, maxChars, onSend, draft, setDraft }: Props) {
  const [composing, setComposing] = useState(false);
  const tooLong = draft.length > maxChars;
  const canSend = !disabled && !busy && draft.trim().length > 0 && !tooLong;

  const submit = () => {
    if (canSend) onSend(draft.trim());
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts a newline; IME composition is left alone.
    if (e.key === "Enter" && !e.shiftKey && !composing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <label htmlFor="message" className="sr-only">
        Message
      </label>
      <textarea
        id="message"
        rows={1}
        value={draft}
        placeholder={disabled ? "Waiting for the model to load…" : "Message the model…"}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKeyDown}
        onCompositionStart={() => setComposing(true)}
        onCompositionEnd={() => setComposing(false)}
        disabled={disabled}
        aria-invalid={tooLong}
      />
      <button type="submit" className="send" disabled={!canSend} aria-label="Send message">
        {busy ? "…" : "Send"}
      </button>
      {tooLong && (
        <p className="hint error" role="alert">
          Message is {draft.length} characters; the limit is {maxChars}.
        </p>
      )}
    </form>
  );
}

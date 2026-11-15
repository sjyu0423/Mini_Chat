import { useState } from "react";
import { stopLabel } from "../chat";
import type { StopReason } from "../api";
import { recorded } from "../showcase/content";

const CATEGORY: Record<string, string> = {
  fact: "fact",
  qa: "fact",
  instruction: "instruction",
  entity: "names",
  multiturn: "multi-turn",
  open: "open question",
};

export function RecordedConversations({ intro }: { intro: string }) {
  const [compare, setCompare] = useState(false);
  return (
    <section className="recorded" aria-label="Recorded conversations">
      <div className="recorded-head">
        <p className="recorded-intro">{intro}</p>
        <label className="toggle">
          <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
          Compare with the 30M research model
        </label>
      </div>
      <ol className="messages">
        {recorded.map((c) => (
          <li key={c.id} className="recorded-item" data-testid="recorded-item">
            <div className="recorded-tag">{CATEGORY[c.category] ?? c.category}</div>
            {c.turns.map((t, i) => (
              <div key={i} className={`msg msg-${t.role}`}>
                <div className="who">{t.role === "user" ? "User" : "Model (given)"}</div>
                <div className="bubble">
                  <p className="text">{t.content}</p>
                </div>
              </div>
            ))}
            <Reply label="Model · 14M (default)" r={c.replies.m5} />
            {compare && <Reply label="Model · 30M (research)" r={c.replies.m6_3} />}
          </li>
        ))}
      </ol>
    </section>
  );
}

function Reply({ label, r }: { label: string; r: (typeof recorded)[number]["replies"]["m5"] }) {
  const s = stopLabel(r.stop_reason as StopReason);
  return (
    <div className="msg msg-assistant">
      <div className="who">{label}</div>
      <div className="bubble">
        <p className="text">{r.reply || "(empty reply)"}</p>
        <div className="meta">
          <span className={`badge badge-${s.tone}`} title={s.title}>
            {s.text}
          </span>
          <span>{r.new_tokens} tokens</span>
          {r.content_check !== null && (
            <span className={r.content_check ? "check-pass" : "check-fail"}>
              {r.content_check ? "passed its content check" : "wrong / failed its content check"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

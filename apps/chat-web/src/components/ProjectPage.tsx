import { fact } from "../showcase/content";

/** Recruiter-friendly project summary. Every number comes from docs/showcase/facts.json. */
export function ProjectPage() {
  return (
    <article className="project" aria-label="About the project">
      <section>
        <h2>What this is</h2>
        <p>
          A GPT-style language model written from scratch in PyTorch — model, tokenizer, data
          pipelines, training, chat fine-tuning and this app — and trained from random weights on a
          16 GB Mac mini. No pretrained weights and no external AI service are used anywhere.
        </p>
      </section>

      <section>
        <h2>How it was built</h2>
        <dl className="facts">
          <div>
            <dt>{fact("m5_parameters")}</dt>
            <dd>parameters: a 6-layer decoder-only Transformer with causal self-attention</dd>
          </div>
          <div>
            <dt>{fact("bpe_merges")}</dt>
            <dd>byte-pair merges learned by my own tokenizer, plus reserved chat tokens</dd>
          </div>
          <div>
            <dt>{fact("pretraining_tokens")}</dt>
            <dd>
              pretraining tokens of educational web and textbook text, each seen once (
              {fact("m41_minutes")} minutes)
            </dd>
          </div>
          <div>
            <dt>{fact("sft_conversations")}</dt>
            <dd>
              human-written conversations for chat fine-tuning ({fact("sft_minutes")} minutes);
              replies that end on their own: {fact("eot_base_vs_sft")}
            </dd>
          </div>
        </dl>
      </section>

      <section>
        <h2>Key experiment: 14M vs 30M parameters</h2>
        <p>
          The model was doubled to {fact("m63_parameters")} parameters with identical data,
          tokenizer, training order and fine-tuning, and compared under rules written down before
          training.
        </p>
        <table className="compare">
          <thead>
            <tr>
              <th scope="col">measure</th>
              <th scope="col">14M → 30M</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>held-out web text, bits per byte (lower is better)</td>
              <td>{fact("heldout_fineweb")}</td>
            </tr>
            <tr>
              <td>chat fine-tuning validation loss</td>
              <td>{fact("sft_val_loss")}</td>
            </tr>
            <tr>
              <td>strict correct answers, 75 prompts</td>
              <td>{fact("strict_success")}</td>
            </tr>
            <tr>
              <td>difference (95% interval, sign test)</td>
              <td>{fact("strict_success_ci")}</td>
            </tr>
            <tr>
              <td>correct in a 20-item blind review</td>
              <td>{fact("blind_review")}</td>
            </tr>
          </tbody>
        </table>
        <p className="finding">
          Doubling model parameters improved language-modeling loss and fluency, but did not produce
          a statistically clear improvement in factual or conversational correctness — so the
          smaller, faster 14M model stays the default.
        </p>
      </section>

      <section>
        <h2>Engineering on Apple Silicon</h2>
        <p>
          Variable-length chat batches made the GPU memory cache balloon ({fact("mps_memory_blowup")}
          ). Shape bucketing, cache release and a memory guard fixed it; the guard later stopped a
          larger run safely ({fact("mps_guard_abort")}) instead of letting the machine swap.
        </p>
      </section>

      <section>
        <h2>Honest limitations</h2>
        <ul>
          <li>Most answers to simple factual questions are wrong.</li>
          <li>Memory of earlier turns is shallow; it often repeats a previous answer.</li>
          <li>English only; small evaluation suites; one training seed per model.</li>
        </ul>
      </section>

      <section>
        <h2>Run it yourself</h2>
        <pre className="code">
          {`uv pip install --python .venv/bin/python -e ".[serve]"
npm --prefix apps/chat-web install && npm --prefix apps/chat-web run build
.venv/bin/python scripts/serve_chat.py   # http://127.0.0.1:8000`}
        </pre>
        <p className="muted-small">
          Median reply latency locally: {fact("app_latency")}. Full reports, data provenance and
          tests are in the repository (<code>docs/showcase/</code> and <code>docs/reports/</code>).
        </p>
      </section>
    </article>
  );
}

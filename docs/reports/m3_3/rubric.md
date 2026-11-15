# M3.3 narrative rubric (pre-registered)

This rubric was written before any M3.3 story was generated. It applies to every
story in the blind packet. The rater sees only the prompt, the continuation and an
anonymous id (S001, …). Each prompt in `entity_prompts.json` lists the facts it
establishes; the rater checks the continuation against them.

Score each story on five graded criteria (0, 1 or 2) and one binary flag.

| criterion | 2 | 1 | 0 |
|---|---|---|---|
| **G: grammar** | no grammatical errors | 1–2 minor slips (agreement, article, tense) | repeated or sentence-breaking errors |
| **E: entity consistency** (names, species, ownership, pronouns, relationships from the prompt and introduced in the story) | every reference stays correct | one lapse (a pronoun or name slip, an entity silently dropped, a vague owner) | an entity is renamed, merged with another or swapped, or ownership or a relationship is reassigned |
| **C: coherence** (cause and effect, event order, actions follow from the setup) | events follow plausibly from the prompt and from each other | weak or partly arbitrary links | non-sequiturs, impossible events, or the order of events is broken |
| **N: ending** | resolves the situation the story set up | generic or abrupt ending | no ending (cut off) or an ending that contradicts the story |
| **R: repetition** | no noticeable repetition | some repeated phrases | loops or the same sentence repeated several times |
| **X: contradiction** (binary) | 1 if the continuation explicitly contradicts a prompt fact or a fact it stated itself; otherwise 0 | | |

**Notes for the rater:**

- E and X overlap by design. E grades the overall tracking. X records whether there
  is at least one explicit contradiction: X is an event count, E is a graded quality.
- Do not reward length or style. A short, correct story can score 2 everywhere.
- If a story stops at the 300-token limit, it gets N = 0 (cut off).
- Story total = G + E + C + N + R (0–10). X is reported separately as a rate.

/**
 * Verbatim model outputs copied from milestone reports. Not rewritten.
 */

export interface ExperimentSample {
  id: string
  title: string
  model: string
  experiment: string
  source: string
  kind: 'limitation' | 'progress' | 'contrast'
  prompt: string
  output: string
  note: string
  unedited: true
}

export const experimentSamples: readonly ExperimentSample[] = [
  {
    id: 'm2-romeo',
    title: 'Character-level continuation',
    model: 'M2 best (10.77M, step 1,750)',
    experiment: 'Tiny Shakespeare pretraining',
    source: 'docs/reports/m2/samples.md',
    kind: 'progress',
    prompt: 'ROMEO:\\n',
    output:
      "Then are comes a man or like to be a whippiness, a\npoint of the steel, a main of my country: a\ngentleman of brave a scandem-poor friend, a\nshall though the mistress of the gentleman: he\nwill place to our reasons sink him.",
    note: 'Unedited sample at temperature 0.7, top-k 40. Looks like play dialogue; the words are often nonsense. Illustrates form without reliable meaning.',
    unedited: true,
  },
  {
    id: 'm3-lily',
    title: 'TinyStories fluency',
    model: 'M3.1 intermediate (12.4M, step 2,400)',
    experiment: 'TinyStories BPE training',
    source: 'docs/reports/m3_1/stories.md',
    kind: 'progress',
    prompt: 'Once upon a time, there was a little girl named Lily.',
    output:
      'She had a big, pretty veil. Lily liked her veil very much.\nOne day, Lily went to the park with her mom. She brought her veil to the park. Lily saw a boy named Tom. Tom asked, "Do you want to play with me?" Lily said, "Yes, yes!" They played all day, laughing and having fun.',
    note: 'Unedited continuation; ended with <|endoftext|>. Readable story shape appears after BPE + TinyStories training.',
    unedited: true,
  },
  {
    id: 'm3-entity',
    title: 'Entity consistency still breaks',
    model: 'M3.3 (27.6M)',
    experiment: 'TinyStories scaling / ownership probe',
    source: 'docs/reports/m3_3/eval_m3_3.json (prompt own1, model C)',
    kind: 'limitation',
    prompt: 'Lily had a red ball. Ben had a blue boat. One day, Ben lost his boat.',
    output:
      'He was very sad. Lily wanted to help him. She thought of a plan. She would use the blue boat to move the blue boat.\nLily went to the blue boat with the blue boat. She used her yellow boat to push the blue boat. The blue boat moved the blue boat across the blue water. Ben was happy. He thought it was his boat.\nBut then, the blue boat moved! The blue boat was not a boat at all. It was a big, friendly, friendly bear. The bear said, "Hello, Lily. I am Cloard. Do you want to be my friends?" Lily and Ben said, "Yes, Cloardy. We like to be friends." They all played together and had a fun day.',
    note: 'Full unedited sampled continuation (seed 1). Fluency improved with scale; ownership and identity did not become reliable.',
    unedited: true,
  },
  {
    id: 'm4-loop',
    title: 'Base model loops on web-style text',
    model: 'M4.1 final (13.99M, base)',
    experiment: 'General-language pretraining',
    source: 'docs/reports/m4_1/samples_excerpt.md (cont13)',
    kind: 'limitation',
    prompt: 'Honeybees communicate with each other by',
    output:
      'sharing information, but they can do what they can do to help them. They can then use a smartphone to see if someone can use your hand to help their friend. They can also be using a smartphone to create a new device.\nLuckily, you would like to help your friend and friend to help her friend to help her friend.\nHoneybeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
    note: 'Full unedited continuation from the report excerpt (seed 11). Shows general English plus degeneration — why chat fine-tuning and decoding controls were needed.',
    unedited: true,
  },
  {
    id: 'm5-planet-base',
    title: 'Base chat: never stops',
    model: 'M4.1 base (13.99M)',
    experiment: 'Conversational evaluation before SFT',
    source: 'docs/reports/m5/transcripts_excerpt.md (qa01, greedy)',
    kind: 'limitation',
    prompt: 'What is the largest planet in our solar system?',
    output:
      'This is the first planet in our solar system. It is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is a planet that is',
    note: 'Unedited greedy transcript; stopped at max_tokens (200). Content check failed.',
    unedited: true,
  },
  {
    id: 'm5-planet-sft',
    title: 'After SFT: stops, still wrong',
    model: 'M5 SFT (14M)',
    experiment: 'Conversational fine-tuning',
    source: 'docs/reports/m5/transcripts_excerpt.md (qa01, greedy)',
    kind: 'contrast',
    prompt: 'What is the largest planet in our solar system?',
    output: 'The largest planet in our solar system is the largest planet in our solar system.',
    note: 'Unedited greedy transcript; stop_reason end_of_turn (16 tokens). Format improved; the answer remains circular and incorrect.',
    unedited: true,
  },
  {
    id: 'm63-france-14m',
    title: '14M factual miss',
    model: 'M5 SFT (14M)',
    experiment: 'Pre-registered conversational scaling',
    source: 'docs/reports/m6_3/transcripts.md (diag_v1/df03)',
    kind: 'limitation',
    prompt: 'What is the capital of France?',
    output: 'The capital of France is Vienna.',
    note: 'Unedited production decoding (T 0.7, top-k 40, repetition penalty 1.2). Blind grade: fluent_wrong.',
    unedited: true,
  },
  {
    id: 'm63-france-30m',
    title: '30M factual miss',
    model: 'M6.3 SFT (29.7M)',
    experiment: 'Pre-registered conversational scaling',
    source: 'docs/reports/m6_3/transcripts.md (diag_v1/df03)',
    kind: 'limitation',
    prompt: 'What is the capital of France?',
    output: 'The capital of France is Sir Arthur and',
    note: 'Unedited matched decoding. Blind grade: malformed. Blind preference vs 14M: tie — both wrong.',
    unedited: true,
  },
] as const

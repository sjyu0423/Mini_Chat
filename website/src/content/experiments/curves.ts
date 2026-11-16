/** Curated recorded eval points only. Regenerated from docs/reports curve_points.json and analysis.json. */

export interface BpbPoint {
  step: number
  tokens_seen: number
  val_bpb: number
  is_best: boolean
}

export interface LossPoint {
  step: number
  val_loss: number
}

export interface SftPoint {
  step: number
  epoch: number
  end_recall: number
  loss: number
}

export const curveSources = {
  "m4_1": "docs/reports/m4_1/curve_points.json",
  "m6_2": "docs/reports/m6_2/curve_points.json",
  "m2": "docs/reports/m2/analysis.json",
  "m5": "docs/reports/m5/curve_points.json",
  "note": "Recorded eval points only; no interpolated values."
} as const

export const generalPretrainingBpb = {
  unit: "bits per byte (validation, exact)",
  compatible: "Same BPE-8192 tokenizer, data mix, and single-pass order.",
  m4_1_14m: [
  {
    "step": 0,
    "tokens_seen": 0,
    "val_bpb": 3.3043,
    "is_best": true
  },
  {
    "step": 800,
    "tokens_seen": 13107200,
    "val_bpb": 1.799,
    "is_best": true
  },
  {
    "step": 1600,
    "tokens_seen": 26214400,
    "val_bpb": 1.5899,
    "is_best": true
  },
  {
    "step": 2400,
    "tokens_seen": 39321600,
    "val_bpb": 1.4748,
    "is_best": true
  },
  {
    "step": 3200,
    "tokens_seen": 52428800,
    "val_bpb": 1.4172,
    "is_best": true
  },
  {
    "step": 4000,
    "tokens_seen": 65536000,
    "val_bpb": 1.3801,
    "is_best": true
  },
  {
    "step": 4800,
    "tokens_seen": 78643200,
    "val_bpb": 1.3505,
    "is_best": true
  },
  {
    "step": 5600,
    "tokens_seen": 91750400,
    "val_bpb": 1.3264,
    "is_best": true
  },
  {
    "step": 6400,
    "tokens_seen": 104857600,
    "val_bpb": 1.3063,
    "is_best": true
  },
  {
    "step": 7200,
    "tokens_seen": 117964800,
    "val_bpb": 1.2909,
    "is_best": true
  },
  {
    "step": 8000,
    "tokens_seen": 131072000,
    "val_bpb": 1.2784,
    "is_best": true
  },
  {
    "step": 8800,
    "tokens_seen": 144179200,
    "val_bpb": 1.2692,
    "is_best": true
  },
  {
    "step": 9600,
    "tokens_seen": 157286400,
    "val_bpb": 1.2641,
    "is_best": true
  }
] as readonly BpbPoint[],
  m6_2_30m: [
  {
    "step": 0,
    "tokens_seen": 0,
    "val_bpb": 3.316,
    "is_best": true
  },
  {
    "step": 800,
    "tokens_seen": 13107200,
    "val_bpb": 1.7588,
    "is_best": true
  },
  {
    "step": 1600,
    "tokens_seen": 26214400,
    "val_bpb": 1.5253,
    "is_best": true
  },
  {
    "step": 2400,
    "tokens_seen": 39321600,
    "val_bpb": 1.4204,
    "is_best": true
  },
  {
    "step": 3200,
    "tokens_seen": 52428800,
    "val_bpb": 1.3673,
    "is_best": true
  },
  {
    "step": 4000,
    "tokens_seen": 65536000,
    "val_bpb": 1.3276,
    "is_best": true
  },
  {
    "step": 4800,
    "tokens_seen": 78643200,
    "val_bpb": 1.2974,
    "is_best": true
  },
  {
    "step": 5600,
    "tokens_seen": 91750400,
    "val_bpb": 1.2717,
    "is_best": true
  },
  {
    "step": 6400,
    "tokens_seen": 104857600,
    "val_bpb": 1.2505,
    "is_best": true
  },
  {
    "step": 7200,
    "tokens_seen": 117964800,
    "val_bpb": 1.2329,
    "is_best": true
  },
  {
    "step": 8000,
    "tokens_seen": 131072000,
    "val_bpb": 1.2203,
    "is_best": true
  },
  {
    "step": 8800,
    "tokens_seen": 144179200,
    "val_bpb": 1.2101,
    "is_best": true
  },
  {
    "step": 9600,
    "tokens_seen": 157286400,
    "val_bpb": 1.2042,
    "is_best": true
  }
] as readonly BpbPoint[],
} as const

export const m2ValLoss = {
  unit: "nats per character (validation, exact)",
  compatible: "Character-level Tiny Shakespeare only; not comparable to BPE bits/byte.",
  points: [
  {
    "step": 0,
    "val_loss": 4.125
  },
  {
    "step": 250,
    "val_loss": 2.0627
  },
  {
    "step": 500,
    "val_loss": 1.7042
  },
  {
    "step": 750,
    "val_loss": 1.5698
  },
  {
    "step": 1000,
    "val_loss": 1.5237
  },
  {
    "step": 1250,
    "val_loss": 1.4898
  },
  {
    "step": 1500,
    "val_loss": 1.4826
  },
  {
    "step": 1750,
    "val_loss": 1.4736
  },
  {
    "step": 2000,
    "val_loss": 1.4869
  },
  {
    "step": 2250,
    "val_loss": 1.4965
  },
  {
    "step": 2500,
    "val_loss": 1.513
  },
  {
    "step": 2750,
    "val_loss": 1.5291
  },
  {
    "step": 3000,
    "val_loss": 1.5512
  },
  {
    "step": 3250,
    "val_loss": 1.5794
  },
  {
    "step": 3500,
    "val_loss": 1.6123
  },
  {
    "step": 3750,
    "val_loss": 1.6393
  },
  {
    "step": 4000,
    "val_loss": 1.6646
  },
  {
    "step": 4250,
    "val_loss": 1.6889
  },
  {
    "step": 4500,
    "val_loss": 1.7048
  },
  {
    "step": 4750,
    "val_loss": 1.7183
  },
  {
    "step": 5000,
    "val_loss": 1.7402
  }
] as readonly LossPoint[],
  best: {
  "step": 1750,
  "val_loss": 1.4736
} as { readonly step: number; readonly val_loss: number },
} as const

export const m5SftEndRecall = {
  unit: "end-token recall on the SFT validation stream (training metric)",
  note: "Distinct from the 97.5% chat-suite natural end-of-turn rate.",
  points: [
  {
    "step": 0,
    "epoch": 0.0,
    "end_recall": 0.0,
    "loss": 3.4382
  },
  {
    "step": 250,
    "epoch": 0.23786869647954328,
    "end_recall": 0.4978,
    "loss": 3.159
  },
  {
    "step": 500,
    "epoch": 0.47573739295908657,
    "end_recall": 0.3704,
    "loss": 3.1389
  },
  {
    "step": 750,
    "epoch": 0.7136060894386299,
    "end_recall": 0.7048,
    "loss": 3.1254
  },
  {
    "step": 1000,
    "epoch": 0.9514747859181731,
    "end_recall": 0.5763,
    "loss": 3.1168
  },
  {
    "step": 1250,
    "epoch": 1.1893434823977165,
    "end_recall": 0.6808,
    "loss": 3.1182
  },
  {
    "step": 1500,
    "epoch": 1.4272121788772598,
    "end_recall": 0.5839,
    "loss": 3.1139
  },
  {
    "step": 1750,
    "epoch": 1.665080875356803,
    "end_recall": 0.5218,
    "loss": 3.0977
  },
  {
    "step": 2000,
    "epoch": 1.9029495718363463,
    "end_recall": 0.6275,
    "loss": 3.0861
  },
  {
    "step": 2250,
    "epoch": 2.14081826831589,
    "end_recall": 0.5861,
    "loss": 3.0937
  },
  {
    "step": 2500,
    "epoch": 2.378686964795433,
    "end_recall": 0.439,
    "loss": 3.0915
  },
  {
    "step": 2750,
    "epoch": 2.616555661274976,
    "end_recall": 0.5621,
    "loss": 3.0854
  },
  {
    "step": 3000,
    "epoch": 2.8544243577545196,
    "end_recall": 0.5839,
    "loss": 3.0893
  },
  {
    "step": 3153,
    "epoch": 3.0,
    "end_recall": 0.6166,
    "loss": 3.0864
  }
] as readonly SftPoint[],
} as const

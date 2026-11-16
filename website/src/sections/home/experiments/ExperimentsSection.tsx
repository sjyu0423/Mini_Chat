import { journeyConclusion } from '@/content/experiments/phases'
import {
  generalPretrainingBpb,
  m2ValLoss,
} from '@/content/experiments/curves'
import { Container } from '@/components/ui/Container'
import { Section } from '@/components/ui/Section'
import { SectionHeading } from '@/components/ui/SectionHeading'

import { JourneyTimeline } from './JourneyTimeline'
import { LineChart } from './LineChart'
import { SamplesPanel } from './SamplesPanel'
import { ScalingCompare } from './ScalingCompare'

function formatTokens(value: number): string {
  if (value >= 1_000_000) {
    const millions = value / 1_000_000
    return `${millions.toFixed(value % 1_000_000 === 0 ? 0 : 1)}M`
  }
  if (value >= 1_000) return `${String(Math.round(value / 1_000))}k`
  return String(value)
}

export function ExperimentsSection() {
  return (
    <Section id="experiments" labelledBy="experiments-title" divider>
      <Container>
        <SectionHeading
          id="experiments-title"
          eyebrow="Experiments"
          title="What training revealed"
          description="A five-phase journey from a character-level Shakespeare model to a fine-tuned chatbot — with recorded curves, a controlled 14M vs 30M comparison, and unedited samples."
        />

        <JourneyTimeline />

        <p className="mt-10 max-w-reading border-l-2 border-accent/50 pl-5 text-lede text-pretty text-fg">
          {journeyConclusion}
        </p>

        <div className="mt-16 space-y-10">
          <h3 className="text-headline text-fg">Recorded training curves</h3>
          <p className="max-w-reading text-body text-pretty text-fg-muted">
            Points are logged evaluation measurements only — no smoothing and no invented
            intermediates. Bits-per-byte charts compare models that share the same tokenizer and
            data; the Shakespeare curve uses a different unit and stays separate.
          </p>

          <LineChart
            title="General pretraining — validation bits per byte"
            description="M4.1 (14M) and M6.2 (30M) on identical FineWeb-Edu / Cosmopedia data, BPE-8192, and single-pass order. Lower is better."
            xLabel="Token presentations"
            yLabel="Val bits/byte"
            formatX={formatTokens}
            formatY={(v) => v.toFixed(2)}
            source="docs/reports/m4_1/curve_points.json · docs/reports/m6_2/curve_points.json"
            series={[
              {
                id: 'm41',
                label: '14M (M4.1)',
                className: 'text-accent-strong',
                points: generalPretrainingBpb.m4_1_14m.map((p) => ({
                  x: p.tokens_seen,
                  y: p.val_bpb,
                })),
              },
              {
                id: 'm62',
                label: '30M (M6.2)',
                className: 'text-fg',
                dasharray: '6 5',
                points: generalPretrainingBpb.m6_2_30m.map((p) => ({
                  x: p.tokens_seen,
                  y: p.val_bpb,
                })),
              },
            ]}
          />

          <LineChart
            title="Character-level Shakespeare — validation loss"
            description={`${m2ValLoss.compatible} Best checkpoint at step ${String(m2ValLoss.best.step)} (val ${String(m2ValLoss.best.val_loss)}), then overfitting.`}
            xLabel="Training step"
            yLabel="Val loss (nats/char)"
            formatX={(v) => String(v)}
            formatY={(v) => v.toFixed(2)}
            source="docs/reports/m2/analysis.json (emergence)"
            series={[
              {
                id: 'm2',
                label: 'M2 gpt_10m (~10.77M)',
                className: 'text-accent',
                points: m2ValLoss.points.map((p) => ({ x: p.step, y: p.val_loss })),
              },
            ]}
          />
        </div>

        <div className="mt-16">
          <ScalingCompare />
        </div>

        <div className="mt-16">
          <SamplesPanel />
        </div>
      </Container>
    </Section>
  )
}

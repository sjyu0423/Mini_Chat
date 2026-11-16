import { engineeringIntro } from '@/content/engineering/modules'
import { Container } from '@/components/ui/Container'
import { Section } from '@/components/ui/Section'
import { SectionHeading } from '@/components/ui/SectionHeading'

import { EngineeringDocs } from './EngineeringDocs'
import { EngineeringModules } from './EngineeringModules'
import { RecordedDemo } from './RecordedDemo'

export function EngineeringSection() {
  return (
    <Section id="engineering" labelledBy="engineering-title" divider>
      <Container>
        <SectionHeading
          id="engineering-title"
          eyebrow="Engineering"
          title="Engineering a GPT from scratch"
          description={engineeringIntro}
        />

        <div className="mt-14">
          <EngineeringModules />
        </div>

        <div className="mt-16">
          <RecordedDemo />
        </div>

        <div className="mt-16">
          <EngineeringDocs />
        </div>
      </Container>
    </Section>
  )
}

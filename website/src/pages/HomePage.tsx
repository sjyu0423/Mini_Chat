import { SiteLayout } from '@/components/layout/SiteLayout'
import { ArchitectureSection } from '@/sections/home/architecture/ArchitectureSection'
import { EngineeringSection } from '@/sections/home/engineering/EngineeringSection'
import { ExperimentsSection } from '@/sections/home/experiments/ExperimentsSection'
import { Hero } from '@/sections/home/Hero'
import { Highlights } from '@/sections/home/Highlights'
import { Motivation } from '@/sections/home/Motivation'
import { StatusRoadmap } from '@/sections/home/StatusRoadmap'
import { WhatIBuilt } from '@/sections/home/WhatIBuilt'

export function HomePage() {
  return (
    <SiteLayout>
      <Hero />
      <div id="overview">
        <Motivation />
        <WhatIBuilt />
        <Highlights />
        <StatusRoadmap />
      </div>
      <ArchitectureSection />
      <ExperimentsSection />
      <EngineeringSection />
    </SiteLayout>
  )
}

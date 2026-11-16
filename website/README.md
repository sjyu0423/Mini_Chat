# Mini_Chat website

Static project website for Mini_Chat, a GPT-style language model built and
trained from scratch. Intended GitHub Pages URL:

`https://sjyu0423.github.io/Mini_Chat/`

The site is fully static: no backend, no database, no authentication, and no
external AI APIs. It is independent of the Python/ML code in the repository
root; nothing here imports, runs, or modifies the training pipeline. The
recorded demo never calls an inference API.

## Stack

| Tool | Version | Purpose |
|---|---|---|
| React | 19.3 | UI |
| TypeScript | 6.0 | strict type checking |
| Vite | 8.3 | dev server and production bundler |
| Tailwind CSS | 4.3 (via `@tailwindcss/vite`) | styling |
| Lucide React | 1.72 | icons |
| IBM Plex Sans (variable) / IBM Plex Mono | Fontsource 5.3 | self-hosted typefaces (latin + latin-ext) |
| ESLint | 10 + `typescript-eslint` 8 (strict, type-checked) | linting |

TypeScript is held at `~6.0` because `typescript-eslint` does not yet support
TypeScript 7. Node.js `^20.19.0` or `>=22.13.0` is required (Vite 8 and
ESLint 10 engine ranges).

## Site architecture

Single-page homepage (`HomePage`) under Vite base `/Mini_Chat/`:

| Section | Content |
|---|---|
| Overview | Hero, motivation, what was built, verified highlights, status |
| Architecture | Five-step 14M BPE walkthrough, Under the hood, historical M1 aside |
| Experiments | Training journey, SVG loss curves, 14M vs 30M, unedited samples |
| Engineering | Deep-dive modules, recorded M7 demo, documentation links |

Evidence for numbers lives in `src/content/facts.ts` (pinned to commit
`8101752` on `v1.0-release`).

## Directory structure

```
website/
  index.html                 SEO / Open Graph entry
  public/                    favicon, og-image, robots.txt, sitemap.xml
  docs/w6_release_review.md  W6 release review notes
  src/
    main.tsx                 React root + latin font CSS
    App.tsx
    config/site.ts           name, URLs, navigation
    content/
      facts.ts               evidence-backed claims
      bpeExample.ts          verified BPE walkthrough fixture
      experiments/           W4 curves, phases, scaling, samples
      engineering/           W5 modules, docs links, recorded demo
    pages/HomePage.tsx
    sections/home/           overview + architecture/experiments/engineering
    components/              layout + UI primitives
    styles/                  design tokens + latin font faces
  vite.config.ts             base: '/Mini_Chat/'
```

Import alias: `@/` → `src/`.

## Local development

```bash
cd website
npm install
npm run dev          # http://localhost:5173/Mini_Chat/
npm run typecheck
npm run lint
```

## Production build

```bash
cd website
npm run build        # type-check + vite build → dist/
npm run preview      # http://localhost:4173/Mini_Chat/
npm run check        # typecheck + lint + build
```

`dist/` is git-ignored. Open `/Mini_Chat/` (not `/`); the root path 404s by
design, matching GitHub Pages project hosting.

## GitHub Pages deployment

Workflow: `.github/workflows/deploy-website.yml`

- Trigger: **manual only** (`workflow_dispatch`)
- Steps: checkout → `npm ci` in `website/` → `npm run check` → upload
  `website/dist` → deploy with `actions/deploy-pages`
- Permissions: `contents: read`, `pages: write`, `id-token: write`

### One-time repository setup (manual)

1. Settings → Pages → Build and deployment → Source: **GitHub Actions**
2. Confirm the site will publish at `https://sjyu0423.github.io/Mini_Chat/`
3. From the Actions tab, run **Deploy website** after the website commit is on
   the branch you want to publish (typically `v1.0-release`)

Do not enable automatic deploy-on-push until you explicitly want that.

Vite `base` must remain `/Mini_Chat/` while the repository name is `Mini_Chat`.

## Experiment and demo data sources

| Content | Source |
|---|---|
| Homepage / architecture metrics | `docs/showcase/facts.json` via `src/content/facts.ts` |
| Training curves | `docs/reports/m4_1/curve_points.json`, `m6_2/curve_points.json`, M2 analysis |
| Scaling table | M6.2 / M6.3 reports |
| Recorded demo (8 chats) | `apps/chat-web/src/showcase/data.json` → `src/content/engineering/recorded.json` + `recordedDemo.ts` |
| BPE walkthrough ids | Release tokenizer fingerprint `e6be77cc…32ce91c8` |

Recorded replies are verbatim. Decoding metadata: temperature 0.7, top-k 40,
repetition penalty 1.2, seed 11. M5 = 13,989,888 params; M6.3 = 29,676,544.

## Known limitations

- This site does **not** host live model inference.
- Answers shown in the demo may be wrong or nonsensical; fluency ≠ correctness.
- Dark theme only (no light mode).
- Social preview (`og-image.png`) is a static branded card, not a live screenshot.
- GitHub Pages must be enabled manually before the workflow can publish.

## Milestones

| Milestone | Scope | State |
|---|---|---|
| W0–W2 | Foundation, design system, homepage | done |
| W3–W3.5 | Architecture walkthrough for the 14M BPE model | done |
| W4 | Experiments: journey, curves, scaling, samples | done |
| W5 | Engineering deep dive and recorded demo | done |
| W6 | Production polish and GitHub Pages preparation | done (awaiting publish approval) |

## Release checklist

- [ ] Review `docs/w6_release_review.md`
- [ ] Approve staging of `website/`, `.github/workflows/deploy-website.yml`, and `.gitignore`
- [ ] Commit on `v1.0-release` (explicit approval required)
- [ ] Push to `origin` (explicit approval required)
- [ ] Enable GitHub Pages → Source: GitHub Actions
- [ ] Run **Deploy website** workflow (manual)
- [ ] Verify `https://sjyu0423.github.io/Mini_Chat/`
- [ ] Confirm recorded demo works offline and makes no `/api/chat` requests
